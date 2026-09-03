from __future__ import annotations

import argparse
import gc
import json
import os
import time
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np

from climate_rag.bm25 import BM25Index
from climate_rag.dense import SentenceTransformerEncoder
from climate_rag.evaluation_protocol import (
    audit_training_serving_contracts,
    stable_id_sha256,
)
from climate_rag.fusion import (
    DEFAULT_FEATURES,
    LightGBMLambdaMART,
    build_candidate_features,
    reciprocal_rank_fusion,
)
from climate_rag.io import (
    iter_evidence,
    load_claims,
    read_json,
    write_json,
    write_jsonl,
)
from climate_rag.metrics import evaluate_predictions
from climate_rag.models import Claim, EvidenceDocument, Prediction, RankedDocument
from climate_rag.public_v2 import file_sha256, load_public_v2_protocol, tree_sha256
from climate_rag.public_v2_runtime import (
    build_faiss_index,
    current_peak_gpu_bytes,
    decisive_claims,
    percentile,
    predictions_from_rows,
    score_dense_index,
)
from climate_rag.representation_eval import evaluate_representation_pair
from climate_rag.rerank import Qwen3CausalLMReranker, weighted_rank_fuse


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the frozen adapted/HNSW/RRF/LambdaMART/Qwen4B comparison."
    )
    parser.add_argument("--protocol", required=True)
    parser.add_argument("--prepared-dir", required=True)
    parser.add_argument("--selection-dir", required=True)
    parser.add_argument("--base-dir", required=True)
    parser.add_argument("--frozen-config", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch-size", type=int, default=32)
    return parser.parse_args()


def _dense_rows(
    claim_ids: list[str],
    query_vectors: np.ndarray,
    index: Any,
    documents: list[EvidenceDocument],
    *,
    width: int,
) -> tuple[dict[str, list[RankedDocument]], list[float]]:
    result: dict[str, list[RankedDocument]] = {}
    latencies: list[float] = []
    for claim_id, vector in zip(claim_ids, query_vectors, strict=True):
        started = time.perf_counter()
        scores, positions = index.search(vector.reshape(1, -1), width)
        latencies.append((time.perf_counter() - started) * 1000.0)
        result[claim_id] = [
            RankedDocument(
                evidence_id=documents[int(position)].evidence_id,
                score=float(score),
                rank=rank,
                text=documents[int(position)].text,
                source="adapted_dense_hnsw",
            )
            for rank, (score, position) in enumerate(
                zip(scores[0], positions[0], strict=True), start=1
            )
            if int(position) >= 0
        ]
    return result, latencies


def _retrieve_bundles(
    claims: dict[str, Claim],
    encoder: SentenceTransformerEncoder,
    hnsw: Any,
    bm25: BM25Index,
    documents: list[EvidenceDocument],
    *,
    recall_width: int,
    candidate_width: int,
    rrf_k: int,
    batch_size: int,
) -> tuple[dict[str, dict[str, list[RankedDocument]]], dict[str, Any]]:
    claim_ids = sorted(claims)
    encode_started = time.perf_counter()
    query_vectors = encoder.encode_queries(
        [claims[claim_id].text for claim_id in claim_ids], batch_size=batch_size
    )
    encode_seconds = time.perf_counter() - encode_started
    dense, dense_latencies = _dense_rows(
        claim_ids, query_vectors, hnsw, documents, width=recall_width
    )
    bundles: dict[str, dict[str, list[RankedDocument]]] = {}
    bm25_latencies: list[float] = []
    rrf_latencies: list[float] = []
    for claim_id in claim_ids:
        started = time.perf_counter()
        lexical = bm25.search(claims[claim_id].text, recall_width)
        bm25_latencies.append((time.perf_counter() - started) * 1000.0)
        started = time.perf_counter()
        rrf = reciprocal_rank_fusion(
            {"bm25": lexical, "dense": dense[claim_id]},
            k=rrf_k,
            top_k=candidate_width,
        )
        rrf_latencies.append((time.perf_counter() - started) * 1000.0)
        bundles[claim_id] = {
            "bm25": lexical,
            "dense": dense[claim_id],
            "rrf": rrf,
        }
    timings = {
        "query_encode_seconds": encode_seconds,
        "query_encode_per_second": len(claims) / max(encode_seconds, 1e-12),
        "hnsw_search_p50_ms": percentile(dense_latencies, 50),
        "hnsw_search_p95_ms": percentile(dense_latencies, 95),
        "bm25_search_p50_ms": percentile(bm25_latencies, 50),
        "bm25_search_p95_ms": percentile(bm25_latencies, 95),
        "rrf_p50_ms": percentile(rrf_latencies, 50),
        "rrf_p95_ms": percentile(rrf_latencies, 95),
    }
    return bundles, timings


def _source(row_id: str, bundle: dict[str, list[RankedDocument]]) -> str:
    lexical = {row.evidence_id for row in bundle["bm25"]}
    dense = {row.evidence_id for row in bundle["dense"]}
    if row_id in lexical and row_id in dense:
        return "both"
    return "bm25_only" if row_id in lexical else "dense_only"


def _feature_row(
    query: str,
    evidence_id: str,
    bundle: dict[str, list[RankedDocument]],
) -> dict[str, float]:
    by_bm25 = {row.evidence_id: row for row in bundle["bm25"]}
    by_dense = {row.evidence_id: row for row in bundle["dense"]}
    by_rrf = {row.evidence_id: row for row in bundle["rrf"]}
    lexical = by_bm25.get(evidence_id)
    dense = by_dense.get(evidence_id)
    rrf = by_rrf[evidence_id]
    return build_candidate_features(
        query,
        rrf.text,
        bm25_score=lexical.score if lexical else 0.0,
        bm25_rank=lexical.rank if lexical else None,
        dense_score=dense.score if dense else 0.0,
        dense_rank=dense.rank if dense else None,
        rrf_score=rrf.score,
        rrf_rank=rrf.rank,
    )


def _rank_ltr(
    query: str,
    bundle: dict[str, list[RankedDocument]],
    ranker: LightGBMLambdaMART,
) -> list[RankedDocument]:
    ids = [row.evidence_id for row in bundle["rrf"]]
    feature_rows = [_feature_row(query, evidence_id, bundle) for evidence_id in ids]
    matrix = np.asarray(
        [[row[name] for name in ranker.feature_names] for row in feature_rows],
        dtype=np.float64,
    )
    scores = ranker.predict(matrix)
    by_id = {row.evidence_id: row for row in bundle["rrf"]}
    ordered = sorted(
        zip(ids, scores, feature_rows, strict=True),
        key=lambda row: (-float(row[1]), row[0]),
    )
    return [
        RankedDocument(
            evidence_id=evidence_id,
            score=float(score),
            rank=rank,
            text=by_id[evidence_id].text,
            source="lambdamart",
            features=features,
        )
        for rank, (evidence_id, score, features) in enumerate(ordered, start=1)
    ]


def _metric_bundle(
    claims: dict[str, Claim], predictions: dict[str, dict[str, Prediction]]
) -> tuple[dict[str, Any], dict[str, list[dict[str, Any]]]]:
    systems: dict[str, Any] = {}
    rows: dict[str, list[dict[str, Any]]] = {}
    for name, system in predictions.items():
        metrics, metric_rows, _ = evaluate_predictions(claims, system, ks=(5, 10))
        systems[name] = metrics
        rows[name] = metric_rows
    return systems, rows


def main() -> int:
    args = parse_args()
    protocol = load_public_v2_protocol(args.protocol)
    frozen = read_json(args.frozen_config)
    if not isinstance(frozen, dict):
        raise TypeError("frozen config must be an object")
    if file_sha256(args.protocol) != str(frozen["protocol_sha256"]):
        raise ValueError("frozen config does not match the protocol")
    output = Path(args.output_dir)
    if output.exists():
        raise FileExistsError(f"output must be unique: {output}")
    output.mkdir(parents=True)
    documents = list(iter_evidence(Path(args.prepared_dir) / "evidence.jsonl"))
    train_all = load_claims(Path(args.selection_dir) / "train-claims.json")
    validation_all = load_claims(Path(args.selection_dir) / "validation-claims.json")
    train = decisive_claims(train_all)
    validation = decisive_claims(validation_all)
    ranking = protocol["downstream_ranking"]
    recall_width = min(int(ranking["recall_width"]), len(documents))
    candidate_width = int(ranking["candidate_width"])
    if candidate_width != int(ranking["reranker_candidate_width"]):
        raise ValueError("LTR and reranker candidate widths must match exactly")
    rrf_k = int(ranking["rrf_k"])
    bm25 = BM25Index.load(Path(args.base_dir) / "bm25.pkl.gz")

    adapter_path = Path(str(frozen["adapter_path"]))
    if tree_sha256(adapter_path) != str(frozen["adapter_sha256"]):
        raise ValueError("selected adapter changed after configuration freeze")
    embedding_path = Path(str(frozen["candidate_embeddings_path"]))
    if file_sha256(embedding_path) != str(frozen["candidate_embeddings_sha256"]):
        raise ValueError("selected embeddings changed after configuration freeze")
    vectors = np.load(embedding_path, mmap_mode="r")
    flat, flat_index_metrics = build_faiss_index(vectors, kind="flat")
    hnsw_path = output / "adapted_hnsw.faiss"
    hnsw, hnsw_index_metrics = build_faiss_index(
        vectors,
        kind="hnsw",
        output_path=hnsw_path,
        hnsw_m=int(ranking["hnsw"]["m"]),
        hnsw_ef_construction=int(ranking["hnsw"]["ef_construction"]),
        hnsw_ef_search=int(ranking["hnsw"]["ef_search"]),
    )
    model = protocol["models"]["dense"]
    encoder = SentenceTransformerEncoder(
        str(model["name"]),
        device=args.device,
        truncate_dim=int(model["dimension"]),
        revision=str(model["revision"]),
        adapter_path=str(adapter_path),
    )
    flat_metrics, _, flat_rows = score_dense_index(
        encoder,
        flat,
        documents,
        validation,
        batch_size=args.batch_size,
        search_width=5,
        final_k=5,
    )
    train_bundles, train_timings = _retrieve_bundles(
        train,
        encoder,
        hnsw,
        bm25,
        documents,
        recall_width=recall_width,
        candidate_width=candidate_width,
        rrf_k=rrf_k,
        batch_size=args.batch_size,
    )
    validation_bundles, serving_timings = _retrieve_bundles(
        validation,
        encoder,
        hnsw,
        bm25,
        documents,
        recall_width=recall_width,
        candidate_width=candidate_width,
        rrf_k=rrf_k,
        batch_size=args.batch_size,
    )

    feature_rows: list[dict[str, Any]] = []
    reachable: list[str] = []
    source_distribution: Counter[str] = Counter()
    all_gold_count = 0
    for claim_id in sorted(train):
        bundle = train_bundles[claim_id]
        candidate_ids = {row.evidence_id for row in bundle["rrf"]}
        supported = [
            evidence_id
            for evidence_id in train[claim_id].evidence_ids
            if evidence_id in candidate_ids
        ]
        all_gold_count += len(train[claim_id].evidence_ids)
        if not supported:
            continue
        reachable.extend(f"{claim_id}\0{evidence_id}" for evidence_id in supported)
        for row in bundle["rrf"]:
            source_distribution[_source(row.evidence_id, bundle)] += 1
            feature_rows.append(
                {
                    "query_id": claim_id,
                    "evidence_id": row.evidence_id,
                    "relevance": 2 if row.evidence_id in supported else 0,
                    "features": _feature_row(
                        train[claim_id].text, row.evidence_id, bundle
                    ),
                }
            )
    if not feature_rows:
        raise ValueError("no candidate-supported LambdaMART groups were generated")
    feature_order = tuple(str(value) for value in ranking["feature_order"])
    if feature_order != DEFAULT_FEATURES:
        raise ValueError("runtime feature order differs from preregistration")
    matrix = np.asarray(
        [
            [float(row["features"][name]) for name in feature_order]
            for row in feature_rows
        ],
        dtype=np.float64,
    )
    labels = np.asarray(
        [float(row["relevance"]) for row in feature_rows], dtype=np.float64
    )
    groups = [str(row["query_id"]) for row in feature_rows]
    train_started = time.perf_counter()
    ranker = LightGBMLambdaMART(feature_order, seed=int(protocol["seed"]))
    ranker.fit(matrix, labels, groups)
    ltr_train_seconds = time.perf_counter() - train_started
    ltr_path = output / "lambdamart_top100.txt"
    ranker.save(ltr_path)
    write_jsonl(output / "ltr_features.jsonl", feature_rows)
    reachable_hash = stable_id_sha256(sorted(reachable))
    training_contract = {
        "schema_version": 1,
        "scope": "retained candidate-supported train groups",
        "candidate_width": candidate_width,
        "feature_names": list(feature_order),
        "reachable_positive_rate": 1.0,
        "reachable_positive_count": len(reachable),
        "reachable_positive_id_sha256": reachable_hash,
        "all_gold_positive_reachability": len(reachable) / max(all_gold_count, 1),
        "positive_policy": "candidate-supported-only",
        "candidate_source_distribution": dict(sorted(source_distribution.items())),
        "feature_rows_sha256": file_sha256(output / "ltr_features.jsonl"),
    }
    write_json(output / "stage_contract.training.json", training_contract)

    serving_distribution: Counter[str] = Counter()
    for bundle in validation_bundles.values():
        for row in bundle["rrf"]:
            serving_distribution[_source(row.evidence_id, bundle)] += 1
    serving_contract = {
        "schema_version": 1,
        "scope": "validation serving candidate union before LambdaMART",
        "candidate_width": candidate_width,
        "feature_names": list(ranker.feature_names),
        "training_reachable_positive_id_sha256": reachable_hash,
        "positive_policy": "candidate-supported-only",
        "candidate_source_distribution": dict(sorted(serving_distribution.items())),
    }
    write_json(output / "stage_contract.serving.json", serving_contract)
    contract_audit = audit_training_serving_contracts(
        training_contract, serving_contract, distribution_tv_limit=0.15
    )
    write_json(output / "stage_contract.audit.json", contract_audit)

    reranker_model = protocol["models"]["reranker"]
    del encoder
    gc.collect()
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except ImportError:
        pass
    reranker_load_started = time.perf_counter()
    reranker = Qwen3CausalLMReranker(
        str(reranker_model["name"]),
        device=args.device,
        max_length=2048,
        batch_size=4,
        dtype=str(reranker_model["dtype"]),
        instruction=str(reranker_model["instruction"]),
        revision=str(reranker_model["revision"]),
    )
    reranker_load_seconds = time.perf_counter() - reranker_load_started

    predictions: dict[str, dict[str, Prediction]] = {
        "bm25": {},
        "adapted_dense_flat": predictions_from_rows(flat_rows),
        "adapted_dense_hnsw": {},
        "rrf": {},
        "top100_lambdamart": {},
        "rrf_qwen3_reranker_4b_fusion": {},
    }
    ltr_latencies: list[float] = []
    reranker_latencies: list[float] = []
    for claim_id in sorted(validation):
        bundle = validation_bundles[claim_id]
        predictions["bm25"][claim_id] = Prediction(
            claim_id, tuple(row.evidence_id for row in bundle["bm25"][:5])
        )
        predictions["adapted_dense_hnsw"][claim_id] = Prediction(
            claim_id, tuple(row.evidence_id for row in bundle["dense"][:5])
        )
        predictions["rrf"][claim_id] = Prediction(
            claim_id, tuple(row.evidence_id for row in bundle["rrf"][:5])
        )
        started = time.perf_counter()
        ltr = _rank_ltr(validation[claim_id].text, bundle, ranker)
        ltr_latencies.append((time.perf_counter() - started) * 1000.0)
        predictions["top100_lambdamart"][claim_id] = Prediction(
            claim_id, tuple(row.evidence_id for row in ltr[:5])
        )
        started = time.perf_counter()
        reranked = reranker.rerank(
            validation[claim_id].text, bundle["rrf"], candidate_width
        )
        fused = weighted_rank_fuse(
            bundle["rrf"],
            reranked,
            5,
            k=int(ranking["reranker_fusion"]["k"]),
            base_weight=float(ranking["reranker_fusion"]["base_weight"]),
            reranker_weight=float(ranking["reranker_fusion"]["reranker_weight"]),
        )
        reranker_latencies.append((time.perf_counter() - started) * 1000.0)
        predictions["rrf_qwen3_reranker_4b_fusion"][claim_id] = Prediction(
            claim_id, tuple(row.evidence_id for row in fused)
        )

    systems, _ = _metric_bundle(validation, predictions)
    paired: dict[str, Any] = {}
    slice_reports: dict[str, Any] = {}
    for name, system in predictions.items():
        if name == "bm25":
            continue
        report, tagged = evaluate_representation_pair(
            validation,
            documents,
            predictions["bm25"],
            system,
            bootstrap_samples=int(protocol["evaluation"]["bootstrap_samples"]),
            seed=int(protocol["seed"]),
        )
        paired[name] = report["paired_bootstrap"]
        slice_reports[name] = report["taxonomy"]
        write_jsonl(output / f"slices_{name}.jsonl", tagged)
    for name, system in predictions.items():
        write_jsonl(
            output / f"predictions_{name}.jsonl",
            (
                {
                    "claim_id": claim_id,
                    "evidence_ids": list(prediction.evidence_ids),
                }
                for claim_id, prediction in sorted(system.items())
            ),
        )
    flat_top5 = {
        claim_id: set(prediction.evidence_ids)
        for claim_id, prediction in predictions["adapted_dense_flat"].items()
    }
    hnsw_top5 = {
        claim_id: set(prediction.evidence_ids)
        for claim_id, prediction in predictions["adapted_dense_hnsw"].items()
    }
    ann_recall = float(
        np.mean(
            [
                len(flat_top5[claim_id] & hnsw_top5[claim_id])
                / max(len(flat_top5[claim_id]), 1)
                for claim_id in sorted(validation)
            ]
        )
    )
    metrics = {
        "schema_version": 1,
        "dataset": "CLIMATE-FEVER-v2-validation",
        "claim_count": len(validation),
        "systems": systems,
        "paired_bootstrap_vs_bm25": paired,
        "diagnostic_slices_vs_bm25": slice_reports,
        "index": {
            "flat": flat_index_metrics,
            "hnsw": hnsw_index_metrics,
            "hnsw_recall_at_5_vs_flat": ann_recall,
        },
        "timing": {
            "train_retrieval": train_timings,
            "validation_serving": serving_timings,
            "lambdamart_train_seconds": ltr_train_seconds,
            "lambdamart_score_p50_ms": percentile(ltr_latencies, 50),
            "lambdamart_score_p95_ms": percentile(ltr_latencies, 95),
            "reranker_model_load_seconds": reranker_load_seconds,
            "reranker_fusion_p50_ms": percentile(reranker_latencies, 50),
            "reranker_fusion_p95_ms": percentile(reranker_latencies, 95),
        },
        "resource": {
            "peak_torch_gpu_bytes": current_peak_gpu_bytes(),
            "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
            "git_commit": os.environ.get("CLIMATE_GIT_COMMIT", "unknown"),
        },
        "lambdamart": {
            "candidate_width": candidate_width,
            "feature_order": list(ranker.feature_names),
            "training_row_count": len(feature_rows),
            "training_query_group_count": len(set(groups)),
            "reachable_positive_count": len(reachable),
            "model_bytes": sum(
                path.stat().st_size for path in output.glob("lambdamart_top100.txt*")
            ),
            "contract_audit": contract_audit,
        },
        "reranker": {
            "model": str(reranker_model["name"]),
            "revision": str(reranker_model["revision"]),
            "candidate_width": candidate_width,
            "fusion": ranking["reranker_fusion"],
        },
        "selected_adapter": {
            "id": frozen["selected_candidate_id"],
            "promoted": frozen["selected_candidate_promoted"],
            "sha256": frozen["adapter_sha256"],
        },
        "truth_boundary": (
            "All effectiveness metrics are CLIMATE-FEVER v2 validation selection. "
            "ANN and stage timings are component benchmarks, not an online SLA. A failed "
            "contract or promotion gate remains a negative result."
        ),
    }
    write_json(output / "metrics.json", metrics)
    write_json(
        output / "artifact_manifest.json",
        {
            "schema_version": 1,
            "metrics_sha256": file_sha256(output / "metrics.json"),
            "payload_tree_sha256": tree_sha256(output),
            "predictions_checkpoints_indexes": "spartan-only",
        },
    )
    print(json.dumps(metrics, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
