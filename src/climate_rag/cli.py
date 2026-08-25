from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

from .ann_benchmark import benchmark_faiss_indices
from .artifacts import write_run_artifacts
from .benchmark import run_five_stage_benchmark
from .bm25 import BM25Index
from .climate_fever import (
    benchmark_public_bm25,
    download_climate_fever,
    prepare_public_benchmark,
)
from .dense import (
    DenseRetriever,
    FaissANNIndex,
    HashDenseEncoder,
    NumpyFlatIndex,
    SentenceTransformerEncoder,
)
from .fusion import (
    DEFAULT_FEATURES,
    LightGBMLambdaMART,
    LinearPairwiseLTR,
    build_candidate_features,
    reciprocal_rank_fusion,
    train_ranker,
)
from .io import (
    iter_evidence,
    load_claims,
    load_predictions,
    read_json,
    read_jsonl,
    write_json,
    write_jsonl,
)
from .metrics import evaluate_predictions, paired_bootstrap
from .models import RankedDocument
from .negatives import mine_hard_negatives
from .pipeline import HybridRetriever
from .rerank import (
    DeterministicFeatureReranker,
    ModelStudioReranker,
    Qwen3CausalLMReranker,
)
from .verification import AbstainingVerifier, ModelStudioStructuredVerifier


def _load_config(path: str | None, command: str) -> dict[str, Any]:
    if not path:
        return {}
    source = Path(path)
    if source.suffix.lower() in {".yaml", ".yml"}:
        try:
            import yaml
        except ImportError as exc:
            raise RuntimeError(
                "PyYAML is required for YAML configs; install the 'yaml' extra"
            ) from exc
        payload = yaml.safe_load(source.read_text(encoding="utf-8")) or {}
    else:
        payload = json.loads(source.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise TypeError("config root must be a mapping")
    section = payload.get(command, payload)
    if not isinstance(section, dict):
        raise TypeError(f"config section '{command}' must be a mapping")
    return section


def _apply_config(args: argparse.Namespace, raw_argv: list[str]) -> None:
    values = _load_config(getattr(args, "config", None), args.command)
    for key, value in values.items():
        flag = "--" + key.replace("_", "-")
        if flag not in raw_argv:
            setattr(args, key, value)


def _required(args: argparse.Namespace, *names: str) -> None:
    missing = [name for name in names if not getattr(args, name, None)]
    if missing:
        raise ValueError(
            "missing required arguments/config keys: " + ", ".join(missing)
        )


def _repository() -> Path:
    return Path(__file__).resolve().parents[2]


def _recorded_arguments(args: argparse.Namespace) -> dict[str, Any]:
    return {
        key: value
        for key, value in vars(args).items()
        if key != "handler" and not callable(value)
    }


def _started_at_utc() -> str:
    return datetime.now(timezone.utc).isoformat()


def command_index(args: argparse.Namespace) -> int:
    _required(args, "evidence", "output_dir")
    started_at = _started_at_utc()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    documents = list(iter_evidence(args.evidence))
    metrics: dict[str, Any] = {"document_count": len(documents)}
    notes: list[str] = []
    if args.backend in {"bm25", "both"}:
        route_started = time.perf_counter()
        bm25 = BM25Index(k1=args.k1, b=args.b).fit(documents)
        bm25_path = output_dir / "bm25.pkl.gz"
        bm25.save(bm25_path)
        metrics.update(
            {
                "bm25_build_seconds": time.perf_counter() - route_started,
                "bm25_index_bytes": bm25_path.stat().st_size,
                "bm25_vocabulary_size": len(bm25.postings),
            }
        )
    if args.backend in {"dense", "both"}:
        route_started = time.perf_counter()
        if args.encoder == "hash":
            encoder = HashDenseEncoder(args.dimension)
            notes.append(
                "Hash embeddings are a deterministic smoke baseline, not a semantic model."
            )
        else:
            encoder = SentenceTransformerEncoder(
                args.model,
                query_prefix=args.query_prefix,
                query_prompt_name=args.query_prompt_name,
                device=args.device,
            )
        if args.ann == "numpy":
            ann = NumpyFlatIndex()
        else:
            ann = FaissANNIndex(
                encoder.dimension,
                kind=args.ann,
                hnsw_m=args.hnsw_m,
                hnsw_ef_construction=args.hnsw_ef_construction,
                nlist=args.nlist,
                pq_m=args.pq_m,
                nbits=args.nbits,
                nprobe=args.nprobe,
            )
        dense = DenseRetriever(encoder, ann)
        mapping_digest = hashlib.sha256(
            "\0".join(document.evidence_id for document in documents).encode("utf-8")
        ).hexdigest()
        if args.embeddings and Path(args.embeddings).exists():
            metadata_path = Path(str(args.embeddings) + ".json")
            if not metadata_path.exists():
                raise ValueError("embedding cache metadata is missing")
            cache_metadata = read_json(metadata_path)
            expected_metadata = {
                "document_count": len(documents),
                "document_id_sha256": mapping_digest,
                "encoder": encoder.name,
                "dimension": encoder.dimension,
            }
            if any(
                cache_metadata.get(key) != value
                for key, value in expected_metadata.items()
            ):
                raise ValueError(
                    "embedding cache does not match corpus mapping or encoder"
                )
            vectors = np.load(args.embeddings, mmap_mode="r")
            metrics["dense_embeddings_reused"] = True
        else:
            vectors = encoder.encode_documents(
                [document.text for document in documents], batch_size=args.batch_size
            )
            metrics["dense_embeddings_reused"] = False
            if args.embeddings:
                embedding_path = Path(args.embeddings)
                embedding_path.parent.mkdir(parents=True, exist_ok=True)
                np.save(embedding_path, vectors)
                write_json(
                    Path(str(embedding_path) + ".json"),
                    {
                        "schema_version": 1,
                        "document_count": len(documents),
                        "document_id_sha256": mapping_digest,
                        "encoder": encoder.name,
                        "dimension": encoder.dimension,
                    },
                )
        training_vectors = None
        if isinstance(ann, FaissANNIndex) and ann.kind == "ivfpq":
            if args.ivf_train_size < 1:
                raise ValueError("ivf_train_size must be positive")
            training_count = min(len(vectors), args.ivf_train_size)
            generator = np.random.default_rng(args.seed)
            training_rows = np.sort(
                generator.choice(len(vectors), size=training_count, replace=False)
            )
            training_vectors = np.asarray(vectors[training_rows], dtype=np.float32)
            metrics["ivf_training_vector_count"] = training_count
            notes.append(
                "IVF-PQ training uses a deterministic sample; the seed and sample size are recorded."
            )
        dense.fit_vectors(documents, vectors, training_vectors=training_vectors)
        dense_dir = output_dir / "dense"
        dense.save(dense_dir)
        metrics.update(
            {
                "dense_build_seconds": time.perf_counter() - route_started,
                "dense_encoder": encoder.name,
                "dense_dimension": encoder.dimension,
                "ann_kind": args.ann,
                "dense_artifact_bytes": sum(
                    path.stat().st_size
                    for path in dense_dir.rglob("*")
                    if path.is_file()
                ),
            }
        )
    metrics["total_build_seconds"] = time.perf_counter() - started
    write_run_artifacts(
        output_dir,
        command="index",
        arguments=_recorded_arguments(args),
        metrics=metrics,
        started_at=started_at,
        inputs=[args.evidence],
        notes=notes,
        repository=_repository(),
    )
    print(json.dumps(metrics, indent=2, sort_keys=True))
    return 0


def _named_paths(values: list[str] | None, argument: str) -> dict[str, Path]:
    result: dict[str, Path] = {}
    for value in values or []:
        if "=" not in value:
            raise ValueError(f"{argument} entries must use name=path")
        name, raw_path = value.split("=", 1)
        if not name or name in result:
            raise ValueError(f"duplicate or empty {argument} name: {name!r}")
        result[name] = Path(raw_path)
    return result


def command_benchmark_ann(args: argparse.Namespace) -> int:
    _required(args, "claims", "output_dir")
    started_at = _started_at_utc()
    indexes = _named_paths(args.index, "--index")
    if "flat" not in indexes:
        raise ValueError("--index must include flat=/path/to/index.faiss")
    claims = load_claims(args.claims)
    claim_ids = sorted(claims)
    encoder = SentenceTransformerEncoder(
        args.model,
        query_prefix=args.query_prefix,
        query_prompt_name=args.query_prompt_name,
        device=args.device,
    )
    encode_started = time.perf_counter()
    query_vectors = encoder.encode_queries(
        [claims[claim_id].text for claim_id in claim_ids], batch_size=args.batch_size
    )
    encode_seconds = time.perf_counter() - encode_started
    metrics, per_query = benchmark_faiss_indices(
        query_vectors,
        indexes,
        top_ks=tuple(int(value) for value in args.ks.split(",") if value),
        repeats=args.repeats,
        latency_sample_size=args.latency_sample_size,
        faiss_threads=args.faiss_threads,
        hnsw_ef_search=args.hnsw_ef_search,
        ivf_nprobe=args.ivf_nprobe,
    )
    metrics.update(
        {
            "encoder": encoder.name,
            "query_embedding_seconds": encode_seconds,
            "query_embedding_qps": len(claim_ids) / max(encode_seconds, 1e-12),
        }
    )
    for claim_id, row in zip(claim_ids, per_query, strict=True):
        row["claim_id"] = claim_id
    manifests = _named_paths(args.index_manifest, "--index-manifest")
    write_run_artifacts(
        args.output_dir,
        command="benchmark-ann",
        arguments=_recorded_arguments(args),
        metrics=metrics,
        started_at=started_at,
        inputs=[args.claims, *manifests.values()],
        predictions=per_query,
        notes=[
            "FlatIP row positions are the exact ANN ground truth on the same embedding mapping.",
            "Batch QPS and single-query P50/P95 are measured separately.",
            "Storage and resource measurements are engineering evidence, not retrieval relevance gains.",
        ],
        repository=_repository(),
    )
    print(json.dumps(metrics, indent=2, sort_keys=True))
    return 0


def _load_rankings(path: str | Path) -> dict[str, dict[str, list[RankedDocument]]]:
    source = Path(path)
    result: dict[str, dict[str, list[RankedDocument]]] = defaultdict(
        lambda: defaultdict(list)
    )
    if source.suffix.lower() == ".jsonl":
        rows = list(read_jsonl(source))
        for row in rows:
            claim_id = str(row["claim_id"])
            source_name = str(row.get("source", "retriever"))
            if "evidence_ids" in row:
                for rank, evidence_id in enumerate(row["evidence_ids"], start=1):
                    result[claim_id][source_name].append(
                        RankedDocument(
                            str(evidence_id), 1.0 / rank, rank, source=source_name
                        )
                    )
            else:
                result[claim_id][source_name].append(
                    RankedDocument(
                        evidence_id=str(row["evidence_id"]),
                        score=float(row.get("score", 0.0)),
                        rank=int(
                            row.get("rank", len(result[claim_id][source_name]) + 1)
                        ),
                        text=str(row.get("text", "")),
                        source=source_name,
                    )
                )
        return result
    payload = read_json(source)
    if not isinstance(payload, dict):
        raise TypeError("rankings JSON must be keyed by claim id")
    for claim_id, routes in payload.items():
        if not isinstance(routes, dict):
            raise TypeError("each claim ranking must map sources to ranked rows")
        for source_name, rows in routes.items():
            for fallback_rank, row in enumerate(rows, start=1):
                if isinstance(row, str):
                    item = RankedDocument(
                        row, 1.0 / fallback_rank, fallback_rank, source=source_name
                    )
                else:
                    item = RankedDocument(
                        evidence_id=str(row["evidence_id"]),
                        score=float(row.get("score", 0.0)),
                        rank=int(row.get("rank", fallback_rank)),
                        text=str(row.get("text", "")),
                        source=str(source_name),
                    )
                result[str(claim_id)][str(source_name)].append(item)
    return result


def command_mine_negatives(args: argparse.Namespace) -> int:
    _required(args, "claims", "output_dir")
    started_at = _started_at_utc()
    claims = load_claims(args.claims)
    live_indexes = bool(args.bm25_index or args.dense_index)
    if live_indexes:
        _required(args, "bm25_index", "dense_index")
        bm25_index = BM25Index.load(args.bm25_index)
        dense_index = DenseRetriever.load(args.dense_index, device=args.device)
        rankings: dict[str, dict[str, list[RankedDocument]]] = {}
        for claim_id in sorted(claims):
            rankings[claim_id] = {
                "bm25": bm25_index.search(claims[claim_id].text, args.recall_k),
                "dense": dense_index.search(claims[claim_id].text, args.recall_k),
            }
        text_by_id = dict(zip(bm25_index.doc_ids, bm25_index.texts, strict=True))
    elif args.rankings:
        rankings = _load_rankings(args.rankings)
        bm25_index = None
    else:
        raise ValueError("provide --rankings or both --bm25-index and --dense-index")
    rows: list[dict[str, Any]] = []
    feature_rows: list[dict[str, Any]] = []
    ranking_rows: list[dict[str, Any]] = []
    missing_rankings = 0
    ltr_supported_positive_count = 0
    ltr_unsupported_positive_count = 0
    ltr_skipped_query_count = 0
    for claim_id in sorted(claims):
        if claim_id not in rankings:
            missing_rankings += 1
            continue
        negatives = mine_hard_negatives(
            rankings[claim_id], claims[claim_id].evidence_ids, limit=args.limit
        )
        for item in negatives:
            rows.append({"claim_id": claim_id, **item})
        for source, source_rows in rankings[claim_id].items():
            ranking_rows.append(
                {
                    "claim_id": claim_id,
                    "source": source,
                    "evidence_ids": [row.evidence_id for row in source_rows],
                    "scores": [row.score for row in source_rows],
                }
            )
        if live_indexes:
            assert bm25_index is not None
            bm25_by_id = {row.evidence_id: row for row in rankings[claim_id]["bm25"]}
            dense_by_id = {row.evidence_id: row for row in rankings[claim_id]["dense"]}
            rrf_by_id = {
                row.evidence_id: row
                for row in reciprocal_rank_fusion(rankings[claim_id], k=args.rrf_k)
            }
            retrieved_ids = bm25_by_id.keys() | dense_by_id.keys()
            supported_gold = [
                evidence_id
                for evidence_id in claims[claim_id].evidence_ids
                if evidence_id in retrieved_ids
            ]
            ltr_supported_positive_count += len(supported_gold)
            ltr_unsupported_positive_count += len(claims[claim_id].evidence_ids) - len(
                supported_gold
            )
            if not supported_gold:
                ltr_skipped_query_count += 1
                continue
            candidate_ids = [
                *supported_gold,
                *(str(row["evidence_id"]) for row in negatives),
            ]
            for evidence_id in dict.fromkeys(candidate_ids):
                text = text_by_id.get(evidence_id)
                if text is None:
                    continue
                bm25_row = bm25_by_id.get(evidence_id)
                dense_row = dense_by_id.get(evidence_id)
                rrf_row = rrf_by_id.get(evidence_id)
                feature_rows.append(
                    {
                        "query_id": claim_id,
                        "evidence_id": evidence_id,
                        "relevance": 2
                        if evidence_id in claims[claim_id].evidence_ids
                        else 0,
                        "features": build_candidate_features(
                            claims[claim_id].text,
                            text,
                            bm25_score=bm25_row.score if bm25_row else 0.0,
                            bm25_rank=bm25_row.rank if bm25_row else None,
                            dense_score=dense_row.score if dense_row else 0.0,
                            dense_rank=dense_row.rank if dense_row else None,
                            rrf_score=rrf_row.score if rrf_row else 0.0,
                            rrf_rank=rrf_row.rank if rrf_row else None,
                        ),
                    }
                )
    output_dir = Path(args.output_dir)
    write_jsonl(output_dir / "hard_negatives.jsonl", rows)
    write_jsonl(output_dir / "rankings.jsonl", ranking_rows)
    if live_indexes:
        write_jsonl(output_dir / "ltr_features.jsonl", feature_rows)
    metrics = {
        "claim_count": len(claims),
        "claim_with_rankings_count": len(claims) - missing_rankings,
        "missing_rankings_count": missing_rankings,
        "hard_negative_count": len(rows),
        "ltr_feature_count": len(feature_rows),
        "ltr_query_group_count": len({str(row["query_id"]) for row in feature_rows}),
        "ltr_supported_positive_count": ltr_supported_positive_count,
        "ltr_unsupported_positive_count": ltr_unsupported_positive_count,
        "ltr_skipped_query_count": ltr_skipped_query_count,
    }
    write_run_artifacts(
        output_dir,
        command="mine-negatives",
        arguments=_recorded_arguments(args),
        metrics=metrics,
        started_at=started_at,
        inputs=[
            args.claims,
            *([args.rankings] if args.rankings else []),
            *([args.bm25_index] if args.bm25_index else []),
            *(
                [Path(args.dense_index) / "dense_index.json"]
                if args.dense_index
                else []
            ),
        ],
        predictions=rows,
        notes=[
            "LTR positives are restricted to the live BM25/dense candidate pool; unretrieved gold rows are never injected with zero retrieval features.",
            "Queries with no candidate-supported positive are excluded from LTR training and counted in metrics.",
            "Training rows record the same RRF score/rank prior used by five-stage inference.",
        ],
        repository=_repository(),
    )
    print(json.dumps(metrics, indent=2, sort_keys=True))
    return 0


def _pairwise_accuracy(
    scores: np.ndarray, labels: np.ndarray, groups: list[str]
) -> float:
    correct = 0
    total = 0
    for left in range(len(labels)):
        for right in range(left + 1, len(labels)):
            if groups[left] != groups[right] or labels[left] == labels[right]:
                continue
            total += 1
            correct += int(
                (scores[left] - scores[right]) * (labels[left] - labels[right]) > 0
            )
    return correct / total if total else 0.0


def command_train_fusion(args: argparse.Namespace) -> int:
    _required(args, "features", "output_dir")
    started_at = _started_at_utc()
    rows = list(read_jsonl(args.features))
    if not rows:
        raise ValueError("feature file is empty")
    feature_names = (
        tuple(args.feature_names.split(",")) if args.feature_names else DEFAULT_FEATURES
    )
    matrix = np.asarray(
        [
            [float(row.get("features", {}).get(name, 0.0)) for name in feature_names]
            for row in rows
        ],
        dtype=np.float64,
    )
    labels = np.asarray([float(row["relevance"]) for row in rows], dtype=np.float64)
    groups = [str(row["query_id"]) for row in rows]
    if args.algorithm == "linear":
        model = LinearPairwiseLTR(feature_names, seed=args.seed)
        model.fit(matrix, labels, groups)
    elif args.algorithm == "lambdamart":
        model = LightGBMLambdaMART(feature_names, seed=args.seed)
        model.fit(matrix, labels, groups)
    else:
        model = train_ranker(
            matrix, labels, groups, feature_names=feature_names, prefer_lightgbm=True
        )
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    if isinstance(model, LightGBMLambdaMART):
        model_path = output_dir / "ltr_model.txt"
        algorithm = "lightgbm_lambdamart"
    else:
        model_path = output_dir / "ltr_model.json"
        algorithm = "linear_pairwise_ranknet_fallback"
    model.save(model_path)
    scores = model.predict(matrix)
    metrics = {
        "training_row_count": len(rows),
        "query_group_count": len(set(groups)),
        "positive_row_count": int(np.sum(labels > 0)),
        "algorithm": algorithm,
        "training_pairwise_accuracy": _pairwise_accuracy(scores, labels, groups),
    }
    write_run_artifacts(
        output_dir,
        command="train-fusion",
        arguments=_recorded_arguments(args),
        metrics=metrics,
        started_at=started_at,
        inputs=[args.features],
        notes=[
            "Training pairwise accuracy is a diagnostic, not held-out ranking quality.",
            "Use claim-grouped train/dev splits before reporting LTR improvements.",
        ],
        repository=_repository(),
    )
    print(json.dumps(metrics, indent=2, sort_keys=True))
    return 0


def command_evaluate(args: argparse.Namespace) -> int:
    _required(args, "claims", "output_dir")
    started_at = _started_at_utc()
    if args.experiment_config:
        metrics, rows = run_five_stage_benchmark(
            claims_path=args.claims,
            config_path=args.experiment_config,
            output_dir=args.output_dir,
            bootstrap_samples=args.bootstrap_samples,
            seed=args.seed,
        )
        write_run_artifacts(
            args.output_dir,
            command="evaluate-five-stage",
            arguments=_recorded_arguments(args),
            metrics=metrics,
            started_at=started_at,
            inputs=[args.claims, args.experiment_config],
            predictions=rows,
            notes=[
                "All configured systems use the same claims and final_k.",
                "The reranker base stage is recorded; RRF and LTR candidates are not interchangeable.",
                "The configured reranker name is recorded; deterministic fallback results must not be described as Qwen3.",
            ],
            repository=_repository(),
        )
        print(json.dumps(metrics, indent=2, sort_keys=True))
        return 0
    _required(args, "predictions")
    claims = load_claims(args.claims)
    predictions = load_predictions(args.predictions)
    ks = tuple(int(item) for item in str(args.ks).split(",") if item)
    metrics, rows, errors = evaluate_predictions(claims, predictions, ks)
    if args.baseline_predictions:
        baseline = load_predictions(args.baseline_predictions)
        _, baseline_rows, _ = evaluate_predictions(claims, baseline, ks)
        baseline_by_id = {row["claim_id"]: row for row in baseline_rows}
        comparisons: dict[str, Any] = {}
        compare_metrics = [
            "evidence_f1",
            "mrr@10",
            "ndcg@10",
            *(f"recall@{k}" for k in ks),
        ]
        for metric in compare_metrics:
            left = [baseline_by_id[row["claim_id"]][metric] for row in rows]
            right = [row[metric] for row in rows]
            comparisons[metric] = paired_bootstrap(
                left, right, samples=args.bootstrap_samples, seed=args.seed
            )
        metrics["paired_bootstrap"] = comparisons
    write_run_artifacts(
        args.output_dir,
        command="evaluate",
        arguments=_recorded_arguments(args),
        metrics=metrics,
        started_at=started_at,
        inputs=[
            args.claims,
            args.predictions,
            *([args.baseline_predictions] if args.baseline_predictions else []),
        ],
        predictions=rows,
        error_cases=errors,
        notes=[
            "All aggregate retrieval metrics are macro averages over claims.",
            "Evidence F1 follows the notebook's per-claim set F1 then macro-average convention.",
        ],
        repository=_repository(),
    )
    print(json.dumps(metrics, indent=2, sort_keys=True))
    return 0


def command_serve(args: argparse.Namespace) -> int:
    _required(args, "bm25_index")
    try:
        import uvicorn
    except ImportError as exc:
        raise RuntimeError("uvicorn is unavailable; install the 'serve' extra") from exc
    from .service import create_app

    bm25 = BM25Index.load(args.bm25_index)
    dense = (
        DenseRetriever.load(args.dense_index, device=args.device)
        if args.dense_index
        else None
    )
    if args.reranker == "none":
        reranker = None
    elif args.reranker == "deterministic":
        reranker = DeterministicFeatureReranker()
    elif args.reranker == "model-studio":
        reranker = ModelStudioReranker(model=args.reranker_model)
    else:
        reranker = Qwen3CausalLMReranker(
            model_name=args.reranker_model,
            device=args.device,
            batch_size=args.reranker_batch_size,
            dtype=args.reranker_dtype,
        )
    retriever = HybridRetriever(bm25=bm25, dense=dense, reranker=reranker)
    verifier = (
        ModelStudioStructuredVerifier(model=args.verifier_model)
        if args.verifier == "model-studio"
        else AbstainingVerifier()
    )
    uvicorn.run(
        create_app(
            retriever,
            verifier=verifier,
            default_top_k=args.top_k,
            max_queries=args.max_queries,
        ),
        host=args.host,
        port=args.port,
    )
    return 0


def command_prepare_public(args: argparse.Namespace) -> int:
    _required(args, "output_dir")
    source = (
        Path(args.input)
        if args.input
        else Path(args.output_dir) / "source" / "climate-fever.jsonl"
    )
    provenance: dict[str, Any] = {}
    if not source.exists():
        provenance = download_climate_fever(source, url=args.url)
    manifest = prepare_public_benchmark(
        source, args.output_dir, seed=args.seed, source_url=args.url
    )
    if provenance:
        manifest["download"] = provenance
        write_json(Path(args.output_dir) / "split_manifest.json", manifest)
    print(json.dumps(manifest, indent=2, sort_keys=True))
    return 0


def command_benchmark_public(args: argparse.Namespace) -> int:
    _required(args, "prepared_dir", "output_dir")
    metrics = benchmark_public_bm25(
        args.prepared_dir,
        args.output_dir,
        split_name=args.split,
        top_k=args.top_k,
    )
    print(json.dumps(metrics, indent=2, sort_keys=True))
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="climate-rag")
    subparsers = parser.add_subparsers(dest="command", required=True)

    index = subparsers.add_parser("index", help="build BM25 and/or dense indexes")
    index.add_argument("--config")
    index.add_argument("--evidence")
    index.add_argument("--output-dir")
    index.add_argument("--backend", choices=("bm25", "dense", "both"), default="bm25")
    index.add_argument("--k1", type=float, default=1.5)
    index.add_argument("--b", type=float, default=0.75)
    index.add_argument(
        "--encoder", choices=("hash", "sentence-transformer"), default="hash"
    )
    index.add_argument("--model", default="Qwen/Qwen3-Embedding-0.6B")
    index.add_argument("--query-prefix", default="")
    index.add_argument("--query-prompt-name")
    index.add_argument("--device")
    index.add_argument("--dimension", type=int, default=256)
    index.add_argument("--batch-size", type=int, default=32)
    index.add_argument(
        "--embeddings",
        help="optional .npy cache shared by Flat/HNSW/IVF-PQ builds; encoder and corpus must match",
    )
    index.add_argument(
        "--ann", choices=("numpy", "flat", "hnsw", "ivfpq"), default="numpy"
    )
    index.add_argument("--hnsw-m", type=int, default=32)
    index.add_argument("--hnsw-ef-construction", type=int, default=200)
    index.add_argument("--nlist", type=int, default=256)
    index.add_argument("--pq-m", type=int, default=32)
    index.add_argument("--nbits", type=int, default=8)
    index.add_argument("--nprobe", type=int, default=16)
    index.add_argument("--ivf-train-size", type=int, default=200000)
    index.add_argument("--seed", type=int, default=17)
    index.set_defaults(handler=command_index)

    ann_benchmark = subparsers.add_parser(
        "benchmark-ann", help="compare FAISS ANN recall/throughput against FlatIP"
    )
    ann_benchmark.add_argument("--config")
    ann_benchmark.add_argument("--claims")
    ann_benchmark.add_argument(
        "--index",
        action="append",
        help="repeat name=/path/index.faiss; flat is required",
    )
    ann_benchmark.add_argument(
        "--index-manifest", action="append", help="repeat name=/path/run_manifest.json"
    )
    ann_benchmark.add_argument("--model", default="Qwen/Qwen3-Embedding-0.6B")
    ann_benchmark.add_argument("--query-prefix", default="")
    ann_benchmark.add_argument("--query-prompt-name")
    ann_benchmark.add_argument("--device")
    ann_benchmark.add_argument("--batch-size", type=int, default=32)
    ann_benchmark.add_argument("--ks", default="5,10,50")
    ann_benchmark.add_argument("--repeats", type=int, default=3)
    ann_benchmark.add_argument("--latency-sample-size", type=int, default=32)
    ann_benchmark.add_argument("--faiss-threads", type=int)
    ann_benchmark.add_argument("--hnsw-ef-search", type=int, default=64)
    ann_benchmark.add_argument("--ivf-nprobe", type=int, default=32)
    ann_benchmark.add_argument("--output-dir")
    ann_benchmark.set_defaults(handler=command_benchmark_ann)

    negatives = subparsers.add_parser(
        "mine-negatives", help="mine high-ranked non-gold evidence"
    )
    negatives.add_argument("--config")
    negatives.add_argument("--claims")
    negatives.add_argument("--rankings")
    negatives.add_argument("--bm25-index")
    negatives.add_argument("--dense-index")
    negatives.add_argument("--device")
    negatives.add_argument("--recall-k", type=int, default=1000)
    negatives.add_argument("--rrf-k", type=int, default=60)
    negatives.add_argument("--output-dir")
    negatives.add_argument("--limit", type=int, default=20)
    negatives.set_defaults(handler=command_mine_negatives)

    fusion = subparsers.add_parser(
        "train-fusion", help="train LambdaMART or pairwise fallback LTR"
    )
    fusion.add_argument("--config")
    fusion.add_argument("--features")
    fusion.add_argument("--output-dir")
    fusion.add_argument(
        "--algorithm", choices=("auto", "lambdamart", "linear"), default="auto"
    )
    fusion.add_argument("--feature-names")
    fusion.add_argument("--seed", type=int, default=17)
    fusion.set_defaults(handler=command_train_fusion)

    evaluate = subparsers.add_parser(
        "evaluate", help="score official-format predictions"
    )
    evaluate.add_argument("--config")
    evaluate.add_argument("--claims")
    evaluate.add_argument("--predictions")
    evaluate.add_argument("--experiment-config")
    evaluate.add_argument("--baseline-predictions")
    evaluate.add_argument("--output-dir")
    evaluate.add_argument("--ks", default="5,10,50")
    evaluate.add_argument("--bootstrap-samples", type=int, default=2000)
    evaluate.add_argument("--seed", type=int, default=17)
    evaluate.set_defaults(handler=command_evaluate)

    prepare_public = subparsers.add_parser(
        "prepare-public",
        help="download/adapt CLIMATE-FEVER and create a leakage-safe split",
    )
    prepare_public.add_argument("--config")
    prepare_public.add_argument("--input")
    prepare_public.add_argument(
        "--url",
        default=(
            "https://raw.githubusercontent.com/tdiggelm/climate-fever-dataset/"
            "main/dataset/climate-fever.jsonl"
        ),
    )
    prepare_public.add_argument("--output-dir")
    prepare_public.add_argument("--seed", type=int, default=20260825)
    prepare_public.set_defaults(handler=command_prepare_public)

    benchmark_public = subparsers.add_parser(
        "benchmark-public", help="run the frozen public BM25 retrieval baseline"
    )
    benchmark_public.add_argument("--config")
    benchmark_public.add_argument("--prepared-dir")
    benchmark_public.add_argument("--output-dir")
    benchmark_public.add_argument(
        "--split", choices=("train", "validation", "test"), default="test"
    )
    benchmark_public.add_argument("--top-k", type=int, default=50)
    benchmark_public.set_defaults(handler=command_benchmark_public)

    serve = subparsers.add_parser(
        "serve", help="serve retrieval and grounded verification"
    )
    serve.add_argument("--config")
    serve.add_argument("--bm25-index")
    serve.add_argument("--dense-index")
    serve.add_argument("--device")
    serve.add_argument(
        "--reranker",
        choices=("none", "deterministic", "qwen-local", "model-studio"),
        default="none",
    )
    serve.add_argument("--reranker-model", default="Qwen/Qwen3-Reranker-4B")
    serve.add_argument("--reranker-batch-size", type=int, default=4)
    serve.add_argument("--reranker-dtype", default="bfloat16")
    serve.add_argument("--verifier", choices=("none", "model-studio"), default="none")
    serve.add_argument("--verifier-model", default="qwen3.7-plus")
    serve.add_argument("--max-queries", type=int, default=2)
    serve.add_argument("--host", default="127.0.0.1")
    serve.add_argument("--port", type=int, default=8000)
    serve.add_argument("--top-k", type=int, default=5)
    serve.set_defaults(handler=command_serve)
    return parser


def main(argv: list[str] | None = None) -> int:
    raw_argv = list(sys.argv[1:] if argv is None else argv)
    parser = build_parser()
    args = parser.parse_args(raw_argv)
    try:
        _apply_config(args, raw_argv)
        return int(args.handler(args))
    except (OSError, RuntimeError, ValueError, KeyError) as exc:
        parser.exit(2, f"climate-rag: error: {exc}\n")


if __name__ == "__main__":
    raise SystemExit(main())
