from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path
from typing import Any

import numpy as np

from climate_rag.bm25 import BM25Index
from climate_rag.dense import SentenceTransformerEncoder
from climate_rag.embedding_training import build_swift_infonce_dataset
from climate_rag.io import (
    iter_evidence,
    load_claims,
    write_json,
    write_jsonl,
)
from climate_rag.metrics import evaluate_predictions
from climate_rag.models import Prediction, RankedDocument
from climate_rag.negatives import mine_hard_negatives
from climate_rag.public_v2 import (
    file_sha256,
    load_public_v2_protocol,
)
from climate_rag.public_v2_runtime import (
    build_dense_vectors,
    build_faiss_index,
    decisive_claims,
    percentile,
    predictions_from_rows,
    score_dense_index,
)
from climate_rag.representation_eval import evaluate_representation_pair


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build frozen BM25/Qwen3 public-v2 bases."
    )
    parser.add_argument("--protocol", required=True)
    parser.add_argument("--prepared-dir", required=True)
    parser.add_argument("--selection-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch-size", type=int, default=32)
    return parser.parse_args()


def _bm25_validation(
    index: BM25Index, claims: dict[str, Any], *, final_k: int
) -> tuple[dict[str, Any], list[dict[str, Any]], dict[str, Prediction]]:
    predictions: dict[str, Prediction] = {}
    rows: list[dict[str, Any]] = []
    latencies: list[float] = []
    for claim_id in sorted(claims):
        started = time.perf_counter()
        ranked = index.search(claims[claim_id].text, final_k)
        latencies.append((time.perf_counter() - started) * 1000.0)
        ids = tuple(row.evidence_id for row in ranked)
        predictions[claim_id] = Prediction(claim_id, ids)
        rows.append({"claim_id": claim_id, "evidence_ids": list(ids)})
    metrics, _, _ = evaluate_predictions(claims, predictions, ks=(5, 10))
    metrics.update(
        {
            "search_p50_ms": percentile(latencies, 50),
            "search_p95_ms": percentile(latencies, 95),
            "final_k": final_k,
        }
    )
    return metrics, rows, predictions


def _dense_rankings(
    claim_ids: list[str],
    query_vectors: np.ndarray,
    flat_index: Any,
    documents: list[Any],
    *,
    width: int,
) -> dict[str, list[RankedDocument]]:
    scores, positions = flat_index.search(query_vectors, width)
    rankings: dict[str, list[RankedDocument]] = {}
    for claim_id, row_scores, row_positions in zip(
        claim_ids, scores, positions, strict=True
    ):
        rankings[claim_id] = [
            RankedDocument(
                documents[int(position)].evidence_id,
                float(score),
                rank,
                documents[int(position)].text,
                "dense",
            )
            for rank, (score, position) in enumerate(
                zip(row_scores, row_positions, strict=True), start=1
            )
            if int(position) >= 0
        ]
    return rankings


def main() -> int:
    args = parse_args()
    protocol = load_public_v2_protocol(args.protocol)
    output = Path(args.output_dir)
    if output.exists():
        raise FileExistsError(f"output must be unique: {output}")
    output.mkdir(parents=True)
    prepared = Path(args.prepared_dir)
    selection = Path(args.selection_dir)
    documents = list(iter_evidence(prepared / "evidence.jsonl"))
    train = load_claims(selection / "train-claims.json")
    validation_all = load_claims(selection / "validation-claims.json")
    validation = decisive_claims(validation_all)
    final_k = int(protocol["downstream_ranking"]["final_k"])

    bm25_started = time.perf_counter()
    bm25 = BM25Index().fit(documents)
    bm25_build_seconds = time.perf_counter() - bm25_started
    bm25_path = output / "bm25.pkl.gz"
    bm25.save(bm25_path)
    bm25_metrics, bm25_rows, bm25_predictions = _bm25_validation(
        bm25, validation, final_k=final_k
    )

    model = protocol["models"]["dense"]
    encoder = SentenceTransformerEncoder(
        str(model["name"]),
        device=args.device,
        truncate_dim=int(model["dimension"]),
        revision=str(model["revision"]),
    )
    vectors, vector_metrics = build_dense_vectors(
        encoder, documents, batch_size=args.batch_size
    )
    embedding_path = output / "base_embeddings.npy"
    np.save(embedding_path, vectors)
    flat_path = output / "base_flat.faiss"
    flat, flat_metrics = build_faiss_index(vectors, kind="flat", output_path=flat_path)
    dense_metrics, _, dense_rows = score_dense_index(
        encoder,
        flat,
        documents,
        validation,
        batch_size=args.batch_size,
        search_width=final_k,
        final_k=final_k,
    )
    dense_predictions = predictions_from_rows(dense_rows, final_k=final_k)
    pair_metrics, tagged_rows = evaluate_representation_pair(
        validation,
        documents,
        bm25_predictions,
        dense_predictions,
        bootstrap_samples=int(protocol["evaluation"]["bootstrap_samples"]),
        seed=int(protocol["seed"]),
    )

    train_decisive = decisive_claims(train)
    train_ids = sorted(train_decisive)
    query_vectors = encoder.encode_queries(
        [train_decisive[claim_id].text for claim_id in train_ids],
        batch_size=args.batch_size,
    )
    recall_width = min(
        int(protocol["downstream_ranking"]["recall_width"]), len(documents)
    )
    dense_train = _dense_rankings(
        train_ids,
        query_vectors,
        flat,
        documents,
        width=recall_width,
    )
    hard_negative_rows: list[dict[str, Any]] = []
    for claim_id in train_ids:
        bm25_rows_for_claim = bm25.search(train_decisive[claim_id].text, recall_width)
        rankings = {
            "bm25": bm25_rows_for_claim,
            "dense": dense_train[claim_id],
        }
        negatives = mine_hard_negatives(
            rankings, train_decisive[claim_id].evidence_ids, limit=8
        )
        for row in negatives:
            hard_negative_rows.append({"claim_id": claim_id, **row})
    hard_negative_path = output / "hard_negatives.jsonl"
    write_jsonl(hard_negative_path, hard_negative_rows)
    training_datasets: dict[str, Any] = {}
    for negative_count in (4, 8):
        dataset = build_swift_infonce_dataset(
            train,
            documents,
            hard_negative_rows,
            negatives_per_positive=negative_count,
            eval_ratio=0.0,
            split_seed=int(protocol["seed"]),
        )
        dataset_dir = output / f"training-n{negative_count}"
        dataset_dir.mkdir()
        write_jsonl(dataset_dir / "train.jsonl", dataset.train_rows)
        # ms-swift requires a non-empty validation dataset. This fixed file is
        # a contract probe only; CLIMATE-FEVER validation remains untouched by training.
        write_jsonl(dataset_dir / "contract-eval.jsonl", dataset.train_rows[:32])
        training_datasets[str(negative_count)] = {
            **dataset.metrics,
            "train_sha256": file_sha256(dataset_dir / "train.jsonl"),
            "contract_eval_sha256": file_sha256(dataset_dir / "contract-eval.jsonl"),
        }

    metrics = {
        "schema_version": 1,
        "dataset": "CLIMATE-FEVER-v2",
        "selection_boundary": {
            "train_claim_count": len(train),
            "validation_claim_count": len(validation_all),
            "validation_decisive_claim_count": len(validation),
            "test_claims_loaded": 0,
        },
        "bm25": {
            **bm25_metrics,
            "index_build_seconds": bm25_build_seconds,
            "index_bytes": bm25_path.stat().st_size,
        },
        "qwen3_embedding_0_6b_base": {
            **dense_metrics,
            "model": str(model["name"]),
            "model_revision": str(model["revision"]),
            "vector_build": vector_metrics,
            "flat_index": flat_metrics,
            "embedding_file_bytes": embedding_path.stat().st_size,
        },
        "dense_vs_bm25": pair_metrics,
        "training": {
            "hard_negative_count": len(hard_negative_rows),
            "hard_negatives_sha256": file_sha256(hard_negative_path),
            "datasets": training_datasets,
        },
        "git_commit": os.environ.get("CLIMATE_GIT_COMMIT", "unknown"),
        "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
    }
    write_jsonl(output / "bm25_predictions_validation.jsonl", bm25_rows)
    write_jsonl(output / "base_predictions_validation.jsonl", dense_rows)
    write_jsonl(output / "base_vs_bm25_slices.jsonl", tagged_rows)
    write_json(output / "metrics.json", metrics)
    print(json.dumps(metrics, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
