from __future__ import annotations

import argparse
import gc
import json
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

from climate_rag.artifacts import write_run_artifacts
from climate_rag.dense import SentenceTransformerEncoder
from climate_rag.encoder_gate import (
    compare_metric_rows,
    evidence_preserving_reservoir_sample,
    required_evidence_ids,
    screening_decision,
)
from climate_rag.io import iter_evidence, load_claims, write_json
from climate_rag.metrics import evaluate_predictions
from climate_rag.models import Prediction


def _evaluate_model(
    model_name: str,
    *,
    claims: dict[str, Any],
    documents: tuple[Any, ...],
    device: str,
    truncate_dim: int | None,
    batch_size: int,
    top_k: int,
) -> tuple[dict[str, Any], list[dict[str, Any]], list[dict[str, Any]]]:
    try:
        import faiss
    except ImportError as exc:
        raise RuntimeError("faiss is required for the dense model gate") from exc
    try:
        import torch
    except ImportError as exc:
        raise RuntimeError("torch is required for the dense model gate") from exc

    if device.startswith("cuda") and torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
    load_started = time.perf_counter()
    encoder = SentenceTransformerEncoder(
        model_name,
        device=device,
        truncate_dim=truncate_dim,
    )
    load_seconds = time.perf_counter() - load_started
    encode_started = time.perf_counter()
    document_vectors = encoder.encode_documents(
        [document.text for document in documents], batch_size=batch_size
    )
    document_encode_seconds = time.perf_counter() - encode_started
    claim_ids = sorted(claims)
    query_started = time.perf_counter()
    query_vectors = encoder.encode_queries(
        [claims[claim_id].text for claim_id in claim_ids], batch_size=batch_size
    )
    query_encode_seconds = time.perf_counter() - query_started
    index = faiss.IndexFlatIP(encoder.dimension)
    build_started = time.perf_counter()
    index.add(np.ascontiguousarray(document_vectors, dtype=np.float32))
    build_seconds = time.perf_counter() - build_started
    search_started = time.perf_counter()
    scores, positions = index.search(
        np.ascontiguousarray(query_vectors, dtype=np.float32), min(top_k, len(documents))
    )
    search_seconds = time.perf_counter() - search_started
    predictions: dict[str, Prediction] = {}
    prediction_rows: list[dict[str, Any]] = []
    for claim_id, row_scores, row_positions in zip(
        claim_ids, scores, positions, strict=True
    ):
        ranked_ids = tuple(documents[int(position)].evidence_id for position in row_positions)
        predictions[claim_id] = Prediction(claim_id, ranked_ids)
        prediction_rows.append(
            {
                "claim_id": claim_id,
                "model": model_name,
                "evidence_ids": list(ranked_ids),
                "scores": [float(score) for score in row_scores],
            }
        )
    aggregate, metric_rows, errors = evaluate_predictions(
        claims, predictions, ks=(5, 10, min(50, top_k))
    )
    peak_gpu_bytes = (
        int(torch.cuda.max_memory_allocated())
        if device.startswith("cuda") and torch.cuda.is_available()
        else 0
    )
    metrics = {
        **aggregate,
        "model": model_name,
        "dimension": encoder.dimension,
        "document_count": len(documents),
        "document_vector_bytes": int(document_vectors.nbytes),
        "model_load_seconds": load_seconds,
        "document_encode_seconds": document_encode_seconds,
        "document_encode_qps": len(documents) / max(document_encode_seconds, 1e-12),
        "query_encode_seconds": query_encode_seconds,
        "query_encode_qps": len(claims) / max(query_encode_seconds, 1e-12),
        "flat_build_seconds": build_seconds,
        "flat_search_seconds": search_seconds,
        "flat_search_qps": len(claims) / max(search_seconds, 1e-12),
        "peak_torch_gpu_bytes": peak_gpu_bytes,
    }
    del index, query_vectors, document_vectors, encoder
    gc.collect()
    if device.startswith("cuda") and torch.cuda.is_available():
        torch.cuda.empty_cache()
    return metrics, metric_rows, [*prediction_rows, *errors]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--claims", required=True)
    parser.add_argument("--evidence", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument(
        "--models",
        default="Qwen/Qwen3-Embedding-0.6B,Qwen/Qwen3-Embedding-4B",
    )
    parser.add_argument("--truncate-dim", type=int, default=1024)
    parser.add_argument("--sample-size", type=int, default=100_000)
    parser.add_argument("--claim-limit", type=int)
    parser.add_argument("--top-k", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--bootstrap-samples", type=int, default=2_000)
    parser.add_argument("--seed", type=int, default=17)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.top_k < 50:
        raise ValueError("top_k must be at least 50")
    started_at = datetime.now(timezone.utc).isoformat()
    claims = load_claims(args.claims)
    if args.claim_limit is not None:
        claims = {key: claims[key] for key in sorted(claims)[: args.claim_limit]}
    sample = evidence_preserving_reservoir_sample(
        iter_evidence(args.evidence),
        required_evidence_ids(claims),
        sample_size=args.sample_size,
        seed=args.seed,
    )
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    print(
        json.dumps(
            {
                "event": "sample_ready",
                "source_document_count": sample.source_document_count,
                "sample_document_count": len(sample.documents),
                "required_evidence_count": len(sample.required_ids),
                "claim_count": len(claims),
            },
            sort_keys=True,
        ),
        flush=True,
    )
    models = [value.strip() for value in args.models.split(",") if value.strip()]
    if len(models) != 2:
        raise ValueError("exactly two comma-separated models are required")
    model_metrics: dict[str, Any] = {}
    metric_rows: dict[str, list[dict[str, Any]]] = {}
    output_rows: list[dict[str, Any]] = []
    for model in models:
        print(json.dumps({"event": "model_started", "model": model}), flush=True)
        metrics, rows, predictions = _evaluate_model(
            model,
            claims=claims,
            documents=sample.documents,
            device=args.device,
            truncate_dim=args.truncate_dim,
            batch_size=args.batch_size,
            top_k=args.top_k,
        )
        model_metrics[model] = metrics
        metric_rows[model] = rows
        output_rows.extend(predictions)
        write_json(
            output_dir / "checkpoint.json",
            {
                "status": "in_progress",
                "completed_models": list(model_metrics),
                "models": model_metrics,
            },
        )
        print(
            json.dumps(
                {
                    "event": "model_completed",
                    "model": model,
                    "recall@5": metrics["recall@5"],
                    "document_encode_seconds": metrics["document_encode_seconds"],
                    "peak_torch_gpu_bytes": metrics["peak_torch_gpu_bytes"],
                },
                sort_keys=True,
            ),
            flush=True,
        )
    comparisons = compare_metric_rows(
        metric_rows[models[0]],
        metric_rows[models[1]],
        samples=args.bootstrap_samples,
        seed=args.seed,
    )
    metrics = {
        "scope": "evidence-preserving sampled-corpus dense encoder model-size gate",
        "source_document_count": sample.source_document_count,
        "sample_document_count": len(sample.documents),
        "required_evidence_count": len(sample.required_ids),
        "claim_count": len(claims),
        "truncate_dim": args.truncate_dim,
        "models": model_metrics,
        "candidate_vs_baseline": comparisons,
        "screening_decision": screening_decision(comparisons),
    }
    write_run_artifacts(
        args.output_dir,
        command="dense-model-size-gate",
        arguments=vars(args),
        metrics=metrics,
        started_at=started_at,
        inputs=[args.claims, args.evidence],
        predictions=output_rows,
        notes=[
            "Every labelled positive for the selected claims is retained; remaining rows are a seeded reservoir sample.",
            "Both models use the same corpus, claims, top-k and output dimension.",
            "This screen cannot replace a full-corpus fixed-dev evaluation.",
        ],
        repository=Path(__file__).resolve().parents[1],
    )
    write_json(
        output_dir / "checkpoint.json",
        {
            "status": "completed",
            "completed_models": models,
            "screening_decision": metrics["screening_decision"],
        },
    )
    print(json.dumps(metrics, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
