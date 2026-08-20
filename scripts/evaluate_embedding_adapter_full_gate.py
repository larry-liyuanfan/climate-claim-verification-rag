from __future__ import annotations

import argparse
import hashlib
import json
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
from evaluate_dense_model_gate import _evaluate_model

from climate_rag.artifacts import write_run_artifacts
from climate_rag.dense import SentenceTransformerEncoder
from climate_rag.embedding_adapter_gate import full_corpus_promotion_decision
from climate_rag.encoder_gate import compare_metric_rows
from climate_rag.io import iter_evidence, load_claims, read_json
from climate_rag.metrics import evaluate_predictions
from climate_rag.models import Prediction


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Full-corpus untouched-dev promotion gate for a Qwen3 embedding adapter."
    )
    parser.add_argument("--claims", required=True)
    parser.add_argument("--evidence", required=True)
    parser.add_argument("--base-index", required=True)
    parser.add_argument("--base-index-metadata", required=True)
    parser.add_argument("--base-embedding-metadata", required=True)
    parser.add_argument("--adapter", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--model", default="Qwen/Qwen3-Embedding-0.6B")
    parser.add_argument("--truncate-dim", type=int, default=1024)
    parser.add_argument("--top-k", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--bootstrap-samples", type=int, default=5_000)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument(
        "--save-index",
        action="store_true",
        help=(
            "persist the rebuildable adapted FlatIP index; disabled by default so the "
            "evaluation can finish under a constrained project artifact quota"
        ),
    )
    return parser.parse_args()


def _document_id_sha256(doc_ids: list[str]) -> str:
    return hashlib.sha256("\0".join(doc_ids).encode("utf-8")).hexdigest()


def _score_base_index(
    *,
    index_path: str,
    doc_ids: list[str],
    claims: dict[str, Any],
    model_name: str,
    device: str,
    truncate_dim: int,
    batch_size: int,
    top_k: int,
) -> tuple[dict[str, Any], list[dict[str, object]], list[dict[str, object]]]:
    import faiss
    import torch

    load_started = time.perf_counter()
    index = faiss.read_index(index_path)
    encoder = SentenceTransformerEncoder(
        model_name,
        device=device,
        truncate_dim=truncate_dim,
    )
    load_seconds = time.perf_counter() - load_started
    claim_ids = sorted(claims)
    query_started = time.perf_counter()
    query_vectors = encoder.encode_queries(
        [claims[claim_id].text for claim_id in claim_ids], batch_size=batch_size
    )
    query_seconds = time.perf_counter() - query_started
    search_started = time.perf_counter()
    scores, positions = index.search(
        np.ascontiguousarray(query_vectors, dtype=np.float32), min(top_k, len(doc_ids))
    )
    search_seconds = time.perf_counter() - search_started
    predictions: dict[str, Prediction] = {}
    rows: list[dict[str, object]] = []
    for claim_id, row_scores, row_positions in zip(
        claim_ids, scores, positions, strict=True
    ):
        ranked_ids = tuple(doc_ids[int(position)] for position in row_positions)
        predictions[claim_id] = Prediction(claim_id, ranked_ids)
        rows.append(
            {
                "claim_id": claim_id,
                "model": "base_reused_full_corpus",
                "evidence_ids": list(ranked_ids),
                "scores": [float(score) for score in row_scores],
            }
        )
    aggregate, metric_rows, errors = evaluate_predictions(
        claims, predictions, ks=(5, 10, min(50, top_k))
    )
    metrics = {
        **aggregate,
        "model": "base_reused_full_corpus",
        "base_model": model_name,
        "adapter_path": None,
        "dimension": int(index.d),
        "document_count": int(index.ntotal),
        "model_and_index_load_seconds": load_seconds,
        "query_encode_seconds": query_seconds,
        "flat_search_seconds": search_seconds,
        "flat_search_qps": len(claims) / max(search_seconds, 1e-12),
        "peak_torch_gpu_bytes": int(torch.cuda.max_memory_allocated()),
    }
    return metrics, metric_rows, [*rows, *errors]


def main() -> int:
    args = parse_args()
    if args.top_k < 50:
        raise ValueError("top_k must be at least 50")
    started_at = datetime.now(timezone.utc).isoformat()
    claims = load_claims(args.claims)
    documents = tuple(iter_evidence(args.evidence))
    doc_ids = [document.evidence_id for document in documents]
    index_metadata = read_json(args.base_index_metadata)
    embedding_metadata = read_json(args.base_embedding_metadata)
    recorded_ids = index_metadata["doc_ids"]
    if doc_ids != recorded_ids:
        raise ValueError("base index document order does not match the evidence corpus")
    if len(doc_ids) != int(embedding_metadata["document_count"]):
        raise ValueError("base embedding document count does not match the evidence corpus")
    if _document_id_sha256(doc_ids) != embedding_metadata["document_id_sha256"]:
        raise ValueError("base embedding document-id hash does not match the evidence corpus")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    base_metrics, base_rows, base_predictions = _score_base_index(
        index_path=args.base_index,
        doc_ids=doc_ids,
        claims=claims,
        model_name=args.model,
        device=args.device,
        truncate_dim=args.truncate_dim,
        batch_size=args.batch_size,
        top_k=args.top_k,
    )
    print(
        json.dumps(
            {
                "event": "base_reused_completed",
                "recall@5": base_metrics["recall@5"],
                "mrr@10": base_metrics["mrr@10"],
            },
            sort_keys=True,
        ),
        flush=True,
    )
    adapted_metrics, adapted_rows, adapted_predictions = _evaluate_model(
        args.model,
        claims=claims,
        documents=documents,
        device=args.device,
        truncate_dim=args.truncate_dim,
        batch_size=args.batch_size,
        top_k=args.top_k,
        adapter_path=args.adapter,
        run_label="adapted_full_corpus",
        save_index_path=output_dir / "adapted-flat.faiss" if args.save_index else None,
    )
    comparisons = compare_metric_rows(
        base_rows,
        adapted_rows,
        samples=args.bootstrap_samples,
        seed=args.seed,
    )
    metrics = {
        "scope": "untouched official dev claims over the complete evidence corpus",
        "claim_count": len(claims),
        "document_count": len(documents),
        "required_evidence_count": len(
            {evidence_id for claim in claims.values() for evidence_id in claim.evidence_ids}
        ),
        "base": base_metrics,
        "adapted": adapted_metrics,
        "adapted_vs_base": comparisons,
        "promotion_decision": full_corpus_promotion_decision(comparisons),
    }
    write_run_artifacts(
        output_dir,
        command="embedding-adapter-full-corpus-gate",
        arguments=vars(args),
        metrics=metrics,
        started_at=started_at,
        inputs=[
            args.claims,
            args.evidence,
            args.base_index,
            args.base_index_metadata,
            args.base_embedding_metadata,
            str(Path(args.adapter) / "adapter_model.safetensors"),
            str(Path(args.adapter) / "adapter_config.json"),
        ],
        predictions=[*base_predictions, *adapted_predictions],
        notes=[
            "The base FlatIP index is reused only after exact document-order/count/hash validation.",
            "The adapter is evaluated on untouched official dev claims and the complete evidence corpus.",
            "The adapted FlatIP index is rebuildable and is persisted only when --save-index is explicit.",
            "The primary gate is paired Recall@5; MRR, nDCG and Evidence F1 must not regress in mean.",
            "Passing remains offline project evidence, not an online production A/B test.",
        ],
        repository=Path(__file__).resolve().parents[1],
    )
    print(json.dumps(metrics, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
