from __future__ import annotations

import time
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np

from .dense import DenseEncoder, FaissANNIndex
from .metrics import evaluate_predictions
from .models import Claim, EvidenceDocument, Prediction


def percentile(values: Sequence[float], quantile: float) -> float:
    if not values:
        return 0.0
    return float(np.percentile(np.asarray(values, dtype=np.float64), quantile))


def current_peak_gpu_bytes() -> int:
    try:
        import torch
    except ImportError:
        return 0
    return int(torch.cuda.max_memory_allocated()) if torch.cuda.is_available() else 0


def reset_peak_gpu_memory() -> None:
    try:
        import torch
    except ImportError:
        return
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()


def decisive_claims(claims: Mapping[str, Claim]) -> dict[str, Claim]:
    return {claim_id: claim for claim_id, claim in claims.items() if claim.evidence_ids}


def build_dense_vectors(
    encoder: DenseEncoder,
    documents: Sequence[EvidenceDocument],
    *,
    batch_size: int,
) -> tuple[np.ndarray, dict[str, Any]]:
    reset_peak_gpu_memory()
    started = time.perf_counter()
    vectors = encoder.encode_documents(
        [document.text for document in documents], batch_size=batch_size
    )
    seconds = time.perf_counter() - started
    return vectors, {
        "document_encode_seconds": seconds,
        "document_encode_per_second": len(documents) / max(seconds, 1e-12),
        "peak_torch_gpu_bytes": current_peak_gpu_bytes(),
        "embedding_bytes": int(vectors.nbytes),
        "dimension": int(vectors.shape[1]),
    }


def build_faiss_index(
    vectors: np.ndarray,
    *,
    kind: str,
    output_path: str | Path | None = None,
    hnsw_m: int = 32,
    hnsw_ef_construction: int = 200,
    hnsw_ef_search: int = 64,
) -> tuple[FaissANNIndex, dict[str, Any]]:
    started = time.perf_counter()
    index = FaissANNIndex(
        int(vectors.shape[1]),
        kind=kind,
        hnsw_m=hnsw_m,
        hnsw_ef_construction=hnsw_ef_construction,
    )
    index.build(vectors)
    if kind == "hnsw":
        index.index.hnsw.efSearch = hnsw_ef_search
    build_seconds = time.perf_counter() - started
    index_bytes = 0
    if output_path is not None:
        target = Path(output_path)
        target.parent.mkdir(parents=True, exist_ok=True)
        index.save(target)
        index_bytes = target.stat().st_size
    return index, {
        "kind": kind,
        "index_build_seconds": build_seconds,
        "index_bytes": index_bytes,
        "document_count": int(index.index.ntotal),
        "hnsw_m": hnsw_m if kind == "hnsw" else None,
        "hnsw_ef_construction": hnsw_ef_construction if kind == "hnsw" else None,
        "hnsw_ef_search": hnsw_ef_search if kind == "hnsw" else None,
    }


def score_dense_index(
    encoder: DenseEncoder,
    index: FaissANNIndex,
    documents: Sequence[EvidenceDocument],
    claims: Mapping[str, Claim],
    *,
    batch_size: int,
    search_width: int = 5,
    final_k: int = 5,
) -> tuple[dict[str, Any], list[dict[str, Any]], list[dict[str, Any]]]:
    if search_width < final_k:
        raise ValueError("search_width must be at least final_k")
    claim_ids = sorted(claims)
    reset_peak_gpu_memory()
    encode_started = time.perf_counter()
    query_vectors = encoder.encode_queries(
        [claims[claim_id].text for claim_id in claim_ids], batch_size=batch_size
    )
    encode_seconds = time.perf_counter() - encode_started
    predictions: dict[str, Prediction] = {}
    rows: list[dict[str, Any]] = []
    latency_ms: list[float] = []
    width = min(search_width, len(documents))
    for claim_id, vector in zip(claim_ids, query_vectors, strict=True):
        started = time.perf_counter()
        scores, positions = index.search(vector.reshape(1, -1), width)
        latency_ms.append((time.perf_counter() - started) * 1000.0)
        ranked_ids = [
            documents[int(position)].evidence_id
            for position in positions[0]
            if int(position) >= 0
        ]
        predictions[claim_id] = Prediction(claim_id, tuple(ranked_ids[:final_k]))
        rows.append(
            {
                "claim_id": claim_id,
                "evidence_ids": ranked_ids,
                "scores": [float(score) for score in scores[0]],
            }
        )
    aggregate, metric_rows, errors = evaluate_predictions(
        claims, predictions, ks=(5, 10)
    )
    metrics = {
        **aggregate,
        "query_count": len(claims),
        "query_encode_seconds": encode_seconds,
        "query_encode_per_second": len(claims) / max(encode_seconds, 1e-12),
        "search_width": width,
        "final_k": final_k,
        "search_p50_ms": percentile(latency_ms, 50),
        "search_p95_ms": percentile(latency_ms, 95),
        "search_total_seconds": sum(latency_ms) / 1000.0,
        "peak_torch_gpu_bytes": current_peak_gpu_bytes(),
    }
    return metrics, metric_rows, [*rows, *errors]


def predictions_from_rows(
    rows: Sequence[Mapping[str, Any]], *, final_k: int = 5
) -> dict[str, Prediction]:
    result: dict[str, Prediction] = {}
    for row in rows:
        if "claim_id" not in row or "evidence_ids" not in row:
            continue
        claim_id = str(row["claim_id"])
        result[claim_id] = Prediction(
            claim_id,
            tuple(str(value) for value in row["evidence_ids"][:final_k]),
        )
    return result
