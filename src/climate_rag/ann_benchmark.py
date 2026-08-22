"""FAISS recall/throughput benchmarking against an exact FlatIP reference."""

from __future__ import annotations

import statistics
import time
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np

from .dense import l2_normalize


def recall_at_k(exact: np.ndarray, candidate: np.ndarray, k: int) -> list[float]:
    """Return per-query set recall against exact top-k row positions."""

    if k <= 0:
        raise ValueError("k must be positive")
    if exact.ndim != 2 or candidate.ndim != 2 or len(exact) != len(candidate):
        raise ValueError("rankings must be two-dimensional with the same query count")
    if exact.shape[1] < k or candidate.shape[1] < k:
        raise ValueError("rankings contain fewer than k results")
    rows: list[float] = []
    for exact_row, candidate_row in zip(exact[:, :k], candidate[:, :k], strict=True):
        gold = {int(value) for value in exact_row if int(value) >= 0}
        found = {int(value) for value in candidate_row if int(value) >= 0}
        rows.append(len(gold & found) / len(gold) if gold else 1.0)
    return rows


def _percentile(values: Sequence[float], percentile: float) -> float:
    if not values:
        return 0.0
    return float(np.percentile(np.asarray(values, dtype=np.float64), percentile))


def _configure_search(index: Any, *, hnsw_ef_search: int, ivf_nprobe: int) -> dict[str, int]:
    configured: dict[str, int] = {}
    if hasattr(index, "hnsw"):
        index.hnsw.efSearch = hnsw_ef_search
        configured["hnsw_ef_search"] = hnsw_ef_search
    if hasattr(index, "nprobe"):
        index.nprobe = ivf_nprobe
        configured["ivf_nprobe"] = ivf_nprobe
    return configured


def benchmark_faiss_indices(
    query_vectors: np.ndarray,
    index_paths: Mapping[str, str | Path],
    *,
    top_ks: Sequence[int] = (5, 10, 50),
    repeats: int = 3,
    latency_sample_size: int = 32,
    faiss_threads: int | None = None,
    hnsw_ef_search: int = 64,
    ivf_nprobe: int = 32,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Benchmark one or more indexes, requiring ``flat`` as exact reference.

    QPS is batched throughput over all query vectors. P50/P95 latency is measured
    one query at a time over a fixed prefix, so the two numbers are not conflated.
    """

    try:
        import faiss
    except ImportError as exc:
        raise RuntimeError("faiss is required for ANN benchmarking") from exc
    if "flat" not in index_paths:
        raise ValueError("index_paths must include a 'flat' exact reference")
    if repeats < 1:
        raise ValueError("repeats must be positive")
    normalized = np.ascontiguousarray(l2_normalize(query_vectors), dtype=np.float32)
    if len(normalized) == 0:
        raise ValueError("at least one query vector is required")
    ks = tuple(sorted({int(value) for value in top_ks}))
    if not ks or ks[0] <= 0:
        raise ValueError("top_ks must contain positive values")
    if faiss_threads is not None:
        if faiss_threads < 1:
            raise ValueError("faiss_threads must be positive")
        faiss.omp_set_num_threads(faiss_threads)

    loaded: dict[str, Any] = {}
    load_metrics: dict[str, dict[str, Any]] = {}
    expected_rows: int | None = None
    expected_dimension: int | None = None
    for name, raw_path in index_paths.items():
        path = Path(raw_path)
        started = time.perf_counter()
        index = faiss.read_index(str(path))
        load_seconds = time.perf_counter() - started
        if expected_rows is None:
            expected_rows = int(index.ntotal)
            expected_dimension = int(index.d)
        if int(index.ntotal) != expected_rows or int(index.d) != expected_dimension:
            raise ValueError("all indexes must share document count and vector dimension")
        if int(index.d) != normalized.shape[1]:
            raise ValueError("query dimension does not match index dimension")
        configured = _configure_search(
            index,
            hnsw_ef_search=hnsw_ef_search,
            ivf_nprobe=ivf_nprobe,
        )
        loaded[name] = index
        load_metrics[name] = {
            "index_path": str(path.resolve()),
            "index_file_bytes": path.stat().st_size,
            "load_seconds": load_seconds,
            "document_count": int(index.ntotal),
            "dimension": int(index.d),
            "storage_bytes_per_document": path.stat().st_size / max(int(index.ntotal), 1),
            **configured,
        }

    max_k = max(ks)
    _, exact_positions = loaded["flat"].search(normalized, max_k)
    metrics_by_index: dict[str, dict[str, Any]] = {}
    per_query: list[dict[str, Any]] = [
        {"query_ordinal": index} for index in range(len(normalized))
    ]
    sample_count = min(max(latency_sample_size, 1), len(normalized))
    for name, index in loaded.items():
        index.search(normalized[: min(4, len(normalized))], max_k)
        durations: list[float] = []
        positions = None
        for _ in range(repeats):
            started = time.perf_counter()
            _, positions = index.search(normalized, max_k)
            durations.append(time.perf_counter() - started)
        assert positions is not None
        single_query_ms: list[float] = []
        for query in normalized[:sample_count]:
            started = time.perf_counter()
            index.search(query.reshape(1, -1), max_k)
            single_query_ms.append((time.perf_counter() - started) * 1000.0)
        median_seconds = statistics.median(durations)
        recalls: dict[str, float] = {}
        for k in ks:
            rows = recall_at_k(exact_positions, positions, k)
            recalls[f"recall@{k}_vs_flat"] = statistics.mean(rows)
            for ordinal, value in enumerate(rows):
                per_query[ordinal][f"{name}_recall@{k}_vs_flat"] = value
        metrics_by_index[name] = {
            **load_metrics[name],
            "batch_search_seconds_median": median_seconds,
            "batch_qps_median": len(normalized) / max(median_seconds, 1e-12),
            "latency_sample_query_count": sample_count,
            "single_query_latency_p50_ms": _percentile(single_query_ms, 50),
            "single_query_latency_p95_ms": _percentile(single_query_ms, 95),
            "search_repeats": repeats,
            **recalls,
        }
    return {
        "query_count": len(normalized),
        "top_ks": list(ks),
        "faiss_threads": faiss_threads,
        "reference": "flat",
        "indexes": metrics_by_index,
    }, per_query
