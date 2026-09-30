"""Serial request profiling, not a network load test or an online SLA."""
from __future__ import annotations

import hashlib
import json
import time
from collections.abc import Callable, Sequence
from typing import Any, TypeVar

import numpy as np

from .benchmark import _rank_with_ltr
from .bm25 import BM25Index
from .dense import DenseRetriever
from .fusion import DEFAULT_FEATURES, Ranker, reciprocal_rank_fusion
from .models import RankedDocument
from .rerank import Reranker, weighted_rank_fuse

T = TypeVar("T")


class ProfiledSearch:
    def __init__(
        self, bm25: BM25Index, dense: DenseRetriever, *,
        ranker: Ranker | None = None, reranker: Reranker | None = None,
        rerank_width: int = 0, recall_width: int = 1000, candidate_width: int = 100,
        rrf_k: int = 60, base_weight: float = 1.0, reranker_weight: float = 1.0,
        synchronize: Callable[[], None] = lambda: None,
        clock: Callable[[], float] = time.perf_counter,
    ) -> None:
        if recall_width < candidate_width or candidate_width != 100:
            raise ValueError("profile must preserve the trained Top-100 candidate contract")
        if (ranker is None) == (reranker is None):
            raise ValueError("choose exactly one of LambdaMART or cross-encoder")
        if ranker is not None and tuple(ranker.feature_names) != DEFAULT_FEATURES:
            raise ValueError("profile feature order differs from the trained contract")
        if reranker is not None and not 1 <= rerank_width <= candidate_width:
            raise ValueError("rerank_width must lie within the fixed candidate pool")
        self.bm25, self.dense = bm25, dense
        self.ranker, self.reranker = ranker, reranker
        self.rerank_width = rerank_width
        self.recall_width, self.candidate_width, self.rrf_k = recall_width, candidate_width, rrf_k
        self.base_weight, self.reranker_weight = base_weight, reranker_weight
        self.synchronize, self.clock = synchronize, clock

    def _measure(self, operation: Callable[[], T]) -> tuple[T, float]:
        self.synchronize()
        started = self.clock()
        result = operation()
        self.synchronize()
        return result, (self.clock() - started) * 1000

    def search(self, query: str) -> tuple[list[RankedDocument], dict[str, Any]]:
        self.synchronize()
        started = self.clock()
        vector, encoding_ms = self._measure(
            lambda: self.dense.encoder.encode_queries([query], batch_size=1)
        )
        lexical, bm25_ms = self._measure(lambda: self.bm25.search(query, self.recall_width))
        dense, ann_ms = self._measure(lambda: self.dense.search_encoded(vector, self.recall_width))
        pool, fusion_ms = self._measure(lambda: reciprocal_rank_fusion(
            {"bm25": lexical, "dense": dense}, k=self.rrf_k, top_k=self.candidate_width,
        ))
        if self.ranker is not None:
            ranked, ranking_ms = self._measure(
                lambda: _rank_with_ltr(query, lexical, dense, pool, self.ranker)
            )
        else:
            reranker = self.reranker
            assert reranker is not None

            def refine() -> list[RankedDocument]:
                reranked = reranker.rerank(query, pool[:self.rerank_width], self.rerank_width)
                # Unscored tail remains reachable and keeps its original RRF evidence.
                return weighted_rank_fuse(
                    pool, reranked, self.candidate_width, k=self.rrf_k,
                    base_weight=self.base_weight, reranker_weight=self.reranker_weight,
                )

            ranked, ranking_ms = self._measure(refine)
        self.synchronize()
        end_to_end_ms = (self.clock() - started) * 1000
        return ranked, {
            "encoding_ms": encoding_ms, "bm25_ms": bm25_ms, "ann_ms": ann_ms,
            "fusion_ms": fusion_ms, "ranking_ms": ranking_ms, "end_to_end_ms": end_to_end_ms,
            "candidate_count": len(pool), "ranking_depth": len(ranked),
            "candidate_ids_sha256": hashlib.sha256(
                json.dumps([row.evidence_id for row in pool]).encode()
            ).hexdigest(),
            "reranker_scored_count": min(self.rerank_width, len(pool)) if self.reranker else 0,
            "recall_width": self.recall_width, "candidate_width": self.candidate_width,
            "feature_order": list(DEFAULT_FEATURES) if self.ranker else None,
        }


def summarize_request_timings(traces: Sequence[dict[str, Any]]) -> dict[str, Any]:
    if not traces:
        raise ValueError("at least one measured request is required")
    stages = ("encoding_ms", "bm25_ms", "ann_ms", "fusion_ms", "ranking_ms", "end_to_end_ms")
    summary: dict[str, Any] = {"request_count": len(traces)}
    for stage in stages:
        values = np.asarray([row[stage] for row in traces], dtype=np.float64)
        if not np.isfinite(values).all() or (values < 0).any():
            raise ValueError("timings must be nonnegative and finite")
        summary[stage] = {"p50": float(np.percentile(values, 50)),
                          "p95": float(np.percentile(values, 95)), "sum": float(values.sum())}
    summary["boundary"] = (
        "batch-size-1 serial in-process requests; E2E is separately timed, not summed "
        "stage percentiles; excludes model load, HTTP, concurrency and serialization"
    )
    return summary
