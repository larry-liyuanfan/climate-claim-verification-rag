from __future__ import annotations

from .bm25 import BM25Index
from .dense import DenseRetriever
from .fusion import reciprocal_rank_fusion
from .models import RankedDocument
from .rerank import Reranker


class HybridRetriever:
    def __init__(
        self,
        *,
        bm25: BM25Index | None = None,
        dense: DenseRetriever | None = None,
        reranker: Reranker | None = None,
        rrf_k: int = 60,
    ) -> None:
        if bm25 is None and dense is None:
            raise ValueError("at least one retrieval route is required")
        self.bm25 = bm25
        self.dense = dense
        self.reranker = reranker
        self.rrf_k = rrf_k

    def retrieve(
        self,
        query: str,
        *,
        recall_k: int = 100,
        rerank_k: int = 50,
        final_k: int = 5,
    ) -> list[RankedDocument]:
        rankings: dict[str, list[RankedDocument]] = {}
        if self.bm25 is not None:
            rankings["bm25"] = self.bm25.search(query, recall_k)
        if self.dense is not None:
            rankings["dense"] = self.dense.search(query, recall_k)
        if len(rankings) == 1:
            candidates = next(iter(rankings.values()))[:rerank_k]
        else:
            candidates = reciprocal_rank_fusion(rankings, k=self.rrf_k, top_k=rerank_k)
        if self.reranker is None:
            return [
                RankedDocument(
                    evidence_id=row.evidence_id,
                    text=row.text,
                    score=row.score,
                    rank=rank,
                    source=row.source,
                    features=row.features,
                )
                for rank, row in enumerate(candidates[:final_k], start=1)
            ]
        return self.reranker.rerank(query, candidates, final_k)

