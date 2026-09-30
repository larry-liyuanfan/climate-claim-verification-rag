"""Gold-free retrieval shared by train sampling and allocated diagnosis."""

from collections.abc import Mapping, Sequence
from typing import Any

from .agent_v3 import Source
from .bm25 import BM25Index
from .models import EvidenceDocument, RankedDocument
from .scifact_grounding import Abstract
from .scifact_terminal import source_from_abstract
from .verification import normalise_claim


class SciFactBM25:
    def __init__(self, corpus: Mapping[int, Abstract]) -> None:
        self.sources = {str(k): source_from_abstract(v) for k, v in sorted(corpus.items())}
        self.index = BM25Index().fit(
            EvidenceDocument(k, source.title + " " + " ".join(source.sentences))
            for k, source in self.sources.items()
        )

    def __call__(self, query: str, width: int) -> list[Source]:
        return [self.sources[r.evidence_id]
                for r in self.index.search(normalise_claim(query), width)]


def rerank_sources(model: Any, query: str, candidates: Sequence[Source]) -> list[Source]:
    """Preserve all candidates; no gold-derived filtering or sentence reordering."""
    lookup = {s.source_id: s for s in candidates}
    rows = [RankedDocument(s.source_id, 0.0, i + 1,
                           s.title + " " + " ".join(s.sentences))
            for i, s in enumerate(candidates)]
    return [lookup[r.evidence_id] for r in model.rerank(query, rows, len(rows))]
