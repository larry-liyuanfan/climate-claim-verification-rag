from __future__ import annotations

from collections.abc import Mapping, Sequence

from .models import RankedDocument


def mine_hard_negatives(
    rankings: Mapping[str, Sequence[RankedDocument]],
    gold_evidence_ids: Sequence[str],
    *,
    limit: int = 20,
) -> list[dict[str, object]]:
    """Select high-ranking non-gold documents from one or more retrieval routes."""
    gold = set(gold_evidence_ids)
    candidates: dict[str, dict[str, object]] = {}
    for source in sorted(rankings):
        for fallback_rank, row in enumerate(rankings[source], start=1):
            if row.evidence_id in gold:
                continue
            item = candidates.setdefault(
                row.evidence_id,
                {
                    "evidence_id": row.evidence_id,
                    "text": row.text,
                    "sources": {},
                    "best_rank": fallback_rank,
                },
            )
            source_ranks = item["sources"]
            assert isinstance(source_ranks, dict)
            source_ranks[source] = row.rank if row.rank > 0 else fallback_rank
            item["best_rank"] = min(int(item["best_rank"]), row.rank or fallback_rank)
    ordered = sorted(
        candidates.values(), key=lambda row: (int(row["best_rank"]), str(row["evidence_id"]))
    )
    return ordered[: max(limit, 0)]

