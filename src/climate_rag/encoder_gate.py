"""Evidence-preserving sampling and comparison helpers for dense model-size gates."""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass

import numpy as np

from .metrics import paired_bootstrap
from .models import Claim, EvidenceDocument


@dataclass(frozen=True, slots=True)
class EvidenceSample:
    documents: tuple[EvidenceDocument, ...]
    required_ids: tuple[str, ...]
    source_document_count: int


def required_evidence_ids(claims: Mapping[str, Claim]) -> set[str]:
    return {evidence_id for claim in claims.values() for evidence_id in claim.evidence_ids}


def evidence_preserving_reservoir_sample(
    documents: Iterable[EvidenceDocument],
    required_ids: set[str],
    *,
    sample_size: int,
    seed: int,
) -> EvidenceSample:
    """Sample a fixed-size corpus while guaranteeing that every labelled positive is retained."""

    if sample_size <= 0:
        raise ValueError("sample_size must be positive")
    if sample_size < len(required_ids):
        raise ValueError("sample_size is smaller than the number of required evidence rows")
    capacity = sample_size - len(required_ids)
    rng = np.random.default_rng(seed)
    required: dict[str, EvidenceDocument] = {}
    reservoir: list[EvidenceDocument] = []
    negative_seen = 0
    source_count = 0
    seen_ids: set[str] = set()
    for document in documents:
        source_count += 1
        if document.evidence_id in seen_ids:
            raise ValueError(f"duplicate evidence id: {document.evidence_id}")
        seen_ids.add(document.evidence_id)
        if document.evidence_id in required_ids:
            required[document.evidence_id] = document
            continue
        negative_seen += 1
        if len(reservoir) < capacity:
            reservoir.append(document)
            continue
        if capacity:
            replacement = int(rng.integers(0, negative_seen))
            if replacement < capacity:
                reservoir[replacement] = document
    missing = required_ids - required.keys()
    if missing:
        preview = ", ".join(sorted(missing)[:5])
        raise ValueError(f"required evidence rows are missing: {preview}")
    rows = sorted([*required.values(), *reservoir], key=lambda row: row.evidence_id)
    return EvidenceSample(tuple(rows), tuple(sorted(required_ids)), source_count)


def compare_metric_rows(
    baseline_rows: Sequence[Mapping[str, float | int | str]],
    candidate_rows: Sequence[Mapping[str, float | int | str]],
    *,
    metrics: Sequence[str] = ("recall@5", "mrr@10", "ndcg@10", "evidence_f1"),
    samples: int = 2_000,
    seed: int = 17,
) -> dict[str, dict[str, float | int]]:
    if [row["claim_id"] for row in baseline_rows] != [
        row["claim_id"] for row in candidate_rows
    ]:
        raise ValueError("metric rows must contain the same ordered claim ids")
    return {
        metric: paired_bootstrap(
            [float(row[metric]) for row in baseline_rows],
            [float(row[metric]) for row in candidate_rows],
            samples=samples,
            seed=seed,
        )
        for metric in metrics
    }


def screening_decision(comparisons: Mapping[str, Mapping[str, float | int]]) -> dict[str, object]:
    """Gate a larger model on sampled Recall@5 without promoting it to production."""

    recall = comparisons["recall@5"]
    passed = float(recall["mean_difference"]) > 0 and float(recall["ci_lower"]) > 0
    return {
        "candidate_passes_sample_screen": passed,
        "next_step": (
            "run a full-corpus fixed-dev gate before changing the production encoder"
            if passed
            else "retain the 0.6B production encoder; the sampled gate does not justify a rebuild"
        ),
        "boundary": (
            "A sampled-corpus gate is a resource-screening result, not full-corpus retrieval evidence."
        ),
    }
