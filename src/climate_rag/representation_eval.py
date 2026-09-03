from __future__ import annotations

import re
from collections import defaultdict
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np

from .io import read_json, read_jsonl
from .metrics import evaluate_predictions, paired_bootstrap
from .models import Claim, EvidenceDocument, Prediction, prediction_from_mapping

TAXONOMY = (
    "entity",
    "numeric_or_year",
    "geographic",
    "lexical_mismatch",
    "semantic_inference",
    "multi_evidence",
    "unanswerable",
)
GEOGRAPHIC_TERMS = frozenset(
    {
        "africa",
        "antarctic",
        "antarctica",
        "arctic",
        "asia",
        "australia",
        "canada",
        "china",
        "europe",
        "global",
        "greenland",
        "india",
        "ocean",
        "pacific",
        "region",
        "russia",
        "uk",
        "united kingdom",
        "united states",
        "us",
    }
)
METRICS = ("recall@5", "mrr@10", "ndcg@10", "evidence_f1")


def _normalise(value: str) -> str:
    return " ".join(re.findall(r"[a-z0-9]+", value.casefold()))


def _tokens(value: str) -> frozenset[str]:
    return frozenset(_normalise(value).split())


def load_prediction_variant(
    path: str | Path, *, variant: str | None = None
) -> dict[str, Prediction]:
    """Load a regular prediction file or one named variant from a mixed run JSONL."""

    source = Path(path)
    rows: Iterable[Mapping[str, Any]]
    if source.suffix.lower() == ".jsonl":
        rows = read_jsonl(source)
    else:
        payload = read_json(source)
        if not isinstance(payload, dict):
            raise TypeError("prediction JSON must be keyed by claim ID")
        rows = (
            {"claim_id": claim_id, **value}
            for claim_id, value in payload.items()
            if isinstance(value, dict)
        )
    result: dict[str, Prediction] = {}
    for row in rows:
        if (
            variant is not None
            and str(row.get("model", row.get("system", ""))) != variant
        ):
            continue
        claim_id = row.get("claim_id", row.get("id"))
        if claim_id is None or "evidence_ids" not in row:
            continue
        claim_id = str(claim_id)
        if claim_id in result:
            raise ValueError(
                f"duplicate prediction for claim {claim_id} and variant {variant}"
            )
        prediction = prediction_from_mapping(claim_id, row)
        if len(prediction.evidence_ids) != len(set(prediction.evidence_ids)):
            raise ValueError(f"prediction {claim_id} contains duplicate evidence IDs")
        result[claim_id] = prediction
    if not result:
        raise ValueError(f"no predictions found for variant {variant!r}")
    return result


def infer_query_taxonomy(
    claim: Claim,
    evidence_by_id: Mapping[str, EvidenceDocument],
    *,
    lexical_mismatch_threshold: float = 0.15,
) -> tuple[str, ...]:
    """Return deterministic, auditable query tags; these are not human labels."""

    text = claim.text
    normalised = _normalise(text)
    tokens = _tokens(text)
    labels: list[str] = []
    if re.search(r"\b(?:18|19|20)\d{2}\b|\b\d+(?:\.\d+)?%?\b", text):
        labels.append("numeric_or_year")
    if any(term in normalised for term in GEOGRAPHIC_TERMS):
        labels.append("geographic")
    if re.search(r"\b[A-Z]{2,}\b|(?:^|[.!?]\s+)[A-Z][a-z]+\s+[A-Z][a-z]+", text):
        labels.append("entity")
    if len(claim.evidence_ids) > 1:
        labels.append("multi_evidence")
    if not claim.evidence_ids:
        labels.append("unanswerable")
    else:
        overlaps: list[float] = []
        for evidence_id in claim.evidence_ids:
            evidence = evidence_by_id.get(evidence_id)
            if evidence is None:
                continue
            evidence_tokens = _tokens(evidence.text)
            union = tokens | evidence_tokens
            overlaps.append(
                len(tokens & evidence_tokens) / len(union) if union else 0.0
            )
        maximum_overlap = max(overlaps, default=0.0)
        if maximum_overlap < lexical_mismatch_threshold:
            labels.append("lexical_mismatch")
        if maximum_overlap < 0.30:
            labels.append("semantic_inference")
    return tuple(name for name in TAXONOMY if name in labels)


def evaluate_representation_pair(
    claims: Mapping[str, Claim],
    evidence: Iterable[EvidenceDocument],
    baseline: Mapping[str, Prediction],
    candidate: Mapping[str, Prediction],
    *,
    bootstrap_samples: int = 5_000,
    seed: int = 17,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Paired retrieval comparison with deterministic query-taxonomy slices."""

    if bootstrap_samples < 5_000:
        raise ValueError(
            "representation evaluation requires at least 5,000 bootstrap samples"
        )
    claim_ids = set(claims)
    if set(baseline) != claim_ids or set(candidate) != claim_ids:
        raise ValueError("base, candidate and claim query sets must match exactly")
    evidence_by_id = {document.evidence_id: document for document in evidence}
    base_metrics, base_rows, _ = evaluate_predictions(claims, baseline, ks=(5, 10, 50))
    candidate_metrics, candidate_rows, _ = evaluate_predictions(
        claims, candidate, ks=(5, 10, 50)
    )
    base_by_id = {str(row["claim_id"]): row for row in base_rows}
    candidate_by_id = {str(row["claim_id"]): row for row in candidate_rows}
    comparisons = {
        metric: paired_bootstrap(
            [float(base_by_id[claim_id][metric]) for claim_id in sorted(claims)],
            [float(candidate_by_id[claim_id][metric]) for claim_id in sorted(claims)],
            samples=bootstrap_samples,
            seed=seed,
        )
        for metric in METRICS
    }
    tagged_rows: list[dict[str, Any]] = []
    by_taxonomy: dict[str, list[str]] = defaultdict(list)
    for claim_id in sorted(claims):
        labels = infer_query_taxonomy(claims[claim_id], evidence_by_id)
        for label in labels:
            by_taxonomy[label].append(claim_id)
        tagged_rows.append(
            {
                "claim_id": claim_id,
                "taxonomy": list(labels),
                "baseline": {
                    metric: float(base_by_id[claim_id][metric]) for metric in METRICS
                },
                "candidate": {
                    metric: float(candidate_by_id[claim_id][metric])
                    for metric in METRICS
                },
            }
        )
    taxonomy_report: dict[str, Any] = {}
    for label in TAXONOMY:
        ids = by_taxonomy.get(label, [])
        taxonomy_report[label] = {
            "query_count": len(ids),
            "baseline": {
                metric: _mean([float(base_by_id[claim_id][metric]) for claim_id in ids])
                for metric in METRICS
            },
            "candidate": {
                metric: _mean(
                    [float(candidate_by_id[claim_id][metric]) for claim_id in ids]
                )
                for metric in METRICS
            },
            "mean_delta": {
                metric: _mean(
                    [
                        float(candidate_by_id[claim_id][metric])
                        - float(base_by_id[claim_id][metric])
                        for claim_id in ids
                    ]
                )
                for metric in METRICS
            },
        }
    return (
        {
            "query_count": len(claims),
            "bootstrap_samples": bootstrap_samples,
            "seed": seed,
            "baseline": {metric: base_metrics[metric] for metric in METRICS},
            "candidate": {metric: candidate_metrics[metric] for metric in METRICS},
            "paired_bootstrap": comparisons,
            "taxonomy": taxonomy_report,
            "taxonomy_boundary": (
                "Taxonomy labels are deterministic diagnostics derived from query/gold text, "
                "not human semantic annotations."
            ),
        },
        tagged_rows,
    )


def build_pareto_report(
    profiles: Sequence[Mapping[str, Any]], *, quality_metric: str = "evidence_f1"
) -> dict[str, Any]:
    """Build Pareto fronts only within profiles with comparable measurement scopes."""

    required = {"name", quality_metric, "p95_ms", "memory_bytes", "comparability_group"}
    parsed: list[dict[str, Any]] = []
    for raw in profiles:
        missing = sorted(required - raw.keys())
        if missing:
            raise ValueError(f"profile is missing required fields: {missing}")
        parsed.append(dict(raw))
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for profile in parsed:
        groups[str(profile["comparability_group"])].append(profile)
    for group_profiles in groups.values():
        for profile in group_profiles:
            dominators = [
                other
                for other in group_profiles
                if other is not profile and _dominates(other, profile, quality_metric)
            ]
            profile["pareto_status"] = "dominated" if dominators else "frontier"
            profile["dominated_by"] = [str(other["name"]) for other in dominators]
    return {
        "quality_metric": quality_metric,
        "profiles": parsed,
        "boundary": (
            "Pareto labels are computed only inside a comparability_group. ANN-only, "
            "feature-scoring-only and cross-encoder-only timings must not be added as if "
            "they were end-to-end service latency."
        ),
    }


def _dominates(
    left: Mapping[str, Any], right: Mapping[str, Any], quality_metric: str
) -> bool:
    left_values = (
        float(left[quality_metric]),
        -float(left["p95_ms"]),
        -float(left["memory_bytes"]),
    )
    right_values = (
        float(right[quality_metric]),
        -float(right["p95_ms"]),
        -float(right["memory_bytes"]),
    )
    return all(
        left_value >= right_value
        for left_value, right_value in zip(left_values, right_values, strict=True)
    ) and any(
        left_value > right_value
        for left_value, right_value in zip(left_values, right_values, strict=True)
    )


def _mean(values: Sequence[float]) -> float | None:
    return float(np.mean(values)) if values else None
