from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np

from .models import Claim, Prediction


def _dcg(relevances: Sequence[int]) -> float:
    return sum(value / math.log2(index + 2) for index, value in enumerate(relevances))


def per_claim_retrieval_metrics(
    claim: Claim, prediction: Prediction, ks: Sequence[int] = (5, 10, 50)
) -> dict[str, float]:
    gold = set(claim.evidence_ids)
    ranked = list(prediction.evidence_ids)
    result: dict[str, float] = {}
    for k in ks:
        selected = ranked[:k]
        correct = len(gold & set(selected))
        result[f"recall@{k}"] = correct / len(gold) if gold else 0.0
        result[f"hit_rate@{k}"] = float(correct > 0)
    reciprocal_rank = 0.0
    for rank, evidence_id in enumerate(ranked[:10], start=1):
        if evidence_id in gold:
            reciprocal_rank = 1.0 / rank
            break
    result["mrr@10"] = reciprocal_rank
    relevances = [1 if evidence_id in gold else 0 for evidence_id in ranked[:10]]
    ideal = [1] * min(len(gold), 10)
    denominator = _dcg(ideal)
    result["ndcg@10"] = _dcg(relevances) / denominator if denominator else 0.0
    correct = len(gold & set(ranked))
    precision = correct / len(ranked) if ranked else 0.0
    recall = correct / len(gold) if gold else 0.0
    result["evidence_precision"] = precision
    result["evidence_recall"] = recall
    result["evidence_f1"] = (
        2.0 * precision * recall / (precision + recall) if precision + recall else 0.0
    )
    return result


def evaluate_predictions(
    claims: Mapping[str, Claim],
    predictions: Mapping[str, Prediction],
    ks: Sequence[int] = (5, 10, 50),
) -> tuple[dict[str, float | int], list[dict[str, Any]], list[dict[str, Any]]]:
    rows: list[dict[str, Any]] = []
    errors: list[dict[str, Any]] = []
    for claim_id in sorted(claims):
        claim = claims[claim_id]
        prediction = predictions.get(claim_id, Prediction(claim_id, ()))
        metrics = per_claim_retrieval_metrics(claim, prediction, ks)
        label_correct: float | None = None
        if claim.label is not None:
            label_correct = float(prediction.label == claim.label)
        row: dict[str, Any] = {
            "claim_id": claim_id,
            "gold_evidence_ids": list(claim.evidence_ids),
            "predicted_evidence_ids": list(prediction.evidence_ids),
            "gold_label": claim.label,
            "predicted_label": prediction.label,
            **metrics,
        }
        if label_correct is not None:
            row["label_correct"] = label_correct
        rows.append(row)
        categories: list[str] = []
        if claim.evidence_ids and not (set(claim.evidence_ids) & set(prediction.evidence_ids)):
            categories.append("retrieval_miss")
        elif metrics["evidence_f1"] < 1.0:
            categories.append("partial_or_over_retrieval")
        if label_correct == 0.0:
            categories.append("classification_error")
        if categories:
            errors.append({"claim_id": claim_id, "categories": categories, **row})
    metric_names = [
        *(f"recall@{k}" for k in ks),
        *(f"hit_rate@{k}" for k in ks),
        "mrr@10",
        "ndcg@10",
        "evidence_precision",
        "evidence_recall",
        "evidence_f1",
    ]
    aggregate: dict[str, float | int] = {"claim_count": len(rows)}
    for metric in metric_names:
        aggregate[metric] = float(np.mean([row[metric] for row in rows])) if rows else 0.0
    labelled = [row for row in rows if "label_correct" in row]
    if labelled:
        accuracy = float(np.mean([row["label_correct"] for row in labelled]))
        evidence_f1 = float(aggregate["evidence_f1"])
        aggregate["labelled_claim_count"] = len(labelled)
        aggregate["claim_accuracy"] = accuracy
        aggregate["harmonic_mean"] = (
            2.0 * evidence_f1 * accuracy / (evidence_f1 + accuracy)
            if evidence_f1 + accuracy
            else 0.0
        )
    return aggregate, rows, errors


def paired_bootstrap(
    baseline: Sequence[float],
    candidate: Sequence[float],
    *,
    samples: int = 2000,
    confidence: float = 0.95,
    seed: int = 17,
) -> dict[str, float | int]:
    left = np.asarray(baseline, dtype=np.float64)
    right = np.asarray(candidate, dtype=np.float64)
    if left.shape != right.shape or left.ndim != 1 or not len(left):
        raise ValueError("paired bootstrap inputs must be non-empty equal-length vectors")
    if samples <= 0:
        raise ValueError("samples must be positive")
    rng = np.random.default_rng(seed)
    differences = np.empty(samples, dtype=np.float64)
    for sample in range(samples):
        indices = rng.integers(0, len(left), size=len(left))
        differences[sample] = np.mean(right[indices] - left[indices])
    alpha = (1.0 - confidence) / 2.0
    probability_non_positive = float(np.mean(differences <= 0.0))
    probability_non_negative = float(np.mean(differences >= 0.0))
    return {
        "pair_count": len(left),
        "samples": samples,
        "baseline_mean": float(left.mean()),
        "candidate_mean": float(right.mean()),
        "mean_difference": float((right - left).mean()),
        "ci_lower": float(np.quantile(differences, alpha)),
        "ci_upper": float(np.quantile(differences, 1.0 - alpha)),
        "two_sided_p": min(1.0, 2.0 * min(probability_non_positive, probability_non_negative)),
    }

