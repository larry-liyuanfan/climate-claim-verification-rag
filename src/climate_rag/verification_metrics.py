from __future__ import annotations

from collections import Counter
from collections.abc import Mapping, Sequence
from itertools import pairwise
from typing import Any

import numpy as np

LABELS = ("SUPPORTS", "REFUTES", "NOT_ENOUGH_INFO")


def verification_metrics(
    gold: Sequence[str],
    predicted: Sequence[str],
    confidences: Sequence[float],
    *,
    bins: int = 10,
) -> dict[str, Any]:
    if not (len(gold) == len(predicted) == len(confidences)) or not gold:
        raise ValueError(
            "gold, predicted and confidences must be non-empty and equally sized"
        )
    if bins <= 0:
        raise ValueError("bins must be positive")
    values = np.asarray(confidences, dtype=np.float64)
    if np.any((values < 0.0) | (values > 1.0)):
        raise ValueError("confidences must be in [0, 1]")
    correct = np.asarray(
        [left == right for left, right in zip(gold, predicted, strict=True)]
    )
    per_label: dict[str, dict[str, float | int]] = {}
    f1_values: list[float] = []
    for label in LABELS:
        tp = sum(
            g == label and p == label for g, p in zip(gold, predicted, strict=True)
        )
        fp = sum(
            g != label and p == label for g, p in zip(gold, predicted, strict=True)
        )
        fn = sum(
            g == label and p != label for g, p in zip(gold, predicted, strict=True)
        )
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        f1 = (
            2 * precision * recall / (precision + recall) if precision + recall else 0.0
        )
        f1_values.append(f1)
        per_label[label] = {
            "support": sum(item == label for item in gold),
            "precision": precision,
            "recall": recall,
            "f1": f1,
        }
    one_hot_error = np.asarray(
        [
            1.0 - confidence if is_correct else confidence
            for confidence, is_correct in zip(values, correct, strict=True)
        ]
    )
    ece = 0.0
    edges = np.linspace(0.0, 1.0, bins + 1)
    for left, right in pairwise(edges):
        selected = (values >= left) & (
            values <= right if right == 1.0 else values < right
        )
        if selected.any():
            ece += float(selected.mean()) * abs(
                float(values[selected].mean()) - float(correct[selected].mean())
            )
    return {
        "count": len(gold),
        "accuracy": float(correct.mean()),
        "macro_f1": float(np.mean(f1_values)),
        "brier": float(np.mean(one_hot_error**2)),
        "ece": ece,
        "label_counts": dict(Counter(gold)),
        "per_label": per_label,
    }


def citation_metrics(
    gold: Mapping[str, set[str]], predicted: Mapping[str, set[str]]
) -> dict[str, float | int]:
    claim_ids = sorted(set(gold) | set(predicted))
    tp = fp = fn = complete = 0
    for claim_id in claim_ids:
        left = gold.get(claim_id, set())
        right = predicted.get(claim_id, set())
        tp += len(left & right)
        fp += len(right - left)
        fn += len(left - right)
        complete += int(bool(left) and left <= right)
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    return {
        "claim_count": len(claim_ids),
        "citation_precision": precision,
        "citation_recall": recall,
        "citation_f1": 2 * precision * recall / (precision + recall)
        if precision + recall
        else 0.0,
        "citation_completeness": complete / len(claim_ids) if claim_ids else 0.0,
    }
