from __future__ import annotations

from collections.abc import Sequence

import numpy as np


def softmax(logits: np.ndarray, temperature: float = 1.0) -> np.ndarray:
    if temperature <= 0:
        raise ValueError("temperature must be positive")
    values = np.asarray(logits, dtype=np.float64) / temperature
    values -= values.max(axis=1, keepdims=True)
    exponentials = np.exp(values)
    return exponentials / exponentials.sum(axis=1, keepdims=True)


def negative_log_likelihood(logits: np.ndarray, labels: Sequence[int], temperature: float) -> float:
    probabilities = softmax(logits, temperature)
    targets = np.asarray(labels, dtype=np.int64)
    if len(probabilities) != len(targets):
        raise ValueError("logits and labels must have equal rows")
    chosen = probabilities[np.arange(len(targets)), targets]
    return float(-np.log(np.maximum(chosen, 1e-15)).mean())


def fit_temperature(
    logits: np.ndarray,
    labels: Sequence[int],
    *,
    minimum: float = 0.05,
    maximum: float = 10.0,
    grid_points: int = 401,
) -> dict[str, float]:
    if minimum <= 0 or maximum <= minimum or grid_points < 2:
        raise ValueError("invalid temperature grid")
    candidates = np.geomspace(minimum, maximum, grid_points)
    losses = np.asarray([negative_log_likelihood(logits, labels, value) for value in candidates])
    best_index = int(np.argmin(losses))
    return {
        "temperature": float(candidates[best_index]),
        "nll_before": negative_log_likelihood(logits, labels, 1.0),
        "nll_after": float(losses[best_index]),
    }


def selective_risk_curve(probabilities: np.ndarray, labels: Sequence[int]) -> list[dict[str, float | int]]:
    values = np.asarray(probabilities, dtype=np.float64)
    targets = np.asarray(labels, dtype=np.int64)
    if values.ndim != 2 or len(values) != len(targets):
        raise ValueError("probabilities must be a 2D matrix aligned with labels")
    confidence = values.max(axis=1)
    predicted = values.argmax(axis=1)
    order = np.argsort(-confidence, kind="stable")
    rows: list[dict[str, float | int]] = []
    errors = 0
    for accepted, index in enumerate(order, start=1):
        errors += int(predicted[index] != targets[index])
        rows.append(
            {
                "accepted": accepted,
                "coverage": accepted / len(targets),
                "risk": errors / accepted,
                "accuracy": 1.0 - errors / accepted,
                "minimum_confidence": float(confidence[index]),
            }
        )
    return rows
