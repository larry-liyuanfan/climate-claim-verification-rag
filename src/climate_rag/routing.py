from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import numpy as np


ROUTER_FEATURE_NAMES = (
    "bm25_dense_overlap_at_1",
    "bm25_dense_overlap_at_3",
    "bm25_dense_overlap_at_5",
    "bm25_rrf_overlap_at_1",
    "bm25_rrf_overlap_at_3",
    "bm25_rrf_overlap_at_5",
    "dense_rrf_overlap_at_1",
    "dense_rrf_overlap_at_3",
    "dense_rrf_overlap_at_5",
    "bm25_dense_rank_agreement",
    "bm25_rrf_rank_agreement",
    "dense_rrf_rank_agreement",
    "rrf_supported_by_both",
    "rrf_supported_by_exactly_one",
)


def _overlap(left: Sequence[str], right: Sequence[str], k: int) -> float:
    return float(len(set(left[:k]) & set(right[:k]))) / float(k)


def _rank_agreement(left: Sequence[str], right: Sequence[str]) -> float:
    right_ranks = {item: rank for rank, item in enumerate(right, start=1)}
    score = 0.0
    for left_rank, item in enumerate(left, start=1):
        right_rank = right_ranks.get(item)
        if right_rank is not None:
            score += 2.0 / float(left_rank + right_rank)
    return score / float(max(len(left), len(right), 1))


def agreement_features(
    bm25_ids: Sequence[str], dense_ids: Sequence[str], rrf_ids: Sequence[str]
) -> np.ndarray:
    bm25 = tuple(bm25_ids)
    dense = tuple(dense_ids)
    rrf = tuple(rrf_ids)
    bm25_set = set(bm25)
    dense_set = set(dense)
    supported_by_both = sum(item in bm25_set and item in dense_set for item in rrf)
    supported_by_one = sum((item in bm25_set) ^ (item in dense_set) for item in rrf)
    values = [
        *(_overlap(bm25, dense, k) for k in (1, 3, 5)),
        *(_overlap(bm25, rrf, k) for k in (1, 3, 5)),
        *(_overlap(dense, rrf, k) for k in (1, 3, 5)),
        _rank_agreement(bm25, dense),
        _rank_agreement(bm25, rrf),
        _rank_agreement(dense, rrf),
        float(supported_by_both) / float(max(len(rrf), 1)),
        float(supported_by_one) / float(max(len(rrf), 1)),
    ]
    return np.asarray(values, dtype=np.float64)


@dataclass(frozen=True)
class RidgeGainModel:
    mean: np.ndarray
    scale: np.ndarray
    coefficients: np.ndarray

    def predict(self, features: np.ndarray) -> np.ndarray:
        matrix = np.atleast_2d(np.asarray(features, dtype=np.float64))
        standardized = (matrix - self.mean) / self.scale
        design = np.column_stack((np.ones(len(matrix)), standardized))
        return design @ self.coefficients


def fit_ridge_gain_model(
    features: np.ndarray, gains: np.ndarray, *, regularization: float = 1.0
) -> RidgeGainModel:
    matrix = np.asarray(features, dtype=np.float64)
    target = np.asarray(gains, dtype=np.float64)
    if matrix.ndim != 2 or target.ndim != 1 or len(matrix) != len(target) or not len(target):
        raise ValueError("features and gains must be non-empty aligned arrays")
    if regularization < 0.0:
        raise ValueError("regularization must be non-negative")
    mean = matrix.mean(axis=0)
    scale = matrix.std(axis=0)
    scale = np.where(scale > 1e-12, scale, 1.0)
    design = np.column_stack((np.ones(len(matrix)), (matrix - mean) / scale))
    penalty = np.eye(design.shape[1], dtype=np.float64) * regularization
    penalty[0, 0] = 0.0
    coefficients = np.linalg.pinv(design.T @ design + penalty) @ design.T @ target
    return RidgeGainModel(mean=mean, scale=scale, coefficients=coefficients)


def select_quality_preserving_threshold(
    predicted_gains: np.ndarray,
    observed_gains: np.ndarray,
    *,
    gain_preservation: float = 0.8,
) -> float:
    predicted = np.asarray(predicted_gains, dtype=np.float64)
    observed = np.asarray(observed_gains, dtype=np.float64)
    if predicted.shape != observed.shape or predicted.ndim != 1 or not len(predicted):
        raise ValueError("predicted and observed gains must be non-empty aligned vectors")
    if not 0.0 <= gain_preservation <= 1.0:
        raise ValueError("gain_preservation must be between zero and one")
    full_gain = float(observed.mean())
    target_gain = max(0.0, gain_preservation * full_gain)
    candidates = [float("inf"), *sorted(set(predicted.tolist()), reverse=True), float("-inf")]
    feasible: list[tuple[float, float, float]] = []
    for threshold in candidates:
        selected = predicted >= threshold
        realised_gain = float(np.mean(np.where(selected, observed, 0.0)))
        rate = float(selected.mean())
        if realised_gain + 1e-12 >= target_gain:
            feasible.append((rate, -realised_gain, threshold))
    if not feasible:
        return float("-inf")
    return min(feasible)[2]


def stable_fold(claim_id: str, fold_count: int) -> int:
    if fold_count < 2:
        raise ValueError("fold_count must be at least two")
    digest = hashlib.sha256(claim_id.encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big") % fold_count


def cross_fit_route(
    claim_ids: Sequence[str],
    features: np.ndarray,
    gains: np.ndarray,
    *,
    fold_count: int = 5,
    regularization: float = 1.0,
    gain_preservation: float = 0.8,
) -> tuple[np.ndarray, np.ndarray, list[dict[str, Any]]]:
    identifiers = tuple(claim_ids)
    matrix = np.asarray(features, dtype=np.float64)
    target = np.asarray(gains, dtype=np.float64)
    if matrix.ndim != 2 or target.ndim != 1 or len(identifiers) != len(matrix) or len(matrix) != len(target):
        raise ValueError("claim_ids, features and gains must be aligned")
    folds = np.asarray([stable_fold(item, fold_count) for item in identifiers], dtype=np.int64)
    predictions = np.empty(len(identifiers), dtype=np.float64)
    selected = np.zeros(len(identifiers), dtype=bool)
    fold_reports: list[dict[str, Any]] = []
    for fold in range(fold_count):
        test_mask = folds == fold
        train_mask = ~test_mask
        if not test_mask.any() or not train_mask.any():
            raise ValueError(f"stable hashing produced an empty fold: {fold}")
        model = fit_ridge_gain_model(
            matrix[train_mask], target[train_mask], regularization=regularization
        )
        train_predictions = model.predict(matrix[train_mask])
        threshold = select_quality_preserving_threshold(
            train_predictions,
            target[train_mask],
            gain_preservation=gain_preservation,
        )
        test_predictions = model.predict(matrix[test_mask])
        test_selected = test_predictions >= threshold
        predictions[test_mask] = test_predictions
        selected[test_mask] = test_selected
        fold_reports.append(
            {
                "fold": fold,
                "train_count": int(train_mask.sum()),
                "test_count": int(test_mask.sum()),
                "threshold": threshold,
                "train_strong_call_rate": float((train_predictions >= threshold).mean()),
                "test_strong_call_rate": float(test_selected.mean()),
            }
        )
    return selected, predictions, fold_reports


def group_prediction_rows(
    rows: Sequence[Mapping[str, Any]],
) -> dict[str, dict[str, Mapping[str, Any]]]:
    grouped: dict[str, dict[str, Mapping[str, Any]]] = {}
    for row in rows:
        claim_id = str(row["claim_id"])
        system = str(row["system"])
        grouped.setdefault(claim_id, {})[system] = row
    return grouped
