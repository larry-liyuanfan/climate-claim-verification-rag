from __future__ import annotations

import numpy as np
import pytest

from climate_rag.routing import (
    ROUTER_FEATURE_NAMES,
    agreement_features,
    cross_fit_route,
    fit_ridge_gain_model,
    select_quality_preserving_threshold,
    stable_fold,
)


def test_agreement_features_are_bounded_and_inference_safe() -> None:
    features = agreement_features(
        ["a", "b", "c", "d", "e"],
        ["a", "x", "c", "y", "e"],
        ["a", "c", "x", "b", "z"],
    )
    assert features.shape == (len(ROUTER_FEATURE_NAMES),)
    assert np.all(features >= 0.0)
    assert np.all(features <= 1.0)


def test_ridge_gain_model_fits_simple_signal() -> None:
    features = np.asarray([[0.0], [1.0], [2.0], [3.0]])
    gains = np.asarray([0.0, 1.0, 2.0, 3.0])
    model = fit_ridge_gain_model(features, gains, regularization=0.0)
    assert model.predict(np.asarray([[1.5]])).item() == pytest.approx(1.5)


def test_threshold_minimises_calls_subject_to_train_quality() -> None:
    predicted = np.asarray([0.9, 0.8, 0.2, 0.1])
    observed = np.asarray([1.0, 1.0, 1.0, -1.0])
    threshold = select_quality_preserving_threshold(predicted, observed, gain_preservation=0.8)
    selected = predicted >= threshold
    assert selected.tolist() == [True, True, False, False]


def test_cross_fit_route_is_deterministic_and_complete() -> None:
    claim_ids = [f"claim-{index}" for index in range(30)]
    features = np.asarray([[float(index % 3), float(index % 5)] for index in range(30)])
    gains = np.asarray([1.0 if index % 3 == 0 else 0.0 for index in range(30)])
    first = cross_fit_route(claim_ids, features, gains, fold_count=3)
    second = cross_fit_route(claim_ids, features, gains, fold_count=3)
    assert np.array_equal(first[0], second[0])
    assert np.array_equal(first[1], second[1])
    assert len(first[2]) == 3
    assert sum(row["test_count"] for row in first[2]) == len(claim_ids)


def test_stable_fold_validates_fold_count() -> None:
    with pytest.raises(ValueError):
        stable_fold("claim", 1)
