from pathlib import Path

import numpy as np

from climate_rag.calibration import fit_temperature, selective_risk_curve, softmax
from climate_rag.io import load_claims, load_predictions
from climate_rag.metrics import evaluate_predictions, paired_bootstrap


FIXTURES = Path(__file__).parents[1] / "fixtures"


def test_candidate_fixture_has_perfect_metrics() -> None:
    metrics, rows, errors = evaluate_predictions(
        load_claims(FIXTURES / "claims.json"),
        load_predictions(FIXTURES / "predictions_candidate.json"),
    )
    assert metrics["recall@5"] == 1.0
    assert metrics["evidence_f1"] == 1.0
    assert metrics["claim_accuracy"] == 1.0
    assert metrics["harmonic_mean"] == 1.0
    assert len(rows) == 4
    assert errors == []


def test_baseline_fixture_surfaces_both_error_layers() -> None:
    metrics, _, errors = evaluate_predictions(
        load_claims(FIXTURES / "claims.json"),
        load_predictions(FIXTURES / "predictions_baseline.json"),
    )
    assert metrics["recall@5"] == 0.5
    assert metrics["claim_accuracy"] == 0.75
    assert metrics["harmonic_mean"] == 0.6
    assert {row["claim_id"] for row in errors} == {"c2", "c3"}


def test_paired_bootstrap_is_seeded_and_positive() -> None:
    result = paired_bootstrap([0, 0, 1, 1], [1, 1, 1, 1], samples=500, seed=3)
    assert result["mean_difference"] == 0.5
    assert result == paired_bootstrap([0, 0, 1, 1], [1, 1, 1, 1], samples=500, seed=3)
    assert result["ci_lower"] >= 0.0


def test_temperature_scaling_and_selective_curve() -> None:
    logits = np.asarray([[8.0, 0.0], [8.0, 0.0], [0.0, 8.0], [8.0, 0.0]])
    labels = [0, 1, 1, 0]
    fitted = fit_temperature(logits, labels)
    assert fitted["nll_after"] <= fitted["nll_before"]
    probabilities = softmax(logits, fitted["temperature"])
    curve = selective_risk_curve(probabilities, labels)
    assert curve[-1]["coverage"] == 1.0
    assert curve[-1]["accuracy"] == 0.75

