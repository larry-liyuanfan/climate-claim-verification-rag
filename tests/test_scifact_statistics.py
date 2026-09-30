"""Synthetic matrices only; never SciFact dev predictions."""

import copy
from dataclasses import replace

import numpy as np
import pytest

from climate_rag.scifact_grounding import Abstract, GoldClaim, Rationale
from climate_rag.scifact_scoring import parse_prediction, score_original
from climate_rag.scifact_statistics import (
    METRICS,
    ROUTES,
    RunRecord,
    component_weights,
    evaluate_matrix,
    micro_f1,
)


def fixture():
    corpus = {
        10: Abstract(
            10, "Synthetic paper", tuple(f"sentence {i}" for i in range(5)), False
        ),
        20: Abstract(20, "Another synthetic paper", ("first", "second"), False),
    }
    gold = [
        GoldClaim(1, "one", {10: (Rationale("SUPPORT", (0,)),)}, ()),
        GoldClaim(
            2,
            "two",
            {10: (Rationale("SUPPORT", (0, 1)),), 20: (Rationale("CONTRADICT", (1,)),)},
            (),
        ),
        GoldClaim(3, "three", {}, ()),
        GoldClaim(4, "four", {10: (Rationale("SUPPORT", (2,)),)}, ()),
        GoldClaim(5, "five", {20: (Rationale("SUPPORT", (0,)),)}, ()),
    ]
    predictions = [
        {"id": 1, "evidence": {"10": {"label": "SUPPORT", "sentences": [0]}}},
        {"id": 2, "evidence": {"10": {"label": "SUPPORT", "sentences": [0]}}},
        {"id": 3, "evidence": {}},
        {"id": 4, "evidence": {}},
        {"id": 5, "evidence": {"20": {"label": "SUPPORT", "sentences": [0]}}},
    ]
    rows = [
        RunRecord(
            route,
            copy.deepcopy(pred),
            "model_abstention:insufficient_evidence"
            if pred["id"] == 3
            else (
                "controller_failure:RuntimeError"
                if pred["id"] == 4
                else "ids_validated_semantics_unmeasured"
            ),
            float(pred["id"] * 100),
            pred["id"] * 30,
            pred["id"] * 5,
            0,
        )
        for route in ROUTES
        for pred in predictions
    ]
    groups = {1: "a", 2: "b", 3: "b", 4: "b", 5: "c", 99: "train-only"}
    return gold, rows, corpus, groups


def test_paired_whole_group_weights_and_micro_not_macro():
    gold, rows, corpus, groups = fixture()
    result = evaluate_matrix(gold, rows, corpus, groups)
    bootstrap = result["bootstrap"]
    assert bootstrap["replicates"] == 5000 and bootstrap["nonempty_target_groups"] == 3
    assert bootstrap["excluded_non_target_mapping_entries"] == 1
    assert bootstrap["weighted_claim_count_range"] == [3, 9]
    assert bootstrap["duplicate_group_draw_replicates"] > 0
    assert result["routes"]["adaptive"]["failures_retained"] == 1
    for comparison in result["paired_comparisons"].values():
        for metric in comparison["quality"].values():
            assert metric["point_delta_f1"] == 0
            assert metric["ci95"] == [0, 0] and metric["p_value"] is None
        assert comparison["mean_whole_question_elapsed_ms"]["ci95"] == [0, 0]
    per_claim_f1 = [
        score_original([g], [parse_prediction(row.prediction, corpus)])["metrics"][
            "abstract_rationalized"
        ]["f1"]
        for g, row in zip(gold, rows[:5], strict=True)
    ]
    actual_micro = result["routes"]["fixed_retrieval"]["official_score"]["metrics"][
        "abstract_rationalized"
    ]["f1"]
    assert actual_micro != np.mean(per_claim_f1)
    assert actual_micro == 0.5  # 2 correct / (3 predicted + 5 relevant), times 2.


def test_duplicate_group_samples_pool_integer_counts_before_f1():
    weights = component_weights(2)
    assert weights.shape == (5000, 2) and np.all(weights.sum(axis=1) == 2)
    assert any(np.array_equal(w, [2, 0]) for w in weights)
    counts = np.array([[1, 1, 1], [0, 4, 8]])
    values = micro_f1(weights @ counts)
    for pattern, expected in [([2, 0], 1), ([0, 2], 0), ([1, 1], 1 / 7)]:
        selected = np.all(weights == pattern, axis=1)
        assert np.allclose(values[selected], expected)
    with pytest.raises(ValueError, match="nonempty"):
        component_weights(0)


def test_empty_and_duplicate_gold_are_not_bootstrapped():
    gold, rows, corpus, groups = fixture()
    with pytest.raises(ValueError, match="empty/duplicate"):
        evaluate_matrix([], rows, corpus, groups)
    with pytest.raises(ValueError, match="empty/duplicate"):
        evaluate_matrix(gold + [gold[0]], rows, corpus, groups)


def test_input_sorting_and_global_unused_components_do_not_change_draws():
    gold, rows, corpus, groups = fixture()
    first = evaluate_matrix(gold, rows, corpus, groups)
    second = evaluate_matrix(
        list(reversed(gold)),
        list(reversed(rows)),
        corpus,
        dict(reversed(list(groups.items()))),
    )
    assert first == second
    assert (
        first["bootstrap"]["integer_weights_sha256"]
        == evaluate_matrix(gold, rows, corpus, {**groups, 100: "another-train-only"})[
            "bootstrap"
        ]["integer_weights_sha256"]
    )


def test_cost_means_use_weighted_claim_count_not_group_count():
    gold, rows, corpus, groups = fixture()
    # Adaptive extra elapsed cost = 30 ms on each member of the 3-claim group.
    rows = [
        replace(r, whole_question_elapsed_ms=r.whole_question_elapsed_ms + 30)
        if r.route == "adaptive" and r.prediction["id"] in {2, 3, 4}
        else r
        for r in rows
    ]
    result = evaluate_matrix(
        gold,
        rows,
        corpus,
        groups,
        separate_phase_costs={"model_load_ms": 999, "runtime_preflight_ms": 888},
    )
    delta = result["paired_comparisons"]["adaptive-minus-fixed_rerank"][
        "mean_whole_question_elapsed_ms"
    ]
    assert delta["point_delta"] == 18
    weights = component_weights(3)
    expected = 90 * weights[:, 1] / (weights @ np.array([1, 3, 1]))
    assert np.allclose(
        delta["ci95"], np.quantile(expected, [0.025, 0.975], method="linear")
    )
    assert result["routes"]["adaptive"]["cost"]["whole_question_p95_ms"] == np.quantile(
        [100, 230, 330, 430, 500], 0.95, method="linear"
    )
    assert result["routes"]["adaptive"]["cost"]["latency_quantile_ci"] is None
    assert result["separate_load_and_preflight_costs"]["model_load_ms"] == 999


def test_unknown_tokens_keep_lower_bounds_no_exact_delta_or_savings():
    gold, rows, corpus, groups = fixture()
    rows = [
        replace(
            r,
            unknown_usage_attempts=2,
            input_tokens_lower_bound=10,
            output_tokens_lower_bound=0,
        )
        if r.route == "adaptive" and r.prediction["id"] == 4
        else r
        for r in rows
    ]
    report = evaluate_matrix(gold, rows, corpus, groups)
    cost = report["routes"]["adaptive"]["cost"]
    assert not cost["token_totals_exact"] and cost["unknown_usage_attempts"] == 2
    assert cost["mean_input_tokens_lower_bound"] == (30 + 60 + 90 + 10 + 150) / 5
    for pair in report["paired_comparisons"].values():
        assert pair["mean_token_delta"] is None
        assert "unknown usage" in pair["token_delta_reason"]


def test_single_component_and_missing_nei_do_not_report_fake_ci():
    gold, rows, corpus, groups = fixture()
    gold = [g for g in gold if g.claim_id != 3]
    rows = [r for r in rows if r.prediction["id"] != 3]
    groups = {g.claim_id: "one" for g in gold}
    report = evaluate_matrix(gold, rows, corpus, groups)
    assert (
        report["routes"]["adaptive"]["official_score"][
            "separate_diagnostics_not_official_f1"
        ]["nei_false_evidence_rate"]
        is None
    )
    for pair in report["paired_comparisons"].values():
        for metric in pair["quality"].values():
            assert metric["ci95"] is None


def test_multi_document_opposite_labels_and_nei_are_not_flattened():
    gold, rows, corpus, groups = fixture()
    exact = {
        "id": 2,
        "evidence": {
            "10": {"label": "SUPPORT", "sentences": [0, 1]},
            "20": {"label": "CONTRADICT", "sentences": [1]},
        },
    }
    rows = [
        replace(r, prediction=exact)
        if r.route == "adaptive" and r.prediction["id"] == 2
        else r
        for r in rows
    ]
    report = evaluate_matrix(gold, rows, corpus, groups)
    adaptive = report["routes"]["adaptive"]["official_score"]
    assert adaptive["metrics"]["abstract_rationalized"]["correct"] == 4
    assert (
        adaptive["separate_diagnostics_not_official_f1"]["mixed_label_claim_count"] == 1
    )
    assert adaptive["separate_diagnostics_not_official_f1"]["nei_claim_count"] == 1
    assert report["claim_accuracy"] is None
    assert set(
        report["paired_comparisons"]["adaptive-minus-fixed_rerank"]["quality"]
    ) == set(METRICS)


@pytest.mark.parametrize(
    "problem",
    [
        "missing",
        "duplicate",
        "unknown_route",
        "extra_id",
        "invalid_sentence",
        "failure_with_evidence",
        "unknown_termination",
        "missing_group",
        "nan",
        "negative",
        "infinite",
        "bool_elapsed",
        "negative_tokens",
        "float_tokens",
        "wrong_scope",
    ],
)
def test_missing_duplicate_and_corrupt_matrices_fail_closed(problem):
    gold, rows, corpus, groups = fixture()
    if problem == "missing":
        rows.pop()
    elif problem == "duplicate":
        rows.append(rows[0])
    elif problem == "missing_group":
        del groups[1]
    else:
        updates = {
            "unknown_route": {"route": "other"},
            "extra_id": {"prediction": {"id": 999, "evidence": {}}},
            "invalid_sentence": {
                "prediction": {
                    "id": 1,
                    "evidence": {"10": {"label": "SUPPORT", "sentences": [99]}},
                }
            },
            "failure_with_evidence": {
                "termination_reason": "controller_failure:RuntimeError"
            },
            "unknown_termination": {"termination_reason": "success"},
            "nan": {"whole_question_elapsed_ms": float("nan")},
            "negative": {"whole_question_elapsed_ms": -1},
            "infinite": {"whole_question_elapsed_ms": float("inf")},
            "bool_elapsed": {"whole_question_elapsed_ms": True},
            "negative_tokens": {"input_tokens_lower_bound": -1},
            "float_tokens": {"unknown_usage_attempts": 0.5},
            "wrong_scope": {"cost_scope": "last_successful_call"},
        }
        rows[0] = replace(rows[0], **updates[problem])
    with pytest.raises(ValueError):
        evaluate_matrix(gold, rows, corpus, groups)
