"""Synthetic disclosure tests, not experiment quality or semantic labels."""
import copy
import json

import pytest

from climate_rag.fair_acquisition import COVERAGE_PROTOCOL, ROUTES
from export_fair_run import (
    BOOTSTRAP, METRICS, QUALITY, project, route_cost, validate_cost_roster, validate_public_identity,
)


def compact():
    g = {"unique_physical_calls": 2, "unknown_usage_attempts": 0, "elapsed_ms": 12,
         "known_token_lower_bound": {"input_tokens": 10, "output_tokens": 5}}
    r = {"physical_requests": 1, "requested_pairs": 20, "completed_pairs": 20,
         "unknown_requests": 0, "elapsed_ms_including_swaps": 7, "known_token_lower_bound": 30}
    row = {"task_id": "PRIVATE_TASK_SENTINEL", "generation": g, "reranker": r,
           "tool_calls": 3, "elapsed_ms": 25, "claim": "PRIVATE_CLAIM_SENTINEL"}
    route = {"all_task_denominator": 32, **dict.fromkeys(QUALITY, .5),
        "behavior": {"answered": 30, "PRIVATE_KEY_SENTINEL": 1},
        "retrieval": {"denominator": 13, "scope": "PRIVATE_SCOPE_SENTINEL",
                      "metrics": dict.fromkeys(METRICS, .5)},
        "physical_cost_by_task": [copy.deepcopy(row) for _ in range(32)]}
    b = dict.fromkeys(BOOTSTRAP, 0)
    return {"protocol": COVERAGE_PROTOCOL, "routes": {a: copy.deepcopy(route) for a in ROUTES},
        "paired_bootstrap": {"autonomous-minus-" + a: {k: b for k in QUALITY[:2]} for a in ROUTES[:2]},
        "PRIVATE_TOP_SENTINEL": "PRIVATE_TEXT_SENTINEL"}


def test_numeric_projection_does_not_copy_private_fields_or_ids():
    result = project(compact())
    assert "PRIVATE" not in json.dumps(result)
    cost = result["routes"]["autonomous"]["cost"]
    assert cost["generation_calls"] == 64
    assert cost["input_tokens_known_lower_bound"] == 320
    assert cost["reranker_completed_pairs"] == 640
    assert cost["slot_p95_nearest_rank_ms"] == 25


def test_incomplete_denominator_and_non_numeric_measurement_fail():
    value = compact()
    value["routes"][ROUTES[0]]["physical_cost_by_task"].pop()
    with pytest.raises(ValueError, match="denominator"):
        project(value)
    value = compact()
    value["routes"][ROUTES[0]]["official_task_correct"] = "PRIVATE_STRING"
    with pytest.raises(ValueError, match="numeric"):
        project(value)


def test_unknown_time_usage_is_not_silently_zeroed():
    rows = compact()["routes"][ROUTES[0]]["physical_cost_by_task"]
    rows[0]["generation"].update(elapsed_ms=None, unknown_usage_attempts=1)
    result = route_cost(rows)
    assert result["generation_ms"] is None
    assert result["unknown_generator_usage"] == 1


def test_public_identity_and_reserved_binding():
    operator = {"source_git": "a" * 40, "release_sha256": "b" * 64, "planned_slots": 96}
    validate_public_identity(operator, dict(operator))
    for key in ("source_git", "release_sha256", "planned_slots"):
        reserved = dict(operator)
        reserved[key] = "PRIVATE_SENTINEL"
        with pytest.raises(ValueError, match="identity"):
            validate_public_identity(operator, reserved)
    operator["source_git"] = "PRIVATE_SENTINEL"
    with pytest.raises(ValueError, match="identity"):
        validate_public_identity(operator, dict(operator))


def test_cost_roster_cannot_duplicate_a_task_even_with_complete_count():
    value = compact()
    for arm in ROUTES:
        value["routes"][arm]["physical_cost_by_task"][0]["task_id"] = "other"
    with pytest.raises(ValueError, match="roster"):
        validate_cost_roster(value, {"runs": []}, {})


def test_exact_raw_roster_and_global_cost_must_both_match():
    value = compact()
    rows = []
    for arm in ROUTES:
        for i, item in enumerate(value["routes"][arm]["physical_cost_by_task"]):
            item.pop("claim")
            item.update(task_id=f"synthetic-{i}", cumulative_model_prompt_tokens=10)
            rows.append({"route": arm, "task_id": item["task_id"],
                "physical_cost": item["generation"], "reranker_cost": item["reranker"],
                "tool_calls": item["tool_calls"], "elapsed_ms": item["elapsed_ms"],
                "cumulative_model_prompt_tokens": item["cumulative_model_prompt_tokens"]})
    cost = {"generation": {"unique_physical_calls": 192, "unknown_usage_attempts": 0,
                "elapsed_ms": 1152, "known_token_lower_bound": {"input_tokens": 960, "output_tokens": 480}},
            "reranker": {"physical_requests": 96, "requested_pairs": 1920, "completed_pairs": 1920,
                "unknown_requests": 0, "elapsed_ms_including_swaps": 672, "known_token_lower_bound": 2880}}
    validate_cost_roster(value, {"runs": rows}, cost)
    cost["generation"]["known_token_lower_bound"]["input_tokens"] += 1
    with pytest.raises(ValueError, match="global_cost"):
        validate_cost_roster(value, {"runs": rows}, cost)
