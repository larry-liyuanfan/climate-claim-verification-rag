"""Fairness contract checks, using explicitly synthetic identities."""

import copy

import pytest

from climate_rag.acquisition_comparison import validate_comparison
from climate_rag.agent_v3 import V3Budget


def arms():
    base = {key: "a" * 64 for key in (
        "initial_frame_sha256", "terminal_contract_sha256", "model_sha256",
        "reranker_sha256", "corpus_sha256", "cohort_sha256")}
    base.update(budget=V3Budget().model_dump(),
                available_tools=["read", "query", "rerank"],
                forced_model_acquisition=False)
    return [{**copy.deepcopy(base), "name": name} for name in (
        "fixed_multiquery", "deterministic_workflow", "autonomous")]


def test_equal_capabilities_do_not_authorize_execution_or_assert_equal_cost():
    result = validate_comparison(arms())
    assert result["model_execution_authorized"] is False
    assert result["equal_actual_cost_assumed"] is False


@pytest.mark.parametrize("key,value,error", [
    ("available_tools", ["read", "query"], "tool_capability"),
    ("terminal_contract_sha256", "b" * 64, "shared_contract"),
    ("initial_frame_sha256", "b" * 64, "shared_contract"),
    ("budget", {"max_calls": 3}, "shared_contract"),
    ("forced_model_acquisition", True, "optional"),
])
def test_refuses_affordance_interface_budget_or_forced_action_confounds(key, value, error):
    candidates = arms()
    candidates[-1][key] = value
    with pytest.raises(ValueError, match=error):
        validate_comparison(candidates)


def test_missing_cohort_binding_cannot_be_a_ready_experiment():
    candidates = arms()
    for arm in candidates:
        arm["cohort_sha256"] = None
    with pytest.raises(ValueError, match="unbound"):
        validate_comparison(candidates)


def test_rejects_equal_but_incomplete_ceilings_and_non_hash_identities():
    candidates = arms()
    for arm in candidates:
        arm["budget"] = {"max_calls": 5}
    with pytest.raises(ValueError, match="incomplete_comparison_budget"):
        validate_comparison(candidates)
    candidates = arms()
    for arm in candidates:
        arm["cohort_sha256"] = "not a digest"
    with pytest.raises(ValueError, match="invalid_comparison_sha256"):
        validate_comparison(candidates)
