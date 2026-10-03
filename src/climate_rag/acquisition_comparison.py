"""CPU preparation for equal-capability comparisons; no model or cloud launch."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
import re
from typing import Any

from .agent_v3 import V3Budget

ARMS = frozenset({"fixed_multiquery", "deterministic_workflow", "autonomous"})
TOOLS = frozenset({"read", "query", "rerank"})


def validate_comparison(arms: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Fail before spending if tools, initial frames, verifier or ceilings differ.

    Actual paid work is separately authorized through the existing operator.
    This check does not certify semantics, a dataset split, or runtime readiness.
    """
    if len(arms) != 3 or {arm.get("name") for arm in arms} != ARMS:
        raise ValueError("comparison_requires_three_named_arms")
    shared = ("initial_frame_sha256", "terminal_contract_sha256", "budget",
              "model_sha256", "reranker_sha256", "corpus_sha256", "cohort_sha256")
    first = arms[0]
    for key in shared:
        if key not in first or first[key] is None:
            raise ValueError("unbound_comparison_identity")
        if any(arm.get(key) != first[key] for arm in arms[1:]):
            raise ValueError("comparison_shared_contract_mismatch")
        if key != "budget" and (not isinstance(first[key], str) or
                                re.fullmatch(r"[0-9a-f]{64}", first[key]) is None):
            raise ValueError("invalid_comparison_sha256")
    if not isinstance(first["budget"], dict) or set(first["budget"]) != set(
        V3Budget().model_dump()
    ):
        raise ValueError("incomplete_comparison_budget")
    V3Budget.model_validate(first["budget"])
    for arm in arms:
        if frozenset(arm.get("available_tools", [])) != TOOLS:
            raise ValueError("comparison_tool_capability_mismatch")
        if arm.get("forced_model_acquisition") is not False:
            raise ValueError("model_acquisition_must_remain_optional")
    return {
        "scope": "CPU_identity_and_capability_check_only",
        "arms": sorted(ARMS), "available_tools": sorted(TOOLS),
        "equal_actual_cost_assumed": False, "model_execution_authorized": False,
        "semantic_support_measured": False,
    }
