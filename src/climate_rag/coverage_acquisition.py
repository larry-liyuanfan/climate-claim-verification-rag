"""Versioned relation-coverage decision, not a requirement to acquire evidence.

Self-reported coverage is not entailment. Only current displayed sentences may
be referenced. The action is still executed by the existing shared controller.
"""
from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

from . import fair_acquisition

PROTOCOL = fair_acquisition.COVERAGE_PROTOCOL
COHORT_SHA = "f2b7c55b1e91f297a613df5762f44d46c2df7666cca8e234d77d5ca53488ec56"
TASKS_SHA = "de088bac5fb93beafd2393a438cbc42ac0f99479ff173817a734c01d071e5b25"
TASK_COUNT = 32
KINDS = ("relation", "time", "quantity", "scope")
STATES = ("covered", "partial", "missing", "uncertain")
STOP_REASONS = ("not_stopping", "sufficient_to_judge", "no_useful_action", "budget", "uncertain_scope")


def validate_frozen_cohort(binding: Mapping[str, Any]) -> None:
    if (binding["cohort_sha256"] != COHORT_SHA or binding["tasks_sha256"] != TASKS_SHA
            or binding["task_count"] != TASK_COUNT):
        raise ValueError("coverage_frozen_consumed_cohort_required")


def schema(readable: Sequence[str], *, can_acquire: bool, can_query: bool,
           can_rerank: bool, max_read: int = 5) -> dict[str, Any]:
    return {"type": "object", "additionalProperties": False,
        "properties": {
            "coverage": {"type": "array", "minItems": 1, "maxItems": 3,
                "items": {"type": "object", "additionalProperties": False,
                    "properties": {
                        "claim_span": {"type": "string", "minLength": 1, "maxLength": 160},
                        "kind": {"type": "string", "enum": list(KINDS)},
                        "status": {"type": "string", "enum": list(STATES)},
                        "sentence_ids": {"type": "array", "minItems": 0, "maxItems": 3,
                                         "items": {"type": "string"}, "uniqueItems": True}},
                    "required": ["claim_span", "kind", "status", "sentence_ids"]}},
            "decision": fair_acquisition.gate_schema(readable, can_acquire=can_acquire,
                can_query=can_query, can_rerank=can_rerank, max_read=max_read),
            "stop_reason": {"type": "string", "enum": list(STOP_REASONS)}},
        "required": ["coverage", "decision", "stop_reason"]}


def parse(value: Any, observation: Mapping[str, Any], *, fully_shown: Sequence[str]) -> dict[str, Any]:
    if not isinstance(value, dict) or set(value) != {"coverage", "decision", "stop_reason"}:
        raise ValueError("coverage_contract_fields")
    rows = value["coverage"]
    if not isinstance(rows, list) or not 1 <= len(rows) <= 3:
        raise ValueError("coverage_count")
    visible = {s["sentence_id"] for s in observation["current_citable"]}
    for item in rows:
        if not isinstance(item, dict) or set(item) != {"claim_span", "kind", "status", "sentence_ids"}:
            raise ValueError("coverage_item_fields")
        span, ids = item["claim_span"], item["sentence_ids"]
        if (not isinstance(span, str) or not 1 <= len(span) <= 160
                or span not in observation["immutable_claim"]
                or item["kind"] not in KINDS or item["status"] not in STATES
                or not isinstance(ids, list) or len(ids) > 3
                or any(not isinstance(s, str) for s in ids)
                or len(set(ids)) != len(ids) or not set(ids) <= visible
                or (item["status"] == "covered" and not ids)):
            raise ValueError("coverage_not_claim_or_current_evidence")
    decision = fair_acquisition.parse_gate(value["decision"], observation, fully_shown=fully_shown)
    reason = value["stop_reason"]
    if reason not in STOP_REASONS or (decision["action"] == "stop") == (reason == "not_stopping"):
        raise ValueError("coverage_stop_reason_mismatch")
    return decision


def system_prompt(phase: str) -> str:
    base = fair_acquisition.system_prompt(phase)
    if phase != "gate":
        return base  # byte-identical shared plan/verifier, unchanged baselines
    return base + (
        " Before the action, return 1-3 exact immutable_claim spans identifying the "
        "decisive relation, time, quantity or scope; mark covered/partial/missing/uncertain "
        "and bind only current displayed sentence IDs. Covered means the relation is "
        "decidable, whether it supports or refutes the claim, not just topic overlap. "
        "Do not invent a missing gap or call tools for activity. Select one available "
        "action if likely to resolve an actual gap. Stop remains legal when evidence "
        "is adequate, scope is unclear, no useful action is available, or budget is low; "
        "return an explicit stop_reason. Self-reported coverage is not semantic proof.")
