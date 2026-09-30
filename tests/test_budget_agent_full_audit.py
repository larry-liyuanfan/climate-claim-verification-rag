"""Synthetic audit tests; no model, private source, gold file or network access."""
import copy
import importlib.util
import json
from pathlib import Path

import pytest

spec = importlib.util.spec_from_file_location(
    "full_audit", Path(__file__).resolve().parents[1] / "scripts/audit_budget_agent_full.py")
audit = importlib.util.module_from_spec(spec)
spec.loader.exec_module(audit)


def row(task="one", route="adaptive", kind="failure"):
    private = "PRIVATE_MODEL_PROSE_NEVER_EXPORT"
    decision = {"action": "abstain", "reason": private}
    diag = {"category": "schema_validation" if kind == "failure" else "validated",
            "eos_observed": True, "reached_max_new_tokens": False,
            "generation_elapsed_ms": 100, "output_tokens": 20,
            "private_attachment": "skipped_total_limit", "output_sha256": "a" * 64,
            "errors": [{"loc": [], "type": "value_error"}] if kind == "failure" else []}
    event = ({"stage": "failure", "error_type": "GeneratedResponseError"} if kind == "failure"
             else {"stage": "decision", "decision": decision})
    return {"task_id": task, "strategy": route, "claim_text": private,
            "answer": None, "reason": "stage_failed" if kind == "failure" else private,
            "provider_kind": "local_model", "candidate_evidence_ids": ["gold", "private_id"],
            "context_evidence_ids": ["gold", "private_id"],
            "delivered_evidence_ids": [] if kind == "failure" else ["gold", "private_id"],
            "events": [{"stage": "retrieve", "query": private, "elapsed_ms": 2}, {**event, "generation_diagnostics": diag}],
            "usage": {"input_tokens": 100, "output_tokens": 20}, "usage_known": True,
            "elapsed_ms": 110, "generation_calls": 1, "model_calls": 1, "tool_calls": 1,
            "retrieval_calls": 1, "rerank_calls": 0, "rerank_candidate_pairs": 0}


def test_failure_cost_is_preserved_and_not_model_abstention():
    value = audit.aggregate_rows([row()], {"one": {"evidence_ids": ["gold"]}}, {})
    assert value["parsed_model_action_counts"] == {}
    assert value["cost_by_outcome"]["controller_failure"]["recorded_output_tokens"] == 20
    assert value["generation_diagnostic_counts"] == {"schema_validation": 1}
    assert value["ungated_retained_context_diagnostic"]["mean_gold_recall_at_context5"] == 1
    assert value["ungated_retained_context_diagnostic"]["queries_with_context_gold_but_empty_delivered"] == 1
    assert "PRIVATE" not in json.dumps(value) and "private_id" not in json.dumps(value)


def test_model_abstention_is_distinct_and_authored_quality_null():
    value = audit.aggregate_rows([row(kind="abstain")], None, {})
    assert value["parsed_model_action_counts"] == {"abstain": 1}
    assert value["cost_by_outcome"]["model_requested_abstention"]["count"] == 1
    assert value["ungated_retained_context_diagnostic"] is None
    assert "PRIVATE" not in json.dumps(value)


def test_payload_field_diagnosis_is_content_free_and_subset_scoped():
    raw = json.dumps({"action": "abstain", "query": "PRIVATE", "reason": "PRIVATE" * 100}).encode()
    item = row()
    item["events"][-1]["generation_diagnostics"]["output_sha256"] = audit.sha(raw)
    value = audit.aggregate_rows([item, row(task="two")], None, {audit.sha(raw): raw})
    check = value["private_failure_field_inspection"]
    assert check["sha_matched_payload_rows"] == 1 and check["unavailable_payload_rows"] == 1
    assert check["structural_flag_counts"] == {"query_on_nonrewrite": 1, "reason_over_500_characters": 1}
    assert "PRIVATE" not in json.dumps(check)


def test_attempts_and_completed_rerank_are_separate_from_model_actions():
    item = row()
    item.update(strategy="fixed_rerank", tool_calls=2, rerank_calls=1, rerank_candidate_pairs=20)
    item["events"].insert(1, {"stage": "rerank", "elapsed_ms": 3, "candidate_count": 20})
    value = audit.aggregate_rows([item], None, {})
    assert value["completed_event_counts"]["rerank"] == 1
    assert value["parsed_model_action_counts"] == {}


def test_unknown_failure_usage_remains_unknown():
    item = row()
    item["usage_known"] = False
    assert audit.aggregate_rows([item], None, {})["cost_by_outcome"]["controller_failure"]["unknown_usage_rows"] == 1


def test_budget_violation_is_reported_not_hidden():
    item = row()
    item.update(tool_calls=4, elapsed_ms=120001)
    checks = audit.aggregate_rows([item], None, {})["budget_violation_counts"]
    assert checks["tool_calls"] == checks["row_elapsed_over_120s"] == 1


def test_case_summaries_are_bounded_anonymous_and_content_free():
    rows = [row(task=str(i), route=s) for i in range(3) for s in audit.STRATEGIES]
    result = audit.select_cases({"validation": rows, "vnext": copy.deepcopy(rows)})
    assert len(result) == 2
    assert len({x["private_locator_sha256"] for x in result}) == 2
    rendered = json.dumps(result)
    assert "PRIVATE" not in rendered and "private_id" not in rendered and '"task_id"' not in rendered


def test_unapproved_action_fails_closed_in_case_export():
    item = row(kind="abstain")
    item["events"][-1]["decision"]["action"] = "PRIVATE"
    with pytest.raises(ValueError, match="compact audit contract"):
        audit.row_summary(item)


def test_quantile_matches_linear_interpolation():
    assert audit.quantile([0, 10, 20], 0.95) == pytest.approx(19)


def test_modified_operator_archive_refused_before_private_reads(tmp_path):
    archive = tmp_path / "wrong.tar"
    archive.write_bytes(b"not the pinned source")
    with pytest.raises(ValueError, match="compact audit contract"):
        audit.build(tmp_path / "missing-private-root", archive)
