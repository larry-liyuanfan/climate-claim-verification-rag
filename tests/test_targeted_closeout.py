"""Only synthetic metadata; no private input fixtures or real model access."""

import json

import pytest

from summarize_targeted_32030221 import diagnose, interval, summarize


def row():
    return {
        "visible_source_ids": {"S0": "private-gold-id", "S1": "private-noise-id"},
        "generation_attempts": [{
            "observation": {"current_citable": [{"sentence_id": "S1:0", "text": "PRIVATE_SENTINEL"}], "tool_feedback": None},
            "allowed_actions": ["answer", "abstain", "rewrite"],
            "decision": {"action": "abstain"}, "status": "valid_decision",
        }],
        "events": [{"status": "completed", "tool": "retrieve", "candidate_ids": ["S1", "S0"]}],
        "answer": None, "delivered_evidence_ids": ["private-noise-id", "private-gold-id"],
        "model_calls": 1, "outcome": "model_abstention:insufficient_evidence",
    }


def test_inventory_is_not_full_text_visibility_and_export_has_no_ids_or_text():
    result = diagnose(row(), {"evidence_ids": ["private-gold-id"], "label": "SUPPORTS"})
    assert result["initial_candidate_gold_hit"]
    assert not result["gold_full_text_shown"]
    assert not result["gold_cited"]
    assert result["queries_available_first"]
    assert result["feedback_then_decision"] == 0
    encoded = json.dumps(result)
    assert "private-gold-id" not in encoded
    assert "private-noise-id" not in encoded
    assert "PRIVATE_SENTINEL" not in encoded


def test_visible_cited_and_wrong_verdict_are_separate():
    value = row()
    value["generation_attempts"][0]["observation"]["current_citable"].append(
        {"sentence_id": "S0:0", "text": "PRIVATE_GOLD_TEXT"})
    value["answer"] = {"label": "REFUTES", "citations": [{"source_id": "private-gold-id"}]}
    result = diagnose(value, {"evidence_ids": ["private-gold-id"], "label": "SUPPORTS"})
    assert result["gold_full_text_shown"] and result["gold_cited"]
    assert not result["correct_label"]


def test_exploratory_paired_cost_interval_and_matrix_guard():
    result = interval([1., 2., 3.], [2., 3., 4.])
    assert result["mean_difference"] == result["ci_lower"] == result["ci_upper"] == 1.
    assert result["samples"] == 5000
    with pytest.raises(ValueError, match="matrix_size"):
        summarize({"runs": []}, {"claims": {}})


def test_rejected_read_is_not_successful_feedback_or_new_tool_execution():
    value = row()
    attempt = value["generation_attempts"][0]
    attempt.update(decision={"action": "read", "source_ids": ["S1"]},
                   requested_context=["S1"], status="validation_failed", error_code="read_loop")
    result = diagnose(value, {"evidence_ids": [], "label": "NOT_ENOUGH_INFO"})
    assert result["read_same_as_current_selection"] == 1
    assert result["tools_after_initial"] == result["feedback_then_decision"] == 0
    assert result["errors"] == ["read_loop"]
