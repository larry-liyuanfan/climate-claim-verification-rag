"""Synthetic corruption/privacy tests; not evidence of real model performance."""

import json

import pytest

from climate_rag import stop_acquire
from replay_recovered_acquisition import STUDY, digest, replay


def episode(tmp_path, *, tamper=False, error="duplicate_sentence_reference"):
    private = "PRIVATE_CLAIM_OR_SOURCE_SENTINEL"
    observation = {
        "immutable_claim": private, "current_citable": [
            {"sentence_id": "c0:0", "text": private}],
        "readable_source_ids": ["c1"], "read_limit": 5,
        "acquisition_available": True, "query_available": True,
        "remaining_calls": 5, "prior_queries": [],
    }
    schema = stop_acquire.gate_schema(["c1"], can_acquire=True, can_query=True)
    attempt = {
        "observation": observation, "schema": schema, "stage": "gate",
        "diagnostics": {"physical_attempt_id": "g0"},
        "decision": {"action": "stop"}, "proposed_decision": {"action": "stop"},
        "error_code": error,
    }
    ledger = tmp_path / "ledger"
    ledger.mkdir()
    recorded = {"observation": observation,
                "schema": {} if tamper else schema}
    (ledger / "g0.reserved.json").write_text(json.dumps(recorded))
    run = {"protocol": stop_acquire.PROTOCOL, "study_kind": STUDY, "runs": [{
        "route": "adaptive", "generation_attempts": [attempt], "events": [{
            "tool": "retrieve", "status": "completed", "trigger_attempt": -1,
        }],
    }]}
    path = tmp_path / "run.json"
    path.write_text(json.dumps(run))
    return path, private


def test_real_transition_replay_exports_only_aggregate_and_no_semantic_claim(tmp_path):
    path, private = episode(tmp_path, error="SECRET_UNRECOGNISED_ERROR")
    result = replay(path, digest(path))
    encoded = json.dumps(result)
    assert private not in encoded and "SECRET_UNRECOGNISED_ERROR" not in encoded
    assert "c0:0" not in encoded and "c1" not in encoded
    route = result["routes"]["adaptive"]
    assert route["tool_events"] == {"retrieve:completed": 1}
    assert route["gate_opportunities"]["query_branch"] == 1
    assert route["validated_actions"] == {"stop": 1}
    assert route["acquisition_feedback_followed_by_model"] == 0
    assert result["model_calls_made"] == 0 and result["scores_recomputed"] is False
    assert result["semantic_support"] == "not_measured_by_this_replay"


def test_rejects_changed_physical_schema_without_reconstructing_a_response(tmp_path):
    path, _ = episode(tmp_path, tamper=True)
    with pytest.raises(ValueError, match="physical_schema_changed"):
        replay(path, digest(path))


def test_rejects_unbound_input_before_parsing_private_records(tmp_path):
    path, _ = episode(tmp_path)
    with pytest.raises(ValueError, match="recorded_run_hash_mismatch"):
        replay(path, "0" * 64)


@pytest.mark.parametrize("physical_tool", ["rewrite", "read"])
def test_synthetic_acquisition_delivery_links_to_the_next_recorded_call(
    tmp_path, physical_tool
):
    path, private = episode(tmp_path)
    run = json.loads(path.read_bytes())
    row = run["runs"][0]
    first = row["generation_attempts"][0]
    first.pop("error_code")
    decision = {"action": "acquire", "tool": "query",
                "purpose": "subquestion", "query": "synthetic second query"}
    if physical_tool == "read":
        decision = {"action": "acquire", "tool": "read", "source_ids": ["c1"]}
    first.update(decision=decision, proposed_decision=decision)
    next_observation = {**first["observation"], "current_citable": [
        *first["observation"]["current_citable"],
        {"sentence_id": "c2:0", "text": "SYNTHETIC_NEW_EVIDENCE"}],
        "tool_feedback": {"status": "completed", "tool": physical_tool,
                          "query_sha256": "b" * 64}}
    next_attempt = {"observation": next_observation, "schema": {},
                    "stage": "verdict", "decision": {"action": "abstain"},
                    "diagnostics": {"physical_attempt_id": "g1"}}
    row["generation_attempts"].append(next_attempt)
    row["events"].append({"tool": physical_tool, "status": "completed",
                          "trigger_attempt": 0, "query_sha256": "b" * 64})
    (path.parent / "ledger/g1.reserved.json").write_text(json.dumps({
        "observation": next_observation, "schema": {}}))
    path.write_text(json.dumps(run))
    result = replay(path, digest(path))
    assert result["routes"]["adaptive"]["acquisition_feedback_followed_by_model"] == 1
    assert result["routes"]["adaptive"]["feedback_with_additional_citable_text"] == 1
    assert private not in json.dumps(result)
    assert "SYNTHETIC_NEW_EVIDENCE" not in json.dumps(result)
    assert result["model_calls_made"] == 0


def test_same_status_from_a_different_tool_is_not_accepted_as_feedback(tmp_path):
    path, _ = episode(tmp_path)
    run = json.loads(path.read_bytes())
    row = run["runs"][0]
    row["generation_attempts"][0].pop("error_code")
    row["generation_attempts"][0].update(
        decision={"action": "acquire", "tool": "read", "source_ids": ["c1"]},
        proposed_decision={"action": "acquire", "tool": "read", "source_ids": ["c1"]})
    second = {**row["generation_attempts"][0], "stage": "verdict",
              "diagnostics": {"physical_attempt_id": "g1"}}
    second["observation"] = {**second["observation"], "tool_feedback": {
        "status": "completed", "tool": "rerank"}}
    row["generation_attempts"].append(second)
    row["events"].append({"tool": "read", "status": "completed",
                          "trigger_attempt": 0})
    (path.parent / "ledger/g1.reserved.json").write_text(json.dumps({
        "observation": second["observation"], "schema": second["schema"]}))
    path.write_text(json.dumps(run))
    with pytest.raises(ValueError, match="feedback_tool_identity_changed"):
        replay(path, digest(path))


def test_model_stop_cannot_be_relabelled_as_executed_acquisition(tmp_path):
    path, _ = episode(tmp_path)
    run = json.loads(path.read_bytes())
    run["runs"][0]["events"].append({
        "tool": "read", "status": "completed", "trigger_attempt": 0})
    path.write_text(json.dumps(run))
    with pytest.raises(ValueError, match="tool_after_non_acquisition_gate"):
        replay(path, digest(path))
