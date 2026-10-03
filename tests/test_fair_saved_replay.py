"""Small synthetic receipt tests only; no real data/model/score execution."""
import hashlib
import json

import pytest

from replay_fair_acquisition_result import PROTOCOL, ROUTES, census, digest


def saved(tmp_path):
    private = "PRIVATE_CLAIM_AND_PASSAGE_SENTINEL"
    ledger = tmp_path / "ledger"
    ledger.mkdir()
    old = {"sentence_id": "c0:0", "text": private}
    new = {"sentence_id": "c1:0", "text": "PRIVATE_NEW_PASSAGE_SENTINEL"}
    old_map = {old["sentence_id"]: hashlib.sha256(old["text"].encode()).hexdigest()}
    new_map = {new["sentence_id"]: hashlib.sha256(new["text"].encode()).hexdigest()}
    rows = []
    key = 0
    for route in ROUTES:
        event0 = {"tool": "retrieve", "status": "completed", "trigger_attempt": -1,
                  "delivery": {"visible_sentence_sha256": old_map, "new_sentence_sha256": old_map}}
        row = {"task_id": "PRIVATE_TASK_ID", "route": route,
               "initial_frame_sha256": "a" * 64, "events": [event0],
               "answer": {"label": "SUPPORTS"}, "generation_attempts": []}
        decisions = [("verdict", {"action": "answer"})]
        if route == "autonomous":
            decisions = [("gate", {"action": "acquire", "tool": "read"}),
                         ("gate", {"action": "stop"}), ("verdict", {"action": "answer"})]
            row["events"].append({"tool": "read", "status": "completed", "trigger_attempt": 0,
                "delivery": {"visible_sentence_sha256": {**old_map, **new_map}, "new_sentence_sha256": new_map}})
        for ordinal, (stage, decision) in enumerate(decisions):
            observation = {"immutable_claim": private, "current_citable": [old],
                           "acquisition_available": True, "query_available": True,
                           "rerank_available": True, "readable_source_ids": ["PRIVATE_SOURCE_ID"]}
            if route == "autonomous" and ordinal > 0:
                observation["current_citable"].append(new)
            attempt = {"stage": stage, "decision": decision, "observation": observation,
                       "schema": {}, "diagnostics": {"physical_attempt_id": f"g{key}"},
                       "received_tool_events": [0, 1] if route == "autonomous" and ordinal > 0 else [0]}
            (ledger / f"g{key}.reserved.json").write_text(json.dumps({"observation": observation, "schema": {}}))
            key += 1
            row["generation_attempts"].append(attempt)
        rows.append(row)
    path = tmp_path / "run.json"
    path.write_text(json.dumps({"protocol": PROTOCOL, "runs": rows}))
    return path


def test_saved_fair_replay_counts_unique_deliveries_and_redacts(tmp_path):
    path = saved(tmp_path)
    result = census(path, digest(path), expected_tasks=1)
    counts = result["routes"]["autonomous"]["counts"]
    assert counts["additional_delivery_events_in_later_model_prompt"] == 1
    assert counts["additional_sentences_delivered"] == 1
    assert counts["gate_calls"] == 2
    encoded = json.dumps(result)
    assert "PRIVATE" not in encoded and "c1:0" not in encoded
    assert result["gold_read"] is False and result["scores_recomputed"] is False
    assert result["model_calls_made"] == 0
    assert result["feedback_causality"].startswith("whole_policy_only")


def test_wrong_hash_rejected_before_private_record_parsing(tmp_path):
    path = saved(tmp_path)
    with pytest.raises(ValueError, match="run_hash_mismatch"):
        census(path, "0" * 64, expected_tasks=1)


def test_case_selection_is_first_within_route_not_global_route_traversal(tmp_path):
    path = saved(tmp_path)
    original = json.loads(path.read_bytes())
    rows = []
    key = 100
    # Original runtime order is task then route, not all tasks within one route.
    for position in range(2):
        for template in original["runs"]:
            row = json.loads(json.dumps(template))
            row["task_id"] = f"PRIVATE_TASK_{position}"
            if (position == 0 and row["route"] == "autonomous") or (
                    position == 1 and row["route"] == "deterministic_workflow"):
                row["generation_attempts"][-1]["error_code"] = "duplicate_sentence_reference"
                row["generation_attempts"][-1]["decision"] = None
                row["answer"] = None
                row["outcome"] = "validation_repair_exhausted"
            for attempt in row["generation_attempts"]:
                attempt["diagnostics"]["physical_attempt_id"] = f"g{key}"
                (path.parent / "ledger" / f"g{key}.reserved.json").write_text(
                    json.dumps({"observation": attempt["observation"], "schema": attempt["schema"]}))
                key += 1
            rows.append(row)
    path.write_text(json.dumps({"protocol": PROTOCOL, "runs": rows}))
    result = census(path, digest(path), expected_tasks=2)
    cases = result["first_cases_by_behavior_only"]
    assert cases["autonomous:validation_error"]["frozen_task_position"] == 0
    assert cases["deterministic_workflow:validation_error"]["frozen_task_position"] == 1
    assert cases["autonomous:validation_repair_exhausted"]["frozen_task_position"] == 0
    assert cases["deterministic_workflow:validation_repair_exhausted"]["frozen_task_position"] == 1
    assert "validation_error" not in cases
    assert result["case_selection"].startswith("first occurrence within each route/behavior")
    assert "PRIVATE" not in json.dumps(result)


def test_unknown_validation_error_is_not_exported_as_private_text(tmp_path):
    path = saved(tmp_path)
    run = json.loads(path.read_bytes())
    run["runs"][-1]["generation_attempts"][-1]["error_code"] = "PRIVATE_ERROR_DETAIL"
    path.write_text(json.dumps(run))
    result = census(path, digest(path), expected_tasks=1)
    assert result["first_cases_by_behavior_only"]["autonomous:validation_error"]["error_kind"] == "other_validation_error"
    assert result["routes"]["autonomous"]["validation_errors"] == {"other_validation_error": 1}
    assert "PRIVATE" not in json.dumps(result)


@pytest.mark.parametrize("corruption", ["schema", "delivery", "future", "post_stop", "initial", "roster"])
def test_corrupt_saved_receipts_rejected(tmp_path, corruption):
    path = saved(tmp_path)
    run = json.loads(path.read_bytes())
    row = run["runs"][-1]
    expected = {"schema": "physical_prompt_receipt_drift", "delivery": "delivery_not_in_physical_prompt",
                "future": "future_delivery", "post_stop": "tool_after_stop",
                "initial": "initial_information_drift", "roster": "route_roster_drift"}[corruption]
    if corruption == "schema":
        row["generation_attempts"][0]["schema"] = {"tampered": True}
    elif corruption == "delivery":
        row["events"][1]["delivery"]["visible_sentence_sha256"]["c1:0"] = "0" * 64
    elif corruption == "future":
        row["events"][1]["trigger_attempt"] = 1
    elif corruption == "post_stop":
        row["events"].append({"tool": "rewrite", "status": "completed", "trigger_attempt": 2})
    elif corruption == "initial":
        row["initial_frame_sha256"] = "b" * 64
    else:
        row["task_id"] = "DIFFERENT_TASK"
    path.write_text(json.dumps(run))
    with pytest.raises(ValueError, match=expected):
        census(path, digest(path), expected_tasks=1)
