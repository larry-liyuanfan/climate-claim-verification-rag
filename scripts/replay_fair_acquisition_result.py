"""Replay saved fair-arm receipts; aggregate only, no gold or model execution.

This complements the original cloud physical/state audit, not a replacement for
it. It counts actual delivery to later recorded prompts, NOT semantic use or
feedback causal benefit. No claim, passage, source ID or task ID is exported.
"""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
from typing import Any

from climate_rag.fair_acquisition import COVERAGE_PROTOCOL

PROTOCOL = "fair-acquisition-three-arm-v1-20261003"
ROUTES = ("fixed_multiquery", "deterministic_workflow", "autonomous")


def digest(path: Path) -> str:
    fingerprint = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            fingerprint.update(block)
    return fingerprint.hexdigest()


def require(value: bool, error: str) -> None:
    if not value:
        raise ValueError(error)


def census(run_path: Path, expected_sha: str, *, expected_tasks: int = 32,
           expected_protocol: str = PROTOCOL) -> dict[str, Any]:
    require(digest(run_path) == expected_sha, "run_hash_mismatch")
    run = json.loads(run_path.read_bytes())
    require(expected_protocol in {PROTOCOL, COVERAGE_PROTOCOL}
            and run["protocol"] == expected_protocol, "unsupported_protocol")
    rows = run["runs"]
    task_ids = [r["task_id"] for r in rows if r["route"] == ROUTES[0]]
    require(len(task_ids) == expected_tasks and len(set(task_ids)) == expected_tasks
            and len(rows) == 3 * expected_tasks, "incomplete_fair_matrix")
    initial: dict[str, str] = {}
    summary: dict[str, Any] = {}
    cases: dict[str, Any] = {}
    for route in ROUTES:
        selected = [r for r in rows if r["route"] == route]
        require([r["task_id"] for r in selected] == task_ids, "route_roster_drift")
        counts: Counter[str] = Counter()
        errors: Counter[str] = Counter()
        actions: Counter[str] = Counter()
        for position, row in enumerate(selected):
            fingerprint = row["initial_frame_sha256"]
            require(initial.setdefault(row["task_id"], fingerprint) == fingerprint,
                    "initial_information_drift")
            attempts, events = row["generation_attempts"], row["events"]
            received: set[int] = set()
            stop_at: int | None = None
            for ordinal, attempt in enumerate(attempts):
                key = attempt["diagnostics"]["physical_attempt_id"]
                require(key.startswith("g") and key.replace("g", "", 1).isdigit(),
                        "physical_id_format")
                reserved = json.loads((run_path.parent / "ledger" / (key + ".reserved.json")).read_bytes())
                require(reserved["observation"] == attempt["observation"]
                        and reserved["schema"] == attempt["schema"], "physical_prompt_receipt_drift")
                decision = attempt.get("decision") or {}
                action = decision.get("action")
                if action is not None:
                    require(action in {"stop", "acquire", "answer", "abstain", "plan_queries"},
                            "unknown_action")
                    actions[action] += 1
                if attempt["stage"] == "gate":
                    counts["gate_calls"] += 1
                    for option in ("acquisition_available", "query_available", "rerank_available"):
                        counts["gate_with_" + option] += bool(attempt["observation"][option])
                    counts["gate_with_unread_initial_or_new_sources"] += bool(
                        attempt["observation"]["readable_source_ids"])
                    if action == "stop":
                        stop_at = ordinal
                    if action == "acquire":
                        counts["acquire_" + decision["tool"]] += 1
                if "error_code" in attempt:
                    code = attempt["error_code"]
                    errors[code if code in {"duplicate_sentence_reference", "invalid_schema", "invalid_json",
                        "invalid_fixed_read_plan", "read_loop", "read_not_unseen_preview"} else "other_validation_error"] += 1
                    cases.setdefault(route + ":validation_error", {"route": route,
                        "frozen_task_position": position, "attempt_ordinal": ordinal,
                        "error_kind": code if code == "duplicate_sentence_reference" else "other_validation_error"})
                visible = {v["sentence_id"]: hashlib.sha256(v["text"].encode()).hexdigest()
                           for v in attempt["observation"]["current_citable"]}
                for index in attempt.get("received_tool_events", []):
                    require(type(index) is int and 0 <= index < len(events), "invalid_event_receipt")
                    event = events[index]
                    require(event["trigger_attempt"] < ordinal, "future_delivery")
                    delivery = event["delivery"]
                    require(all(visible.get(k) == v for k, v in delivery["visible_sentence_sha256"].items()),
                            "delivery_not_in_physical_prompt")
                    if index > 0 and delivery["new_sentence_sha256"]:
                        require(all(visible.get(k) == v for k, v in delivery["new_sentence_sha256"].items()),
                                "new_text_not_in_physical_prompt")
                        received.add(index)
                        if route == "autonomous":
                            category = "autonomous_new_text_then_" + attempt["stage"]
                            cases.setdefault(route + ":" + category, {"route": route, "frozen_task_position": position,
                                                       "event_ordinal": index, "next_attempt": ordinal})
            for index, event in enumerate(events):
                if "tool" not in event:
                    continue
                require(event["tool"] in {"retrieve", "read", "rewrite", "rerank"}
                        and event["status"] in {"completed", "failed"}, "unknown_tool_event")
                counts["tool_" + event["tool"] + "_" + event["status"]] += 1
                if route == "autonomous" and stop_at is not None:
                    require(event["trigger_attempt"] < stop_at, "tool_after_stop")
                if event["status"] == "failed" or event.get("search_empty"):
                    category = "tool_error" if event["status"] == "failed" else "empty_query"
                    cases.setdefault(route + ":" + category, {"route": route, "frozen_task_position": position,
                                                "event_ordinal": index})
                delivery = event.get("delivery", {})
                if index > 0 and delivery.get("new_sentence_sha256"):
                    counts["additional_full_text_delivery_events"] += 1
                    counts["additional_sentences_delivered"] += len(delivery["new_sentence_sha256"])
            counts["additional_delivery_events_in_later_model_prompt"] += len(received)
            if route == "autonomous" and stop_at is not None:
                cases.setdefault(route + ":stop_then_verdict", {"route": route,
                    "frozen_task_position": position, "stop_attempt": stop_at,
                    "final_outcome_kind": "answered" if row["answer"] else "abstained_or_failed"})
            if row.get("outcome") == "validation_repair_exhausted":
                cases.setdefault(route + ":validation_repair_exhausted", {"route": route,
                    "frozen_task_position": position})
            counts["slots_with_additional_text_received"] += bool(received)
            counts["recorded_model_calls"] += len(attempts)
        summary[route] = {"slots": len(selected), "counts": dict(counts),
                          "validated_actions": dict(actions), "validation_errors": dict(errors)}
    return {"schema": "fair-acquisition-saved-receipt-census-v1", "protocol": expected_protocol,
        "run_sha256": expected_sha, "total_slots": len(rows), "routes": summary,
        "first_cases_by_behavior_only": cases,
        "case_selection": "first occurrence within each route/behavior in frozen task order; no score/gold filter",
        "model_calls_made": 0, "gold_read": False, "scores_recomputed": False,
        "semantic_support": "unmeasured", "reasonable_stop": "not_adjudicated_by_action_validity",
        "feedback_causality": "whole_policy_only; prompt receipt is not causal or semantic use"}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--run-sha256", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--protocol", choices=(PROTOCOL, COVERAGE_PROTOCOL), default=PROTOCOL)
    args = parser.parse_args()
    result = census(args.run, args.run_sha256, expected_protocol=args.protocol)
    with args.output.open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(result, stream, indent=2)
        stream.write("\n")
    print(json.dumps({"output_sha256": digest(args.output), "slots": result["total_slots"],
                      "model_calls_made": 0, "gold_read": False}))


if __name__ == "__main__":
    main()
