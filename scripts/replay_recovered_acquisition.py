"""Read saved real episodes; export counts without gold, text, IDs or inference."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Any

from climate_rag import stop_acquire
from climate_rag.targeted_query import ROUTES

STUDY = "public_v2_repeatedly_used_validation_replay_not_independent_test"


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def require(value: bool, message: str) -> None:
    if not value:
        raise ValueError(message)


def visible(observation: dict[str, Any]) -> dict[str, str]:
    return {item["sentence_id"]: item["text"]
            for item in observation["current_citable"]}


def replay(run_path: Path, expected_sha: str) -> dict[str, Any]:
    """Check recorded transition identities, not model quality or entailment."""
    require(digest(run_path) == expected_sha, "recorded_run_hash_mismatch")
    run = json.loads(run_path.read_bytes())
    require(run["protocol"] == stop_acquire.PROTOCOL, "unsupported_protocol")
    require(run["study_kind"] == STUDY, "unsupported_study")
    rows = run["runs"]
    require(all(row["route"] in ROUTES for row in rows), "unknown_route")
    routes: dict[str, Any] = {}
    for route in sorted({row["route"] for row in rows}):
        selected = [row for row in rows if row["route"] == route]
        actions: Counter[str] = Counter()
        errors: Counter[str] = Counter()
        tools: Counter[str] = Counter()
        feedback = 0
        added_text = 0
        query_branches = 0
        read_branches = 0
        rerank_branches = 0
        acquire_available = 0
        read_counts: list[int] = []
        gates = 0
        remaining: list[int] = []
        for row in selected:
            attempts = row["generation_attempts"]
            for ordinal, attempt in enumerate(attempts):
                observation = attempt["observation"]
                # Journal identities bind the saved physical call to this episode.
                key = attempt["diagnostics"]["physical_attempt_id"]
                reserved = json.loads((run_path.parent / "ledger" /
                                       f"{key}.reserved.json").read_bytes())
                require(reserved["observation"] == observation,
                        "physical_observation_changed")
                require(reserved["schema"] == attempt["schema"],
                        "physical_schema_changed")
                decision = attempt.get("decision")
                if decision is not None:
                    require(decision["action"] in {
                        "stop", "acquire", "answer", "abstain", "plan_queries",
                        "read", "rewrite", "rerank"}, "unknown_action")
                    actions[decision["action"]] += 1
                if "error_code" in attempt:
                    error = attempt["error_code"]
                    errors[error if error in {
                        "duplicate_sentence_reference", "invalid_schema", "invalid_json",
                        "read_loop", "read_not_unseen_preview",
                    } else "other_validation_error"] += 1
                if route == "adaptive" and attempt["stage"] == "gate":
                    gates += 1
                    remaining.append(observation["remaining_calls"])
                    read_counts.append(len(observation["readable_source_ids"]))
                    acquire_available += bool(observation["acquisition_available"])
                    expected = stop_acquire.gate_schema(
                        observation["readable_source_ids"],
                        can_acquire=observation["acquisition_available"],
                        can_query=observation["query_available"],
                        max_read=observation["read_limit"],
                    )
                    require(expected == attempt["schema"], "gate_schema_changed")
                    branches = attempt["schema"]["anyOf"]
                    query_branches += any(
                        b["properties"].get("tool", {}).get("enum") == ["query"]
                        for b in branches)
                    read_branches += any(
                        b["properties"].get("tool", {}).get("enum") == ["read"]
                        for b in branches)
                    rerank_branches += any(
                        b["properties"].get("tool", {}).get("enum") == ["rerank"]
                        for b in branches)
                    if decision is not None:
                        shown = [s.split(":")[0] for s in visible(observation)]
                        require(stop_acquire.parse_gate(
                            attempt["proposed_decision"], observation,
                            fully_shown=shown) == decision, "gate_parse_changed")
                for event in row["events"]:
                    if event.get("trigger_attempt") != ordinal:
                        continue
                    require(event["tool"] in {"retrieve", "read", "rewrite", "rerank"}
                            and event["status"] in {"completed", "failed"},
                            "unknown_tool_event")
                    tools[f'{event["tool"]}:{event["status"]}'] += 1
                    if route == "adaptive" and attempt["stage"] == "gate":
                        require(decision is not None and decision["action"] == "acquire",
                                "tool_after_non_acquisition_gate")
                        require(event["tool"] == (
                            "read" if decision["tool"] == "read" else "rewrite"),
                            "gate_tool_execution_identity_changed")
                    # A policy's query action is executed/journalled as rewrite;
                    # retrieve is also a physical acquisition event, never query.
                    if event["tool"] not in ("retrieve", "rewrite", "read"):
                        continue
                    if ordinal + 1 >= len(attempts):
                        continue
                    subsequent = attempts[ordinal + 1]["observation"]
                    delivered = subsequent.get("tool_feedback", {})
                    if delivered.get("status") != event["status"]:
                        continue
                    require(delivered.get("tool") == event["tool"],
                            "feedback_tool_identity_changed")
                    if event.get("query_sha256") is not None:
                        require(delivered.get("query_sha256") == event["query_sha256"],
                                "feedback_query_identity_changed")
                    before, after = visible(observation), visible(subsequent)
                    require(all(after.get(k) == v for k, v in before.items()),
                            "retained_text_changed_after_acquisition")
                    feedback += 1
                    added_text += bool(set(after) - set(before))
            # Initial retrieve has no triggering model call; count it separately.
            for event in row["events"]:
                if event.get("trigger_attempt") is None or event["trigger_attempt"] < 0:
                    require(event["tool"] in {"retrieve", "read", "rewrite", "rerank"}
                            and event["status"] in {"completed", "failed"},
                            "unknown_initial_tool_event")
                    tools[f'{event["tool"]}:{event["status"]}'] += 1
        routes[route] = {
            "slots": len(selected),
            "recorded_model_calls": sum(len(r["generation_attempts"]) for r in selected),
            "validated_actions": dict(actions),
            "validation_errors": dict(errors),
            "tool_events": dict(tools),
            "acquisition_feedback_followed_by_model": feedback if gates else None,
            "feedback_with_additional_citable_text": added_text if gates else None,
            "event_link_scope": "trigger-bound adaptive transitions" if gates else
                                "control events lack per-attempt triggers; not counted",
            "gate_opportunities": {
                "count": gates, "acquisition_available": acquire_available,
                "query_branch": query_branches, "read_branch": read_branches,
                "rerank_branch": rerank_branches,
                "readable_source_count_min": min(read_counts, default=0),
                "readable_source_count_max": max(read_counts, default=0),
                "remaining_calls_min": min(remaining, default=0),
                "remaining_calls_max": max(remaining, default=0),
            },
        }
    return {
        "schema": "recovered-acquisition-census-v1",
        "protocol": run["protocol"], "study_kind": run["study_kind"],
        "run_sha256": expected_sha, "total_slots": len(rows), "routes": routes,
        "gold_read": False, "model_calls_made": 0, "scores_recomputed": False,
        "semantic_support": "not_measured_by_this_replay",
        "reasonable_stop": "not_adjudicated_by_action_validity",
        "boundary": "Recorded physical transitions only; not a new model experiment. "
                    "Fixed control feedback is not autonomous acquisition benefit.",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--run-sha256", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = replay(args.run, args.run_sha256)
    # Existing files and original episodes must never be overwritten.
    with args.output.open("x", encoding="utf-8", newline="\n") as handle:
        json.dump(result, handle, indent=2, ensure_ascii=False)
        handle.write("\n")
    print(json.dumps({"output_sha256": digest(args.output),
                      "slots": result["total_slots"], "model_calls_made": 0}))


if __name__ == "__main__":
    main()
