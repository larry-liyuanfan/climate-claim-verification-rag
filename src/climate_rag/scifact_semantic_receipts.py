"""Stdlib-only completed-arm receipt checks, callable before runtime installation."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .scifact_semantic_contract import (
    COMPACT_SHA, MODEL_SHA, PAIR, POLICIES, PREPARATION_GIT, PROMPTS, RELEASE,
    RERANKER_SHA, checked, sha,
)

def read_completed_run(path: Path, protocol: dict[str, Any], protocol_sha: str,
                       source_sha: str, inference_sha: str, policy: str) -> dict[str, Any]:
    run: dict[str, Any] = json.loads(path.read_bytes())
    expected = {"release_id": RELEASE, "arm": policy, "comparison_protocol": PAIR,
        "source_git": protocol["execution_source_git"], "source_archive_sha256": source_sha,
        "preparation_source_git": PREPARATION_GIT, "preparation_compact_sha256": COMPACT_SHA,
        "protocol_sha256": protocol_sha, "inference_archive_sha256": inference_sha,
        "inference_file_sha256": protocol["inference_file_sha256"], "model_sha256": MODEL_SHA,
        "reranker_sha256": RERANKER_SHA, "prompt_sha256": PROMPTS[policy], "gold_loaded": False}
    if any(run.get(k) != v for k, v in expected.items()):
        raise ValueError("semantic_completed_run_identity")
    flight = json.loads(checked(path.parent / "runtime-preflight.json", run["preflight_sha256"]))
    if (flight.get("policy") != policy or flight.get("prompt_sha256") != PROMPTS[policy]
            or flight.get("gap") is not True or flight.get("source_git") != run["source_git"]
            or flight.get("protocol_sha256") != protocol_sha or flight.get("status") != "passed"
            or flight.get("attempted_calls") != 4 or len(flight.get("records", [])) != 4
            or flight.get("execution_kind") != "local_model"):
        raise ValueError("semantic_preflight_binding")
    slots = protocol["slots_per_arm"]
    if len(run["runs"]) != len(slots):
        raise ValueError("semantic_matrix_incomplete")
    for index, (row, spec) in enumerate(zip(run["runs"], slots, strict=True), 1):
        identity = {**spec, "arm": policy, "source_git": run["source_git"],
                    "comparison_protocol": PAIR, "prompt_sha256": PROMPTS[policy]}
        if any(row.get(k) != v or row["result"].get(k) != v for k, v in identity.items()):
            raise ValueError("semantic_slot_identity")
        if json.loads((path.parent / f"slot-{index:02d}.json").read_bytes()) != row:
            raise ValueError("semantic_durable_slot_mismatch")
        raw = json.loads((path.parent / f"slot-{index:02d}-raw.json").read_bytes())
        finalizer_added = {"decision_execution_audit", "trace_unlinked_events",
                           "model_tool_links", "initial_context_identity"}
        expected_raw = {k: v for k, v in row["result"].items() if k not in finalizer_added}
        if raw != expected_raw:
            raise ValueError("semantic_durable_raw_mismatch")
    return run


def previous_arm(root: Path, protocol: dict[str, Any], protocol_sha: str,
                 source_sha: str, inference_sha: str) -> str:
    previous = root / "runs" / (RELEASE + "-" + POLICIES[0])
    state = json.loads((previous / "operator-status.json").read_bytes())
    path = previous / "inference/run.json"
    run_sha = sha(path.read_bytes())
    if (state.get("status") != "complete" or state.get("source_git") != protocol["execution_source_git"]
            or state.get("source_archive_sha256") != source_sha or state.get("protocol_sha256") != protocol_sha
            or state.get("inference_sha256") != run_sha or state.get("arm") != POLICIES[0]):
        raise ValueError("previous_semantic_arm_not_complete")
    read_completed_run(path, protocol, protocol_sha, source_sha, inference_sha, POLICIES[0])
    return run_sha
