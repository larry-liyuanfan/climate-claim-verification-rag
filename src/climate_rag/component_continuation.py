"""One explicit post-observation continuation, not generic retry or resume.

Only r2 preflight 0 may be carried. Old records stay immutable and semantically
negative. New model calls exclude that slot and are bounded to 3 + 33.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .component_execution import PROTOCOL_SHA, SLOTS_SHA, TARGETS_SHA, complete_usage, durable, execute_slot, skeleton
from .component_preflight import preflight_cases
from .scifact_component_contract import LIMITS, PROMPTS, packing, require
from .scifact_semantic_contract import MODEL_SHA, TOKENIZER_SHA, checked, encoded, sha

RELEASE = "scifact-component-technical-continuation-20261001-v1"
POLICY = "technical-preflight-semantic-measurement-v1"
PREVIOUS_RELEASE = "scifact-component-single-attempt-20261001-r2"
PREVIOUS_SOURCE = "81917f2c9a1b2060bd84a0688bd65a40b1c4e134"
PREVIOUS_ARCHIVE = "a393a8e21c53542a851feae480ea42d7c331d7e91d1ce7e85b7a386fbe848a5e"
PREVIOUS_JOB = "31734906"
FILES = {
    "compact.json": "5e21126d04dff703a3b5a294d207048e1b8254118b87c956d7bdb3f3349fac57",
    "inference-exited.json": "453ce6ae649915d296296ad33af9ae43678145a6881816218763a319a5e797d0",
    "worker-identity.json": "f5efe10472c923b45c366dc51a3221943281f57b344261ac287bfbaa1a1d421b",
    "operator-status.json": "391a95cf25905da5c865afee7a99382bf0e15ef85b37da666a80f41f59c7f33b",
    "inference/run.json": "6f809e673e43ec5ffa7d1b0c72cac8f368203f9d4d97990e98f015515fe3971b",
    "inference/preflight-00/wire.json": "9b1f87d8d4833befb68bab93b611b322570c71a7e3a28d328de73f0ab1563b95",
    "inference/preflight-00/response.json": "befb197653cee7971c0169105b37c72a1dff829ddd684907140c55ab3d74eb50",
    "inference/preflight-00/completed.json": "3a7dde5dd001cd2ae151b6d43e4fbd3fd9b7dfef094899cb595395a5ee6e4a79",
    "inference/preflight-00/started.json": "4ac98c5256c6437761fa9852bda1b8b5c4d7c8b744d30c1f65cf07889eb996b0",
}
HISTORICAL_COST = {"v1_job": "31729507", "v1_source": "79f069d2eb4c8db3b82b152fa8a698b5ae3230ad",
    "v1_allocated_a100_seconds": 81, "r2_job": PREVIOUS_JOB, "r2_source": PREVIOUS_SOURCE,
    "r2_allocated_a100_seconds": 96, "historical_allocated_a100_seconds": 177,
    "historical_batch_cpu_seconds": 154.310}


def policy_identity() -> dict[str, Any]:
    spec = {"version": POLICY, "release": RELEASE, "predecessor_release": PREVIOUS_RELEASE,
        "predecessor_source": PREVIOUS_SOURCE, "predecessor_archive": PREVIOUS_ARCHIVE,
        "predecessor_files": FILES, "carried_slots": ["preflight-00"], "max_new_calls": 36,
        "max_cumulative_calls": 37, "synthetic_technical_gate": "valid_with_complete_verified_cost",
        "frozen_slots": SLOTS_SHA, "frozen_protocol": PROTOCOL_SHA, "frozen_targets": TARGETS_SHA,
        "model_manifest": MODEL_SHA, "tokenizer_files": TOKENIZER_SHA, "limits": LIMITS,
        "prompts": {key: sha(value.encode()) for key, value in PROMPTS.items()},
        "synthetic_inputs_and_expected_sha256": sha(encoded(preflight_cases())),
        "semantic_match": "measurement_only_no_target_change", "formal_schema_failure": "paid_continue_no_retry",
        "scope": "post_observation_already_consumed_train_diagnostic_not_original_protocol_completion"}
    return {"version": POLICY, "sha256": sha(encoded(spec))}


def audit_physical_layout(root: Path, slot_names: set[str], root_files: set[str]) -> None:
    """Reject orphan generation artifacts, including those without reservations."""
    require(not root.is_symlink(), "inference_symlink")
    if not root.exists():
        return  # A failure before the inference directory was created.
    require(root.is_dir(), "inference_directory")
    for child in root.iterdir():
        require(not child.is_symlink(), "inference_symlink")
        if child.name in root_files:
            require(child.is_file(), "inference_root_file")
            continue
        require(child.name in slot_names and child.is_dir(), "unexpected_inference_artifact")
        allowed = {"initialized.json", "wire.json", "started.json", "response.json",
                   "failure.json", "completed.json", "private"}
        private_files = []
        for item in child.iterdir():
            require(not item.is_symlink() and item.name in allowed, "unexpected_slot_artifact")
            if item.name == "private":
                require(item.is_dir(), "private_directory")
                private_files = list(item.iterdir())
                require(all(not p.is_symlink() and p.is_file() and
                    (p.name.endswith("-response.txt") or p.name.endswith("-grammar.txt"))
                    for p in private_files), "unexpected_private_artifact")
                require(all(sum(p.name.endswith(f"-{kind}.txt") for p in private_files) <= 1
                    for kind in ("response", "grammar")), "duplicate_private_artifact")
            else:
                require(item.is_file(), "slot_regular_file")
        if not (child / "started.json").exists():
            require(not (child / "response.json").exists() and not (child / "failure.json").exists()
                    and not private_files, "orphan_generation_without_reservation")


def load_carried(prior: Path, tokenizer: Any = None) -> dict[str, Any]:
    from .component_audit import reconcile
    require(prior.name == PREVIOUS_RELEASE and not prior.is_symlink(), "fixed_predecessor_path")
    checked(prior.parent.parent / "envs" / ("component-execution-" + PREVIOUS_ARCHIVE) / "source.tar", PREVIOUS_ARCHIVE)
    data = {name: json.loads(checked(prior / name, digest)) for name, digest in FILES.items()}
    identity, marker, run = data["worker-identity.json"], data["inference-exited.json"], data["inference/run.json"]
    state, compact = data["operator-status.json"], data["compact.json"]
    require(identity["release"] == PREVIOUS_RELEASE and identity["source_git"] == PREVIOUS_SOURCE
        and identity["source_archive_sha256"] == PREVIOUS_ARCHIVE and identity["model_manifest_sha256"] == MODEL_SHA
        and identity["slots_sha256"] == SLOTS_SHA and identity["scoring_targets_loaded"] is False,
        "carried_source_identity")
    require(marker["release"] == PREVIOUS_RELEASE and marker["child_reaped"] is True and marker["returncode"] == 0
        and marker["worker_identity_sha256"] == FILES["worker-identity.json"], "carried_exit_proof")
    require(state["job_id"] == PREVIOUS_JOB and state["compact_sha256"] == FILES["compact.json"]
        and compact["total_attempted"] == 1 and compact["unknown_cost_attempts"] == 0
        and compact["status"] == "stopped_no_semantic_effect_claim", "carried_terminal_identity")
    require(len(run["preflight"]) == 4 and len(run["diagnostic"]) == 33
        and run["preflight"][0]["attempted"] is True
        and all(not r["attempted"] and r["status"] == "not_attempted_after_stop"
                for r in run["preflight"][1:] + run["diagnostic"]), "carried_attempt_history")
    inference = prior / "inference"
    audit_physical_layout(inference, {"preflight-00"}, {"run.json"})
    require({p.name for p in inference.iterdir()} == {"preflight-00", "run.json"}
        and all(not p.is_symlink() for p in inference.rglob("*"))
        and {p.relative_to(inference).as_posix() for p in inference.rglob("started.json")}
            == {"preflight-00/started.json"}, "carried_hidden_attempt")
    require([r["slot"] for r in identity["preflight"]] == list(range(4)) and
        [r["input"] for r in identity["preflight"]] == [s for s, _ in preflight_cases()], "carried_preflight_input")
    first = identity["preflight"][0]
    if tokenizer is not None:
        require(packing(tokenizer, first["input"]) == first["packing"], "carried_runtime_packing")
    record = reconcile(first, prior / "inference", "preflight")
    require(record["status"] == "valid" and record["usage_known"] is True
        and record["usage"] == {"input_tokens": 277, "output_tokens": 15}
        and record["prediction"] == {"decision": "abstain", "document_ids": []}, "carried_exact_negative")
    reference = {"predecessor_release": PREVIOUS_RELEASE, "predecessor_job": PREVIOUS_JOB,
        "predecessor_source": PREVIOUS_SOURCE, "predecessor_archive": PREVIOUS_ARCHIVE,
        "files": FILES, "record_sha256": sha(encoded(record)), "policy": policy_identity()}
    return {"record": record, "reference": reference}


def measurement(record: dict[str, Any], index: int, *, carried: bool) -> dict[str, Any]:
    valid = record["status"] == "valid" and record["usage_known"] and complete_usage(record["usage"], {})
    return record | {"execution_origin": "carried" if carried else "new",
        "technical_ready": bool(valid) if record["attempted"] else None,
        "semantic_match": record.get("prediction") == preflight_cases()[index][1] if valid else None}


def cost_partition(preflight: list[dict[str, Any]], records: list[dict[str, Any]]) -> dict[str, Any]:
    require(len(preflight) == 4 and [r["slot"] for r in preflight] == list(range(4)) and len(records) == 33
            and [r["slot"] for r in records] == list(range(33)), "continuation_denominators")
    require(preflight[0]["execution_origin"] == "carried" and preflight[0]["attempted"]
        and all(r["execution_origin"] == "new" for r in preflight[1:] + records), "carried_counted_once")
    def cost(rows: list[dict[str, Any]]) -> dict[str, Any]:
        return {"attempted": sum(bool(r["attempted"]) for r in rows),
            "tokens": {k: sum((r.get("usage") or r.get("known_usage_lower_bound") or {}).get(k, 0) for r in rows)
                       for k in ("input_tokens", "output_tokens")},
            "unknown_cost_attempts": sum(r["attempted"] and not r["usage_known"] for r in rows),
            "slot_elapsed_ms": sum(r.get("elapsed_ms", 0) for r in rows)}
    carried, new, total = cost(preflight[:1]), cost(preflight[1:] + records), cost(preflight + records)
    require(carried["attempted"] == 1 and new["attempted"] <= 36 and total["attempted"] <= 37, "continuation_call_ceiling")
    return {"carried": carried, "new": new, "cumulative": total, "historical_jobs": HISTORICAL_COST}


def execute_continuation(rows: list[dict[str, Any]], provider: Any, root: Path, prior: Path) -> dict[str, Any]:
    require(len(rows) == 33 and [r["slot"] for r in rows] == list(range(33)), "matrix_shape")
    carried = load_carried(prior, provider.tokenizer)
    root.mkdir(mode=0o700)
    # Only a reference is persisted; never copy old wire/raw/receipt or create a
    # new preflight-00 directory. The scorer reopens and verifies the predecessor.
    durable(root / "carried-reference.json", carried["reference"])
    preflight = [measurement(carried["record"], 0, carried=True)]
    stopped = not preflight[0]["technical_ready"]
    for i, (spec, _) in enumerate(preflight_cases()[1:], start=1):
        slot = {"slot": i, "input": spec, "packing": packing(provider.tokenizer, spec)}
        record = skeleton(slot, "preflight") if stopped else execute_slot(slot, provider, root, "preflight")
        item = measurement(record, i, carried=False)
        preflight.append(item)
        stopped = stopped or item["technical_ready"] is not True
    records = []
    for slot in rows:
        record = skeleton(slot, "diagnostic") if stopped else execute_slot(slot, provider, root, "diagnostic")
        records.append(record | {"execution_origin": "new"})
        stopped = stopped or record["status"] == "provider_failed"
    report = {"release": RELEASE, "policy": policy_identity(), "preflight": preflight,
        "diagnostic": records, "status": "stopped" if stopped else "complete",
        "scoring_targets_loaded": False, "cost_partition": cost_partition(preflight, records)}
    durable(root / "run.json", report)
    return report
