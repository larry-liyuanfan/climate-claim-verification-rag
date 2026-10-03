"""Independent, post-exit audit for the one-carried/36-new diagnostic protocol."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from climate_rag.component_audit import aggregate, reconcile
from climate_rag.component_continuation import (
    POLICY, RELEASE, PREVIOUS_RELEASE, audit_physical_layout, cost_partition, load_carried, measurement, policy_identity,
)
from climate_rag.component_execution import SLOTS_SHA, TARGETS_SHA, durable, frozen_slots
from climate_rag.component_preflight import preflight_cases
from climate_rag.scifact_component_contract import require
from climate_rag.scifact_semantic_contract import MODEL_SHA, checked, sha
from run_budget_agent_full_operator import ROOT


def score_run(result: Path, slots_path: Path, target_path: Path, exit_sha: str) -> dict[str, Any]:
    require(result.name == RELEASE and not result.is_symlink(), "fixed_continuation_root")
    marker = json.loads(checked(result / "inference-exited.json", exit_sha))
    require(marker.get("child_reaped") is True and marker.get("release") == RELEASE, "inference_not_exited")
    identity = json.loads(checked(result / "worker-identity.json", marker["worker_identity_sha256"]))
    require(identity["release"] == RELEASE and identity["scoring_targets_loaded"] is False
        and identity["preflight_policy"] == policy_identity() and identity["slots_sha256"] == SLOTS_SHA
        and identity["model_manifest_sha256"] == MODEL_SHA, "continuation_worker_identity")
    carried = load_carried(ROOT / "runs" / PREVIOUS_RELEASE)
    require(identity["carried_reference"] == carried["reference"], "worker_carried_identity")
    inference = result / "inference"
    require(not (inference / "preflight-00").exists() and not (inference / "preflight-00").is_symlink(),
            "duplicate_carried_slot")
    audit_physical_layout(inference, {f"preflight-{i:02d}" for i in (1, 2, 3)} |
        {f"diagnostic-{i:02d}" for i in range(33)}, {"carried-reference.json", "run.json"})
    if (inference / "carried-reference.json").exists():
        require(json.loads((inference / "carried-reference.json").read_bytes()) == carried["reference"], "carried_reference")
    else:
        # Failure before any new model invocation may have no inference directory.
        require(not inference.exists(), "missing_carried_reference")
    preflight_slots = identity["preflight"]
    require([s["slot"] for s in preflight_slots] == list(range(4)) and
        [s["input"] for s in preflight_slots] == [spec for spec, _ in preflight_cases()]
        and preflight_slots[0]["packing"] == carried["record"]["packing"], "preflight_identity")
    preflight = [measurement(carried["record"], 0, carried=True)] + [
        measurement(reconcile(slot, inference, "preflight"), slot["slot"], carried=False)
        for slot in preflight_slots[1:]]
    slots = frozen_slots(slots_path)
    records = [reconcile(slot, inference, "diagnostic") | {"execution_origin": "new"} for slot in slots]
    expected_started = {f'preflight-{r["slot"]:02d}/started.json' for r in preflight[1:] if r["attempted"]} | {
        f'diagnostic-{r["slot"]:02d}/started.json' for r in records if r["attempted"]}
    require(all(not p.is_symlink() for p in inference.rglob("*")) and
        {p.relative_to(inference).as_posix() for p in inference.rglob("started.json")} == expected_started,
        "unaccounted_new_attempt")
    # No scoring targets enter executor/provider; only this process reads them,
    # after the hash-bound actual child-exit proof above.
    targets = json.loads(checked(target_path, TARGETS_SHA))
    summary, private = aggregate(slots, targets, records, preflight, preflight_policy=POLICY)
    partition = cost_partition(preflight, records)
    summary.update(release=RELEASE, preflight_policy=policy_identity(), cost_partition=partition,
        new_attempted=partition["new"]["attempted"], carried_attempted=1,
        cumulative_attempted=partition["cumulative"]["attempted"],
        cross_job_continuation=True, original_protocol_completed=False,
        semantic_measurements=[{k: r[k] for k in ("slot", "execution_origin", "technical_ready", "semantic_match")}
                               for r in preflight],
        carried_reference=carried["reference"], source_git=identity["source_git"],
        source_archive_sha256=identity["source_archive_sha256"], inference_exit_sha256=exit_sha,
        slots_sha256=SLOTS_SHA, targets_sha256=TARGETS_SHA, model_manifest_sha256=MODEL_SHA)
    if (inference / "run.json").exists():
        run = json.loads((inference / "run.json").read_bytes())
        require(run["policy"] == policy_identity() and run["cost_partition"] == partition, "executor_audit_disagree")
    durable(result / "private-score.json", private)
    summary["private_score_sha256"] = sha((result / "private-score.json").read_bytes())
    durable(result / "compact.json", summary)
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("result", "slots", "targets"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--exit-sha", required=True)
    args = parser.parse_args()
    score_run(args.result, args.slots, args.targets, args.exit_sha)
