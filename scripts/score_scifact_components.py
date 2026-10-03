"""Scoring-only process: validate private wire artifacts after inference exits."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from climate_rag.component_audit import aggregate, reconcile
from climate_rag.component_execution import INFRASTRUCTURE_LINEAGE, RELEASE, TARGETS_SHA, durable, frozen_slots
from climate_rag.component_preflight import preflight_cases
from climate_rag.scifact_component_contract import require
from climate_rag.scifact_semantic_contract import checked, sha


def score_run(result: Path, slots_path: Path, target_path: Path, exit_sha: str) -> dict[str, Any]:
    marker = json.loads(checked(result / "inference-exited.json", exit_sha))
    require(marker.get("child_reaped") is True and marker.get("release") == RELEASE, "inference_not_exited")
    slots = frozen_slots(slots_path)
    # The scorer is the only process that reads this hash-bound target file.
    targets = json.loads(checked(target_path, TARGETS_SHA))
    identity = json.loads(checked(result / "worker-identity.json", marker["worker_identity_sha256"]))
    require(identity["release"] == RELEASE and identity["scoring_targets_loaded"] is False, "worker_identity")
    require(identity.get("infrastructure_lineage") == INFRASTRUCTURE_LINEAGE, "infrastructure_lineage")
    require([s["slot"] for s in identity["preflight"]] == list(range(4)) and
            [s["input"] for s in identity["preflight"]] == [spec for spec, _ in preflight_cases()], "preflight_identity")
    preflight = [reconcile(slot, result / "inference", "preflight") for slot in identity["preflight"]]
    records = [reconcile(slot, result / "inference", "diagnostic") for slot in slots]
    summary, private_rows = aggregate(slots, targets, records, preflight)
    summary.update(source_git=identity["source_git"], source_archive_sha256=identity["source_archive_sha256"],
                   preparation_git=identity["preparation_git"], inference_exit_sha256=exit_sha,
                   slots_sha256=identity["slots_sha256"], targets_sha256=TARGETS_SHA,
                   model_manifest_sha256=identity["model_manifest_sha256"],
                   release=RELEASE, infrastructure_lineage=INFRASTRUCTURE_LINEAGE)
    durable(result / "private-score.json", private_rows)
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
