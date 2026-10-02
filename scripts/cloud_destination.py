"""Destination CPU byte/capacity gate; a receipt is evidence, not authority."""
from __future__ import annotations

import json
import hashlib
import shutil
from pathlib import Path
from typing import Any

from cloud_replay_contract import check_paths, digest, validate_release, verify_source
from cloud_capacity import observe_capacity
from preflight_cloud_assets import inventory

GIB = 1024**3


def plan_sha(value: dict[str, Any]) -> str:
    ignored = {"authorization", "model_execution_authorized", "runtime_receipt_sha256", "asset_receipt_sha256"}
    return hashlib.sha256(json.dumps({k: v for k, v in value.items() if k not in ignored}, sort_keys=True).encode()).hexdigest()


def disk_capacity(value: dict[str, Any], extracted_bytes: int) -> list[dict[str, Any]]:
    """Already uploaded bytes consume used space; don't subtract them again.

    Sum additional reservations on each actual device. Different quota/mount
    views of one device use the smallest observed free space, conservatively.
    """
    if extracted_bytes <= 0:
        raise ValueError("invalid_extraction_size")
    reservations = {"input_archive": 0, "gold_path": 0, "source": 2 * GIB,
                    "python_executable": 2 * GIB, "work": extracted_bytes + 4 * GIB,
                    "output": 8 * GIB, "asset_receipt": GIB}
    groups: dict[int, dict[str, Any]] = {}
    for key, needed in reservations.items():
        path = Path(value[key])
        # A venv Python may be a symlink to the system interpreter: reserve on
        # the environment's filesystem, not the target interpreter's filesystem.
        anchor = path.parent if key == "python_executable" else path
        while not anchor.exists() and anchor != anchor.parent:
            anchor = anchor.parent
        device = anchor.stat().st_dev
        usage = shutil.disk_usage(anchor)
        item = groups.setdefault(device, {"device": device, "free_bytes": usage.free,
                                         "additional_required_bytes": 0, "paths": []})
        item["free_bytes"] = min(item["free_bytes"], usage.free)
        item["additional_required_bytes"] += needed
        item["paths"].append({"key": key, "path": str(path), "observed_at": str(anchor),
                              "additional_bytes": needed})
    for group in groups.values():
        if group["free_bytes"] < group["additional_required_bytes"]:
            raise ValueError("destination_filesystem_capacity")
    return list(groups.values())


def destination_preflight(value: dict[str, Any], source: Path) -> dict[str, Any]:
    validate_release(value, execution=False)
    check_paths(value, new_run=True)
    verify_source(value, source)
    effective_compute = observe_capacity()
    if not Path(value["python_executable"]).is_file():
        raise ValueError("destination_environment_missing")
    for key in ("work", "output", "asset_receipt"):
        if not Path(value[key]).parent.is_dir():
            raise ValueError("dedicated_run_parent_required")
    assets = inventory(Path(value["input_archive"]), Path(value["gold_path"]), source,
                       full=True, input_sha=value["input_transport_sha256"])
    if assets["asset_status"] != "verified_local_assets":
        raise ValueError("destination_assets_unverified")
    return {"scope": "destination_cpu_assets_not_execution_authority", "plan_sha256": plan_sha(value),
            "model_execution_authorized": False, "assets": assets["assets"],
            "capacity": disk_capacity(value, assets["assets"]["input"]["extracted_bytes"]),
            "effective_compute": effective_compute,
            "gold_labels_parsed": False, "model_weights_loaded": False}


def verify_destination(value: dict[str, Any]) -> None:
    path = Path(value["asset_receipt"])
    if digest(path) != value["asset_receipt_sha256"]:
        raise ValueError("destination_receipt_hash")
    receipt = json.loads(path.read_bytes())
    if (receipt.get("scope") != "destination_cpu_assets_not_execution_authority"
            or receipt.get("plan_sha256") != plan_sha(value)
            or receipt.get("model_execution_authorized") is not False):
        raise ValueError("destination_receipt_binding")
    assets = receipt["assets"]
    if (assets["input"]["status"] != "verified"
            or assets["input"]["transport_sha256"] != value["input_transport_sha256"]
            or assets["gold"]["sha256"] != value["gold_sha256"]
            or digest(Path(value["gold_path"])) != value["gold_sha256"]):
        raise ValueError("destination_asset_identity")
    disk_capacity(value, assets["input"]["extracted_bytes"])
    # prepare() independently rehashes transport and each extracted member
    # before the model child can start. No mtime/inode trust shortcut.
