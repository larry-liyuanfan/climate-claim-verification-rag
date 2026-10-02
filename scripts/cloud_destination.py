"""Destination CPU byte/capacity gate; a receipt is evidence, not authority."""
from __future__ import annotations

import json
import hashlib
import shutil
import os
from pathlib import Path
from typing import Any

from cloud_replay_contract import check_paths, digest, validate_release, verify_source
from cloud_capacity import observe_capacity, provider_for_release, RUNPOD_VISIBLE, _unescape
from preflight_cloud_assets import inventory

GIB = 1024**3
RUNPOD_WORKSPACE = Path("/workspace")


def workspace_mount() -> str:
    matches = []
    for line in Path("/proc/self/mountinfo").read_text().splitlines():
        before, after = line.split(" - ", 1)
        path = Path(_unescape(before.split()[4]))
        if path == RUNPOD_WORKSPACE:
            matches.append(after.split()[0])
        elif path.is_relative_to(RUNPOD_WORKSPACE):
            raise ValueError("workspace_nested_mount_unverified")
    if len(matches) != 1:
        raise ValueError("workspace_mount_unverified")
    return matches[0]


def provider_volume_budget(value: dict[str, Any], additional: int) -> dict[str, Any]:
    provider = provider_for_release(value)
    if provider is None:
        raise ValueError("runpod_provider_required")
    filesystem = workspace_mount()
    volume = RUNPOD_WORKSPACE
    if volume.resolve() != volume:
        raise ValueError("workspace_symlink")
    device = volume.stat().st_dev
    seen: set[tuple[int, int]] = set()
    used = 0

    def unreadable(exc: OSError) -> None:
        raise exc

    # Entire visible Pod volume, not only this project: metadata only, no
    # content hashing. Links count as stored links and are never followed.
    for directory, dirs, files in os.walk(volume, onerror=unreadable, followlinks=False):
        for path in [Path(directory), *(Path(directory) / name for name in dirs + files)]:
            info = path.lstat()
            if info.st_dev != device:
                raise ValueError("workspace_submount_unverified")
            identity = (info.st_dev, info.st_ino)
            if identity not in seen:
                seen.add(identity)
                used += max(info.st_size, getattr(info, "st_blocks", 0) * 512)
    allocated = provider["allocation"]["workspace_volume_gb"] * 10**9
    if used + additional > allocated:
        raise ValueError("provider_workspace_allocation_budget")
    return {"provider_allocation_sha256": value["provider_allocation_sha256"],
            "workspace_volume": str(volume), "filesystem": filesystem,
            "provider_workspace_allocated_bytes": allocated, "observed_volume_used_bytes": used,
            "additional_reserved_bytes": additional, "budget_fits_observation": True,
            "usage_method": "visible_volume_metadata_max_apparent_or_allocated_deduplicated_inode",
            "provider_container_allocated_bytes": provider["allocation"]["container_gb"] * 10**9,
            "container_usage_verified": False, "container_space_not_pooled": True,
            "quota_independently_verified": False, "provider_overhead_independently_verified": False,
            "df_scope": "provider_shared_filesystem_not_per_pod_quota"}


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
    if value.get("capacity_contract") == RUNPOD_VISIBLE:
        if any(not Path(value[key]).is_relative_to(RUNPOD_WORKSPACE) for key in reservations):
            raise ValueError("runpod_reserved_paths_outside_workspace")
        budget = provider_volume_budget(value, sum(g["additional_required_bytes"] for g in groups.values()))
        for group in groups.values():
            group["provider_volume_budget"] = budget
    return list(groups.values())


def destination_preflight(value: dict[str, Any], source: Path) -> dict[str, Any]:
    validate_release(value, execution=False)
    check_paths(value, new_run=True)
    verify_source(value, source)
    effective_compute = observe_capacity(value)
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
            "capacity_contract": value["capacity_contract"],
            "provider_allocation_sha256": value["provider_allocation_sha256"],
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
