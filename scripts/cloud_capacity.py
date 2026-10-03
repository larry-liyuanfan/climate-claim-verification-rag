"""Two explicit capacity contracts; neither guarantees exclusive resources."""
from __future__ import annotations

from fractions import Fraction
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import re
from typing import Any
from datetime import datetime, timezone

COMPLETE_HOST = "complete-host-unified-v2"
RUNPOD_VISIBLE = "runpod-namespaced-visible-v1"


def validate_provider(value: dict[str, Any]) -> None:
    expected = {"schema", "provider", "instance_id", "source_evidence_sha256", "allocation"}
    if (set(value) != expected or value["schema"] != "coordinator-runpod-allocation-v1"
            or value["provider"] != "Runpod"
            or not re.fullmatch(r"[a-z0-9]{6,40}", value["instance_id"])
            or not re.fullmatch(r"[0-9a-f]{64}", value["source_evidence_sha256"])):
        raise ValueError("provider_allocation_schema")
    allocation = value["allocation"]
    if set(allocation) != {"vcpu", "memory_gb", "container_gb", "workspace_volume_gb", "image"}:
        raise ValueError("provider_allocation_fields")
    for key in ("vcpu", "memory_gb", "container_gb", "workspace_volume_gb"):
        if type(allocation[key]) is not int or allocation[key] <= 0:
            raise ValueError("provider_allocation_units")
    if not isinstance(allocation["image"], str) or not allocation["image"]:
        raise ValueError("provider_image_missing")


def provider_from_evidence(path: Path, expected_sha: str) -> dict[str, Any]:
    """Local CPU projection: omit billing, SSH and dynamic kernel observations."""
    raw = path.read_bytes()
    if hashlib.sha256(raw).hexdigest() != expected_sha:
        raise ValueError("provider_source_evidence_hash")
    original = json.loads(raw)
    result = {"schema": "coordinator-runpod-allocation-v1", "provider": original["provider"],
              "instance_id": original["instance_id"], "source_evidence_sha256": expected_sha,
              "allocation": original["control_plane"]}
    validate_provider(result)
    return result


def provider_for_release(value: dict[str, Any]) -> dict[str, Any] | None:
    if value["capacity_contract"] == COMPLETE_HOST:
        return None
    if value["capacity_contract"] != RUNPOD_VISIBLE:
        raise ValueError("unknown_capacity_contract")
    raw = Path(value["provider_allocation_receipt"]).read_bytes()
    if hashlib.sha256(raw).hexdigest() != value["provider_allocation_sha256"]:
        raise ValueError("provider_allocation_hash")
    result: dict[str, Any] = json.loads(raw)
    validate_provider(result)
    if result["source_evidence_sha256"] != value["provider_evidence_sha256"]:
        raise ValueError("provider_evidence_binding")
    return result


def check_provider_instance(configuration: dict[str, Any], instance_id: str) -> None:
    provider = configuration.get("provider_allocation")
    if provider is not None and (provider["instance_id"] != instance_id
            or os.environ.get("RUNPOD_POD_ID") != instance_id):
        raise ValueError("provider_instance_identity_mismatch")


def _unescape(value: str) -> str:
    return re.sub(r"\\([0-7]{3})", lambda match: chr(int(match[1], 8)), value)


def effective_capacity(membership: str, mounts: str, *, host_memory: int,
                       host_cpus: int, affinity: int,
                       capacity_contract: str = COMPLETE_HOST,
                       provider: dict[str, Any] | None = None) -> dict[str, Any]:
    """Pure metadata entry. Namespace semantics require explicit opt-in."""
    if capacity_contract not in (COMPLETE_HOST, RUNPOD_VISIBLE):
        raise ValueError("unknown_capacity_contract")
    namespaced = capacity_contract == RUNPOD_VISIBLE
    if namespaced:
        if provider is None:
            raise ValueError("provider_allocation_required")
        validate_provider(provider)
    lines = membership.strip().splitlines()
    if len(lines) != 1 or not lines[0].startswith("0::"):
        raise ValueError("unverified_cgroup_layout:v1_hybrid_or_missing")
    group = PurePosixPath(lines[0][3:])
    if not group.is_absolute() or ".." in group.parts or group.as_posix() != lines[0][3:]:
        raise ValueError("unverified_cgroup_layout:membership")
    candidates = []
    for line in mounts.splitlines():
        before, after = line.split(" - ", 1)
        fields, description = before.split(), after.split()
        if description[0] == "cgroup2":
            candidates.append((_unescape(fields[3]), Path(_unescape(fields[4]))))
    if len(candidates) != 1 or candidates[0][0] != "/":
        raise ValueError("unverified_cgroup_layout:hidden_or_ambiguous_ancestors")
    root = candidates[0][1]
    current = root.joinpath(*group.parts[1:])
    if current.resolve() != current or root.resolve() != root:
        raise ValueError("unverified_cgroup_layout:symlink")
    memory, cpus = host_memory, Fraction(min(host_cpus, affinity))
    if namespaced and provider is not None:
        memory = min(memory, provider["allocation"]["memory_gb"] * 10**9)
        cpus = min(cpus, Fraction(provider["allocation"]["vcpu"]))
    if memory <= 0 or cpus <= 0:
        raise ValueError("unverified_host_capacity")
    observed = []
    while True:
        try:
            # A genuine host hierarchy root has these controller files but no
            # memory.max/cpu.max. A namespaced child root exposes limit files:
            # reject rather than assume hidden ancestor limits are unbounded.
            if current == root and not namespaced:
                controllers = (root / "cgroup.controllers").read_text().split()
                if not {"memory", "cpu"} <= set(controllers):
                    raise ValueError("unverified_cgroup_layout:controllers")
                if (root / "memory.max").exists() or (root / "cpu.max").exists():
                    raise ValueError("unverified_cgroup_layout:namespaced_root")
                observed.append({"path": str(current), "host_hierarchy_root": True})
                break
            raw_memory = (current / "memory.max").read_text().strip()
            quota, period = (current / "cpu.max").read_text().split()
            if not period.isdecimal() or int(period) <= 0:
                raise ValueError("unverified_cgroup_layout:cpu_period")
            if namespaced and current == root and (raw_memory == "max" or quota == "max"):
                raise ValueError("finite_visible_root_limits_required")
            if raw_memory != "max":
                if not raw_memory.isdecimal() or int(raw_memory) <= 0:
                    raise ValueError("unverified_cgroup_layout:memory_limit")
                memory = min(memory, int(raw_memory))
            if quota != "max":
                if not quota.isdecimal() or int(quota) <= 0:
                    raise ValueError("unverified_cgroup_layout:cpu_quota")
                cpus = min(cpus, Fraction(int(quota), int(period)))
            observed.append({"path": str(current), "memory.max": raw_memory, "cpu.max": [quota, period]})
            if current == root:
                break
        except OSError as exc:
            raise ValueError("unverified_cgroup_layout:unreadable_ancestor") from exc
        current = current.parent
    if namespaced:
        return {"capacity_contract": RUNPOD_VISIBLE, "membership": str(group), "mount": str(root),
                "visible_ancestors": observed, "provider_allocation": provider,
                "host_memory_bytes": host_memory, "host_cpus": host_cpus, "affinity_cpus": affinity,
                "checked_upper_memory_bytes": memory, "checked_upper_cpu_fraction": [cpus.numerator, cpus.denominator],
                "capacity_basis": "visible_limits_and_provider_allocation",
                "hidden_ancestors_verified": False, "exclusive_capacity_guaranteed": False}
    return {"layout": "complete_host_unified_v2", "membership": str(group),
            "mount": str(root), "ancestors": observed, "host_memory_bytes": host_memory,
            "host_cpus": host_cpus, "affinity_cpus": affinity,
            "effective_memory_bytes": memory, "effective_cpu_fraction": [cpus.numerator, cpus.denominator],
            "not_exclusive_capacity_guarantee": True}


def require_capacity(receipt: dict[str, Any]) -> None:
    visible = receipt.get("capacity_contract") == RUNPOD_VISIBLE
    if (receipt["checked_upper_memory_bytes" if visible else "effective_memory_bytes"] < 64 * 1024**3
            or Fraction(*receipt["checked_upper_cpu_fraction" if visible else "effective_cpu_fraction"]) < 8):
        raise ValueError("host_allocation_too_small")


def observe_capacity(value: dict[str, Any] | None = None) -> dict[str, Any]:
    contract = COMPLETE_HOST if value is None else value["capacity_contract"]
    provider = None if value is None else provider_for_release(value)
    try:
        result = effective_capacity(Path("/proc/self/cgroup").read_text(),
            Path("/proc/self/mountinfo").read_text(),
            host_memory=os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES"),
            host_cpus=os.cpu_count() or 0, affinity=len(os.sched_getaffinity(0)),
            capacity_contract=contract, provider=provider)
    except OSError as exc:
        raise ValueError("unverified_cgroup_layout:proc_unreadable") from exc
    require_capacity(result)
    if provider is not None:
        check_provider_instance(result, provider["instance_id"])
    # Informational usage sample, never a stable identity or free-RAM guarantee.
    current = Path(result["mount"]).joinpath(*PurePosixPath(result["membership"]).parts[1:]) / "memory.current"
    sample = current.read_text().strip() if current.is_file() else None
    if sample is not None and (not sample.isdecimal() or int(sample) < 0):
        raise ValueError("invalid_memory_current")
    return {"configuration": result,
            "observations": {"observed_at_utc": datetime.now(timezone.utc).isoformat(),
                             "memory_current_bytes": None if sample is None else int(sample),
                             "work_probe_performed": False, "available_capacity_proven": False}}
