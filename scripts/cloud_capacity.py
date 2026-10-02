"""Fail-closed effective quotas from a complete unified cgroup v2 hierarchy."""
from __future__ import annotations

from fractions import Fraction
import os
from pathlib import Path, PurePosixPath
import re
from typing import Any


def _unescape(value: str) -> str:
    return re.sub(r"\\([0-7]{3})", lambda match: chr(int(match[1], 8)), value)


def effective_capacity(membership: str, mounts: str, *, host_memory: int,
                       host_cpus: int, affinity: int) -> dict[str, Any]:
    """Pure metadata/fixture entry. v1/hybrid or hidden ancestors are refused."""
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
    if memory <= 0 or cpus <= 0:
        raise ValueError("unverified_host_capacity")
    observed = []
    while True:
        try:
            # A genuine host hierarchy root has these controller files but no
            # memory.max/cpu.max. A namespaced child root exposes limit files:
            # reject rather than assume hidden ancestor limits are unbounded.
            if current == root:
                controllers = (root / "cgroup.controllers").read_text().split()
                if not {"memory", "cpu"} <= set(controllers):
                    raise ValueError("unverified_cgroup_layout:controllers")
                if (root / "memory.max").exists() or (root / "cpu.max").exists():
                    raise ValueError("unverified_cgroup_layout:namespaced_root")
                observed.append({"path": str(current), "host_hierarchy_root": True})
                break
            raw_memory = (current / "memory.max").read_text().strip()
            quota, period = (current / "cpu.max").read_text().split()
            if int(period) <= 0:
                raise ValueError("unverified_cgroup_layout:cpu_period")
            if raw_memory != "max":
                if not raw_memory.isdecimal() or int(raw_memory) <= 0:
                    raise ValueError("unverified_cgroup_layout:memory_limit")
                memory = min(memory, int(raw_memory))
            if quota != "max":
                if not quota.isdecimal() or int(quota) <= 0:
                    raise ValueError("unverified_cgroup_layout:cpu_quota")
                cpus = min(cpus, Fraction(int(quota), int(period)))
            observed.append({"path": str(current), "memory.max": raw_memory, "cpu.max": [quota, period]})
        except OSError as exc:
            raise ValueError("unverified_cgroup_layout:unreadable_ancestor") from exc
        current = current.parent
    return {"layout": "complete_host_unified_v2", "membership": str(group),
            "mount": str(root), "ancestors": observed, "host_memory_bytes": host_memory,
            "host_cpus": host_cpus, "affinity_cpus": affinity,
            "effective_memory_bytes": memory, "effective_cpu_fraction": [cpus.numerator, cpus.denominator],
            "not_exclusive_capacity_guarantee": True}


def require_capacity(receipt: dict[str, Any]) -> None:
    if (receipt["effective_memory_bytes"] < 64 * 1024**3
            or Fraction(*receipt["effective_cpu_fraction"]) < 8):
        raise ValueError("host_allocation_too_small")


def observe_capacity() -> dict[str, Any]:
    try:
        result = effective_capacity(Path("/proc/self/cgroup").read_text(),
            Path("/proc/self/mountinfo").read_text(),
            host_memory=os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES"),
            host_cpus=os.cpu_count() or 0, affinity=len(os.sched_getaffinity(0)))
    except OSError as exc:
        raise ValueError("unverified_cgroup_layout:proc_unreadable") from exc
    require_capacity(result)
    return result
