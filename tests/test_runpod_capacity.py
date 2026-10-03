"""CPU-only Runpod namespace fixtures; no cloud access or resource guarantees."""
import copy
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

import cloud_capacity as capacity
import cloud_destination as destination
import cloud_replay_contract as contract
import run_cloud_replay as entry
from test_cloud_replay import approved


def provider():
    return {"schema": "coordinator-runpod-allocation-v1", "provider": "Runpod",
            "instance_id": "syntheticpod", "source_evidence_sha256": "a" * 64,
            "allocation": {"vcpu": 16, "memory_gb": 250, "workspace_volume_gb": 120,
                           "container_gb": 30, "image": "synthetic-image"}}


def visible(tmp_path, *, memory="249999998976", cpu="1360000 100000"):
    root = tmp_path / "cgroup"
    root.mkdir()
    (root / "memory.max").write_text(memory)
    (root / "cpu.max").write_text(cpu)
    (root / "memory.current").write_text("36962304")
    (root / "cgroup.controllers").write_text("memory cpu")
    mounts = f"1 2 0:1 / {root} rw - cgroup2 cgroup rw"
    return root, mounts


def test_observed_shape_13_6_cpu_passes_only_explicit_visible_contract(tmp_path):
    root, mounts = visible(tmp_path)
    args = {"host_memory": 2101188784 * 1024, "host_cpus": 128, "affinity": 128}
    with pytest.raises(ValueError, match="namespaced_root"):
        capacity.effective_capacity("0::/", mounts, **args)
    result = capacity.effective_capacity("0::/", mounts, **args,
                                        capacity_contract=capacity.RUNPOD_VISIBLE, provider=provider())
    capacity.require_capacity(result)
    assert result["checked_upper_cpu_fraction"] == [68, 5]
    assert result["checked_upper_memory_bytes"] == 249999998976
    assert result["hidden_ancestors_verified"] is False
    assert result["exclusive_capacity_guaranteed"] is False
    assert "effective_memory_bytes" not in result


@pytest.mark.parametrize("case", ["no_provider", "max_memory", "max_cpu", "bad_period", "v1", "missing", "fraction", "ui_memory", "ui_cpu", "affinity", "physical"])
def test_visible_limits_still_fail_closed(tmp_path, case):
    root, mounts = visible(tmp_path)
    metadata = provider()
    membership = "0::/"
    if case == "max_memory":
        (root / "memory.max").write_text("max")
    elif case == "max_cpu":
        (root / "cpu.max").write_text("max 100000")
    elif case == "bad_period":
        (root / "cpu.max").write_text("100000 0")
    elif case == "v1":
        membership = "2:cpu:/"
    elif case == "missing":
        (root / "memory.max").unlink()
    elif case == "fraction":
        (root / "cpu.max").write_text("799000 100000")
    elif case == "ui_memory":
        metadata["allocation"]["memory_gb"] = 64  # GB is not GiB.
    elif case == "ui_cpu":
        metadata["allocation"]["vcpu"] = 7
    with pytest.raises(ValueError):
        result = capacity.effective_capacity(membership, mounts, host_memory=(63 if case == "physical" else 512) * 1024**3,
            host_cpus=128, affinity=7 if case == "affinity" else 128,
            capacity_contract=capacity.RUNPOD_VISIBLE, provider=None if case == "no_provider" else metadata)
        capacity.require_capacity(result)


def test_visible_root_and_child_both_constrain(tmp_path):
    root, mounts = visible(tmp_path)
    child = root / "nested"
    child.mkdir()
    (child / "memory.max").write_text("max")
    (child / "cpu.max").write_text("800000 100000")
    result = capacity.effective_capacity("0::/nested", mounts, host_memory=512 * 1024**3, host_cpus=128,
                                        affinity=128, capacity_contract=capacity.RUNPOD_VISIBLE, provider=provider())
    capacity.require_capacity(result)
    assert result["checked_upper_cpu_fraction"] == [8, 1]
    assert len(result["visible_ancestors"]) == 2


def test_provider_projection_and_both_hash_bindings(tmp_path):
    original = {"provider": "Runpod", "instance_id": "syntheticpod", "control_plane": provider()["allocation"],
                "pricing": {"unneeded": True}, "ssh": {"unneeded": True}, "kernel_observation": {"memory_current_bytes": 1}}
    evidence = tmp_path / "evidence.json"
    evidence.write_text(json.dumps(original))
    fingerprint = contract.digest(evidence)
    safe = capacity.provider_from_evidence(evidence, fingerprint)
    assert set(safe) == {"schema", "provider", "instance_id", "source_evidence_sha256", "allocation"}
    receipt = tmp_path / "allocation.json"
    receipt.write_text(json.dumps(safe))
    value = {"capacity_contract": capacity.RUNPOD_VISIBLE, "provider_allocation_receipt": str(receipt),
             "provider_allocation_sha256": contract.digest(receipt), "provider_evidence_sha256": fingerprint}
    assert capacity.provider_for_release(value) == safe
    for key in ("provider_allocation_sha256", "provider_evidence_sha256"):
        changed = {**value, key: "f" * 64}
        with pytest.raises(ValueError):
            capacity.provider_for_release(changed)
    with pytest.raises(ValueError, match="source_evidence_hash"):
        capacity.provider_from_evidence(evidence, "f" * 64)


def test_draft_and_worker_projection_bind_named_contract():
    value = contract.draft("a" * 40, "b" * 64, "c" * 64, root="/workspace/climate-replay", run_id="namespace-fixture",
                           capacity_contract=capacity.RUNPOD_VISIBLE, provider_allocation_sha="d" * 64,
                           provider_evidence_sha="e" * 64)
    contract.validate_release(value, execution=False)
    projected = contract.worker_projection(approved(value), "f" * 64)
    assert projected["capacity_contract"] == capacity.RUNPOD_VISIBLE
    assert projected["provider_allocation_sha256"] == "d" * 64
    assert not any("gold" in key for key in projected)
    changed = {**value, "capacity_contract": capacity.COMPLETE_HOST}
    with pytest.raises(ValueError):
        contract.validate_release(changed, execution=False)


def test_pod_identity_must_match_coordinator_metadata(monkeypatch):
    configuration = {"provider_allocation": provider()}
    monkeypatch.delenv("RUNPOD_POD_ID", raising=False)
    with pytest.raises(ValueError, match="instance_identity"):
        capacity.check_provider_instance(configuration, "syntheticpod")
    monkeypatch.setenv("RUNPOD_POD_ID", "syntheticpod")
    capacity.check_provider_instance(configuration, "syntheticpod")
    with pytest.raises(ValueError, match="instance_identity"):
        capacity.check_provider_instance(configuration, "anotherpod")
    monkeypatch.setenv("RUNPOD_POD_ID", "anotherpod")
    with pytest.raises(ValueError, match="instance_identity"):
        capacity.check_provider_instance(configuration, "syntheticpod")


@pytest.mark.parametrize("change", ["usage", "quota", "allocation", "gpu"])
def test_dynamic_samples_do_not_mask_stable_identity_drift(tmp_path, monkeypatch, change):
    saved = {"runtime_identity_schema": "stable-configuration-v2", "backend": contract.BACKEND,
             "status": "observed", "synthetic": False, "instance_id": "syntheticpod",
             "capacity_configuration": {"cpu": [68, 5]}, "provider_allocation_sha256": "a" * 64,
             "gpu": "synthetic-GPU", "observations": {"memory.current": 1, "observed_at": "before"}}
    path = tmp_path / "runtime.json"
    path.write_text(json.dumps(saved))
    fresh = copy.deepcopy(saved)
    fresh["observations"] = {"memory.current": 10**9, "observed_at": "after"}
    if change == "quota":
        fresh["capacity_configuration"]["cpu"] = [12, 1]
    elif change == "allocation":
        fresh["provider_allocation_sha256"] = "b" * 64
    elif change == "gpu":
        fresh["gpu"] = "different-GPU"
    called = []
    monkeypatch.setattr(entry, "observe_runtime", lambda *args: (called.append(True), fresh)[1])
    value = {"runtime_receipt": str(path), "runtime_receipt_sha256": contract.digest(path)}
    if change == "usage":
        assert entry.verify_runtime(value) == fresh
    else:
        with pytest.raises(ValueError, match="runtime_changed"):
            entry.verify_runtime(value)
    assert called == [True]


def test_real_capacity_observer_separates_memory_current(tmp_path, monkeypatch):
    root, mounts = visible(tmp_path)
    original = Path.read_text

    def read(path, *args, **kwargs):
        if path == Path("/proc/self/cgroup"):
            return "0::/"
        if path == Path("/proc/self/mountinfo"):
            return mounts
        return original(path, *args, **kwargs)

    monkeypatch.setattr(Path, "read_text", read)
    monkeypatch.setattr(capacity.os, "sysconf", lambda name: 4096 if name == "SC_PAGE_SIZE" else 512 * 1024**3 // 4096, raising=False)
    monkeypatch.setattr(capacity.os, "cpu_count", lambda: 128)
    monkeypatch.setattr(capacity.os, "sched_getaffinity", lambda pid: set(range(128)), raising=False)
    monkeypatch.setattr(capacity, "provider_for_release", lambda value: provider())
    monkeypatch.setenv("RUNPOD_POD_ID", "syntheticpod")
    value = {"capacity_contract": capacity.RUNPOD_VISIBLE}
    before = capacity.observe_capacity(value)
    (root / "memory.current").write_text("99999999")
    after = capacity.observe_capacity(value)
    assert before["configuration"] == after["configuration"]
    assert before["observations"]["memory_current_bytes"] == 36962304
    assert after["observations"]["memory_current_bytes"] == 99999999
    assert after["observations"]["available_capacity_proven"] is False
    (root / "memory.current").write_text("invalid")
    with pytest.raises(ValueError, match="memory_current"):
        capacity.observe_capacity(value)


def test_shared_df_cannot_override_provider_volume_budget(tmp_path, monkeypatch):
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    outside_project = workspace / "other-pod-volume-files"
    outside_project.write_bytes(b"synthetic existing volume file")
    monkeypatch.setattr(destination, "RUNPOD_WORKSPACE", workspace)
    monkeypatch.setattr(destination, "workspace_mount", lambda: "mfs")
    monkeypatch.setattr(destination, "provider_for_release", lambda value: provider())
    monkeypatch.setattr(destination.shutil, "disk_usage", lambda path: SimpleNamespace(free=10**15))
    value = {"provider_allocation_sha256": "a" * 64}
    result = destination.provider_volume_budget(value, 10**9)
    assert result["observed_volume_used_bytes"] >= outside_project.stat().st_size
    assert result["provider_workspace_allocated_bytes"] == 120_000_000_000
    assert result["provider_container_allocated_bytes"] == 30_000_000_000
    assert result["quota_independently_verified"] is False
    with pytest.raises(ValueError, match="workspace_allocation_budget"):
        destination.provider_volume_budget(value, 120_000_000_000)
