"""Synthetic private-storage delta; no production input/gold or model weights."""
import copy
import json
import os
from pathlib import Path
from types import SimpleNamespace

import pytest

import cloud_capacity as capacity
import cloud_destination as destination
import cloud_private_storage as private
import cloud_replay_contract as contract
import run_cloud_replay as entry
from test_cloud_replay import approved
from test_runpod_capacity import provider

POSIX = pytest.mark.skipif(os.name != "posix", reason="real POSIX permissions in Linux CI")


def draft():
    return contract.draft("a" * 40, "b" * 64, "c" * 64,
        root="/workspace/climate-replay", run_id="private-storage-fixture",
        capacity_contract=capacity.RUNPOD_VISIBLE, provider_allocation_sha="d" * 64,
        provider_evidence_sha="e" * 64, storage_contract=private.PRIVATE_POSIX,
        private_root="/root/climate-private")


def fixture_paths(tmp_path, monkeypatch):
    """Only the mount observation is synthetic; uid/mode/probe IO remain real."""
    value = draft()
    for key, item in value.items():
        if isinstance(item, str):
            value[key] = item.replace("/workspace/climate-replay", str(tmp_path / "persistent")).replace(
                "/root/climate-private", str(tmp_path / "private"))
    Path(value["root"]).mkdir()
    for path in private.required_parents(value):
        path.mkdir(mode=0o700)
    monkeypatch.setattr(private, "private_mount", lambda root, persistent: {
        "mount": "/", "filesystem": "synthetic-POSIX", "device": root.stat().st_dev,
        "ephemeral_on_container_stop": True})
    return value


def test_split_contract_all_sensitive_outputs_and_projection():
    value = draft()
    contract.validate_release(value, execution=False)
    assert value["input"].startswith("/workspace/")
    assert value["source"].startswith("/workspace/")
    for key in ("output", "gold_path", "private_scratch", "runtime_receipt",
                "asset_receipt", "deadline_receipt", "storage_receipt"):
        assert value[key].startswith("/root/climate-private/")
    with pytest.raises((ValueError, TypeError)):
        contract.validate_release(approved(value))
    value = {**approved(value), "storage_receipt_sha256": "f" * 64}
    contract.validate_release(value)
    projection = contract.worker_projection(value, "1" * 64)
    assert not any("gold" in key for key in projection)
    assert value["gold_path"] not in json.dumps(projection)
    for key in ("storage_contract", "private_root", "private_scratch", "storage_receipt_sha256"):
        assert projection[key] == value[key]
    legacy = contract.draft("a" * 40, "b" * 64, "c" * 64,
                            root="/workspace/climate-replay", run_id="legacy-fixture")
    for key in ("policy", "purpose", "generation_contract", "resource_cap",
                "max_worker_seconds", "max_operator_seconds", "gold_sha256"):
        assert value[key] == legacy[key]


@pytest.mark.parametrize("field", ["output", "gold_path", "private_root", "private_scratch", "deadline_receipt", "storage_contract"])
def test_private_mapping_mutation_refused(field):
    value = draft()
    value[field] = "/workspace/leaked" if field != "storage_contract" else private.SINGLE_ROOT
    with pytest.raises(ValueError):
        contract.validate_release(value, execution=False)


@POSIX
def test_real_probe_and_gold_modes(tmp_path, monkeypatch):
    value = fixture_paths(tmp_path, monkeypatch)
    result = private.observe_storage(value, probe=True)
    assert result["raw_recovery_required_before_stop"]
    assert not list(Path(value["private_root"]).rglob(".permission-probe-*"))
    gold = Path(value["gold_path"])
    gold.write_bytes(b"synthetic-not-labels")
    gold.chmod(0o600)
    private.observe_storage(value, require_gold=True)
    gold.chmod(0o644)
    with pytest.raises(ValueError, match="private_file_uid_mode"):
        private.observe_storage(value, require_gold=True)


@POSIX
@pytest.mark.parametrize("kind", ["0777", "uid", "symlink", "ancestor"])
def test_private_root_fails_closed(tmp_path, monkeypatch, kind):
    value = fixture_paths(tmp_path, monkeypatch)
    root = Path(value["private_root"])
    if kind == "0777":
        root.chmod(0o777)
    elif kind == "uid":
        monkeypatch.setattr(private.os, "getuid", lambda: root.stat().st_uid + 1)
    elif kind == "symlink":
        link = tmp_path / "linked-private"
        link.symlink_to(root, target_is_directory=True)
        value["private_root"] = str(link)
    else:
        tmp_path.chmod(0o777)
    with pytest.raises(ValueError):
        private.observe_storage(value, probe=True)


@POSIX
def test_successful_chmod_call_is_not_permission_evidence(tmp_path, monkeypatch):
    root = tmp_path / "private"
    root.mkdir(mode=0o700)
    real = os.chmod

    def mfs_noop(path, mode):
        real(path, 0o777)  # faithfully models the observed post-chmod result

    monkeypatch.setattr(private.os, "chmod", mfs_noop)
    with pytest.raises(ValueError, match="private_directory_uid_mode"):
        private.permission_probe(root)
    assert not list(root.iterdir())


@POSIX
def test_worker_fails_before_runtime_or_model_if_ledger_is_exposed(tmp_path, monkeypatch):
    value = fixture_paths(tmp_path, monkeypatch)
    configuration = private.observe_storage(value, probe=True)
    storage = Path(value["storage_receipt"])
    storage.write_text(json.dumps({"scope": "private_posix_preflight_not_execution_authority",
        "source_git": value["source_git"], "run_id": value["run_id"],
        "provider_allocation_sha256": value["provider_allocation_sha256"],
        "probe_passed": True, "configuration": configuration}))
    storage.chmod(0o600)
    value["storage_receipt_sha256"] = contract.digest(storage)
    output = Path(value["output"])
    output.mkdir(mode=0o700)
    (output / "allocation").mkdir(mode=0o700)
    (output / "inference").mkdir(mode=0o700)
    (output / "inference/ledger").mkdir(mode=0o777)
    (output / "inference/ledger").chmod(0o777)
    scratch = Path(value["private_scratch"])
    scratch.mkdir(mode=0o700)
    projected = {key: item for key, item in value.items() if "gold" not in key}
    projected["authorization"] = "validated_worker_projection"
    projection = scratch / "worker-projection.json"
    projection.write_text(json.dumps(projected))
    monkeypatch.setattr(entry, "validate_projection", lambda *args: None)  # covered separately by exact schema tests
    monkeypatch.setattr(entry, "verify_source", lambda *args: None)
    monkeypatch.setattr(entry, "verify_runtime", lambda *args: pytest.fail("runtime/model must not be reached"))
    with pytest.raises(ValueError, match="private_directory_uid_mode"):
        entry.worker_stage(projection, contract.digest(projection))


@POSIX
def test_output_loader_slots_and_ledgers_all_owner_only(tmp_path, monkeypatch):
    from climate_rag.local_targeted_provider import LocalTargetedProvider
    from climate_rag.scifact_read_continuation import ordered_write

    value = fixture_paths(tmp_path, monkeypatch)
    output = Path(value["output"])
    previous = os.umask(0o077)
    try:
        for folder in ("private-loader", "allocation", "reranker-ledger", "inference/ledger", "inference/slot-0"):
            (output / folder).mkdir(mode=0o700, parents=True, exist_ok=True)
        backend = object.__new__(LocalTargetedProvider)  # no model loader
        backend.start_slot(output / "inference/slot-0/private")
        backend.private_store.sink("response", 128).write("synthetic response")
        ordered_write(output / "inference/ledger/g0.json", {"observation": "synthetic"})
        ordered_write(output / "reranker-ledger/r0.json", {"pairs": "synthetic"})
        (output / "allocation/worker.log").write_text("synthetic")
        private.check_output_tree(output)
        (output / "inference/ledger/g0.json").chmod(0o644)
        with pytest.raises(ValueError, match="private_file_uid_mode"):
            private.check_output_tree(output)
    finally:
        os.umask(previous)


def test_container_capacity_does_not_borrow_workspace_free(tmp_path, monkeypatch):
    root = tmp_path / "private"
    root.mkdir()
    value = {"storage_contract": private.PRIVATE_POSIX, "private_root": str(root), "root": str(tmp_path)}
    monkeypatch.setattr(destination, "provider_for_release", lambda value: provider())
    monkeypatch.setattr(destination, "private_mount", lambda *args: {"device": root.stat().st_dev})
    monkeypatch.setattr(destination.shutil, "disk_usage", lambda path: SimpleNamespace(free=10**15))
    with pytest.raises(ValueError, match="private_container_capacity"):
        destination.container_budget(value, 31 * 10**9)
    good = destination.container_budget(value, 15 * destination.GIB)
    assert good["workspace_space_not_pooled"] is True
    assert good["service_quota_verified"] is False
    monkeypatch.setattr(destination.shutil, "disk_usage", lambda path: SimpleNamespace(free=1))
    with pytest.raises(ValueError, match="private_container_capacity"):
        destination.container_budget(value, 15 * destination.GIB)


@POSIX
def test_private_mount_rejects_same_device_and_fuse(tmp_path, monkeypatch):
    root, persistent = tmp_path / "root", tmp_path / "persistent"
    root.mkdir()
    persistent.mkdir()
    with pytest.raises(ValueError, match="not_separate"):
        private.private_mount(root, persistent)
    original = Path.stat

    def different_device(path, *args, **kwargs):
        info = original(path, *args, **kwargs)
        return SimpleNamespace(st_dev=info.st_dev + 1) if path == persistent else info

    monkeypatch.setattr(Path, "stat", different_device)
    monkeypatch.setattr(Path, "read_text", lambda *args, **kwargs: "1 0 0:1 / / rw - fuse.mfs mfs rw")
    with pytest.raises(ValueError, match="posix_mount"):
        private.private_mount(root, persistent)


def test_storage_receipt_hash_is_not_plan_drift_but_private_root_is():
    value = draft()
    same = copy.deepcopy(value)
    same["storage_receipt_sha256"] = "f" * 64
    assert destination.plan_sha(value) == destination.plan_sha(same)
    same["private_root"] = "/root/other-private"
    assert destination.plan_sha(value) != destination.plan_sha(same)


@POSIX
def test_asset_permission_gate_precedes_large_inventory(tmp_path, monkeypatch):
    value = fixture_paths(tmp_path, monkeypatch)
    Path(value["private_root"]).chmod(0o777)
    monkeypatch.setattr(destination, "validate_release", lambda *args, **kwargs: None)
    monkeypatch.setattr(destination, "verify_source", lambda *args: None)
    monkeypatch.setattr(destination, "inventory", lambda *args, **kwargs: pytest.fail("large inventory must not run"))
    with pytest.raises(ValueError, match="private_directory_uid_mode"):
        destination.destination_preflight(value, Path(value["source"]))


@POSIX
def test_tempfile_cached_path_is_reset_and_all_caches_are_private(tmp_path, monkeypatch):
    import tempfile

    value = fixture_paths(tmp_path, monkeypatch)
    scratch = Path(value["private_scratch"])
    scratch.mkdir(mode=0o700)
    expected = private.private_environment(value)
    for key in expected:
        monkeypatch.setenv(key, "/tmp/old-cached-path")
    monkeypatch.setattr(tempfile, "tempdir", "/tmp/old-cached-path")
    previous = os.umask(0o077)
    try:
        private.configure_private_environment(value)
        assert tempfile.gettempdir() == str(scratch)
        assert all(Path(item).is_relative_to(scratch) for item in expected.values())
    finally:
        os.umask(previous)


def test_disk_reservations_split_into_two_nonpooled_budgets(tmp_path, monkeypatch):
    workspace, sensitive = tmp_path / "workspace", tmp_path / "private"
    workspace.mkdir()
    sensitive.mkdir()
    value = {"capacity_contract": capacity.RUNPOD_VISIBLE, "storage_contract": private.PRIVATE_POSIX,
             "private_root": str(sensitive)}
    for key in ("input_archive", "source", "python_executable", "work"):
        value[key] = str(workspace / key)
    for key in ("gold_path", "output", "asset_receipt", "private_scratch"):
        value[key] = str(sensitive / key)
    monkeypatch.setattr(destination, "RUNPOD_WORKSPACE", workspace)
    original = Path.stat

    def devices(path, *args, **kwargs):
        info = original(path, *args, **kwargs)
        return SimpleNamespace(st_dev=2 if path.is_relative_to(sensitive) else 1, st_mode=info.st_mode)

    monkeypatch.setattr(Path, "stat", devices)
    monkeypatch.setattr(destination.shutil, "disk_usage", lambda path: SimpleNamespace(free=100 * destination.GIB))
    seen = {}
    monkeypatch.setattr(destination, "provider_volume_budget", lambda value, n: seen.setdefault("workspace", n))
    monkeypatch.setattr(destination, "container_budget", lambda value, n: seen.setdefault("container", n))
    result = destination.disk_capacity(value, 16 * destination.GIB)
    assert len(result) == 2
    assert seen == {"workspace": 24 * destination.GIB, "container": 15 * destination.GIB}
    monkeypatch.setattr(destination.shutil, "disk_usage", lambda path: SimpleNamespace(
        free=destination.GIB if path.is_relative_to(sensitive) else 1000 * destination.GIB))
    with pytest.raises(ValueError, match="filesystem_capacity"):
        destination.disk_capacity(value, 16 * destination.GIB)
