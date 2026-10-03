"""Small synthetic prelaunch regressions; no production archives or weights."""
import json
import os
from pathlib import Path
import py_compile
import shutil
import subprocess
import sys
import time
from types import SimpleNamespace

import pytest

import cloud_capacity as capacity
import cloud_destination as destination
import cloud_replay_contract as contract
import preflight_cloud_assets as assets
import run_cloud_replay as entry
from cloud_deadline import supervise, write_receipt
from test_cloud_replay import approved, synthetic_tar


def hierarchy(tmp_path):
    root = tmp_path / "cg"
    leaf = root / "parent/child"
    leaf.mkdir(parents=True)
    (root / "cgroup.controllers").write_text("memory cpu")
    for path in (leaf, leaf.parent):
        (path / "memory.max").write_text("max")
        (path / "cpu.max").write_text("max 100000")
    mounts = f"1 2 0:1 / {root} rw - cgroup2 cgroup rw"
    return root, leaf, mounts


def observe(mounts):
    return capacity.effective_capacity("0::/parent/child", mounts,
                                      host_memory=512 * 1024**3, host_cpus=64, affinity=32)


def test_effective_ancestor_limits_and_exact_boundary(tmp_path):
    root, leaf, mounts = hierarchy(tmp_path)
    (leaf.parent / "memory.max").write_text(str(64 * 1024**3))
    (leaf / "cpu.max").write_text("800000 100000")
    result = observe(mounts)
    capacity.require_capacity(result)
    assert result["effective_memory_bytes"] == 64 * 1024**3
    assert result["effective_cpu_fraction"] == [8, 1]
    assert len(result["ancestors"]) == 3
    (leaf.parent / "cpu.max").write_text("799000 100000")
    with pytest.raises(ValueError, match="too_small"):
        capacity.require_capacity(observe(mounts))


@pytest.mark.parametrize("case", ["missing", "period", "memory", "namespace", "hidden", "v1", "affinity", "ram"])
def test_unknown_or_insufficient_capacity_never_unlimited(tmp_path, case):
    root, leaf, mounts = hierarchy(tmp_path)
    if case == "missing":
        (leaf.parent / "memory.max").unlink()
    elif case == "period":
        (leaf / "cpu.max").write_text("max 0")
    elif case == "memory":
        (leaf / "memory.max").write_text("invalid")
    elif case == "namespace":
        (root / "cpu.max").write_text("max 100000")
    elif case == "hidden":
        mounts = mounts.replace("0:1 / ", "0:1 /container ")
    elif case == "ram":
        (leaf.parent / "memory.max").write_text(str(63 * 1024**3))
    with pytest.raises(ValueError):
        result = capacity.effective_capacity("2:cpu:/parent/child" if case == "v1" else "0::/parent/child",
                                            mounts, host_memory=512 * 1024**3,
                                            host_cpus=64, affinity=7 if case == "affinity" else 32)
        capacity.require_capacity(result)


@pytest.mark.parametrize("extension", [".pyc", ".so", ".pyd"])
def test_isolated_bootstrap_rejects_alternate_bytes_before_import(tmp_path, extension):
    scripts = tmp_path / "scripts"
    scripts.mkdir()
    (tmp_path / "src").mkdir()
    copied = scripts / "run_cloud_replay.py"
    shutil.copyfile(entry.__file__, copied)
    marker = tmp_path / "executed"
    if extension == ".pyc":
        source = tmp_path / "shadow.py"
        source.write_text(f"open({str(marker)!r}, 'w').write('BAD')", encoding="utf-8")
        py_compile.compile(str(source), cfile=str(scripts / "argparse.pyc"), doraise=True)
    else:
        (scripts / ("argparse" + extension)).write_bytes(b"not a native extension")
    result = subprocess.run([sys.executable, "-X", "utf8", "-IB", str(copied), "--help"], capture_output=True, text=True, encoding="utf-8")
    assert result.returncode != 0 and "unbound_project_bytecode_or_extension" in result.stderr
    assert not marker.exists()
    assert "No module named" not in result.stderr


@pytest.mark.skipif(os.name != "posix", reason="Linux symlink contract")
def test_symlink_package_rejected_before_import(tmp_path):
    (tmp_path / "scripts").mkdir()
    (tmp_path / "src").mkdir()
    outside = tmp_path / "outside"
    outside.mkdir()
    (tmp_path / "src/climate_rag").symlink_to(outside, target_is_directory=True)
    copied = tmp_path / "scripts/run_cloud_replay.py"
    shutil.copyfile(entry.__file__, copied)
    result = subprocess.run([sys.executable, "-IB", str(copied), "--help"], capture_output=True, text=True)
    assert result.returncode != 0 and "project_symlink" in result.stderr


def test_nonisolated_entry_refuses_before_project_imports(tmp_path):
    copied = tmp_path / "entry.py"
    shutil.copyfile(entry.__file__, copied)
    result = subprocess.run([sys.executable, "-B", str(copied)], capture_output=True, text=True)
    assert result.returncode != 0 and "isolated_entry_required" in result.stderr


def test_same_filesystem_reservations_sum_and_existing_transport_not_double_counted(tmp_path, monkeypatch):
    value = {key: str(tmp_path / key) for key in ("input_archive", "gold_path", "source", "python_executable", "work", "output", "asset_receipt")}
    monkeypatch.setattr(destination.shutil, "disk_usage", lambda path: SimpleNamespace(free=18 * destination.GIB))
    result = destination.disk_capacity(value, destination.GIB)
    assert len(result) == 1 and result[0]["additional_required_bytes"] == 18 * destination.GIB
    monkeypatch.setattr(destination.shutil, "disk_usage", lambda path: SimpleNamespace(free=18 * destination.GIB - 1))
    with pytest.raises(ValueError, match="filesystem_capacity"):
        destination.disk_capacity(value, destination.GIB)


@pytest.mark.skipif(os.name != "posix", reason="Linux destination path contract")
@pytest.mark.parametrize("case", ["success", "wrong_gold", "damaged_input", "missing_input", "reused", "stale_plan", "space_dropped"])
def test_destination_exact_assets_and_receipt(tmp_path, monkeypatch, case):
    value = contract.draft("a" * 40, "b" * 64, "c" * 64, root=str(tmp_path), run_id="destination-fixture")
    for key in ("work", "output", "asset_receipt", "input_archive", "gold_path", "python_executable"):
        Path(value[key]).parent.mkdir(parents=True, exist_ok=True)
    Path(value["python_executable"]).write_bytes(b"synthetic interpreter")
    source = Path(value["source"])
    (source / contract.SELECTION).parent.mkdir(parents=True)
    (source / contract.SELECTION).write_bytes(b"synthetic selection")
    gold = Path(value["gold_path"])
    gold.write_bytes(b"hash-only-not-json")
    monkeypatch.setattr(assets, "GOLD_SHA", contract.digest(gold))
    monkeypatch.setattr(assets, "SELECTION_SHA", contract.digest(source / contract.SELECTION))
    monkeypatch.setattr(destination, "verify_source", lambda *args: None)
    monkeypatch.setattr(destination, "observe_capacity", lambda *args: {"synthetic": True})
    tar = synthetic_tar(tmp_path, monkeypatch)
    tar.rename(value["input_archive"])
    value["input_transport_sha256"] = contract.digest(Path(value["input_archive"]))
    # Keep the production release schema, swapping only synthetic identities.
    monkeypatch.setattr(contract, "GOLD_SHA", contract.digest(gold))
    value["gold_sha256"] = contract.digest(gold)
    monkeypatch.setattr(destination.shutil, "disk_usage", lambda path: SimpleNamespace(free=100 * destination.GIB))
    if case == "wrong_gold":
        gold.write_bytes(b"changed")
    elif case == "damaged_input":
        with Path(value["input_archive"]).open("ab") as stream:
            stream.write(b"changed")
    elif case == "missing_input":
        Path(value["input_archive"]).unlink()
    elif case == "reused":
        Path(value["output"]).mkdir()
    if case in ("wrong_gold", "damaged_input", "missing_input", "reused"):
        with pytest.raises(ValueError):
            destination.destination_preflight(value, source)
        assert not Path(value["work"]).exists()
        return
    receipt = destination.destination_preflight(value, source)
    assert receipt["model_execution_authorized"] is False and receipt["gold_labels_parsed"] is False
    assert not Path(value["work"]).exists() and not Path(value["output"]).exists()
    Path(value["asset_receipt"]).write_text(json.dumps(receipt))
    authorized = approved(value)
    authorized["asset_receipt_sha256"] = contract.digest(Path(value["asset_receipt"]))
    if case == "stale_plan":
        authorized["source_git"] = "f" * 40
    elif case == "space_dropped":
        monkeypatch.setattr(destination.shutil, "disk_usage", lambda path: SimpleNamespace(free=0))
    if case in ("stale_plan", "space_dropped"):
        with pytest.raises(ValueError):
            destination.verify_destination(authorized)
    else:
        destination.verify_destination(authorized)


@pytest.mark.skipif(not sys.platform.startswith("linux"), reason="independent Linux fork/subreaper deadline")
@pytest.mark.parametrize("case", ["success", "slow_ledger", "ignore_term_final", "new_session", "slow_receipt", "slow_fsync", "late_success", "receipt_interrupt"])
def test_absolute_deadline_covers_closeout_and_reaps(tmp_path, case):
    import signal

    raw = tmp_path / "physical-cost.jsonl"
    child_record = tmp_path / "child.pid"

    def operation():
        raw.write_text('{"synthetic_cost":1}\n')
        if case in ("ignore_term_final", "new_session"):
            signal.signal(signal.SIGTERM, signal.SIG_IGN)
        if case == "late_success":
            def delayed_success(signum, frame):
                os._exit(0)
            signal.signal(signal.SIGTERM, delayed_success)
        if case == "new_session":
            child = os.fork()
            if child == 0:
                os.setsid()
                child_record.write_text(str(os.getpid()))
                time.sleep(30)
                os._exit(0)
        if case not in ("success", "slow_receipt", "slow_fsync", "receipt_interrupt"):
            time.sleep(30)
        return {"status": "completed"}

    def slow_writer(path, result):
        time.sleep(30)

    def fsync_writer(path, result):
        os.fsync = lambda fd: time.sleep(30)
        write_receipt(path, result)

    def interrupt_writer(path, result):
        os.kill(os.getppid(), signal.SIGTERM)
        write_receipt(path, result)

    kwargs = ({"receipt_writer": slow_writer} if case == "slow_receipt" else
              {"receipt_writer": fsync_writer} if case == "slow_fsync" else
              {"receipt_writer": interrupt_writer} if case == "receipt_interrupt" else {})
    begin = time.monotonic()
    result = supervise(operation, tmp_path / "deadline.json", total_seconds=2,
                       cleanup_seconds=1, term_seconds=0.1, receipt_seconds=0.2, **kwargs)
    assert time.monotonic() - begin < 2.5
    assert raw.read_text() == '{"synthetic_cost":1}\n'
    assert result["all_children_reaped"]
    assert result["status"] == ("completed" if case == "success" else
                                "failed_deadline_receipt_or_reaping" if case in ("slow_receipt", "slow_fsync") else
                                "interrupted_unscored" if case == "receipt_interrupt" else "timeout_unscored")
    if child_record.exists():
        assert not Path("/proc/" + child_record.read_text()).exists()
    if case not in ("slow_receipt", "slow_fsync"):
        receipt = json.loads((tmp_path / "deadline.json").read_text())
        if case != "receipt_interrupt":
            assert receipt["status"] == result["status"]
        else:
            assert result["interrupts"] and result["status"] != "completed"
    else:
        assert not (tmp_path / "deadline.json").exists()


@pytest.mark.skipif(not sys.platform.startswith("linux"), reason="actual shared operator finalization")
def test_real_operator_final_ledger_is_inside_independent_deadline(tmp_path, monkeypatch):
    import run_targeted_replay_operator as operator

    value = approved(contract.draft("a" * 40, "b" * 64, "c" * 64,
                                    root=str(tmp_path), run_id="final-ledger-test"))
    Path(value["output"]).parent.mkdir()
    Path(value["work"]).parent.mkdir()
    raw = tmp_path / "retained-cost"

    def prep(*args):
        raw.write_text("physical synthetic cost")
        raise ValueError("force real final ledger closeout")

    def slow_ledger(*args):
        time.sleep(30)

    monkeypatch.setattr(operator, "ledger_cost", slow_ledger)
    result = supervise(lambda: operator.run_supervised(
        value, tmp_path / "release", "a" * 64, tmp_path, Path(value["work"]),
        validate_fn=contract.validate_release, prepare_fn=prep),
        tmp_path / "deadline.json", total_seconds=2, cleanup_seconds=1,
        term_seconds=0.1, receipt_seconds=0.2)
    assert result["status"] == "timeout_unscored" and result["all_children_reaped"]
    assert raw.read_text() == "physical synthetic cost"
    assert not (Path(value["output"]) / "compact.json").exists()


def test_output_filesystem_cannot_borrow_free_space_from_input(tmp_path, monkeypatch):
    root = tmp_path / "main"
    other = tmp_path / "separate-output"
    root.mkdir()
    other.mkdir()
    value = {key: str(root / key) for key in ("input_archive", "gold_path", "source", "python_executable", "work", "output", "asset_receipt")}
    value["output"] = str(other / "new-run")
    original_stat = Path.stat

    def stat(path, **kwargs):
        fields = list(original_stat(path, **kwargs))
        fields[2] = 502 if path.is_relative_to(other) else 501
        return os.stat_result(fields)

    monkeypatch.setattr(Path, "stat", stat)
    monkeypatch.setattr(destination.shutil, "disk_usage", lambda path: SimpleNamespace(
        free=(7 if path.is_relative_to(other) else 100) * destination.GIB))
    with pytest.raises(ValueError, match="filesystem_capacity"):
        destination.disk_capacity(value, destination.GIB)


@pytest.mark.skipif(not sys.platform.startswith("linux"), reason="real standalone bootstrap/fork")
def test_cli_watchdog_starts_before_project_imports(tmp_path):
    # A synthetic copy with short constants and deliberately blocked project
    # import. Production has no CLI/environment override for its7200s ceiling.
    scripts = tmp_path / "source/scripts"
    scripts.mkdir(parents=True)
    (tmp_path / "source/src").mkdir()
    (tmp_path / "runtime").mkdir()
    copied = scripts / "run_cloud_replay.py"
    code = Path(entry.__file__).read_text(encoding="utf-8")
    code = code.replace("from climate_rag.scifact_read_continuation import ordered_write",
                        "time.sleep(30)\nfrom climate_rag.scifact_read_continuation import ordered_write")
    copied.write_text(code, encoding="utf-8")
    watchdog = Path(entry.__file__).with_name("cloud_deadline.py").read_text(encoding="utf-8")
    for old, new in (("total_seconds: float = 7200", "total_seconds: float = 2"),
                     ("cleanup_seconds: float = 20", "cleanup_seconds: float = 1"),
                     ("term_seconds: float = 3", "term_seconds: float = 0.1"),
                     ("receipt_seconds: float = 3", "receipt_seconds: float = 0.2")):
        watchdog = watchdog.replace(old, new)
    (scripts / "cloud_deadline.py").write_text(watchdog, encoding="utf-8")
    value = approved(contract.draft("a" * 40, "b" * 64, "c" * 64,
                                    root=str(tmp_path), run_id="bootstrap-fixture"))
    release = tmp_path / "release.json"
    release.write_text(json.dumps(value))
    result = subprocess.run([sys.executable, "-IB", str(copied), "run", "--release", str(release),
                             "--release-sha", contract.digest(release)], capture_output=True, timeout=5)
    assert result.returncode != 0
    receipt = json.loads(Path(value["deadline_receipt"]).read_text())
    assert receipt["status"] == "timeout_unscored" and receipt["all_children_reaped"]
    assert receipt["identity"]["release_sha256"] == contract.digest(release)
