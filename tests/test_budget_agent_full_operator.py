"""Tiny synthetic operator tests. No real archive expansion, gold or model use."""
import importlib.util
import io
import json
import tarfile
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("full_operator", ROOT / "scripts/run_budget_agent_full_operator.py")
operator = importlib.util.module_from_spec(spec)
spec.loader.exec_module(operator)


def archive(path, names, link=False):
    with tarfile.open(path, "w") as bundle:
        for name in names:
            member = tarfile.TarInfo(name)
            if link:
                member.type = tarfile.SYMTYPE
                member.linkname = "/etc/passwd"
                bundle.addfile(member)
            else:
                member.size = 4
                bundle.addfile(member, io.BytesIO(b"test"))


def state():
    return {"status": "preparing", "phases": {p: {"status": "not_started"} for p in operator.PHASES}}


def test_serial_phases_share_one_readonly_input(tmp_path):
    source = tmp_path / "tiny.tar"
    archive(source, ["models/tiny", "evidence.jsonl"])
    assert operator.safe_extract(source, tmp_path / "input") == 8
    operator.read_only_tree(tmp_path / "input")
    inputs = tmp_path / "input"
    files = {p.name: operator.digest(p) for p in inputs.rglob("*") if p.is_file()}
    calls = []
    args = {p: ["--phase", p, "--evidence", str(inputs / "evidence.jsonl"),
                "--model-dir", str(inputs / "models")] for p in operator.PHASES}
    commands = operator.phase_commands(tmp_path, tmp_path / "result", tmp_path / "gold.json", args)
    receipt = state()
    operator.run_phases(receipt, tmp_path / "state.json", lambda p: calls.append((p, "infer")),
                        lambda p: calls.append((p, "score")), lambda p: calls.append((p, "validate")))
    assert calls == [(p, action) for p in operator.PHASES for action in ("infer", "score", "validate")]
    assert receipt["status"] == "complete"
    for phase in operator.PHASES:
        cmd = commands[phase]["infer"]
        assert cmd[cmd.index("--evidence") + 1] == str(inputs / "evidence.jsonl")
        assert cmd[cmd.index("--model-dir") + 1] == str(inputs / "models")
        assert "--gold" not in cmd
        assert cmd[cmd.index("--output") + 1] == str(tmp_path / "result" / phase / "run.json")
    assert "--gold" in commands["validation"]["score"]
    assert "--gold" not in commands["vnext"]["score"]
    assert files == {p.name: operator.digest(p) for p in inputs.rglob("*") if p.is_file()}


@pytest.mark.parametrize("point", ["inference", "score", "matrix", "signal"])
def test_failure_does_not_start_second_phase_or_mark_complete(tmp_path, point):
    receipt, calls = state(), []

    def stage(phase, at):
        calls.append((phase, at))
        if point == at:
            raise ValueError("synthetic incomplete")
        if point == "signal" and at == "inference":
            raise InterruptedError("synthetic TERM")

    with pytest.raises((ValueError, InterruptedError)):
        operator.run_phases(receipt, tmp_path / "state.json", lambda p: stage(p, "inference"),
                            lambda p: stage(p, "score"), lambda p: stage(p, "matrix"))
    saved = json.loads((tmp_path / "state.json").read_text())
    assert saved["status"] != "complete"
    assert saved["phases"]["vnext"]["status"] == "not_started"
    assert all(p == "validation" for p, _ in calls)


def test_second_phase_failure_keeps_first_completed_receipt(tmp_path):
    receipt = state()

    def infer(phase):
        if phase == "vnext":
            raise ValueError("synthetic failure")

    with pytest.raises(ValueError):
        operator.run_phases(receipt, tmp_path / "state.json", infer, lambda p: None, lambda p: None)
    saved = json.loads((tmp_path / "state.json").read_text())
    assert saved["phases"]["validation"]["status"] == "complete"
    assert saved["phases"]["vnext"]["status"] != "complete"
    assert saved["status"] != "complete"


@pytest.mark.parametrize("names,link", [(["../outside"], False), (["same", "same"], False), (["link"], True)])
def test_unsafe_archive_rejected_before_extraction(tmp_path, names, link):
    source = tmp_path / "bad.tar"
    archive(source, names, link)
    with pytest.raises(ValueError):
        operator.safe_extract(source, tmp_path / "output")
    assert not (tmp_path / "output").exists()


def test_existing_output_and_low_scratch_refused(tmp_path, monkeypatch):
    source = tmp_path / "tiny.tar"
    archive(source, ["small"])
    destination = tmp_path / "output"
    destination.mkdir()
    with pytest.raises(ValueError, match="destination_exists"):
        operator.safe_extract(source, destination)
    from types import SimpleNamespace
    monkeypatch.setattr(operator.shutil, "disk_usage", lambda p: SimpleNamespace(free=1))
    with pytest.raises(ValueError, match="scratch_capacity"):
        operator.safe_extract(source, tmp_path / "other")
    assert not (tmp_path / "other").exists()


@pytest.mark.parametrize("prefix", ["", "./"])
def test_runtime_selects_normalized_site_packages_only(tmp_path, prefix):
    source = tmp_path / "runtime.tar"
    archive(source, [prefix + "lib/python3.10/site-packages/package.py", prefix + "bin/python"])
    target = tmp_path / "runtime"
    assert operator.safe_extract(source, target, runtime=True) == 4
    assert (target / "lib/python3.10/site-packages/package.py").read_bytes() == b"test"
    assert not (target / "bin").exists()


@pytest.mark.parametrize("names,link,code", [
    (["./lib/python3.10/site-packages/../../outside"], False, "archive_traversal"),
    (["./lib/python3.10/site-packages/link"], True, "archive_link_or_special"),
    (["./lib/python3.10/site-packages/same", "lib/python3.10/site-packages/same"], False, "archive_duplicate"),
    (["./bin/python"], False, "runtime_site_packages_absent"),
])
def test_runtime_normalization_preserves_safety_checks(tmp_path, names, link, code):
    source = tmp_path / "runtime.tar"
    archive(source, names, link)
    with pytest.raises(operator.OperatorError, match=code):
        operator.safe_extract(source, tmp_path / "runtime", runtime=True)
    assert not (tmp_path / "runtime").exists()


@pytest.mark.parametrize("error,expected", [
    (operator.OperatorError("archive_traversal"), "archive_traversal"),
    (operator.OperatorError("PRIVATE response or credential"), "unexpected_error"),
    (ValueError("PRIVATE response or credential"), "unexpected_error"),
    (operator.OperatorError({"PRIVATE": "value"}), "unexpected_error"),
    (InterruptedError("PRIVATE response or credential"), "allocation_interrupted"),
])
def test_error_code_never_exports_unapproved_exception_payload(error, expected):
    assert operator.error_code(error) == expected


def test_main_persists_runtime_failure_stage_and_safe_code(tmp_path, monkeypatch):
    root, work = tmp_path / "project", tmp_path / "climate-agent-full-synthetic"
    (root / "runs").mkdir(parents=True)
    (root / "envs").mkdir()
    work.mkdir()
    gold = root / "envs" / f"budget-agent-validation-gold-{operator.GOLD_SHA}.json"
    gold.write_text("synthetic")
    runtime = root / "envs/runtime.tar"
    archive(runtime, ["./bin/python"])
    original_digest = operator.digest
    monkeypatch.setattr(operator, "digest", lambda p: operator.GOLD_SHA if p == gold else original_digest(p))
    monkeypatch.setattr(operator, "ROOT", root)
    monkeypatch.setattr(operator, "ARCHIVES", {"runtime": ("runtime.tar", original_digest(runtime))})
    monkeypatch.setattr(operator.os, "umask", lambda mask: None)
    monkeypatch.setattr(operator.signal, "SIGUSR1", 10, raising=False)
    monkeypatch.setattr(operator.signal, "signal", lambda *args: None)
    for key, value in {
        "SLURM_JOB_ID": "synthetic", "CUDA_VISIBLE_DEVICES": "synthetic",
        "CLIMATE_FULL_RELEASE_ID": "climate-full-72eaa90-synthetic-test",
        "CLIMATE_FULL_TASK_ROOT": str(work), "CLIMATE_FULL_SCRATCH_PARENT": str(tmp_path),
        "CLIMATE_OPERATOR_GIT": "synthetic", "CLIMATE_OPERATOR_SHA256": "synthetic",
        "CLIMATE_FULL_WRAPPER_SHA256": "synthetic",
    }.items():
        monkeypatch.setenv(key, value)
    with pytest.raises(operator.OperatorError, match="runtime_site_packages_absent"):
        operator.main()
    saved = json.loads((root / "runs/climate-full-72eaa90-synthetic-test/operator-status.json").read_text())
    assert saved["status"] == "failed"
    assert saved["stage"] == "extract_runtime"
    assert saved["error_code"] == "runtime_site_packages_absent"
    assert saved["file_receipts"] == {}
    assert saved["input_extractions"] == 0
    assert all(row["status"] == "not_started" for row in saved["phases"].values())


def phase_fixture(directory, phase):
    directory.mkdir()
    protocol = operator.PROTOCOLS[phase][1]
    slots = operator.PROTOCOLS[phase][2]
    run = {"phase": phase, "code_sha": operator.INFERENCE, "working_tree_dirty": False,
           "protocol_sha256": protocol, "prompt_identity": {"system_prompt_sha256": operator.PROMPT_SHA,
           "schema_json_sha256": operator.SCHEMA_SHA}, "corpus_sha256": operator.CORPUS_SHA,
           "document_count": 5240, "model_sha256": operator.MODEL_SHA, "reranker_sha256": operator.RERANKER_SHA,
           "dense_enabled": False, "dense_file_hashes": None}
    (directory / "run.json").write_text(json.dumps(run))
    (directory / "run.consumed.json").write_text(json.dumps(
        {"phase": phase, "protocol_sha256": protocol, "provider": "local-qwen"}))
    score = {"phase": phase, "inference_source_git_sha": operator.INFERENCE,
             "protocol_sha256": protocol, "scorer_identity": {"git_sha": operator.SCORER},
             "run_sha256": operator.digest(directory / "run.json"),
             "matrix": {"expected_slots": slots, "actual_slots": slots,
                        "task_denominator_per_strategy": slots // 3, "validated": True},
             "gold_sha256": operator.GOLD_SHA if phase == "validation" else None,
             "selection_manifest_sha256": operator.SELECTION_SHA if phase == "validation" else None}
    (directory / "score.json").write_text(json.dumps(score))
    return score


@pytest.mark.parametrize("phase", operator.PHASES)
def test_phase_receipt_checks_frozen_source_and_slots(tmp_path, phase):
    directory = tmp_path / phase
    phase_fixture(directory, phase)
    result = operator.verify_phase(directory, phase)
    assert set(result) == {"run.json", "score.json", "run.consumed.json"}


@pytest.mark.parametrize("corruption", ["missing_slot", "wrong_run_hash", "wrong_scorer", "phase_leak", "gold_leak"])
def test_bad_score_is_not_accepted(tmp_path, corruption):
    directory = tmp_path / "vnext"
    score = phase_fixture(directory, "vnext")
    if corruption == "missing_slot":
        score["matrix"]["actual_slots"] = 23
    elif corruption == "wrong_run_hash":
        score["run_sha256"] = "a" * 64
    elif corruption == "wrong_scorer":
        score["scorer_identity"]["git_sha"] = operator.INFERENCE
    elif corruption == "phase_leak":
        score["phase"] = "validation"
    else:
        score["gold_sha256"] = operator.GOLD_SHA
    (directory / "score.json").write_text(json.dumps(score))
    with pytest.raises(ValueError):
        operator.verify_phase(directory, "vnext")


def test_frozen_wrapper_and_one_input_extraction_contract():
    wrapper = (ROOT / "hpc/budget_agent_full.sbatch").read_text()
    source = (ROOT / "scripts/run_budget_agent_full_operator.py").read_text()
    for flag in ("--gres=gpu:A100:1", "--cpus-per-task=8", "--mem=32G", "--tmp=30G",
                 "--time=02:00:00", "--no-requeue", "--signal=B:USR1@90"):
        assert "#SBATCH " + flag in wrapper
    assert "CLIMATE_FULL_RELEASE_ID:?" in wrapper
    assert 'exec python "${CLIMATE_FULL_TASK_ROOT}/operator.py"' in wrapper
    assert "sbatch " not in source
    assert source.count('safe_extract(archive, work / name') == 1
    assert 'result.mkdir(mode=0o700)' in source
    assert 'shutil.rmtree' not in source
    assert operator.PHASES == ("validation", "vnext")
    assert operator.INFERENCE.startswith("72eaa90") and operator.SCORER.startswith("208ff93")
