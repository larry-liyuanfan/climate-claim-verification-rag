"""Execution wiring only: original runner/scorer, synthetic inputs/model outputs."""
import copy
import json
import os
import signal
from pathlib import Path
import subprocess
import sys
import time

import pytest

from climate_rag.scifact_read_continuation import ordered_write
from climate_rag.scifact_semantic_contract import sha
import run_scifact_bottleneck_operator as operator
import scifact_bottleneck_execution as execution
from bottleneck_execution_fixture import REPO, fixture_release, validate_fixture


def prepared(tmp_path, monkeypatch):
    release = fixture_release(tmp_path)
    path = Path(release["execution_release_file"])
    ordered_write(path, release)
    monkeypatch.setattr(operator, "validate_release", validate_fixture)
    work = tmp_path / "scratch"
    work.mkdir()
    def prepare(release, work):
        (work / execution.MODEL_RELATIVE).mkdir(parents=True)
    return release, sha(path.read_bytes()), work, prepare


@pytest.mark.parametrize("mode", ["normal", "worker_exit", "worker_timeout", "load_failure", "score_timeout"])
def test_supervised_real_runner_cli_score_export_and_faults(tmp_path, monkeypatch, mode):
    release, digest, work, prepare = prepared(tmp_path, monkeypatch)
    def worker(command, allocation, inference, seconds, *, watchdog):
        phase = "worker" if "run_scifact_evidence_bottleneck.py" in command[1] else "score"
        command = [sys.executable, str(REPO / "tests/bottleneck_execution_fixture.py"), phase, mode, *command[2:]]
        timeout = min(seconds, 12) if ((phase == "worker" and mode == "worker_timeout")
                                    or (phase == "score" and mode == "score_timeout")) else seconds
        if os.name == "posix":
            return operator.bounded_worker(command, allocation, inference, timeout, watchdog=watchdog)
        # Windows verifies the same CLI dataflow; Linux CI additionally uses
        # the real production process-group supervisor and reap semantics.
        interrupted = None
        with (allocation / "worker.log").open("xb") as log:
            process = subprocess.Popen(command, stdout=log, stderr=log)
            try:
                process.wait(timeout=timeout)
            except subprocess.TimeoutExpired:
                interrupted = "TimeoutError"
                process.kill()
                process.wait()
        return {"child_started": True, "child_reaped": True, "returncode": process.returncode,
                "interrupted": interrupted}
    result = operator.run_supervised(release, digest, REPO, work, prepare_fn=prepare, worker=worker)
    root = Path(release["run_directory"])
    complete = json.loads((root / "complete.json").read_bytes())
    compact = json.loads((root / "compact.json").read_bytes())
    assert compact == result and complete["worker_exit"]["child_reaped"]
    proof = json.loads((root / "inference/worker-exit.json").read_bytes())
    assert proof["release_sha256"] == digest
    assert proof["runtime_paths_sha256"] == sha((root / "runtime-paths.json").read_bytes())
    assert (root / "release.json").read_bytes() == Path(release["execution_release_file"]).read_bytes()
    assert not any(s in json.dumps(compact) for s in ("Original sentence", "immutable fixture", '"claim_id"'))
    if mode == "normal":
        assert result["status"] == "scored", (root / "score-process/worker.log").read_text()
        assert result["accounting"]["physical"]["unique_physical_calls"] == 96
        assert complete["scorer_exit"]["child_reaped"]
        assert result["mechanism"] == "not_supported"
        assert all(v["planned"] == 24 for v in result["routes"].values())
    else:
        assert result["status"] == "no_quality" and result["planned_route_results"] == 72
        assert result["gold_read"] == (mode == "score_timeout")
        if mode in ("worker_exit", "score_timeout"):
            assert result["physical"]["unique_physical_calls"] == 96
        if mode == "worker_timeout":
            assert proof["interrupted"] == "TimeoutError" and proof["returncode"] != 0
            assert result["physical"]["unique_physical_calls"] == 2
            assert result["physical"]["unknown_usage_attempts"] == 1
        if mode == "load_failure":
            assert result["physical"]["unique_physical_calls"] == 0
    with pytest.raises(FileExistsError):
        operator.run_supervised(release, digest, REPO, work, prepare_fn=prepare, worker=worker)


def test_prepare_failure_retains_all_slots_no_worker_no_gold(tmp_path, monkeypatch):
    release, digest, work, _ = prepared(tmp_path, monkeypatch)
    def fail(*args):
        raise RuntimeError("synthetic prepare failed")
    def forbidden(*args, **kwargs):
        pytest.fail("worker/scorer must not start")
    result = operator.run_supervised(release, digest, REPO, work, prepare_fn=fail, worker=forbidden)
    assert result["status"] == "no_quality" and not result["gold_read"]
    assert result["planned_route_results"] == 72 and result["physical"]["unique_physical_calls"] == 0


def test_expired_total_budget_does_not_prepare(tmp_path, monkeypatch):
    release, digest, work, _ = prepared(tmp_path, monkeypatch)
    def forbidden(*args, **kwargs):
        pytest.fail("expired preparation cannot start")
    result = operator.run_supervised(release, digest, REPO, work,
        prepare_fn=forbidden, worker=forbidden, initial_seconds=0)
    assert result["status"] == "no_quality" and result["planned_route_results"] == 72


def test_draft_rejected_before_import_config_or_reservation(tmp_path, monkeypatch):
    release = fixture_release(tmp_path)
    release["status"] = "draft_not_authorized"
    def forbidden():
        pytest.fail("draft must not expand config")
    monkeypatch.setattr(execution, "frozen_contract", forbidden)
    with pytest.raises(ValueError, match="new_exact_source_authorization_required"):
        operator.run_supervised(release, "a"*64, REPO, tmp_path)
    assert not Path(release["run_directory"]).exists()


def test_watchdog_matches_new_nested_layout_only_pending_stages(tmp_path):
    pending = tmp_path / "1/selector"
    pending.mkdir(parents=True)
    ordered_write(pending / "reserved.json", {"started_unix": time.time()-121})
    with pytest.raises(TimeoutError, match="bottleneck_stage_deadline"):
        execution.stage_watchdog(tmp_path)
    ordered_write(pending / "result.json", {})
    execution.stage_watchdog(tmp_path)


def test_runtime_model_escape_and_exit_proof_binding(tmp_path, monkeypatch):
    release = fixture_release(tmp_path)
    root = Path(release["run_directory"])
    root.mkdir()
    ordered_write(root / "reserved.json", {"release_sha256": "a"*64})
    work = tmp_path / "scratch"
    model = work / execution.MODEL_RELATIVE
    model.mkdir(parents=True)
    monkeypatch.setattr(execution, "trusted_work", lambda: work)
    binding = root / "runtime-paths.json"
    row = execution.bind_paths(release, "a"*64, work)
    ordered_write(binding, row)
    execution.check_paths(release, "a"*64, binding, model)
    with pytest.raises(ValueError, match="runtime_model_binding"):
        execution.check_paths(release, "a"*64, binding, tmp_path)
    inference = Path(release["output"])
    inference.mkdir()
    proof = {"child_reaped": True, "returncode": 0, "interrupted": None,
             "release_sha256": "a"*64, "runtime_paths_sha256": "b"*64}
    ordered_write(inference / "worker-exit.json", proof)
    with pytest.raises(ValueError, match="supervised_exit_required"):
        execution.verify_exit(release, "a"*64, binding, model)
    other = tmp_path / "other-scratch"
    other_model = other / execution.MODEL_RELATIVE
    other_model.mkdir(parents=True)
    binding.write_text(json.dumps(execution.bind_paths(release, "a"*64, other)))
    with pytest.raises(ValueError, match="runtime_model_binding"):
        execution.check_paths(release, "a"*64, binding, other_model)


@pytest.mark.skipif(os.name != "posix", reason="TERM transition/reap contract uses real POSIX signals in Linux CI")
@pytest.mark.parametrize("transition", ["worker-exit.json", "exit.json"])
def test_SIGTERM_after_reap_durable_cost_and_no_premature_gold(tmp_path, monkeypatch, transition):
    release, digest, work, prepare = prepared(tmp_path, monkeypatch)
    original_write = operator.ordered_write
    fired = []
    def inject(path, value):
        original_write(path, value)
        if path.name == transition and not fired:
            fired.append(path.name)
            os.kill(os.getpid(), signal.SIGTERM)
    monkeypatch.setattr(operator, "ordered_write", inject)
    def worker(command, allocation, inference, seconds, *, watchdog):
        phase = "worker" if "run_scifact_evidence_bottleneck.py" in command[1] else "score"
        command = [sys.executable, str(REPO / "tests/bottleneck_execution_fixture.py"), phase, "normal", *command[2:]]
        return operator.bounded_worker(command, allocation, inference, seconds, watchdog=watchdog)
    result = operator.run_supervised(release, digest, REPO, work, prepare_fn=prepare, worker=worker)
    assert fired and result["status"] == "no_quality" and result["planned_route_results"] == 72
    assert result["physical"]["unique_physical_calls"] == 96
    assert result["gold_read"] == (transition == "exit.json")
    assert (Path(release["run_directory"]) / "complete.json").exists()


def test_real_release_contract_and_bounds_no_execution(tmp_path, monkeypatch):
    from build_scifact_bottleneck_draft import build
    from climate_rag.scifact_generation import frozen_contract
    from run_scifact_evidence_commit_operator import release_fields
    from climate_rag.scifact_relation_verifier import RELATION_PROTOCOL
    import scifact_evidence_input as adapter
    fields = release_fields({"protocol": RELATION_PROTOCOL, "input_protocol": adapter.PROTOCOL})
    receipt = dict(fields, protocol=execution.PROTOCOL, claims=24, overflow_count=0, model_calls=0,
        gold_read=False, tokenizer_sha256=execution.TOKENIZER_SHA, frames_identity="c"*64,
        source_git="d"*40, job_id="31996384")
    draft = build(receipt, fields, frozen_contract(), "e"*40, "f"*64, "a"*64)
    del draft["model_directory"]
    draft.update(execution.execution_fields(draft), wrapper_sha256="b"*64,
        status="authorized_model_run", model_execution_authorized=True,
        authorization="coordinator_exact_hash_release",
        execution_release_file=(execution.ROOT / "envs" / ("evidence-bottleneck-"+"e"*12) / "release.json").as_posix())
    execution.validate_release(draft)
    for field, value in (("max_generations", 97), ("planned_route_results", 69),
                         ("scoring_reserve_seconds", 0), ("max_operator_seconds", 1801),
                         ("claims_sha256", "0"*64), ("model_directory", "future-model-input")):
        altered = copy.deepcopy(draft)
        altered[field] = value
        with pytest.raises(ValueError):
            execution.validate_release(altered)


def test_scorer_cli_help_and_draft_refusal(tmp_path):
    # CLI actually parses a candidate, not only a direct validator unit test.
    release = fixture_release(tmp_path)
    release["status"] = "draft_not_authorized"
    p = tmp_path / "draft.json"
    ordered_write(p, release)
    command = [sys.executable, str(REPO / "scripts/score_scifact_evidence_bottleneck.py"),
        "--release", str(p), "--release-sha", sha(p.read_bytes()),
        "--runtime-paths", str(tmp_path / "absent.json"), "--model-dir", str(tmp_path / "absent")]
    result = subprocess.run(command, capture_output=True, timeout=30)
    assert result.returncode != 0 and b"new_exact_source_authorization_required" in result.stderr
    assert not Path(release["reports"]).exists()
