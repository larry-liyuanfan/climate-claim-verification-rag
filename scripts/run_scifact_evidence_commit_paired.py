"""One supervisor; serial isolated workers; original scorer only after both exit."""
from __future__ import annotations

import json
from contextlib import contextmanager
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
from typing import Any

from climate_rag.scifact_read_continuation import ordered_write
from climate_rag.scifact_utility_runtime import ledger_cost
from run_budget_agent_full_operator import ARCHIVES, read_only_tree
from run_scifact_component_operator import generator_only
from run_scifact_evidence_commit_operator import start_attempt, output_path
from run_scifact_evidence_note_operator import bounded_worker
from run_scifact_grounding_train_operator import ROOT, require, sha
from run_scifact_natural_operator import slot_watchdog
from scifact_paired_comparison import WRAPPER, validate_pair, sum_costs, compare_quality
import scifact_evidence_input as inputs


@contextmanager
def supervisor_signals():
    """Cover preparation/phase transitions too, not just the GPU child's wait."""
    previous = {}
    def interrupted(number, frame):
        raise InterruptedError("paired_supervisor_signal")
    if os.name == "posix":
        for number in (signal.SIGTERM, signal.SIGINT):
            previous[number] = signal.signal(number, interrupted)
    try:
        yield
    finally:
        for number, handler in previous.items():
            signal.signal(number, handler)


def cpu_stage(seconds, function, *args, **kwargs):
    """No nested GPU supervisor: bound synchronous metadata/extraction only."""
    require(seconds > 0, "cpu_stage_budget_exhausted")
    if os.name != "posix":  # Windows synthetic tests; executable main requires POSIX.
        return function(*args, **kwargs)
    def expired(number, frame):
        raise TimeoutError("paired_cpu_stage_deadline")
    previous = signal.signal(signal.SIGALRM, expired)
    signal.setitimer(signal.ITIMER_REAL, seconds)
    try:
        return function(*args, **kwargs)
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        signal.signal(signal.SIGALRM, previous)


def run_pair(release: dict[str, Any], release_sha: str, source: Path, work: Path,
             *, clock: Any = time.monotonic, worker: Any = bounded_worker,
             prepare: Any = inputs.copy_prepared, reserve: Any = start_attempt,
             extract: Any = generator_only, readonly: Any = read_only_tree,
             score: Any = None, initial_seconds: float | None = None) -> dict[str, Any]:
    validate_pair(release)
    began = clock()
    deadline = began + min(release["max_operator_seconds"], initial_seconds if initial_seconds is not None else release["max_operator_seconds"])
    output = Path(release["output"])
    output.mkdir(mode=0o700)
    ordered_write(output / "reserved.json", {"release_sha256": release_sha,
        "planned_slots": 144, "max_generator_calls": 480, "source_git": release["source_git"],
        "job_id": os.environ.get("SLURM_JOB_ID"), "automatic_retry": False})
    phases: list[dict[str, Any]] = []
    shared_failure, shared_seconds = None, 0.0
    for index, child in enumerate(release["children"]):
        child_file = output / f"release-v{index+2}.json"
        ordered_write(child_file, child)
        child_sha = sha(child_file)
        proof: dict[str, Any] = {"child_started": False, "child_reaped": False, "returncode": None,
                                 "interrupted": "not_started", "automatic_retry": False}
        phases.append({"protocol": child["protocol"], "release_sha256": child_sha,
                       "planned_slots": 72, "status": "not_started", "proof": proof, "elapsed_seconds": 0})
    # Every slot exists before extraction/first model load; outer failures cannot
    # drop the unstarted protocol or the cost of a previously completed worker.
    active = None
    with supervisor_signals():
        try:
            shared_began = clock()
            shared_deadline = min(deadline-release["scoring_reserve_seconds"], clock()+300)
            cpu_stage(shared_deadline-clock(), extract, ROOT / "envs" / ARCHIVES["input"][0],
                      work / "input", release["model_archive_sha256"])
            cpu_stage(shared_deadline-clock(), readonly, work / "input")
            shared_seconds = clock()-shared_began
            for index, (child,record) in enumerate(zip(release["children"],phases,strict=True)):
                # Give each launched phase its full registered ceiling.
                if deadline-clock() < release["phase_seconds"]+release["scoring_reserve_seconds"]:
                    break
                active = record
                phase_began = clock()
                phase_deadline = phase_began+release["phase_seconds"]
                phase, child_sha = output_path(child), record["release_sha256"]
                child_file = output / f"release-v{index+2}.json"
                record["status"] = "preparing"
                cpu_stage(phase_deadline-clock()-420, reserve, child,child_sha,os.environ.get("SLURM_JOB_ID","test"))
                cpu_stage(phase_deadline-clock()-420, prepare, phase/"prepared",child)
                cpu_stage(phase_deadline-clock()-420, readonly, phase/"prepared/inference")
                seconds = min(child["max_worker_seconds"],phase_deadline-clock()-420)
                require(seconds > 0,"phase_preparation_exhausted")
                command = [sys.executable,str(source/"scripts/run_scifact_evidence_commit.py"),
                    "--release",str(child_file),"--release-sha",child_sha,
                    "--inference-dir",str(phase/"prepared/inference"),
                    "--model-root",str(work/"input/models"),"--output",str(phase/"inference")]
                proof = worker(command,phase,phase/"inference",seconds,watchdog=slot_watchdog)
                record.update(proof=proof,elapsed_seconds=clock()-phase_began)
                require(not proof["child_started"] or proof["child_reaped"],"unreaped_child")
                proof.update(release_sha256=child_sha,preparation_sha256=sha(phase/"prepared/preparation.json"))
                ordered_write(phase/"worker-exit.json",proof)
                record["status"] = "worker_exited"
                cost = ledger_cost(phase/"inference/ledger")
                if (proof["returncode"] != 0 or proof["interrupted"] is not None
                        or cost["unknown_usage_attempts"] or cost["unique_physical_calls"] > 240):
                    break
                active = None
        except Exception as exc:
            if active is not None:
                active.update(status="infrastructure_failed",error_type=type(exc).__name__,
                              elapsed_seconds=clock()-phase_began)
            else:
                shared_failure = type(exc).__name__
                shared_seconds = clock()-shared_began
        # Ignore repeated termination only for the short durable receipt section.
        prior = {}
        if os.name == "posix":
            prior = {s:signal.signal(s,signal.SIG_IGN) for s in (signal.SIGTERM,signal.SIGINT)}
        try:
            for index,(child,record) in enumerate(zip(release["children"],phases,strict=True)):
                record["physical_generation_cost"] = ledger_cost(output_path(child)/"inference/ledger")
                ordered_write(output/f"phase-v{index+2}.json",record)
            costs = {"release_sha256":release_sha,**sum_costs([p["physical_generation_cost"] for p in phases]),
                     "phases":phases,"shared_preparation_seconds":shared_seconds,
                     "shared_preparation_error":shared_failure,"gold_read":False}
            ordered_write(output/"cost-before-gold.json",costs)
        finally:
            for s,handler in prior.items():
                signal.signal(s,handler)
    # A failed/unlaunched phase retains its 72 slots and physical lower bounds.
    reports = []
    for child, phase_record in zip(release["children"], phases, strict=True):
        phase = output_path(child)
        report: dict[str, Any] = {"status": "no_quality", "protocol": child["protocol"], "planned_slots": 72}
        if (phase_record["status"] == "worker_exited" and phase_record["proof"]["child_reaped"]
                and all(not p["proof"]["child_started"] or p["proof"]["child_reaped"] for p in phases)
                and deadline-clock() >= release["scoring_seconds_per_phase"]):
            try:
                if score is not None:
                    report = score(child, phase_record)
                else:
                    command = [sys.executable, str(source / "scripts/score_scifact_evidence_commit.py"),
                        "--release", str(output / ("release-v2.json" if child is release["children"][0] else "release-v3.json")),
                        "--release-sha", phase_record["release_sha256"],
                        "--model-dir", str(work / "input/models/generator/model")]
                    with (phase / "score.log").open("xb") as stream:
                        scored = subprocess.run(command, stdout=stream, stderr=stream,
                                                timeout=release["scoring_seconds_per_phase"], check=False)
                    name = "quality.json" if scored.returncode == 0 else "no-quality.json"
                    report = json.loads((phase / name).read_bytes())
            except Exception as exc:
                report.update(status="no_quality", error_type=type(exc).__name__)
        reports.append(report)
    result = {**compare_quality(reports), "release_sha256": release_sha,
              "cost_sha256": sha(output / "cost-before-gold.json"),
              "protocol_results": reports, "elapsed_seconds": clock()-began,
              "training_authorized": False, "automatic_retry": False}
    ordered_write(output / "comparison.json", result)
    return result


def main() -> None:
    require(os.name == "posix" and bool(os.environ.get("SLURM_JOB_ID"))
            and bool(os.environ.get("CUDA_VISIBLE_DEVICES")), "allocated_gpu_only")
    source = Path(__file__).resolve().parents[1]
    work = Path(os.environ["CLIMATE_GROUNDING_WORK"]).resolve()
    release_file = Path(os.environ["CLIMATE_GROUNDING_RELEASE_FILE"])
    release_sha = os.environ["CLIMATE_GROUNDING_RELEASE_SHA"]
    require(sha(release_file) == release_sha, "paired_release_identity")
    release = json.loads(release_file.read_bytes())
    validate_pair(release)
    require(source.parent == work and work.name.startswith("climate-grounding-")
            and (source / "SOURCE_REVISION").read_text().strip() == release["source_git"] == os.environ["CLIMATE_SOURCE_GIT"]
            and release["source_archive_sha256"] == os.environ["CLIMATE_SOURCE_SHA256"]
            and sha(source / WRAPPER) == release["wrapper_sha256"], "paired_source_binding")
    wrapper_elapsed = max(0,time.time()-float(os.environ["CLIMATE_PAIR_STARTED_UNIX"]))
    result = run_pair(release, release_sha, source, work,
                      initial_seconds=release["max_operator_seconds"]-wrapper_elapsed)
    raise SystemExit(0 if result["status"] == "paired_scored" else 2)


if __name__ == "__main__":
    main()
