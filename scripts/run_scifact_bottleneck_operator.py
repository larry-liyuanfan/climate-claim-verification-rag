"""One bounded parent: prepare -> bottleneck child -> reaped proof -> score CLI."""
from __future__ import annotations

import argparse
from contextlib import contextmanager
import json
import os
from pathlib import Path
import signal
import sys
import time
from typing import Any, Callable, Iterator, cast

from climate_rag.scifact_natural_contract import require
from climate_rag.scifact_read_continuation import ordered_write
from climate_rag.scifact_utility_runtime import ledger_cost
from prepare_scifact_semantic_pair import verify_source_tree
from run_budget_agent_full_operator import ARCHIVES, read_only_tree
from run_scifact_component_operator import generator_only
from run_scifact_evidence_commit_paired import cpu_stage, supervisor_signals
from run_scifact_evidence_note_operator import bounded_worker
from run_scifact_grounding_train_operator import ROOT, sha
from run_scifact_utility8_operator import verify_runtime_receipt
import scifact_evidence_input as adapter
from scifact_bottleneck_execution import (
    WRAPPER, MODEL_RELATIVE, bind_paths, export_failure, load_release, stage_watchdog, validate_release,
)


def prepare(release: Any, work: Path) -> None:
    verify_runtime_receipt(release)
    adapter.check_prepared(Path(release["prepared"]), release)
    generator_only(ROOT / "envs" / ARCHIVES["input"][0], work / "input", release["model_archive_sha256"])
    read_only_tree(work / "input")


@contextmanager
def durable_write(*, final: bool = False) -> Iterator[None]:
    """Finish short receipts before handling TERM; final failure flush is bounded.

    Outside final closeout, defer (not lose) a signal so it prevents the next
    stage. Existing supervisor_signals remains installed across all transitions.
    """
    previous: dict[Any, Any] = {}
    pending = False
    def defer(number: Any, frame: Any) -> None:
        nonlocal pending
        pending = True
    if os.name == "posix":
        for number in (signal.SIGTERM, signal.SIGINT):
            previous[number] = signal.signal(number, signal.SIG_IGN if final else defer)
    try:
        yield
    finally:
        for number, handler in previous.items():
            signal.signal(number, handler)
    if pending:
        raise InterruptedError("deferred_receipt_signal")


def run_supervised(release: Any, release_sha: str, source: Path, work: Path, *,
                   prepare_fn: Any = prepare, worker: Any = bounded_worker,
                   clock: Any = time.monotonic, initial_seconds: float | None = None) -> Any:
    validate_release(release)
    began = clock()
    seconds = release["max_operator_seconds"] if initial_seconds is None else min(initial_seconds, release["max_operator_seconds"])
    deadline = began + seconds
    root, output = Path(release["run_directory"]), Path(release["output"])
    root.mkdir(mode=0o700)  # single lifetime reservation, never retry/reuse
    ordered_write(root / "reserved.json", {"release_sha256": release_sha, "source_git": release["source_git"],
        "planned_route_results": 72, "max_generations": 96, "automatic_retry": False,
        "job_id": os.environ.get("SLURM_JOB_ID")})
    # Exact bytes are passed in from the checked release, not a runtime rewrite.
    release_path = root / "release.json"
    from climate_rag.scifact_semantic_contract import checked
    raw = checked(Path(release["execution_release_file"]), release_sha)
    with release_path.open("xb") as stream:
        stream.write(raw)
    binding = root / "runtime-paths.json"
    proof = {"child_started": False, "child_reaped": False, "returncode": None,
             "interrupted": "not_started", "automatic_retry": False}
    failure, scored, result = None, None, None
    model = work / MODEL_RELATIVE
    def flush_exit_cost() -> None:
        output.mkdir(mode=0o700, exist_ok=True)
        proof.update(release_sha256=release_sha, runtime_paths_sha256=sha(binding) if binding.exists() else None)
        if not (output / "worker-exit.json").exists():
            ordered_write(output / "worker-exit.json", proof)
        if not (root / "cost-before-gold.json").exists():
            ordered_write(root / "cost-before-gold.json", {"release_sha256": release_sha, "gold_read": False,
                "planned_route_results": 72, "physical": ledger_cost(output / "ledger"),
                "worker_exit_sha256": sha(output / "worker-exit.json")})
    with supervisor_signals():
        try:
            cast(Callable[..., Any], cpu_stage)(min(release["max_prepare_seconds"], deadline-clock()-release["scoring_reserve_seconds"]),
                      prepare_fn, release, work)
            require(model.is_dir(), "actual_extracted_model_missing")
            ordered_write(binding, bind_paths(release, release_sha, work))
            command = [sys.executable, str(source / "scripts/run_scifact_evidence_bottleneck.py"),
                "--release", str(release_path), "--release-sha", release_sha,
                "--runtime-paths", str(binding), "--model-dir", str(model)]
            proof = worker(command, root, output,
                min(release["max_worker_seconds"], deadline-clock()-release["scoring_reserve_seconds"]),
                watchdog=stage_watchdog)
            require(not proof["child_started"] or proof["child_reaped"], "unreaped_worker")
            with durable_write():
                flush_exit_cost()
            cost = ledger_cost(output / "ledger")
            require(proof["child_reaped"] and proof["returncode"] == 0 and proof["interrupted"] is None
                    and cost["unknown_usage_attempts"] == 0, "worker_failed_or_unknown_cost")
            score_allocation = root / "score-process"
            score_allocation.mkdir(mode=0o700)
            command = [sys.executable, str(source / "scripts/score_scifact_evidence_bottleneck.py"),
                "--release", str(release_path), "--release-sha", release_sha,
                "--runtime-paths", str(binding), "--model-dir", str(model)]
            scored = worker(command, score_allocation, Path(release["reports"]),
                min(release["scoring_reserve_seconds"], deadline-clock()), watchdog=lambda _: None)
            with durable_write():
                ordered_write(score_allocation / "exit.json", dict(scored, release_sha256=release_sha))
            require(scored["child_reaped"] and scored["returncode"] == 0 and scored["interrupted"] is None,
                    "scorer_failed_or_timed_out")
            result = json.loads((Path(release["reports"]) / "compact.json").read_bytes())
            require(result["status"] == "scored" and result["release_sha256"] == release_sha, "scorer_compact_binding")
        except Exception as exc:
            failure = type(exc).__name__
        finally:
            # No model/gold work here. Ignore repeated termination for the short
            # final cost/failure receipt section, within outer kill-after grace.
            with durable_write(final=True):
                flush_exit_cost()
                if failure is not None or result is None:
                    result = export_failure(release, release_sha, failure or "incomplete_execution")
                else:
                    ordered_write(root / "compact.json", result)
                ordered_write(root / "complete.json", {"status": result["status"], "source_git": release["source_git"],
                    "release_sha256": release_sha, "elapsed_seconds": clock()-began,
                    "worker_exit": proof, "scorer_exit": scored, "automatic_retry": False,
                    "compact_sha256": sha(root / "compact.json"), "training_authorized": False})
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--release", type=Path, required=True)
    parser.add_argument("--release-sha", required=True)
    args = parser.parse_args()
    release = load_release(args.release, args.release_sha)
    require(os.name == "posix" and os.environ.get("SLURM_JOB_ID") and os.environ.get("CUDA_VISIBLE_DEVICES"), "allocated_gpu_only")
    source = Path(__file__).resolve().parents[1]
    work = Path(os.environ["CLIMATE_BOTTLENECK_WORK"]).resolve()
    require(source.parent == work and work.name.startswith("climate-bottleneck-"), "isolated_scratch_required")
    require(sha(source / WRAPPER) == release["wrapper_sha256"], "wrapper_binding")
    verify_source_tree(source, Path(release["source_archive"]), release["source_archive_sha256"], release["source_git"])
    require(args.release.resolve() == Path(release["execution_release_file"]).resolve(), "release_file_binding")
    elapsed = max(0, time.time()-float(os.environ["CLIMATE_BOTTLENECK_STARTED_UNIX"]))
    result = run_supervised(release, args.release_sha, source, work,
        initial_seconds=release["max_operator_seconds"]-elapsed)
    raise SystemExit(0 if result["status"] == "scored" else 2)


if __name__ == "__main__":
    main()
