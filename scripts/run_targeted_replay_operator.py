"""Reuse bounded worker/process-group supervisor; no automatic retries or warmups."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from typing import Any

from climate_rag.scifact_generation import frozen_contract
from climate_rag.scifact_read_continuation import ordered_write
from climate_rag.scifact_utility_runtime import ledger_cost, reranker_cost
from climate_rag.targeted_query import PROTOCOL
from climate_rag import stop_acquire
from climate_rag.targeted_replay import (
    ROOT,
    INPUT_NAME,
    INPUT_SHA,
    policy,
    sha,
    slot_watchdog,
)
from run_budget_agent_full_operator import digest, safe_extract, read_only_tree
from run_scifact_bottleneck_operator import durable_write
from run_scifact_evidence_commit_paired import cpu_stage, supervisor_signals
from run_scifact_evidence_note_operator import bounded_worker
from run_scifact_utility8_operator import verify_runtime_receipt

WRAPPER = "hpc/climate_targeted_replay.sbatch"
RESOURCE = {
    "gpu": "A100:1",
    "cpus": 8,
    "host_ram_gib": 32,
    "scratch_gib": 30,
    "slurm_seconds": 7200,
}


def draft(source_git: str, archive_sha: str, wrapper_sha: str, *, protocol: str = PROTOCOL) -> dict[str, Any]:
    result: dict[str, Any] = {
        "authorization": "none_draft",
        "model_execution_authorized": False,
        "purpose": protocol,
        "source_git": source_git,
        "source_archive_sha256": archive_sha,
        "wrapper_sha256": wrapper_sha,
        "output": (ROOT / "runs" / ("targeted-feedback-" + source_git[:12])).as_posix(),
        "policy": policy(protocol),
        "resource_cap": dict(RESOURCE),
        "generation_contract": frozen_contract(),
        "max_worker_seconds": 6000,
        "max_operator_seconds": 6900,
        "score_seconds": 300,
        "preparation_seconds": 600,
        "resource_basis": {
            "old_operator_receipt_sha256": "4f1e7a0e2e3d2fcf2dc64ddf5d4ac128e65dda59e41af2b880a16a65f9e12f34",
            "old_public_compact_sha256": "0094b372d98144d90fe9c09c4998bca29dda9d189cc482753aa50f363a86082f",
            "old_validation_generation_seconds_per_call": 406.105126 / 96,
            "old_validation_rerank_seconds_per_request": 18.078303 / 32,
            "generation_bound": 800,
            "rerank_bound": 128,
            "runtime_multiplier": 1.5,
            "not_prediction_or_sla": True,
        },
        "automatic_retry": False,
        "training_authorized": False,
        "protected_split_read": False,
        "runtime_receipt_sha256": "9878d3e4d45735828b4260d87656fd2827481663266a875725bfe7abab7d390a",
        "runtime_files_sha256": "8d231f4980e0bf94fe26273074588a0f6c0ea67646117871fd97443ad1142697",
        "python_executable": "/apps/easybuild-2022/easybuild/software/Compiler/GCCcore/11.3.0/Python/3.10.4/bin/python",
    }
    if protocol == stop_acquire.PROTOCOL:
        result["output"] = (ROOT / "runs" / ("stop-acquire-" + source_git[:12])).as_posix()
        result["resource_cap"]["host_ram_gib"] = 48
        result["resource_basis"] = {
            "job_id": "32030221",
            "compact_sha256": "a07721e8f5a9cf477b2f3052dc65bfc64ebd7da8642f212af593c15ad8c3f19b",
            "peak_host_ram_gib": 33530392 / 1024**2,
            "ram_safety_factor": 1.5,
            "generation_seconds_per_call": 276.610 / 194,
            "rerank_seconds_per_request_including_swaps": 791.349 / 96,
            "generation_bound": 800, "rerank_bound": 128,
            "runtime_multiplier": 1.5,
            "historical_scaled_seconds": (800 * 276.610 / 194 + 128 * 791.349 / 96) * 1.5,
            "worker_seconds_ceiling": 6000,
            "new_gate_latency_unmeasured": True,
            "max_input_tokens_per_call": 8192, "max_output_tokens_per_call": 512,
            "not_prediction_or_sla": True,
            "cost_feasibility": ">=2 calls even when stopping; cannot claim efficiency vs fixed_rerank without total gate+verdict+repair measurements",
        }
    return result


def wrapper_for(protocol: str) -> str:
    if protocol == stop_acquire.PROTOCOL:
        return "hpc/climate_stop_acquire.sbatch"
    if protocol == PROTOCOL:
        return WRAPPER
    raise ValueError("unknown_release_protocol")


def validate_release(value: dict[str, Any]) -> None:
    if (
        value.get("authorization") != "coordinator_exact_hash_release"
        or value.get("model_execution_authorized") is not True
    ):
        raise ValueError("unauthorized_draft_no_model_or_preparation")
    expected = draft(
        value["source_git"], value["source_archive_sha256"], value["wrapper_sha256"], protocol=value["purpose"]
    )
    expected.update(
        authorization="coordinator_exact_hash_release", model_execution_authorized=True
    )
    if value != expected:
        raise ValueError("frozen_release_changed")
    for key, length in (
        ("source_git", 40),
        ("source_archive_sha256", 64),
        ("wrapper_sha256", 64),
    ):
        if len(value[key]) != length or any(
            c not in "0123456789abcdef" for c in value[key]
        ):
            raise ValueError("release_hash_format")


def load_release(path: Path, expected_sha: str, source: Path) -> dict[str, Any]:
    raw = path.read_bytes()
    if sha(raw) != expected_sha:
        raise ValueError("execution_release_hash")
    value = json.loads(raw)
    validate_release(value)
    if (source / "SOURCE_REVISION").read_text().strip() != value[
        "source_git"
    ] or digest(source / wrapper_for(value["purpose"])) != value["wrapper_sha256"]:
        raise ValueError("exact_source_wrapper")
    return dict(value)


def prepare(release: dict[str, Any], work: Path) -> None:
    verify_runtime_receipt(release)
    archive = ROOT / "envs" / INPUT_NAME
    if digest(archive) != INPUT_SHA:
        raise ValueError("input_archive_hash")
    safe_extract(archive, work / "input")
    read_only_tree(work / "input")


def run_supervised(
    release: dict[str, Any],
    release_path: Path,
    release_sha: str,
    source: Path,
    work: Path,
    *,
    prepare_fn: Any = prepare,
    worker_fn: Any = bounded_worker,
    scorer_fn: Any = subprocess.run,
    validate_fn: Any = validate_release,
    stage_command: Any = None,
    execution_identity: dict[str, Any] | None = None,
) -> dict[str, Any]:
    validate_fn(release)  # Must precede mkdir, extraction or any model access.
    planned_slots = release["policy"]["planned_slots"]
    max_generator_calls = release["policy"]["max_generator_calls"]
    output = Path(release["output"])
    output.mkdir(mode=0o700)
    allocation = output / "allocation"
    allocation.mkdir(mode=0o700)
    ordered_write(
        output / "reserved.json",
        {
            "release_sha256": release_sha,
            "source_git": release["source_git"],
            "planned_slots": planned_slots,
            "max_generator_calls": max_generator_calls,
            "automatic_retry": False,
            **({"job_id": os.environ.get("SLURM_JOB_ID")} if execution_identity is None else execution_identity),
        },
    )
    began = time.monotonic()
    deadline = began + release["max_operator_seconds"]
    common = [
        "--release",
        str(release_path),
        "--release-sha",
        release_sha,
        "--input",
        str(work / "input"),
    ]
    command = [sys.executable, str(source / "scripts/run_targeted_replay.py")]
    proof: dict[str, Any] = {
        "child_started": False,
        "child_reaped": False,
        "returncode": None,
        "interrupted": "not_started",
        "automatic_retry": False,
    }
    error, scored = None, False
    try:
        bounded_prepare: Any = cpu_stage
        bounded_prepare(release["preparation_seconds"], prepare_fn, release, work)
        proof = worker_fn(
            [*command, "worker", *common] if stage_command is None else stage_command("worker"),
            allocation,
            output / "inference",
            min(
                release["max_worker_seconds"],
                deadline - time.monotonic() - release["score_seconds"],
            ),
            watchdog=slot_watchdog,
        )
        with durable_write():
            ordered_write(allocation / "worker-exit.json", proof)
            ordered_write(
                output / "cost-before-quality.json",
                {
                    "planned_slots": planned_slots,
                    "completed_slots": len(
                        list((output / "inference").glob("slot-*/finished.json"))
                    ),
                    "generation": ledger_cost(output / "inference/ledger"),
                    "reranker": reranker_cost(output / "reranker-ledger"),
                    "worker_proof": proof,
                },
            )
        if not (
            proof["child_started"]
            and proof["child_reaped"]
            and proof["returncode"] == 0
            and proof["interrupted"] is None
        ):
            raise ValueError("worker_failed_no_quality")
        seconds = min(release["score_seconds"], deadline - time.monotonic())
        if seconds <= 0:
            raise TimeoutError("score_budget_exhausted")
        with (allocation / "score.log").open("xb") as stream:
            scorer_fn(
                [*command, "score", *common] if stage_command is None else stage_command("score"),
                check=True,
                timeout=seconds,
                stdout=stream,
                stderr=stream,
            )
        if not (output / "compact.json").is_file():
            raise ValueError("score_missing_compact")
        scored = True
    except BaseException as exc:
        error = type(exc).__name__
    finally:
        with durable_write(final=True):
            if not (allocation / "worker-exit.json").exists():
                ordered_write(allocation / "worker-exit.json", proof)
            if not (output / "cost-before-quality.json").exists():
                ordered_write(
                    output / "cost-before-quality.json",
                    {
                        "planned_slots": planned_slots,
                        "completed_slots": len(
                            list((output / "inference").glob("slot-*/finished.json"))
                        ),
                        "generation": ledger_cost(output / "inference/ledger"),
                        "reranker": reranker_cost(output / "reranker-ledger"),
                        "worker_proof": proof,
                    },
                )
            result = {
                "status": "completed" if scored else "failed_unscored",
                "error_type": error,
                "planned_slots": planned_slots,
                "source_git": release["source_git"],
                "release_sha256": release_sha,
                "automatic_retry": False,
                "elapsed_seconds": time.monotonic() - began,
                "cost_sha256": digest(output / "cost-before-quality.json"),
                "compact_sha256": digest(output / "compact.json") if scored else None,
            }
            ordered_write(output / "operator-status.json", result)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--release", type=Path, required=True)
    parser.add_argument("--release-sha", required=True)
    args = parser.parse_args()
    source = Path(__file__).resolve().parents[1]
    release = load_release(args.release, args.release_sha, source)
    if (
        os.name != "posix"
        or not os.environ.get("SLURM_JOB_ID")
        or not os.environ.get("CUDA_VISIBLE_DEVICES")
    ):
        raise ValueError("allocated_gpu_required")
    work = Path(os.environ["CLIMATE_TARGETED_WORK"]).resolve()
    if source.parent != work or not work.name.startswith("climate-targeted-"):
        raise ValueError("private_scratch_required")
    if (
        release["source_archive_sha256"] != os.environ["CLIMATE_SOURCE_SHA256"]
        or release["source_git"] != os.environ["CLIMATE_SOURCE_GIT"]
    ):
        raise ValueError("source_environment_identity")
    with supervisor_signals():
        result = run_supervised(release, args.release, args.release_sha, source, work)
    if result["status"] != "completed":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
