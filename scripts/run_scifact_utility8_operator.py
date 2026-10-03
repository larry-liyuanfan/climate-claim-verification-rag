"""Exact-release GPU operator; DRAFT never executes; no submission or retry."""
from __future__ import annotations

import json
import os
from pathlib import Path
import signal
import sys
import time
from typing import Any

from climate_rag.scifact_read_continuation import ordered_write
from climate_rag.scifact_utility_contract import MAX_GENERATIONS, PROTOCOL
from prepare_scifact_utility8 import prepare
from run_budget_agent_full_operator import ARCHIVES, read_only_tree
from run_scifact_adapter_regression_operator import bounded_child, extract_models
from run_scifact_grounding_train_operator import DEPS, ROOT, import_observation, require, sha

OUTPUT = ROOT / "runs/scifact-utility8-20261001-v1"
RESOURCE = {"gpu": "A100:1", "cpus": 8, "host_ram_gib": 32, "scratch_gib": 30, "slurm_seconds": 3600}


def validate_release(release: dict[str, Any]) -> None:
    require(release["authorization"] == "coordinator_exact_hash_release", "draft_is_not_executable")
    require(release["protocol"] == PROTOCOL and release["output"] == str(OUTPUT)
            and release["resource_cap"] == RESOURCE and release["max_generator_calls"] == MAX_GENERATIONS
            and release["model_archive_sha256"] == ARCHIVES["input"][1]
            and release["warmup_generation_calls"] == 0 and release["max_rerank_requests"] == 16
            and release["training_authorized"] is False and release["protected_split_read"] is False
            and release["automatic_retry"] is False, "utility8_frozen_limits")
    for key in ("source_archive_sha256", "wrapper_sha256", "runtime_receipt_sha256", "runtime_files_sha256"):
        require(len(release[key]) == 64 and all(c in "0123456789abcdef" for c in release[key]), "release_hash_format")
    require(len(release["source_git"]) == 40, "exact_source_git")


def verify_runtime_receipt(release: dict[str, Any]) -> dict[str, Any]:
    # Reuse the audited immutable inventory receipt; do not rehash 11k cache files.
    require(sha(DEPS / "final-receipt.json") == release["runtime_receipt_sha256"], "runtime_receipt_hash")
    receipt = json.loads((DEPS / "final-receipt.json").read_bytes())
    require(receipt["status"] == "imports_verified" and not receipt["dependency_errors"]
            and receipt["stderr_empty"] is True, "runtime_receipt_status")
    require(sha(DEPS / "runtime-files.json") == release["runtime_files_sha256"] == receipt["runtime_files_sha256"],
            "runtime_inventory_receipt_hash")
    observed = import_observation(list(receipt["versions"]))
    require(observed["python_executable"] == release["python_executable"]
            and all(observed[k] == receipt[k] for k in ("python", "os_name", "torch", "versions", "module_files")),
            "actual_runtime_import_drift")
    return dict(observed, full_inventory_rehashed_this_run=False)


def main() -> None:
    started = time.monotonic()
    require(os.name == "posix" and bool(os.environ.get("SLURM_JOB_ID"))
            and bool(os.environ.get("CUDA_VISIBLE_DEVICES")), "allocated_gpu_only")
    def interrupted(number: int, frame: Any) -> None:
        raise InterruptedError(f"utility8_operator_signal_{number}")
    for number in (signal.SIGINT, signal.SIGTERM):
        signal.signal(number, interrupted)
    source = Path(__file__).resolve().parents[1]
    work = Path(os.environ["CLIMATE_GROUNDING_WORK"]).resolve()
    require(source.parent == work and work.name.startswith("climate-grounding-"), "private_scratch_source")
    release_path = Path(os.environ["CLIMATE_GROUNDING_RELEASE_FILE"])
    release_sha = os.environ["CLIMATE_GROUNDING_RELEASE_SHA"]
    require(sha(release_path) == release_sha, "release_identity")
    release = json.loads(release_path.read_bytes())
    validate_release(release)
    require((source / "SOURCE_REVISION").read_text().strip() == release["source_git"] == os.environ["CLIMATE_SOURCE_GIT"]
            and release["source_archive_sha256"] == os.environ["CLIMATE_SOURCE_SHA256"]
            and sha(source / "hpc/scifact_utility8.sbatch") == release["wrapper_sha256"], "source_wrapper_binding")
    OUTPUT.mkdir(mode=0o700)  # Exclusive lifetime reservation, never overwrite/restart.
    ordered_write(OUTPUT / "reserved.json", {"protocol": PROTOCOL, "job_id": os.environ["SLURM_JOB_ID"],
        "release_sha256": release_sha, "planned_claims": 8, "planned_slots": 24,
        "source_git": release["source_git"], "automatic_retry": False})
    prepared = prepare(OUTPUT / "prepared")
    ordered_write(OUTPUT / "runtime.json", verify_runtime_receipt(release))
    extract_models(ROOT / "envs" / ARCHIVES["input"][0], work / "input", release["model_archive_sha256"])
    read_only_tree(work / "input")
    read_only_tree(OUTPUT / "prepared/inference")
    command = [sys.executable, str(source / "scripts/run_scifact_utility8.py"),
        "--release", str(release_path), "--release-sha", release_sha,
        "--inference-dir", str(OUTPUT / "prepared/inference"), "--model-root", str(work / "input/models"),
        "--output", str(OUTPUT / "inference")]
    proof = bounded_child(command, OUTPUT / "worker.log", min(3200, 3540 - (time.monotonic() - started) - 120))
    proof.update(release_sha256=release_sha, preparation_sha256=sha(OUTPUT / "prepared/preparation.json"))
    ordered_write(OUTPUT / "worker-exit.json", proof)
    # Import and open official gold only after the inference child is fully reaped.
    require(proof["child_reaped"] is True and proof["parent_wait_interrupted"] is None, "no_score_before_exit")
    from score_scifact_utility8 import score_after_exit
    scored = score_after_exit(OUTPUT)
    quality = OUTPUT / "quality.json"
    ordered_write(OUTPUT / "complete.json", {"source_git": release["source_git"], "prepared": prepared,
        "inference_returncode": proof["returncode"], "status": scored.get("status", "scored"),
        "quality_sha256": sha(quality) if quality.exists() else None,
        "no_quality_sha256": sha(OUTPUT / "no-quality.json") if (OUTPUT / "no-quality.json").exists() else None,
        "cost_sha256": sha(OUTPUT / "cost-before-gold.json"), "elapsed_seconds": time.monotonic() - started,
        "further_execution_authorized": False})
    raise SystemExit(proof["returncode"] or 0)


if __name__ == "__main__":
    main()
