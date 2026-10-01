"""Exact-release operator. CPU-ready is not permission to submit this GPU run."""
from __future__ import annotations

import json
import os
from pathlib import Path
import sys
import time
from typing import Any

from climate_rag.scifact_document_verifier import DECODER_IMPLEMENTATION, PROTOCOL
from climate_rag.scifact_mixed_launch import PYTHON_EXECUTABLE, RUNTIME_FILES_SHA, RUNTIME_RECEIPT_SHA
from climate_rag.scifact_read_continuation import ordered_write
from prepare_scifact_natural import SELECTION_SHA, prepare
from run_budget_agent_full_operator import ARCHIVES, read_only_tree
from run_scifact_component_operator import generator_only
from run_scifact_evidence_note_operator import bounded_worker
from run_scifact_grounding_train_operator import ROOT, require, sha
from run_scifact_natural_operator import slot_watchdog
from run_scifact_utility8_operator import verify_runtime_receipt

ATTEMPT = "bounded-decoder-v1"
OUTPUT = ROOT / "runs" / (PROTOCOL + "-" + ATTEMPT)
RESOURCE = {"gpu": "A100:1", "cpus": 8, "host_ram_gib": 32, "scratch_gib": 30, "slurm_seconds": 7200}
WRAPPER = "hpc/scifact_document_verifier.sbatch"
RUNTIME_OBSERVATION_SHA = "6153e39d11dbf44b57b311359cf2b5846dcf6a45ef70cd9069f69e4c8c68d455"
FROZEN_FIELDS: dict[str, Any] = {
    "protocol": PROTOCOL, "attempt_id": ATTEMPT, "infrastructure_retry": 0,
    "decoder_implementation": DECODER_IMPLEMENTATION, "comparison_job_id": "31918065",
    "comparison_quality_sha256": "95c43ad2e30cd65b0193a6a8bc42a9c97e2dd5796169c6a97c95743ce08a58e9",
    "output": OUTPUT.as_posix(), "resource_cap": RESOURCE, "selection_sha256": SELECTION_SHA,
    "initial_inventory_sha256": "0c4b663184acabc0a4b0421f37f92182a5aea2ab414dcad7d3a5840e3b57c028",
    "max_generator_calls": 240, "planned_episodes": 48, "max_episode_seconds": 120,
    "max_episode_tools": 5, "max_episode_generations": 5, "max_worker_seconds": 6360,
    "warmup_generation_calls": 0, "training_authorized": False, "protected_split_read": False,
    "automatic_retry": False, "adapter_loaded": False, "reranker_loaded": False,
    "model_archive_sha256": ARCHIVES["input"][1], "python_executable": PYTHON_EXECUTABLE,
    "runtime_receipt_sha256": RUNTIME_RECEIPT_SHA, "runtime_files_sha256": RUNTIME_FILES_SHA,
    "runtime_observation_sha256": RUNTIME_OBSERVATION_SHA,
}
REQUIRED_RELEASE_KEYS = frozenset(FROZEN_FIELDS) | {
    "authorization", "source_git", "source_archive_sha256", "wrapper_sha256",
}


def validate_release(release: dict[str, Any]) -> None:
    missing = REQUIRED_RELEASE_KEYS - release.keys()
    require(not missing, "release_missing_fields:" + ",".join(sorted(missing)))
    require(release["authorization"] == "coordinator_exact_hash_release", "draft_is_not_executable")
    require(all(type(release[k]) is type(v) and release[k] == v for k, v in FROZEN_FIELDS.items()),
            "document_verifier_frozen_contract")
    for key, size in (("source_archive_sha256", 64), ("wrapper_sha256", 64), ("source_git", 40)):
        require(isinstance(release[key], str) and len(release[key]) == size
                and all(c in "0123456789abcdef" for c in release[key]), "release_hash:" + key)


def start_attempt(release: dict[str, Any], release_sha: str, job_id: str) -> dict[str, Any]:
    """Exercise the real consumer before data preparation or model loading."""
    validate_release(release)
    observed = verify_runtime_receipt(release)
    OUTPUT.mkdir(mode=0o700)  # new implementation comparison, never reuse r1/r2
    ordered_write(OUTPUT / "reserved.json", {"protocol": PROTOCOL, "attempt_id": ATTEMPT,
        "infrastructure_retry": 0, "comparison_job_id": release["comparison_job_id"],
        "decoder_implementation": DECODER_IMPLEMENTATION,
        "release_sha256": release_sha, "job_id": job_id, "source_git": release["source_git"],
        "planned_episodes": 48, "automatic_retry": False})
    ordered_write(OUTPUT / "runtime.json", observed)
    return observed


def main() -> None:
    began = time.monotonic()
    require(os.name == "posix" and bool(os.environ.get("SLURM_JOB_ID"))
            and bool(os.environ.get("CUDA_VISIBLE_DEVICES")), "allocated_gpu_only")
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
            and sha(source / WRAPPER) == release["wrapper_sha256"], "source_binding")
    start_attempt(release, release_sha, os.environ["SLURM_JOB_ID"])
    prepared = prepare(OUTPUT / "prepared")
    generator_only(ROOT / "envs" / ARCHIVES["input"][0], work / "input", release["model_archive_sha256"])
    read_only_tree(work / "input")
    read_only_tree(OUTPUT / "prepared/inference")
    command = [sys.executable, str(source / "scripts/run_scifact_document_verifier.py"),
        "--release", str(release_path), "--release-sha", release_sha,
        "--inference-dir", str(OUTPUT / "prepared/inference"), "--model-root", str(work / "input/models"),
        "--output", str(OUTPUT / "inference")]
    proof = bounded_worker(command, OUTPUT, OUTPUT / "inference",
        min(6360, 7200 - (time.monotonic() - began) - 420), watchdog=slot_watchdog)
    proof.update(release_sha256=release_sha, preparation_sha256=sha(OUTPUT / "prepared/preparation.json"))
    ordered_write(OUTPUT / "worker-exit.json", proof)
    require(proof["child_reaped"] is True, "no_scoring_before_reap")
    from score_scifact_document_verifier import score_after_exit
    def load_tokenizer() -> Any:
        import transformers
        return getattr(transformers, "AutoTokenizer").from_pretrained(work / "input/models/generator/model", local_files_only=True)
    scored = score_after_exit(OUTPUT, load_tokenizer, release)
    ordered_write(OUTPUT / "complete.json", {"source_git": release["source_git"], "prepared": prepared,
        "status": scored["status"], "inference_returncode": proof["returncode"],
        "cost_sha256": sha(OUTPUT / "cost-before-gold.json"),
        "quality_sha256": sha(OUTPUT / "quality.json") if (OUTPUT / "quality.json").exists() else None,
        "elapsed_seconds": time.monotonic() - began, "training_authorized": False,
        "attempt_id": ATTEMPT, "infrastructure_retry": 0,
        "decoder_implementation": DECODER_IMPLEMENTATION, "automatic_retry": False})
    raise SystemExit(proof["returncode"] or 0)


if __name__ == "__main__":
    main()
