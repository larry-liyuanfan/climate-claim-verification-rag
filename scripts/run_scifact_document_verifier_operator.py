"""Exact-release operator. CPU-ready is not permission to submit this GPU run."""
from __future__ import annotations

import json
import os
from pathlib import Path
import sys
import time
from typing import Any

from climate_rag.scifact_document_verifier import PROTOCOL
from climate_rag.scifact_read_continuation import ordered_write
from prepare_scifact_natural import SELECTION_SHA, prepare
from run_budget_agent_full_operator import ARCHIVES, read_only_tree
from run_scifact_component_operator import generator_only
from run_scifact_evidence_note_operator import bounded_worker
from run_scifact_grounding_train_operator import ROOT, require, sha
from run_scifact_natural_operator import slot_watchdog
from run_scifact_utility8_operator import verify_runtime_receipt

OUTPUT = ROOT / "runs" / PROTOCOL
RESOURCE = {"gpu": "A100:1", "cpus": 8, "host_ram_gib": 32, "scratch_gib": 30, "slurm_seconds": 7200}
WRAPPER = "hpc/scifact_document_verifier.sbatch"


def validate_release(release: dict[str, Any]) -> None:
    require(release["authorization"] == "coordinator_exact_hash_release", "draft_is_not_executable")
    require(release["protocol"] == PROTOCOL and release["output"] == str(OUTPUT)
            and release["resource_cap"] == RESOURCE and release["selection_sha256"] == SELECTION_SHA
            and release["max_generator_calls"] == 240 and release["planned_episodes"] == 48
            and release["max_episode_seconds"] == 120 and release["max_episode_tools"] == 5
            and release["max_episode_generations"] == 5 and release["max_worker_seconds"] == 6360
            and release["warmup_generation_calls"] == 0 and release["training_authorized"] is False
            and release["protected_split_read"] is False and release["automatic_retry"] is False
            and release["adapter_loaded"] is False and release["reranker_loaded"] is False
            and release["model_archive_sha256"] == ARCHIVES["input"][1], "document_verifier_frozen_contract")
    for key in ("source_archive_sha256", "wrapper_sha256", "runtime_receipt_sha256",
                "runtime_files_sha256", "initial_inventory_sha256"):
        require(len(release[key]) == 64 and all(c in "0123456789abcdef" for c in release[key]), "release_hash")
    require(len(release["source_git"]) == 40 and all(c in "0123456789abcdef" for c in release["source_git"]), "exact_source")


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
    OUTPUT.mkdir(mode=0o700)  # one lifetime reservation; no retry/replacement
    ordered_write(OUTPUT / "reserved.json", {"protocol": PROTOCOL, "release_sha256": release_sha,
        "job_id": os.environ["SLURM_JOB_ID"], "source_git": release["source_git"], "planned_episodes": 48})
    prepared = prepare(OUTPUT / "prepared")
    ordered_write(OUTPUT / "runtime.json", verify_runtime_receipt(release))
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
        "elapsed_seconds": time.monotonic() - began, "training_authorized": False, "retry": False})
    raise SystemExit(proof["returncode"] or 0)


if __name__ == "__main__":
    main()
