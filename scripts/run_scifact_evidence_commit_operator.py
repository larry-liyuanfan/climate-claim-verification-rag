"""Exact-release operator. CPU-ready is not permission to submit this GPU run."""
from __future__ import annotations

import json
import os
from pathlib import Path
import sys
import time
from typing import Any

from climate_rag.scifact_evidence_commit import ASSEMBLER, PROTOCOL, ISOLATED_PROTOCOL, PROTOCOLS, ARMS, CALL_CAPS
from climate_rag.scifact_relation_verifier import RELATION_PROTOCOL
from climate_rag.scifact_mixed_launch import PYTHON_EXECUTABLE, RUNTIME_FILES_SHA, RUNTIME_RECEIPT_SHA
from climate_rag.scifact_read_continuation import ordered_write
from prepare_scifact_natural import SELECTION_SHA, prepare
from run_budget_agent_full_operator import ARCHIVES, read_only_tree
from run_scifact_component_operator import generator_only
from run_scifact_evidence_note_operator import bounded_worker
from run_scifact_grounding_train_operator import ROOT, require, sha
from run_scifact_natural_operator import slot_watchdog
from run_scifact_utility8_operator import verify_runtime_receipt
import scifact_evidence_input as inputs

ATTEMPT = "evidence-commit-v1"
OUTPUT = ROOT / "runs" / (PROTOCOL + "-" + ATTEMPT)
RESOURCE = {"gpu": "A100:1", "cpus": 8, "host_ram_gib": 32, "scratch_gib": 30, "slurm_seconds": 2400}
WRAPPER = "hpc/scifact_evidence_commit.sbatch"
RUNTIME_OBSERVATION_SHA = "6153e39d11dbf44b57b311359cf2b5846dcf6a45ef70cd9069f69e4c8c68d455"
FROZEN_FIELDS: dict[str, Any] = {
    "protocol": PROTOCOL, "attempt_id": ATTEMPT, "infrastructure_retry": 0,
    "assembler": ASSEMBLER, "comparison_job_id": "31930176",
    "comparison_quality_sha256": "fe982fc480769db8e9bf6eaa438c62b6ace8851085b787ca3857640606b7c4d1",
    "output": OUTPUT.as_posix(), "resource_cap": RESOURCE, "selection_sha256": SELECTION_SHA,
    "initial_inventory_sha256": "0c4b663184acabc0a4b0421f37f92182a5aea2ab414dcad7d3a5840e3b57c028",
    "arms": list(ARMS), "call_caps": CALL_CAPS,
    "fixed_empty_positive": "unresolved", "selection_order": "issued_ref_order_all_legal_subsets",
    "verify_prerequisite_not_spontaneous_demand": True,
    "max_generator_calls": 240, "planned_episodes": 72, "max_episode_seconds": 120,
    "max_episode_tools": 5, "max_episode_generations": 5, "max_worker_seconds": 1980,
    "warmup_generation_calls": 0, "training_authorized": False, "protected_split_read": False,
    "automatic_retry": False, "adapter_loaded": False, "reranker_loaded": False,
    "model_archive_sha256": ARCHIVES["input"][1], "python_executable": PYTHON_EXECUTABLE,
    "runtime_receipt_sha256": RUNTIME_RECEIPT_SHA, "runtime_files_sha256": RUNTIME_FILES_SHA,
    "runtime_observation_sha256": RUNTIME_OBSERVATION_SHA,
}
REQUIRED_RELEASE_KEYS = frozenset(FROZEN_FIELDS) | {
    "authorization", "source_git", "source_archive_sha256", "wrapper_sha256",
}


def release_fields(release: dict[str, Any]) -> dict[str, Any]:
    protocol = release.get("protocol", PROTOCOL)
    require(protocol in PROTOCOLS, "unknown_commit_protocol")
    fields = inputs.frozen_fields(FROZEN_FIELDS) if inputs.prospective(release) else FROZEN_FIELDS
    if protocol in (ISOLATED_PROTOCOL, RELATION_PROTOCOL):
        require(inputs.prospective(release), "isolated_protocol_requires_frozen_prospective_inputs")
        fields = dict(fields, protocol=protocol, attempt_id="semantic-input-v2",
            output=(ROOT / "runs" / (protocol + "-prospective24-v1")).as_posix(),
            comparison_job_id="31956320",
            comparison_quality_sha256="247fcdd764d5de4d39469786c1ef8b770e87a29fcf03033ab90fe47ed8d29eae",
            preparation_source_git="da243036871f61eef2e039a1618b8a3e1e1a00ac",
            preparation_source_archive_sha256="d4c2ef55a9fc8675df2aba3e1bf537c22fcbd71fc8ee0c8f1f54df0a1421dbce",
            decision_policy_delta="semantic_verifier_input_isolation_v2",
            selection_sha256="351523c9c44bb418e24ed17b40372e6bef7b87792f8ad0b78dcaeb2d22e22754",
            component_reservations_sha256="275bd306c9a154b583987ae63df148e3264c2989b671b92ac7b453fc9610a20f",
            preparation_sha256="23e1835a0e4f1330c2d11d0d70a2dbfa5c4655c2192169b05240433f5e5aae72",
            claims_sha256="bf5cd947e26ac2b8f83546106fbea673efbf71a84f7a4ea00adb31d40b7a5b7c",
            frames_sha256="e20a5188ba51bbaf940009338ee167d6d07bf4429652eaa06967265f37a4e29c",
            ordered_ids_sha256="41ef532f8c1a49caf81118164c615b5d5df6de04b22a7795baa563e37cad3469")
        if protocol == RELATION_PROTOCOL:
            fields.update(attempt_id="relation-rationale-v3",
                          comparison_protocol=ISOLATED_PROTOCOL,
                          comparison_status="v2_implemented_not_model_evaluated",
                          decision_policy_delta="single_generation_relation_rationale_v3")
    return dict(fields)


def validate_release(release: dict[str, Any]) -> None:
    fields = release_fields(release)
    missing = (frozenset(fields) | {"authorization", "source_git", "source_archive_sha256", "wrapper_sha256"}) - release.keys()
    require(not missing, "release_missing_fields:" + ",".join(sorted(missing)))
    require(release["authorization"] == "coordinator_exact_hash_release", "draft_is_not_executable")
    require(all(type(release[k]) is type(v) and release[k] == v for k, v in fields.items()),
            "evidence_commit_frozen_contract")
    if inputs.prospective(release):
        inputs.validate_hashes(release)
    for key, size in (("source_archive_sha256", 64), ("wrapper_sha256", 64), ("source_git", 40)):
        require(isinstance(release[key], str) and len(release[key]) == size
                and all(c in "0123456789abcdef" for c in release[key]), "release_hash:" + key)


def start_attempt(release: dict[str, Any], release_sha: str, job_id: str) -> dict[str, Any]:
    """Exercise the real consumer before data preparation or model loading."""
    validate_release(release)
    observed = verify_runtime_receipt(release)
    output = output_path(release)
    output.mkdir(mode=0o700)  # exclusive reservation, never reuse another run
    ordered_write(output / "reserved.json", {"protocol": release["protocol"], "attempt_id": release["attempt_id"],
        "infrastructure_retry": 0, "comparison_job_id": release["comparison_job_id"],
        "assembler": ASSEMBLER,
        "release_sha256": release_sha, "job_id": job_id, "source_git": release["source_git"],
        "planned_episodes": 72, "automatic_retry": False})
    ordered_write(output / "runtime.json", observed)
    return dict(observed)


def output_path(release: Any) -> Path:
    if release.get("protocol") in (ISOLATED_PROTOCOL, RELATION_PROTOCOL):
        return Path(release_fields(release)["output"])
    return Path(inputs.OUTPUT) if inputs.prospective(release) else OUTPUT


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
    output = output_path(release)
    prepared = (inputs.copy_prepared(output / "prepared", release) if inputs.prospective(release)
                else prepare(output / "prepared"))
    generator_only(ROOT / "envs" / ARCHIVES["input"][0], work / "input", release["model_archive_sha256"])
    read_only_tree(work / "input")
    read_only_tree(output / "prepared/inference")
    command = [sys.executable, str(source / "scripts/run_scifact_evidence_commit.py"),
        "--release", str(release_path), "--release-sha", release_sha,
        "--inference-dir", str(output / "prepared/inference"), "--model-root", str(work / "input/models"),
        "--output", str(output / "inference")]
    proof = bounded_worker(command, output, output / "inference",
        min(1980, 2400 - (time.monotonic() - began) - 420), watchdog=slot_watchdog)
    proof.update(release_sha256=release_sha, preparation_sha256=sha(output / "prepared/preparation.json"))
    ordered_write(output / "worker-exit.json", proof)
    require(proof["child_reaped"] is True, "no_scoring_before_reap")
    from score_scifact_evidence_commit import score_after_exit
    def load_tokenizer() -> Any:
        import transformers
        return getattr(transformers, "AutoTokenizer").from_pretrained(work / "input/models/generator/model", local_files_only=True)
    scored = score_after_exit(output, load_tokenizer, release, release_sha=release_sha)
    ordered_write(output / "complete.json", {"source_git": release["source_git"], "prepared": prepared,
        "status": scored["status"], "inference_returncode": proof["returncode"],
        "cost_sha256": sha(output / "cost-before-gold.json"),
        "quality_sha256": sha(output / "quality.json") if (output / "quality.json").exists() else None,
        "elapsed_seconds": time.monotonic() - began, "training_authorized": False,
        "attempt_id": release["attempt_id"], "infrastructure_retry": 0,
        "assembler": ASSEMBLER, "automatic_retry": False})
    raise SystemExit(proof["returncode"] or 0)


if __name__ == "__main__":
    main()
