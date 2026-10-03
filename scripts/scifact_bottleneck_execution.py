"""Thin execution bindings around the already frozen bottleneck experiment."""
from __future__ import annotations

import json
import os
from pathlib import Path
import time
from typing import Any

from climate_rag.scifact_evidence_bottleneck import PROTOCOL, ROUTES, STAGES
from climate_rag.scifact_generation import frozen_contract
from climate_rag.scifact_natural_contract import require
from climate_rag.scifact_read_continuation import ordered_write
from climate_rag.scifact_semantic_contract import MODEL_SHA, TOKENIZER_SHA, checked
from climate_rag.scifact_utility_runtime import ledger_cost
from run_budget_agent_full_operator import ARCHIVES
from run_scifact_evidence_commit_operator import release_fields
from run_scifact_grounding_train_operator import ROOT, sha
from climate_rag.scifact_relation_verifier import RELATION_PROTOCOL
import scifact_evidence_input as adapter

EXECUTION = "bottleneck-supervised-execution-v1"
WRAPPER = "hpc/scifact_evidence_bottleneck.sbatch"
RESOURCE = {"gpu": "A100:1", "cpus": 8, "host_ram_gib": 32, "scratch_gib": 30, "slurm_seconds": 1800}
MODEL_RELATIVE = "input/models/generator/model"
LIMITS = {"max_operator_seconds": 1740, "max_prepare_seconds": 300,
          "max_worker_seconds": 1200, "scoring_reserve_seconds": 180}


def execution_fields(draft: Any) -> dict[str, Any]:
    run = ROOT / "runs" / (PROTOCOL + "-" + draft["source_git"][:12])
    fields = release_fields({"protocol": RELATION_PROTOCOL, "input_protocol": adapter.PROTOCOL})
    return {"execution_protocol": EXECUTION, "run_directory": run.as_posix(),
        "output": (run / "inference").as_posix(), "reports": (run / "reports").as_posix(),
        "model_relative_directory": MODEL_RELATIVE, "resource_cap": RESOURCE, **LIMITS,
        "model_archive_sha256": ARCHIVES["input"][1], "automatic_retry": False,
        "training_authorized": False, "protected_split_read": False,
        **{k: fields[k] for k in ("python_executable", "runtime_receipt_sha256", "runtime_files_sha256")}}


def validate_release(release: Any) -> None:
    # Authorization first: no GenerationConfig/Torch import, filesystem read or
    # exclusive reservation for an unapproved candidate.
    require(release.get("status") == "authorized_model_run"
            and release.get("model_execution_authorized") is True
            and release.get("authorization") == "coordinator_exact_hash_release",
            "new_exact_source_authorization_required")
    expected = execution_fields(release)
    expected.update(protocol=PROTOCOL, scope=adapter.SCOPE, routes=list(ROUTES),
        max_generations=96, planned_route_results=72, input_token_cap=8192,
        output_token_cap=512, per_stage_seconds=120, model_sha256=MODEL_SHA,
        generation_contract=frozen_contract(), tokenizer_sha256=TOKENIZER_SHA)
    fields = release_fields({"protocol": RELATION_PROTOCOL, "input_protocol": adapter.PROTOCOL})
    expected.update({k: fields[k] for k in adapter.HASH_KEYS})
    require(all(release.get(k) == v for k, v in expected.items()), "frozen_execution_contract")
    require("model_directory" not in release, "model_path_must_be_runtime_bound_not_invented")
    for key, size in (("source_git", 40), ("source_archive_sha256", 64), ("wrapper_sha256", 64)):
        require(isinstance(release.get(key), str) and len(release[key]) == size
                and set(release[key]) <= set("0123456789abcdef"), "release_hash:"+key)
    require(release["prepared"] == (ROOT / "runs/scifact-evidence-commit-prospective24-v1-20261002-confirmation-v1/prepared").as_posix(),
            "frozen_prepared_location")
    require(Path(release["source_archive"]).resolve().is_relative_to((ROOT / "envs").resolve()), "source_archive_scope")
    stage = ROOT / "envs" / ("evidence-bottleneck-"+release["source_git"][:12])
    require(release["source_archive"] == (stage / "source.tar").as_posix()
            and release["execution_release_file"] == (stage / "release.json").as_posix(), "exact_release_locations")


def bind_paths(release: Any, release_sha: str, work: Path) -> dict[str, Any]:
    return {"execution_protocol": EXECUTION, "release_sha256": release_sha,
        "source_git": release["source_git"], "scratch": str(work.resolve()),
        "model_directory": str((work / MODEL_RELATIVE).resolve()),
        "prepared": release["prepared"], "output": release["output"],
        "reports": release["reports"], "model_sha256": MODEL_SHA}


def trusted_work() -> Path:
    work = Path(os.environ["CLIMATE_BOTTLENECK_WORK"]).resolve()
    require(Path(__file__).resolve().parents[1].parent == work
            and work.name.startswith("climate-bottleneck-"), "executing_source_scratch_binding")
    return work


def check_paths(release: Any, release_sha: str, binding: Path, model: Path) -> dict[str, Any]:
    row = json.loads(binding.read_bytes())
    require(binding.resolve() == (Path(release["run_directory"]) / "runtime-paths.json").resolve(), "runtime_binding_location")
    require(row == bind_paths(release, release_sha, trusted_work())
            and model.resolve() == Path(row["model_directory"]).resolve(), "runtime_model_binding")
    require(model.resolve().is_relative_to(Path(row["scratch"]).resolve()), "model_escapes_scratch")
    require(model.is_dir() and not model.is_symlink(), "extracted_model_required")
    reservation = json.loads((binding.parent / "reserved.json").read_bytes())
    require(reservation["release_sha256"] == release_sha, "parent_release_binding")
    return dict(row)


def stage_watchdog(inference: Path) -> None:
    # Old slot_watchdog scans a different layout. Reuse the same reservation
    # semantics but adapt its path, without changing the model-facing receipt.
    for path in inference.glob("*/*/reserved.json"):
        require(path.parent.name in STAGES, "unexpected_stage_reservation")
        if (path.parent / "result.json").exists():
            continue
        try:
            stamp = json.loads(path.read_bytes())["started_unix"]
        except json.JSONDecodeError:
            stamp = path.stat().st_mtime
        if time.time() - stamp >= 120:
            raise TimeoutError("bottleneck_stage_deadline")


def failure_report(release: Any, reason: str, *, gold_read: Any = False) -> dict[str, Any]:
    return {"protocol": PROTOCOL, "scope": release["scope"], "status": "no_quality",
        "source_git": release["source_git"], "reason": reason, "gold_read": gold_read,
        "planned_route_results": 72, "planned_per_route": 24, "unscored_route_results": 72,
        "physical": ledger_cost(Path(release["output"]) / "ledger"), "Agent_gain_established": False}


def export_failure(release: Any, release_sha: str, reason: str) -> dict[str, Any]:
    reports = Path(release["reports"])
    reports.mkdir(mode=0o700, exist_ok=True)
    gold = (reports / "gold-read-started.json").exists()
    result = dict(failure_report(release, reason, gold_read=gold), release_sha256=release_sha)
    # Distinct name: never overwrite a partially completed scorer's report.
    ordered_write(reports / "execution-failure.json", result)
    ordered_write(Path(release["run_directory"]) / "compact.json", result)
    return result


def verify_exit(release: Any, release_sha: str, binding: Path, model: Path) -> Any:
    check_paths(release, release_sha, binding, model)
    proof = json.loads((Path(release["output"]) / "worker-exit.json").read_bytes())
    require(proof["child_reaped"] is True and proof["returncode"] == 0 and proof["interrupted"] is None
            and proof["release_sha256"] == release_sha
            and proof["runtime_paths_sha256"] == sha(binding), "supervised_exit_required")
    return proof


def load_release(path: Path, digest: str) -> Any:
    release = json.loads(checked(path, digest))
    validate_release(release)
    return release
