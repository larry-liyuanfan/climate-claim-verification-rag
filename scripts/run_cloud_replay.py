"""Opt-in standalone Linux entry; no SSH, scheduler, purchase or automatic retry."""
# ruff: noqa: E402 -- bind exact local source before importing project modules
from __future__ import annotations

import sys

if __name__ == "__main__" and not sys.flags.isolated:
    raise SystemExit("isolated_entry_required: use absolute venv/python -IB absolute/entry.py")

import argparse
import importlib
import importlib.metadata
import json
import hashlib
import os
import platform
import shutil
import subprocess
import time
import runpy
from pathlib import Path
from typing import Any

# Always resolve source modules from this exact checkout/archive, not an editable
# install or caller PYTHONPATH. The source tree is verified before model access.
SOURCE = Path(__file__).resolve().parents[1]
BOOT_STARTED = time.monotonic()
sys.dont_write_bytecode = True
os.environ["PYTHONDONTWRITEBYTECODE"] = "1"


def early_supervised_run() -> None:
    """Stdlib parent: model/project imports and source checks stay in child."""
    parser = argparse.ArgumentParser()
    parser.add_argument("stage", choices=("run",))
    parser.add_argument("--release", type=Path, required=True)
    parser.add_argument("--release-sha", required=True)
    args = parser.parse_args()
    with args.release.open("rb") as stream:
        raw = stream.read(65537)
    if len(raw) > 65536 or hashlib.sha256(raw).hexdigest() != args.release_sha:
        raise ValueError("release_hash_or_size")
    value = json.loads(raw)
    if (value.get("authorization") != "standalone_exact_hash_release"
            or value.get("model_execution_authorized") is not True
            or value.get("backend") != "standalone-linux-v1"):
        raise ValueError("unauthorized_draft_no_model_or_preparation")
    private_storage = value.get("storage_contract") == "runpod-private-posix-v1"
    root = Path(value["private_root"] if private_storage else value["root"])
    receipt = Path(value["deadline_receipt"])
    run_id = value["run_id"]
    if (not root.is_absolute() or root.resolve() != root or len(root.parts) < 3
            or not isinstance(run_id, str) or not run_id or any(c not in "abcdefghijklmnopqrstuvwxyz0123456789-" for c in run_id)
            or receipt != root / "runtime" / (run_id + "-deadline.json") or receipt.resolve() != receipt
            or not receipt.parent.is_dir()):
        raise ValueError("deadline_receipt_path")
    if private_storage:
        guard = runpy.run_path(str(SOURCE / "scripts/cloud_private_storage.py"))
        guard["checked_directory"](root)
        guard["check_ancestors"](root)
        guard["checked_directory"](receipt.parent)
        os.umask(0o077)

    def child() -> dict[str, Any]:
        try:
            runpy.run_path(str(SOURCE / "scripts/run_cloud_replay.py"), run_name="__main__",
                           init_globals={"_CLOUD_SUPERVISED_CHILD": True})
        except SystemExit as exc:
            if exc.code not in (None, 0):
                raise
        return {"status": "completed"}

    # Read this stdlib-only helper explicitly as source, not via an importable
    # bytecode/native candidate. No project search path is installed in parent.
    watchdog = runpy.run_path(str(SOURCE / "scripts/cloud_deadline.py"))
    result = watchdog["supervise"](child, receipt, started=BOOT_STARTED,
                                   identity={"release_sha256": args.release_sha,
                                             "source_git": value["source_git"]})
    # Receipt AND observed exit0 are required; never block the watchdog on a
    # full stdout pipe. No automatic retry or quality promotion here.
    raise SystemExit(0 if result["status"] == "completed" else 1)


if (__name__ == "__main__" and len(sys.argv) > 1 and sys.argv[1] == "run"
        and not globals().get("_CLOUD_SUPERVISED_CHILD")):
    early_supervised_run()


def reject_project_binaries(source: Path) -> None:
    """Stdlib bootstrap: reject alternate executable bytes BEFORE project imports."""
    def unreadable(error: OSError) -> None:
        raise error

    for folder in ("src", "scripts"):
        base = source / folder
        if base.is_symlink():
            raise ValueError("project_symlink")
        for directory, dirs, files in os.walk(base, onerror=unreadable, followlinks=False):
            for name in dirs + files:
                path = Path(directory) / name
                if path.is_symlink():
                    raise ValueError("project_symlink")
                if path.suffix.lower() in {".pyc", ".pyo", ".so", ".pyd", ".dll", ".dylib"}:
                    raise ValueError("unbound_project_bytecode_or_extension:" + str(path))


if __name__ == "__main__":
    reject_project_binaries(SOURCE)
sys.path[:0] = [str(SOURCE / "src"), str(SOURCE / "scripts")]

from climate_rag.scifact_read_continuation import ordered_write
from cloud_replay_contract import (
    BACKEND, DEPENDENCIES, ENTRY, check_paths, digest, inspect_input, load_release,
    validate_release, validate_projection, verify_source, worker_projection,
)
from run_budget_agent_full_operator import read_only_tree
from run_scifact_evidence_commit_paired import cpu_stage, supervisor_signals
from run_targeted_replay_operator import run_supervised
from cloud_capacity import observe_capacity, check_provider_instance
from cloud_destination import destination_preflight, verify_destination, container_budget, GIB
from cloud_private_storage import (
    is_split, observe_storage, verify_storage, checked_file,
    configure_private_environment,
)


def imported_runtime(runtime_files: dict[str, str]) -> dict[str, Any]:
    """The inventory and actually imported modules must describe the same bytes."""
    modules = ("torch", "torch._C", "transformers", "transformers.generation.utils",
               "transformers.models.qwen3.modeling_qwen3", "lmformatenforcer",
               "lmformatenforcer.integrations.transformers", "interegular", "numpy",
               "pydantic", "pydantic_core", "tokenizers", "accelerate", "safetensors", "jsonschema")
    result = {}
    for name in modules:
        module = importlib.import_module(name)
        origin = getattr(module, "__file__", None)
        if origin is None:
            raise ValueError("runtime_module_origin_absent")
        path = str(Path(origin).resolve())
        if path not in runtime_files:
            raise ValueError("runtime_import_outside_inventory")
        result[name] = {"path": path, "sha256": runtime_files[path]}
    return result


def observe_runtime(value: dict[str, Any], instance_id: str) -> dict[str, Any]:
    """FUTURE owner-controlled GPU host only. No weights are loaded by this probe."""
    if platform.system() != "Linux" or os.environ.get("SLURM_JOB_ID"):
        raise ValueError("standalone_linux_not_slurm_required")
    if not instance_id.strip() or instance_id.lower().startswith(("synthetic", "fixture", "unknown")):
        raise ValueError("observed_instance_identity_required")
    if sys.executable != value["python_executable"] or platform.python_version_tuple()[:2] != ("3", "11"):
        raise ValueError("cloud_python_identity")
    storage = observe_storage(value)
    versions = {key: importlib.metadata.version(key) for key in DEPENDENCIES}
    if versions != DEPENDENCIES:
        raise ValueError("cloud_dependency_versions")
    for name, module in tuple(sys.modules.items()):
        if name == "climate_rag" or name.startswith("climate_rag."):
            origin = getattr(module, "__file__", None)
            if origin is None or not Path(origin).resolve().is_relative_to(Path(value["source"]) / "src"):
                raise ValueError("imported_source_origin")
    import torch

    if not torch.cuda.is_available() or torch.cuda.device_count() != 1 or torch.version.cuda != "12.6":
        raise ValueError("one_cuda126_gpu_required")
    props = torch.cuda.get_device_properties(0)
    if "A100" not in props.name or props.total_memory < 39 * 1024**3:
        raise ValueError("a100_capacity")
    capacity = observe_capacity(value)
    check_provider_instance(capacity["configuration"], instance_id)
    gpu = subprocess.run(["nvidia-smi", "--query-gpu=uuid,name,driver_version,memory.total",
                          "--format=csv,noheader,nounits"], capture_output=True, text=True,
                         timeout=15, check=True).stdout.strip()
    machine = Path("/etc/machine-id")
    if not machine.is_file():
        raise ValueError("machine_identity_unavailable")
    runtime_files = {}
    for package in DEPENDENCIES:
        distribution = importlib.metadata.distribution(package)
        for relative in distribution.files or []:
            if relative.suffix in {".py", ".so"}:
                file = Path(str(distribution.locate_file(relative))).resolve()
                runtime_files[str(file)] = digest(file)
    return {"backend": BACKEND, "status": "observed", "synthetic": False,
            "runtime_identity_schema": "stable-configuration-v2",
            "capacity_contract": value["capacity_contract"],
            "storage_configuration": storage,
            "provider_allocation_sha256": value["provider_allocation_sha256"],
            "provider_evidence_sha256": value["provider_evidence_sha256"],
            "instance_id": instance_id, "machine_id_sha256": digest(machine),
            "source_git": value["source_git"], "source_archive_sha256": value["source_archive_sha256"],
            "entry_sha256": value["entry_sha256"], "python_executable": sys.executable,
            "python_binary_sha256": digest(Path(sys.executable)), "python": platform.python_version(),
            "system": platform.system(), "kernel": platform.release(),
            "dependencies": versions, "runtime_files": runtime_files,
            "imported_runtime": imported_runtime(runtime_files),
            "cuda": torch.version.cuda, "gpu": gpu,
            "gpu_bytes": props.total_memory, "capacity_configuration": capacity["configuration"],
            "observations": {"capacity": capacity["observations"]},
            "weights_loaded": False}


def runtime_identity(receipt: dict[str, Any]) -> dict[str, Any]:
    if receipt.get("runtime_identity_schema") != "stable-configuration-v2":
        raise ValueError("runtime_identity_schema")
    return {key: value for key, value in receipt.items() if key != "observations"}


def verify_runtime(value: dict[str, Any]) -> dict[str, Any]:
    receipt_path = Path(value["runtime_receipt"])
    if digest(receipt_path) != value["runtime_receipt_sha256"]:
        raise ValueError("cloud_runtime_receipt_hash")
    receipt: dict[str, Any] = json.loads(receipt_path.read_bytes())
    if receipt.get("backend") != BACKEND or receipt.get("status") != "observed" or receipt.get("synthetic") is not False:
        raise ValueError("unobserved_or_legacy_runtime")
    observed = observe_runtime(value, receipt["instance_id"])
    if runtime_identity(receipt) != runtime_identity(observed):
        raise ValueError("runtime_changed_since_preflight")
    return observed


def prepare(value: dict[str, Any], work: Path) -> None:
    work.mkdir(mode=0o700)  # exclusive; never reuse failed or previous output
    if shutil.disk_usage(work).free < 35 * 1024**3:
        raise ValueError("scratch_capacity")
    inspect_input(Path(value["input_archive"]), value["input_transport_sha256"],
                  full=True, target=work / "input")
    read_only_tree(work / "input")


def projection_command(value: dict[str, Any], release_path: Path, release_sha: str,
                       stage: str) -> list[str]:
    if stage not in ("worker", "score"):
        raise ValueError("unknown_stage")
    path, fingerprint = release_path, release_sha
    if stage == "worker":
        path = Path(value["private_scratch"] if is_split(value) else value["work"]) / "worker-projection.json"
        ordered_write(path, worker_projection(value, release_sha))
        fingerprint = digest(path)
        ordered_write(Path(value["output"]) / "allocation/worker-projection-binding.json",
                      {"parent_release_sha256": release_sha, "projection_sha256": fingerprint})
    return [value["python_executable"], "-IB", str(Path(value["source"]) / ENTRY), stage,
            "--release", str(path), "--release-sha", fingerprint]


def launch(value: dict[str, Any], release_path: Path, release_sha: str) -> dict[str, Any]:
    validate_release(value)
    check_paths(value, new_run=True)
    source = Path(__file__).resolve().parents[1]
    def prelaunch() -> dict[str, Any]:
        verify_source(value, source)
        verify_storage(value)
        verify_destination(value)
        if is_split(value):
            os.umask(0o077)
            Path(value["private_scratch"]).mkdir(mode=0o700)
            configure_private_environment(value)
        return verify_runtime(value)

    bounded: Any = cpu_stage
    runtime = bounded(240, prelaunch)  # Outer process enforces total including final writes.
    # Gold is checksum-only at preflight; label parsing remains scorer-only.
    if not Path(value["gold_path"]).is_file() or not Path(value["input_archive"]).is_file():
        raise ValueError("needs_assets")
    for key in ("output", "work"):
        if not Path(value[key]).parent.is_dir():
            raise ValueError("dedicated_run_parent_required")
    os.environ.update(HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1", HF_DATASETS_OFFLINE="1",
                      TOKENIZERS_PARALLELISM="false", OMP_NUM_THREADS="8",
                      OPENBLAS_NUM_THREADS="8", MKL_NUM_THREADS="8", PYTHONNOUSERSITE="1",
                      HF_HOME=str(Path(value["work"]) / "hf-cache"),
                      PYTHONPATH=os.pathsep.join((str(source / "src"), str(source / "scripts"))))
    if is_split(value):
        configure_private_environment(value)
    with supervisor_signals():
        return run_supervised(value, release_path, release_sha, source, Path(value["work"]),
                              validate_fn=validate_release, prepare_fn=prepare,
                              stage_command=lambda stage: projection_command(value, release_path, release_sha, stage),
                              execution_identity={"backend": BACKEND, "instance_id": runtime["instance_id"],
                                                  "runtime_receipt_sha256": value["runtime_receipt_sha256"]})


def worker_stage(path: Path, expected_sha: str) -> None:
    if digest(path) != expected_sha:
        raise ValueError("worker_projection_hash")
    value = json.loads(path.read_bytes())
    if value.get("authorization") != "validated_worker_projection" or value.get("backend") != BACKEND:
        raise ValueError("worker_projection_authorization")
    if any("gold" in key for key in value):
        raise ValueError("worker_must_not_receive_gold")
    validate_projection(value, path, expected_sha)
    verify_source(value, Path(__file__).resolve().parents[1])
    verify_storage(value, require_gold=False, require_output=True)
    if is_split(value):
        configure_private_environment(value)
    verify_runtime(value)
    input_dir = Path(value["input"])
    projection_parent = Path(value["private_scratch"]) if is_split(value) else input_dir.parent
    if input_dir.resolve() != input_dir or path.resolve().parent != projection_parent:
        raise ValueError("worker_private_input_path")
    from run_targeted_replay import worker_body

    # Provider does not see release metadata, scoring paths or labels.
    worker_body(value, input_dir)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("storage-preflight", "asset-preflight", "runtime-preflight", "run", "worker", "score"))
    parser.add_argument("--release", required=True, type=Path)
    parser.add_argument("--release-sha", required=True)
    parser.add_argument("--instance-id")
    args = parser.parse_args()
    if args.stage == "worker":
        worker_stage(args.release, args.release_sha)
        return
    value = load_release(args.release, args.release_sha, execution=args.stage not in ("storage-preflight", "asset-preflight", "runtime-preflight"))
    if is_split(value):
        os.umask(0o077)
    if args.stage == "storage-preflight":
        if not is_split(value):
            raise ValueError("explicit_private_storage_contract_required")
        check_paths(value, new_run=True)
        verify_source(value, SOURCE)
        receipt = {"scope": "private_posix_preflight_not_execution_authority",
                   "source_git": value["source_git"], "run_id": value["run_id"],
                   "provider_allocation_sha256": value["provider_allocation_sha256"],
                   "configuration": observe_storage(value, probe=True), "probe_passed": True,
                   "container_budget_observation": container_budget(value, 15 * GIB),
                   "model_weights_loaded": False, "model_execution_authorized": False}
        path = Path(value["storage_receipt"])
        if path.exists():
            raise ValueError("storage_receipt_already_exists")
        ordered_write(path, receipt)
        checked_file(path)
        print(json.dumps({"storage_receipt_sha256": digest(path), "model_execution_authorized": False}))
    elif args.stage == "asset-preflight":
        receipt = destination_preflight(value, SOURCE)
        path = Path(value["asset_receipt"])
        if path.exists():
            raise ValueError("destination_receipt_already_exists")
        ordered_write(path, receipt)
        print(json.dumps({"asset_receipt_sha256": digest(path), "model_execution_authorized": False}))
    elif args.stage == "runtime-preflight":
        check_paths(value, new_run=True)
        verify_source(value, Path(__file__).resolve().parents[1])
        observe_storage(value)
        configure_private_environment(value, preflight=True)
        receipt = observe_runtime(value, args.instance_id or "")
        path = Path(value["runtime_receipt"])
        if path.exists():
            raise ValueError("runtime_receipt_already_exists")
        ordered_write(path, receipt)
        print(json.dumps({"runtime_receipt_sha256": digest(path), "model_execution_authorized": False}))
    elif args.stage == "run":
        if not globals().get("_CLOUD_SUPERVISED_CHILD"):
            raise ValueError("independent_supervisor_required")
        check_paths(value, new_run=True)
        result = launch(value, args.release, args.release_sha)
        print(json.dumps(result))
        if result["status"] != "completed":
            raise SystemExit(1)
    else:
        check_paths(value, new_run=False)
        verify_source(value, Path(__file__).resolve().parents[1])
        verify_storage(value, require_output=True)
        if is_split(value):
            configure_private_environment(value)
        verify_runtime(value)
        from run_targeted_replay import score_after_exit

        score_after_exit(value, Path(value["input"]), gold_path=Path(value["gold_path"]))


if __name__ == "__main__":
    main()
