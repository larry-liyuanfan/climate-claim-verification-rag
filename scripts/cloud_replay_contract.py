"""Explicit standalone release; CPU preparation never grants model execution."""
from __future__ import annotations

import hashlib
import json
import re
import tarfile
from pathlib import Path, PurePosixPath
from typing import Any, IO
from cloud_capacity import COMPLETE_HOST, RUNPOD_VISIBLE

from climate_rag import stop_acquire
from climate_rag.scifact_generation import frozen_contract
from climate_rag.targeted_replay import (
    CORPUS_SHA, GOLD_SHA, INPUT_SHA, MANIFEST_BYTES, OLD_PROTOCOL_SHA,
    MODEL_SHA, RERANKER_SHA, SELECTION_SHA, policy, sha,
)

BACKEND = "standalone-linux-v1"
ENTRY = "scripts/run_cloud_replay.py"
SELECTION = "docs/verified-runs/budget-agent-validation-selection-20260929.json"
DEPENDENCIES = {"torch": "2.7.1+cu126", "transformers": "4.51.3",
                "lm-format-enforcer": "0.11.3", "interegular": "0.3.3",
                "numpy": "1.26.4", "pydantic": "2.11.7",
                "pydantic-core": "2.33.2", "tokenizers": "0.21.4",
                "accelerate": "1.6.0", "safetensors": "0.5.3", "jsonschema": "4.23.0"}


def stream_sha(stream: IO[bytes]) -> str:
    result = hashlib.sha256()
    for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
        result.update(chunk)
    return result.hexdigest()


def digest(path: Path) -> str:
    with path.open("rb") as stream:
        return stream_sha(stream)


def hex_id(value: str, size: int = 64) -> None:
    if not isinstance(value, str) or not re.fullmatch(r"[0-9a-f]{" + str(size) + r"}", value):
        raise ValueError("invalid_identity")


def linux_path(value: str) -> PurePosixPath:
    path = PurePosixPath(value)
    if not path.is_absolute() or path.as_posix() != value or ".." in path.parts or "\\" in value or path == PurePosixPath("/"):
        raise ValueError("canonical_absolute_linux_path_required")
    return path


def draft(source_git: str, source_sha: str, entry_sha: str, *, root: str,
          run_id: str, input_sha: str = INPUT_SHA, capacity_contract: str = COMPLETE_HOST,
          provider_allocation_sha: str | None = None,
          provider_evidence_sha: str | None = None) -> dict[str, Any]:
    base = linux_path(root)
    if len(base.parts) < 3 or not re.fullmatch(r"[a-z0-9][a-z0-9-]{5,79}", run_id):
        raise ValueError("dedicated_root_and_unique_run_id_required")
    for value, length in ((source_git, 40), (source_sha, 64), (entry_sha, 64), (input_sha, 64)):
        hex_id(value, length)
    if capacity_contract == RUNPOD_VISIBLE:
        if provider_allocation_sha is None or provider_evidence_sha is None or not base.is_relative_to("/workspace"):
            raise ValueError("runpod_workspace_and_provider_binding_required")
        hex_id(provider_allocation_sha)
        hex_id(provider_evidence_sha)
    elif capacity_contract != COMPLETE_HOST or provider_allocation_sha is not None or provider_evidence_sha is not None:
        raise ValueError("capacity_contract_or_provider_binding")
    work = base / "work" / run_id
    return {
        "backend": BACKEND, "authorization": "none_draft", "model_execution_authorized": False,
        "source_git": source_git, "source_archive_sha256": source_sha, "entry_sha256": entry_sha,
        "root": root, "run_id": run_id,
        "source": str(base / "source" / source_git),
        "source_archive": str(base / "packages" / "source.tar"),
        "input_archive": str(base / "assets" / "input.tar"),
        "input_transport_sha256": input_sha,
        "work": str(work), "input": str(work / "input"),
        "output": str(base / "runs" / run_id),
        "gold_path": str(base / "scoring" / "validation-gold.json"),
        "gold_sha256": GOLD_SHA,
        "python_executable": str(base / "venv" / "bin" / "python"),
        "runtime_receipt": str(base / "runtime" / (run_id + ".json")),
        "runtime_receipt_sha256": None,
        "capacity_contract": capacity_contract,
        "provider_allocation_receipt": str(base / "runtime" / (run_id + "-provider-allocation.json")) if capacity_contract == RUNPOD_VISIBLE else None,
        "provider_allocation_sha256": provider_allocation_sha,
        "provider_evidence_sha256": provider_evidence_sha,
        "asset_receipt": str(base / "runtime" / (run_id + "-assets.json")),
        "asset_receipt_sha256": None,
        "deadline_receipt": str(base / "runtime" / (run_id + "-deadline.json")),
        "purpose": stop_acquire.PROTOCOL, "policy": policy(stop_acquire.PROTOCOL),
        "generation_contract": frozen_contract(),
        "runtime_requirements": {"system": "Linux", "python_minor": "3.11",
                                 "dependencies": DEPENDENCIES, "cuda": "12.6"},
        "resource_cap": {"gpus": 1, "gpu_family": "A100", "minimum_gpu_gib": 39,
                         "host_ram_gib": 64, "cpus": 8, "disk_gib": 100,
                         "compute_seconds_ceiling": 7200},
        "max_worker_seconds": 6000, "max_operator_seconds": 6900,
        "score_seconds": 300, "preparation_seconds": 600,
        "automatic_retry": False, "training_authorized": False, "protected_split_read": False,
    }


def validate_release(value: dict[str, Any], *, execution: bool = True) -> None:
    if value.get("backend") != BACKEND:
        raise ValueError("cross_backend_release")
    expected = draft(value["source_git"], value["source_archive_sha256"], value["entry_sha256"],
                     root=value["root"], run_id=value["run_id"], input_sha=value["input_transport_sha256"],
                     capacity_contract=value["capacity_contract"], provider_allocation_sha=value["provider_allocation_sha256"],
                     provider_evidence_sha=value["provider_evidence_sha256"])
    if execution:
        if value.get("authorization") != "standalone_exact_hash_release" or value.get("model_execution_authorized") is not True:
            raise ValueError("unauthorized_draft_no_model_or_preparation")
        hex_id(value["runtime_receipt_sha256"])
        hex_id(value["asset_receipt_sha256"])
        expected.update(authorization="standalone_exact_hash_release", model_execution_authorized=True,
                        runtime_receipt_sha256=value["runtime_receipt_sha256"],
                        asset_receipt_sha256=value["asset_receipt_sha256"])
    if value != expected:
        raise ValueError("frozen_cloud_release_changed")


def load_release(path: Path, expected_sha: str, *, execution: bool = True) -> dict[str, Any]:
    if digest(path) != expected_sha:
        raise ValueError("release_hash")
    value: dict[str, Any] = json.loads(path.read_bytes())
    validate_release(value, execution=execution)
    return value


def check_paths(value: dict[str, Any], *, new_run: bool) -> None:
    """Reject symlinks, including existing ancestors of not-yet-created children."""
    root = Path(value["root"])
    if value.get("provider_allocation_receipt") is not None:
        provider_path = Path(value["provider_allocation_receipt"])
        if provider_path.resolve() != provider_path or not provider_path.is_relative_to(root):
            raise ValueError("provider_path_escape_or_symlink")
    for key in ("root", "source", "source_archive", "input_archive", "work", "input", "output", "gold_path", "runtime_receipt", "asset_receipt", "deadline_receipt"):
        path = Path(value[key])
        if path.resolve() != path or not path.is_relative_to(root):
            raise ValueError("path_escape_or_symlink:" + key)
    # venv/bin/python is normally a symlink to the base interpreter. Its lexical
    # location is bound and its resolved executable identity is recorded at preflight.
    if Path(value["python_executable"]).parent.resolve() != Path(value["python_executable"]).parent:
        raise ValueError("environment_ancestor_symlink")
    if new_run and any(Path(value[key]).exists() for key in ("work", "output", "deadline_receipt")):
        raise ValueError("run_directory_already_exists")


def verify_source(value: dict[str, Any], source: Path) -> None:
    for folder in ("src", "scripts"):
        if any(p.suffix.lower() in {".pyc", ".pyo", ".so", ".pyd", ".dll", ".dylib"}
               for p in (source / folder).rglob("*")):
            raise ValueError("unbound_project_bytecode_or_extension")
    if source != Path(value["source"]) or digest(Path(value["source_archive"])) != value["source_archive_sha256"]:
        raise ValueError("source_location_or_archive_hash")
    names: set[str] = set()
    with tarfile.open(value["source_archive"]) as bundle:
        for member in bundle.getmembers():
            path = PurePosixPath(member.name)
            if path.is_absolute() or ".." in path.parts or "\\" in member.name:
                raise ValueError("source_member_path")
            if member.isdir():
                continue
            if not member.isfile() or member.name in names:
                raise ValueError("source_member_type_or_duplicate")
            names.add(member.name)
            target = source / member.name
            stream = bundle.extractfile(member)
            if stream is None or target.resolve() != target or not target.is_file() or digest(target) != stream_sha(stream):
                raise ValueError("source_member_drift")
    actual_python = {p.relative_to(source).as_posix() for folder in ("src", "scripts") for p in (source / folder).rglob("*.py")}
    if not actual_python <= names or (source / "SOURCE_REVISION").read_text().strip() != value["source_git"]:
        raise ValueError("source_revision_or_extra_code")
    if digest(source / ENTRY) != value["entry_sha256"] or digest(source / SELECTION) != SELECTION_SHA:
        raise ValueError("source_entry_or_selection_hash")


def input_members(bundle: tarfile.TarFile) -> dict[str, str]:
    """Only pinned inference bytes may be extracted; no gold or task sidecars."""
    members = bundle.getmembers()
    names: set[str] = set()
    for member in members:
        path = PurePosixPath(member.name)
        if (not member.isfile() or path.is_absolute() or path.as_posix() != member.name
                or ".." in path.parts or "\\" in member.name or member.name in names):
            raise ValueError("input_member_type_path_or_duplicate")
        names.add(member.name)
    wanted = {"evidence.jsonl": CORPUS_SHA, "validation-protocol.json": OLD_PROTOCOL_SHA}
    optional = {"authored-protocol.json", "models/asset_report.json"}
    optional.update(f"{kind}-{phase}.json" for kind in ("args", "execution") for phase in ("pilot", "validation", "vnext"))
    for key in ("generator", "reranker"):
        name = f"models/{key}/model_manifest.json"
        stream = bundle.extractfile(name)
        if stream is None:
            raise ValueError("missing_model_manifest")
        raw = stream.read()
        if sha(raw) != MANIFEST_BYTES[key]:
            raise ValueError("model_manifest_hash")
        manifest = json.loads(raw)
        wanted[name] = MANIFEST_BYTES[key]
        for filename, fingerprint in manifest.items():
            wanted[f"models/{key}/model/{filename}"] = fingerprint
        optional.update((f"models/{key}/source.json", f"models/{key}/model/README.md"))
    if not set(wanted) <= names or not names <= set(wanted) | optional:
        raise ValueError("missing_or_unapproved_input_member")
    return wanted


def inspect_input(archive: Path, expected_sha: str, *, full: bool = False,
                  target: Path | None = None) -> dict[str, Any]:
    """Read-only unless an explicit new extraction target is supplied; never loads weights."""
    if target is not None and (not full or target.exists()):
        raise ValueError("verified_new_input_target_required")
    if full and digest(archive) != expected_sha:
        raise ValueError("input_transport_hash")
    with tarfile.open(archive) as bundle:
        wanted = input_members(bundle)
        extracted_bytes = sum(bundle.getmember(name).size for name in wanted)
        if target is not None:
            target.mkdir(mode=0o700)
        for name, expected in wanted.items():
            member = bundle.getmember(name)
            if not full and member.size > 10 * 1024 * 1024:
                continue
            stream = bundle.extractfile(member)
            if stream is None:
                raise ValueError("missing_input_stream")
            result = hashlib.sha256()
            destination = target / name if target is not None else None
            if destination is not None:
                destination.parent.mkdir(parents=True, exist_ok=True)
            out = destination.open("xb") if destination is not None else None
            try:
                for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
                    result.update(chunk)
                    if out is not None:
                        out.write(chunk)
            finally:
                if out is not None:
                    out.close()
            if result.hexdigest() != expected:
                raise ValueError("input_logical_file_hash")
    return {"status": "verified" if full else "present_unverified_large_bytes",
            "bytes": archive.stat().st_size, "transport_sha256": expected_sha if full else None,
            "required_members": len(wanted), "gold_extracted": False,
            "extracted_bytes": extracted_bytes,
            "model_weights_loaded": False, "historical_transport_sha256": INPUT_SHA,
            "logical_identity": {"corpus": CORPUS_SHA, "protocol": OLD_PROTOCOL_SHA,
                                 "generator": MODEL_SHA, "reranker": RERANKER_SHA}}


def worker_projection(value: dict[str, Any], release_sha: str) -> dict[str, Any]:
    validate_release(value)
    keys = ("backend", "root", "run_id", "source_git", "source", "source_archive", "source_archive_sha256",
            "entry_sha256", "purpose", "generation_contract", "output", "input",
            "input_transport_sha256", "python_executable", "runtime_receipt", "runtime_receipt_sha256",
            "asset_receipt_sha256", "capacity_contract", "provider_allocation_receipt",
            "provider_allocation_sha256", "provider_evidence_sha256")
    return {**{key: value[key] for key in keys}, "parent_release_sha256": release_sha,
            "authorization": "validated_worker_projection"}


def validate_projection(value: dict[str, Any], path: Path, expected_sha: str) -> None:
    """Projection is not an authority token; bind it back to supervisor reservation."""
    full = draft(value["source_git"], value["source_archive_sha256"], value["entry_sha256"],
                 root=value["root"], run_id=value["run_id"], input_sha=value["input_transport_sha256"],
                 capacity_contract=value["capacity_contract"], provider_allocation_sha=value["provider_allocation_sha256"],
                 provider_evidence_sha=value["provider_evidence_sha256"])
    full.update(authorization="standalone_exact_hash_release", model_execution_authorized=True,
                runtime_receipt_sha256=value["runtime_receipt_sha256"],
                asset_receipt_sha256=value["asset_receipt_sha256"])
    hex_id(value["parent_release_sha256"])
    if value != worker_projection(full, value["parent_release_sha256"]):
        raise ValueError("frozen_worker_projection_changed")
    check_paths(full, new_run=False)
    if path.resolve() != Path(full["work"]) / "worker-projection.json":
        raise ValueError("worker_projection_path")
    output = Path(full["output"])
    reserved = json.loads((output / "reserved.json").read_bytes())
    binding = json.loads((output / "allocation/worker-projection-binding.json").read_bytes())
    if (reserved["release_sha256"] != value["parent_release_sha256"]
            or reserved["backend"] != BACKEND
            or binding != {"parent_release_sha256": value["parent_release_sha256"],
                           "projection_sha256": expected_sha}):
        raise ValueError("worker_projection_parent_binding")
