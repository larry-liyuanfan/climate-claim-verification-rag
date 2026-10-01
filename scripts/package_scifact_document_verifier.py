"""Freeze a new decoder implementation candidate; never submit or run models."""
from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path
import tarfile
from typing import Any

from climate_rag.scifact_read_continuation import ordered_write
from package_scifact_source import package, run_shell_guard, sha256
from run_scifact_document_verifier_operator import FROZEN_FIELDS, WRAPPER, validate_release


def build_release(revision: str, archive_sha: str, wrapper_sha: str,
                  runtime_receipt: Path, runtime_observation: Path) -> dict[str, Any]:
    # The original inventory receipt has no interpreter field. The successful
    # FIT24 runtime observation does; bind both instead of inventing a path.
    if (sha256(runtime_receipt) != FROZEN_FIELDS["runtime_receipt_sha256"]
            or sha256(runtime_observation) != FROZEN_FIELDS["runtime_observation_sha256"]):
        raise ValueError("accepted_runtime_receipts_required")
    receipt = json.loads(runtime_receipt.read_bytes())
    observed = json.loads(runtime_observation.read_bytes())
    if (receipt["status"] != "imports_verified" or receipt["dependency_errors"]
            or receipt["stderr_empty"] is not True
            or receipt["runtime_files_sha256"] != FROZEN_FIELDS["runtime_files_sha256"]
            or observed.get("python_executable") != FROZEN_FIELDS["python_executable"]
            or observed.get("model_loaded") is not False or observed.get("generation_calls") != 0
            or any(observed.get(k) != receipt[k] for k in ("python", "os_name", "torch", "versions", "module_files"))):
        raise ValueError("runtime_observation_contract")
    release = dict(copy.deepcopy(FROZEN_FIELDS), source_git=revision,
        source_archive_sha256=archive_sha, wrapper_sha256=wrapper_sha,
        python_executable=observed["python_executable"], authorization="coordinator_exact_hash_release")
    validate_release(release)
    release["authorization"] = "DRAFT_CPU_READY_NOT_AUTHORIZED"
    return release


def freeze(repo: Path, revision: str, output: Path, bash: str,
           runtime_receipt: Path, runtime_observation: Path) -> dict[str, Any]:
    receipt = package(repo, revision, output, bash)
    with tarfile.open(output / "source.tar") as archive:
        source = archive.extractfile(WRAPPER)
        if source is None:
            raise ValueError("document_wrapper_missing")
        raw = source.read()
    wrapper = output / "document-verifier.sbatch"
    with wrapper.open("xb") as stream:
        stream.write(raw)
    run_shell_guard(output / "source.tar", revision, wrapper, bash)
    release = build_release(revision, str(receipt["source_archive_sha256"]), sha256(wrapper),
                            runtime_receipt, runtime_observation)
    ordered_write(output / "release.draft.json", release)
    result = {"source_git": revision, "source_archive_sha256": receipt["source_archive_sha256"],
        "source_archive_bytes": receipt["source_archive_bytes"], "actual_wrapper": "document-verifier.sbatch",
        "wrapper_sha256": sha256(wrapper), "release_draft_sha256": sha256(output / "release.draft.json"),
        "attempt_id": release["attempt_id"], "output": release["output"],
        "shell_guard_passed": True, "runtime_observation_sha256": release["runtime_observation_sha256"],
        "job_submitted": False, "model_calls": 0, "training": False, "gold_read": False}
    ordered_write(output / "document-verifier-source-receipt.json", result)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("repo", "output", "runtime-receipt", "runtime-observation"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--commit", required=True)
    parser.add_argument("--bash", required=True)
    args = parser.parse_args()
    print(json.dumps(freeze(args.repo, args.commit, args.output, args.bash,
                           args.runtime_receipt, args.runtime_observation)))


if __name__ == "__main__":
    main()
