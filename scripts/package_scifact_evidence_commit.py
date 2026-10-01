"""Freeze a new evidence-commit candidate; never submit or run models."""
from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path
import tarfile
from typing import Any

from climate_rag.scifact_evidence_commit import PROTOCOL, PROTOCOLS
from climate_rag.scifact_read_continuation import ordered_write
from package_scifact_source import package, run_shell_guard, sha256
from run_scifact_evidence_commit_operator import FROZEN_FIELDS, WRAPPER, release_fields, validate_release
import scifact_evidence_input as inputs


def build_release(revision: str, archive_sha: str, wrapper_sha: str,
                  runtime_receipt: Path, runtime_observation: Path,
                  input_compact: Path | None = None, *, protocol: str = PROTOCOL) -> dict[str, Any]:
    if protocol not in PROTOCOLS or (protocol != PROTOCOL and input_compact is None):
        raise ValueError("unsupported_protocol_or_missing_frozen_inputs")
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
    fields = copy.deepcopy(FROZEN_FIELDS)
    if input_compact is not None:
        prepared = json.loads(input_compact.read_bytes())
        isolated = protocol != PROTOCOL
        expected = release_fields({"protocol":protocol, "input_protocol":inputs.PROTOCOL})
        if (prepared['protocol'] != inputs.PROTOCOL or prepared['scope'] != inputs.SCOPE
                or prepared['source_git'] != (expected['preparation_source_git'] if isolated else revision)
                or prepared['source_archive_sha256'] != (expected['preparation_source_archive_sha256'] if isolated else archive_sha)
                or prepared['selected_claim_count'] != 24 or prepared['selected_component_count'] != 24
                or prepared['model_calls'] != 0 or prepared['official_gold_decoded'] is not False
                or prepared['protected_split_read'] is not False or prepared['future_gpu_authorized'] is not False
                or prepared['prompt_probe']['overflow_count'] != 0):
            raise ValueError('prospective_cpu_compact_contract')
        fields = expected
        fields.update({key:prepared[key] for key in inputs.HASH_KEYS})
    release = dict(fields, source_git=revision,
        source_archive_sha256=archive_sha, wrapper_sha256=wrapper_sha,
        python_executable=observed["python_executable"], authorization="coordinator_exact_hash_release")
    validate_release(release)
    release["authorization"] = "DRAFT_CPU_READY_NOT_AUTHORIZED"
    return release


def freeze(repo: Path, revision: str, output: Path, bash: str,
           runtime_receipt: Path, runtime_observation: Path,
           input_compact: Path | None = None, *, protocol: str = PROTOCOL) -> dict[str, Any]:
    receipt = package(repo, revision, output, bash)
    with tarfile.open(output / "source.tar") as archive:
        source = archive.extractfile(WRAPPER)
        if source is None:
            raise ValueError("evidence_commit_wrapper_missing")
        raw = source.read()
    wrapper = output / "evidence-commit.sbatch"
    with wrapper.open("xb") as stream:
        stream.write(raw)
    run_shell_guard(output / "source.tar", revision, wrapper, bash)
    release = build_release(revision, str(receipt["source_archive_sha256"]), sha256(wrapper),
                            runtime_receipt, runtime_observation, input_compact, protocol=protocol)
    ordered_write(output / "release.draft.json", release)
    result = {"source_git": revision, "source_archive_sha256": receipt["source_archive_sha256"],
        "source_archive_bytes": receipt["source_archive_bytes"], "actual_wrapper": "evidence-commit.sbatch",
        "wrapper_sha256": sha256(wrapper), "release_draft_sha256": sha256(output / "release.draft.json"),
        "attempt_id": release["attempt_id"], "output": release["output"],
        "shell_guard_passed": True, "runtime_observation_sha256": release["runtime_observation_sha256"],
        "job_submitted": False, "model_calls": 0, "training": False, "gold_read": False}
    ordered_write(output / "evidence-commit-source-receipt.json", result)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("repo", "output", "runtime-receipt", "runtime-observation"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--commit", required=True)
    parser.add_argument("--bash", required=True)
    parser.add_argument("--input-compact", type=Path)
    parser.add_argument("--protocol", choices=PROTOCOLS, default=PROTOCOL)
    args = parser.parse_args()
    print(json.dumps(freeze(args.repo, args.commit, args.output, args.bash,
                           args.runtime_receipt, args.runtime_observation, args.input_compact,
                           protocol=args.protocol)))


if __name__ == "__main__":
    main()
