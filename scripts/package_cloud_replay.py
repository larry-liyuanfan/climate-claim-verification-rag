"""CPU-only exact-source package plus UNAUTHORIZED standalone release."""
from __future__ import annotations

import argparse
import json
import hashlib
from pathlib import Path
from typing import Any

from cloud_replay_contract import ENTRY, draft
from climate_rag.targeted_replay import INPUT_SHA
from package_scifact_source import git, require_clean_source, validate_archive
from cloud_capacity import COMPLETE_HOST, RUNPOD_VISIBLE, provider_from_evidence
from cloud_private_storage import SINGLE_ROOT, PRIVATE_POSIX


def package(repo: Path, revision: str, output: Path, *, root: str, run_id: str,
            input_sha: str = INPUT_SHA, capacity_contract: str = COMPLETE_HOST,
            provider_evidence: Path | None = None, provider_evidence_sha: str | None = None,
            storage_contract: str = SINGLE_ROOT, private_root: str | None = None) -> dict[str, Any]:
    require_clean_source(repo, revision)
    allocation_bytes: bytes | None = None
    allocation_sha = None
    if capacity_contract == RUNPOD_VISIBLE:
        if provider_evidence is None or provider_evidence_sha is None:
            raise ValueError("provider_evidence_required")
        allocation_bytes = (json.dumps(provider_from_evidence(provider_evidence, provider_evidence_sha),
                                       indent=2, sort_keys=True) + "\n").encode()
        allocation_sha = hashlib.sha256(allocation_bytes).hexdigest()
    elif provider_evidence is not None or provider_evidence_sha is not None:
        raise ValueError("provider_evidence_requires_explicit_namespace_contract")
    output.mkdir(parents=True, exist_ok=False)
    archive = output / "source.tar"
    raw = git(repo, "-c", "core.autocrlf=false", "-c", "core.eol=lf",
              "-c", "tar.umask=0022", "archive", "--format=tar", revision)
    with archive.open("xb") as stream:
        stream.write(raw)
    receipt = validate_archive(repo, revision, archive, wrapper=ENTRY)
    value = draft(revision, str(receipt["source_archive_sha256"]), str(receipt["wrapper_sha256"]),
                  root=root, run_id=run_id, input_sha=input_sha, capacity_contract=capacity_contract,
                  provider_allocation_sha=allocation_sha, provider_evidence_sha=provider_evidence_sha,
                  storage_contract=storage_contract, private_root=private_root)
    receipt.update(backend=value["backend"], entry=ENTRY, entry_sha256=value["entry_sha256"],
                   model_execution_authorized=False, runtime_status="unobserved",
                   capacity_contract=capacity_contract, provider_allocation_sha256=allocation_sha,
                   provider_evidence_sha256=provider_evidence_sha,
                   storage_contract=storage_contract, private_root=private_root,
                   private_storage_status="unobserved")
    if allocation_bytes is not None:
        with (output / "provider-allocation.json").open("xb") as stream:
            stream.write(allocation_bytes)
    for name, payload in (("source-receipt.json", receipt), ("release.unauthorized.json", value)):
        with (output / name).open("x", encoding="utf-8", newline="\n") as stream:
            stream.write(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    return receipt


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-git", required=True)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--root", required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--input-sha", default=INPUT_SHA)
    parser.add_argument("--capacity-contract", choices=(COMPLETE_HOST, RUNPOD_VISIBLE), default=COMPLETE_HOST)
    parser.add_argument("--provider-evidence", type=Path)
    parser.add_argument("--provider-evidence-sha")
    parser.add_argument("--storage-contract", choices=(SINGLE_ROOT, PRIVATE_POSIX), default=SINGLE_ROOT)
    parser.add_argument("--private-root")
    args = parser.parse_args()
    print(json.dumps(package(Path(__file__).resolve().parents[1], args.source_git, args.output,
                             root=args.root, run_id=args.run_id, input_sha=args.input_sha,
                             capacity_contract=args.capacity_contract, provider_evidence=args.provider_evidence,
                             provider_evidence_sha=args.provider_evidence_sha,
                             storage_contract=args.storage_contract, private_root=args.private_root), indent=2))


if __name__ == "__main__":
    main()
