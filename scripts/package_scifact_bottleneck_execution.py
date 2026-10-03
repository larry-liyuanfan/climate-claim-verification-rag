"""Package one exact-source candidate. Does not authorize, upload or submit."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import tarfile

from build_scifact_bottleneck_draft import build, same_probe_implementation
from climate_rag.scifact_generation import frozen_contract
from climate_rag.scifact_read_continuation import ordered_write
from climate_rag.scifact_relation_verifier import RELATION_PROTOCOL
from climate_rag.scifact_semantic_contract import checked
from package_scifact_source import package, run_shell_guard, sha256
from run_scifact_evidence_commit_operator import release_fields
from run_scifact_grounding_train_operator import ROOT
from scifact_bottleneck_execution import WRAPPER, execution_fields
import scifact_evidence_input as adapter


def freeze(repo, revision, output, bash, preflight, preflight_sha):
    receipt = json.loads(checked(preflight, preflight_sha))
    same_probe_implementation(repo, receipt["source_git"], revision)
    archive = package(repo, revision, output, bash)
    with tarfile.open(output / "source.tar") as bundle:
        stream = bundle.extractfile(WRAPPER)
        if stream is None:
            raise ValueError("bottleneck_wrapper_missing")
        with (output / "bottleneck.sbatch").open("xb") as target:
            target.write(stream.read())
    run_shell_guard(output / "source.tar", revision, output / "bottleneck.sbatch", bash)
    fields = release_fields({"protocol": RELATION_PROTOCOL, "input_protocol": adapter.PROTOCOL})
    draft = build(receipt, fields, frozen_contract(), revision, archive["source_archive_sha256"], preflight_sha)
    del draft["model_directory"]
    del draft["model_directory_status"]
    draft.update(execution_fields(draft), wrapper_sha256=sha256(output / "bottleneck.sbatch"),
        authorization="DRAFT_NOT_AUTHORIZED",
        execution_release_file=(ROOT / "envs" / ("evidence-bottleneck-"+revision[:12]) / "release.json").as_posix(),
        release_requires="coordinator exact-source authorization; no inference performed by this package",
        scratch_binding="parent records actual extracted path; immutable release is never rewritten")
    ordered_write(output / "release.draft.json", draft)
    result = {"source_git": revision, "source_archive_sha256": archive["source_archive_sha256"],
        "source_archive_bytes": archive["source_archive_bytes"], "wrapper_sha256": draft["wrapper_sha256"],
        "release_draft_sha256": sha256(output / "release.draft.json"), "shell_guard_passed": True,
        "tokenizer_receipt_sha256": preflight_sha, "model_calls": 0, "job_submitted": False,
        "gold_read": False, "status": "draft_not_authorized"}
    ordered_write(output / "execution-source-receipt.json", result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("repo", "output", "preflight"):
        parser.add_argument("--"+name, type=Path, required=True)
    for name in ("commit", "bash", "preflight-sha"):
        parser.add_argument("--"+name, required=True)
    args = parser.parse_args()
    print(json.dumps(freeze(args.repo, args.commit, args.output, args.bash, args.preflight, args.preflight_sha)))


if __name__ == "__main__":
    main()
