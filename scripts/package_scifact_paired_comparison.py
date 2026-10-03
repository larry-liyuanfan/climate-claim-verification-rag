"""CPU-only exact-source packaging; drafts cannot execute without later authorization."""
from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path
import tarfile

from climate_rag.scifact_read_continuation import ordered_write
from package_scifact_source import package, run_shell_guard, sha256
from package_scifact_evidence_commit import build_release
from scifact_paired_comparison import VERSIONS, WRAPPER, pair_fields, child_extensions, validate_pair


def build_pair(revision, archive_sha, wrapper_sha, runtime_receipt, runtime_observation, input_compact):
    children = []
    for protocol in VERSIONS:
        child = build_release(revision, archive_sha, wrapper_sha, runtime_receipt,
                              runtime_observation, input_compact, protocol=protocol)
        child.update(child_extensions(protocol), authorization="coordinator_exact_hash_release")
        children.append(child)
    candidate = dict(pair_fields(children), authorization="coordinator_exact_hash_release")
    validate_pair(candidate)
    draft = copy.deepcopy(candidate)
    for row in [draft, *draft["children"]]:
        row["authorization"] = "DRAFT_CPU_READY_NOT_AUTHORIZED"
    return draft


def freeze(args):
    receipt = package(args.repo, args.commit, args.output, args.bash)
    with tarfile.open(args.output / "source.tar") as archive:
        member = archive.extractfile(WRAPPER)
        if member is None:
            raise ValueError("paired_wrapper_missing")
        raw = member.read()
    wrapper = args.output / "paired-comparison.sbatch"
    with wrapper.open("xb") as stream:
        stream.write(raw)
    run_shell_guard(args.output / "source.tar", args.commit, wrapper, args.bash)
    draft = build_pair(args.commit, receipt["source_archive_sha256"], sha256(wrapper),
                       args.runtime_receipt, args.runtime_observation, args.input_compact)
    ordered_write(args.output / "paired-release.draft.json", draft)
    for tag, child in zip(("v2","v3"), draft["children"], strict=True):
        ordered_write(args.output / f"release-{tag}.draft.json", child)
    result = {**receipt, "actual_wrapper": "paired-comparison.sbatch", "wrapper_sha256": sha256(wrapper),
              "paired_release_draft_sha256": sha256(args.output / "paired-release.draft.json"),
              "child_release_draft_sha256": {tag: sha256(args.output / f"release-{tag}.draft.json") for tag in ("v2","v3")},
              "cpu_preflight_reused": draft["cpu_preflight_job"], "generation_calls": 0,
              "new_slurm_jobs": 0, "protected_split_read": False, "status": "prepared_not_authorized"}
    ordered_write(args.output / "paired-source-receipt.json", result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("repo", "output", "runtime-receipt", "runtime-observation", "input-compact"):
        parser.add_argument("--"+name, type=Path, required=True)
    parser.add_argument("--commit", required=True)
    parser.add_argument("--bash", required=True)
    print(json.dumps(freeze(parser.parse_args())))


if __name__ == "__main__":
    main()
