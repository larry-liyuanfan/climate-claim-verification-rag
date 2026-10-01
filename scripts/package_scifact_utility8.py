"""Local source freeze and actual utility8 wrapper guard checks; never sbatch."""
from __future__ import annotations

import argparse
import hashlib
from pathlib import Path
import subprocess
import tarfile
from typing import Any

from climate_rag.scifact_read_continuation import ordered_write
from package_scifact_source import package, run_shell_guard

WRAPPER = "hpc/scifact_utility8.sbatch"


def freeze(repo: Path, revision: str, output: Path, bash: str) -> dict[str, Any]:
    receipt = package(repo, revision, output, bash)
    archive = output / "source.tar"
    with tarfile.open(archive) as bundle:
        stream = bundle.extractfile(WRAPPER)
        if stream is None:
            raise ValueError("utility8_wrapper_missing")
        raw = stream.read()
    wrapper = output / "utility8.sbatch"
    with wrapper.open("xb") as stream:
        stream.write(raw)
    run_shell_guard(archive, revision, wrapper, bash)
    negatives = []
    bad = output / "wrong-wrapper.sbatch"
    with bad.open("xb") as stream:
        stream.write(raw + b"# deliberate synthetic mismatch\n")
    empty = output / "wrong-archive.tar"
    with tarfile.open(empty, "w"):
        pass
    for name, tar, commit, shell in (("revision", archive, "0" * 40, wrapper),
            ("wrapper", archive, revision, bad), ("archive", empty, revision, wrapper)):
        try:
            run_shell_guard(tar, commit, shell, bash)
        except subprocess.CalledProcessError:
            negatives.append(name)
        else:
            raise ValueError("utility8_negative_guard_accepted:" + name)
    result = {"source_git": revision, "source_archive_sha256": receipt["source_archive_sha256"],
        "source_archive_bytes": receipt["source_archive_bytes"], "wrapper_path": WRAPPER,
        "wrapper_sha256": hashlib.sha256(raw).hexdigest(), "actual_utility8_guard_passed": True,
        "rejected_mismatches": negatives, "legacy_guard_is_not_utility8_evidence": True,
        "job_submitted": False, "generator_calls": 0}
    ordered_write(output / "utility8-source-receipt.json", result)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--commit", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--bash", required=True)
    args = parser.parse_args()
    print(freeze(args.repo, args.commit, args.output, args.bash))


if __name__ == "__main__":
    main()
