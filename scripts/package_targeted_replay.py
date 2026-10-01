"""Exact source package plus explicitly unauthorized draft. No remote submission."""

from __future__ import annotations

import argparse
from pathlib import Path

import package_scifact_source as source_package
from climate_rag.scifact_read_continuation import ordered_write
from run_targeted_replay_operator import WRAPPER, draft


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-git", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--bash", default="bash")
    args = parser.parse_args()
    repo = Path(__file__).resolve().parents[1]
    source_package.WRAPPER = WRAPPER
    result = source_package.package(repo, args.source_git, args.output, args.bash)
    release = draft(
        args.source_git,
        str(result["source_archive_sha256"]),
        str(result["wrapper_sha256"]),
    )
    ordered_write(args.output / "release.unauthorized.json", release)
    print("Frozen source and unauthorized draft; model execution remains disabled.")


if __name__ == "__main__":
    main()
