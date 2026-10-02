"""CPU-only exact-source package plus UNAUTHORIZED standalone release."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from cloud_replay_contract import ENTRY, draft
from climate_rag.targeted_replay import INPUT_SHA
from package_scifact_source import git, require_clean_source, validate_archive


def package(repo: Path, revision: str, output: Path, *, root: str, run_id: str,
            input_sha: str = INPUT_SHA) -> dict[str, Any]:
    require_clean_source(repo, revision)
    output.mkdir(parents=True, exist_ok=False)
    archive = output / "source.tar"
    raw = git(repo, "-c", "core.autocrlf=false", "-c", "core.eol=lf",
              "-c", "tar.umask=0022", "archive", "--format=tar", revision)
    with archive.open("xb") as stream:
        stream.write(raw)
    receipt = validate_archive(repo, revision, archive, wrapper=ENTRY)
    value = draft(revision, str(receipt["source_archive_sha256"]), str(receipt["wrapper_sha256"]),
                  root=root, run_id=run_id, input_sha=input_sha)
    receipt.update(backend=value["backend"], entry=ENTRY, entry_sha256=value["entry_sha256"],
                   model_execution_authorized=False, runtime_status="unobserved")
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
    args = parser.parse_args()
    print(json.dumps(package(Path(__file__).resolve().parents[1], args.source_git, args.output,
                             root=args.root, run_id=args.run_id, input_sha=args.input_sha), indent=2))


if __name__ == "__main__":
    main()
