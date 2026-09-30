"""Freeze exact Git source and exercise the real shell guard; never submit a job."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import posixpath
import re
import shutil
import subprocess
import tarfile
from pathlib import Path

WRAPPER = "hpc/scifact_train_diagnostic.sbatch"
PACKAGER = "scripts/package_scifact_source.py"
MARKER = "SOURCE_REVISION"
BEGIN = "# BEGIN SOURCE_ARCHIVE_GUARD\n"
END = "# END SOURCE_ARCHIVE_GUARD\n"


def git(repo: Path, *arguments: str) -> bytes:
    return subprocess.run(
        ["git", "-C", str(repo), *arguments], check=True, capture_output=True,
    ).stdout


def exact_tree(repo: Path, revision: str) -> dict[str, tuple[int, bytes]]:
    if not re.fullmatch(r"[0-9a-f]{40}", revision):
        raise ValueError("exact 40-hex source commit required")
    if git(repo, "rev-parse", revision + "^{commit}").decode().strip() != revision:
        raise ValueError("source is not an exact commit")
    rows = git(repo, "ls-tree", "-rz", "--full-tree", revision).split(b"\0")
    entries: dict[str, tuple[int, bytes]] = {}
    for row in filter(None, rows):
        meta, raw_name = row.split(b"\t", 1)
        mode, kind, oid = meta.split()
        if kind != b"blob" or mode not in (b"100644", b"100755"):
            raise ValueError("source tree permits only regular Git blobs")
        name = raw_name.decode("utf-8")
        entries[name] = (int(mode, 8) & 0o777, git(repo, "cat-file", "blob", oid.decode()))
    if MARKER not in entries or entries[MARKER][1] != b"$Format:%H$\n":
        raise ValueError("tracked export-subst revision marker required")
    entries[MARKER] = (0o644, (revision + "\n").encode("ascii"))
    return entries


def validate_archive(repo: Path, revision: str, archive: Path) -> dict[str, object]:
    expected = exact_tree(repo, revision)
    directories = {
        "/".join(name.split("/")[:i])
        for name in expected for i in range(1, len(name.split("/")))
    }
    seen: set[str] = set()
    files: set[str] = set()
    actual_directories: set[str] = set()
    with tarfile.open(archive) as bundle:
        for member in bundle.getmembers():
            name = member.name[:-1] if member.isdir() and member.name.endswith("/") else member.name
            normalized = posixpath.normpath(name)
            if normalized in seen:
                raise ValueError("duplicate normalized source member")
            seen.add(normalized)
            if (not name or name != normalized or name.startswith("/")
                    or "\\" in name or ":" in name
                    or any(p in ("", ".", "..") for p in name.split("/"))):
                raise ValueError("non-canonical source member")
            if member.isdir():
                if name not in directories:
                    raise ValueError("directory outside exact Git allowlist")
                actual_directories.add(name)
                continue
            if not member.isfile():
                raise ValueError("source members must be regular files/directories")
            if name not in expected:
                raise ValueError("file outside exact Git allowlist")
            mode, raw = expected[name]
            stream = bundle.extractfile(member)
            if stream is None or stream.read() != raw or member.mode != mode:
                raise ValueError("source member differs from exact Git blob/mode")
            files.add(name)
    if files != set(expected) or actual_directories != directories:
        raise ValueError("incomplete exact Git allowlist")
    return {
        "source_git": revision, "source_archive_sha256": sha256(archive),
        "source_archive_bytes": archive.stat().st_size,
        "regular_files": len(files), "directories": len(directories),
        "source_revision_members": 1, "source_revision_bytes": 41,
        "normalized_members_unique": True, "complete_git_blob_allowlist": True,
        "wrapper_sha256": hashlib.sha256(expected[WRAPPER][1]).hexdigest(),
    }


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def shell_path(path: Path) -> str:
    resolved = path.resolve()
    if os.name == "nt":
        # GNU tar treats C:/... as a remote host. Git Bash needs /c/... instead.
        if not re.fullmatch(r"[A-Za-z]:", resolved.drive):
            raise ValueError("Git Bash packaging requires a local drive path")
        return "/" + resolved.drive[0].lower() + resolved.as_posix()[2:]
    return resolved.as_posix()


def run_shell_guard(archive: Path, revision: str, wrapper: Path, bash: str) -> None:
    text = wrapper.read_bytes().decode("utf-8")
    if text.count(BEGIN) != 1 or text.count(END) != 1 or "\r" in text:
        raise ValueError("unique LF shell guard boundaries required")
    guard = text.split(BEGIN)[1].split(END)[0]
    env = dict(os.environ)
    env.update(CLIMATE_SOURCE_TAR=shell_path(archive), CLIMATE_SOURCE_GIT=revision)
    subprocess.run(
        [bash, "-c", "set -euo pipefail\n" + guard, shell_path(wrapper)],
        env=env, check=True, capture_output=True,
    )


def require_clean_source(repo: Path, revision: str) -> None:
    if git(repo, "rev-parse", "HEAD").decode().strip() != revision:
        raise ValueError("package exact current HEAD only")
    if git(repo, "status", "--porcelain", "--untracked-files=all"):
        raise ValueError("clean tracked and nonignored untracked source required")
    git(repo, "ls-files", "--error-unmatch", PACKAGER)
    if Path(__file__).resolve() != (repo / PACKAGER).resolve():
        raise ValueError("run the tracked packager from the selected source checkout")


def package(repo: Path, revision: str, output: Path, bash: str) -> dict[str, object]:
    repo = repo.resolve()
    require_clean_source(repo, revision)
    output.mkdir(parents=True, exist_ok=False)
    archive = output / "source.tar"
    # export-subst expands the already tracked marker. Never append a second one.
    raw = git(repo, "-c", "core.autocrlf=false", "-c", "core.eol=lf",
              "-c", "tar.umask=0022", "archive", "--format=tar", revision)
    with archive.open("xb") as stream:
        stream.write(raw)
    receipt = validate_archive(repo, revision, archive)
    wrapper = output / "wrapper.sbatch"
    with tarfile.open(archive) as bundle:
        source = bundle.extractfile(WRAPPER)
        if source is None:
            raise ValueError("missing source wrapper")
        with wrapper.open("xb") as stream:
            stream.write(source.read())
    run_shell_guard(archive, revision, wrapper, bash)
    receipt.update(schema_version="scifact-source-package-v1", shell_guard_passed=True,
                   job_submitted=False, model_calls=0)
    with (output / "source-receipt.json").open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(receipt, stream, indent=2, sort_keys=True)
        stream.write("\n")
    return receipt


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--commit", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--bash", default=shutil.which("bash"))
    args = parser.parse_args()
    if not args.bash:
        parser.error("Bash with tar, grep, wc, sha256sum, cut and tr is required")
    print(json.dumps(package(args.repo, args.commit, args.output, args.bash), indent=2))


if __name__ == "__main__":
    main()
