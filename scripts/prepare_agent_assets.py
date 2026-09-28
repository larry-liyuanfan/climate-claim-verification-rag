"""Fetch only revision-pinned public model files; never import or execute weights."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import subprocess
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path, PurePosixPath
from typing import Any

LOAD_SUFFIXES = {".json", ".safetensors", ".model", ".txt", ".jinja"}


def download_ranges(url: str, partial: Path, size: int) -> None:
    """Bounded 3-connection ranges; retain only the ordered resumable prefix."""
    offset = partial.stat().st_size if partial.exists() else 0
    if offset > size:
        raise ValueError("partial file is larger than the pinned source")
    chunk_bytes = 64 * 1024 * 1024

    def fetch(start: int) -> Path:
        end = min(start + chunk_bytes, size) - 1
        chunk = partial.with_name(partial.name + f".chunk-{start}")
        reply = subprocess.run([
            "curl", "-q", "--fail", "--location", "--silent", "--show-error",
            "--retry", "3", "--connect-timeout", "30", "--max-time", "180",
            "--range", f"{start}-{end}", "--output", str(chunk),
            "--write-out", "%{http_code}", url,
        ], check=True, capture_output=True, text=True)
        if reply.stdout != "206" or chunk.stat().st_size != end - start + 1:
            raise ValueError("range response does not match requested bytes")
        return chunk

    with ThreadPoolExecutor(max_workers=3) as pool:
        while offset < size:
            starts = list(range(offset, min(offset + 3 * chunk_bytes, size), chunk_bytes))
            chunks = list(pool.map(fetch, starts))
            with partial.open("ab") as destination:
                for chunk in chunks:
                    with chunk.open("rb") as source:
                        for block in iter(lambda: source.read(8 * 1024 * 1024), b""):
                            destination.write(block)
                    chunk.unlink()  # only this function's exact generated chunk
            offset = partial.stat().st_size
            print(json.dumps({"file": partial.name, "verified_length": offset,
                              "total_bytes": size, "sha256_check_pending": True}), flush=True)


def eligible(row: dict[str, Any]) -> bool:
    name = row["rfilename"]
    path = PurePosixPath(name)
    if path.is_absolute() or ".." in path.parts or "\\" in name or ":" in name:
        raise ValueError("unsafe remote filename")
    return path.suffix in LOAD_SUFFIXES or name in {"README.md", "LICENSE"}


def checked_digest(path: Path, row: dict[str, Any]) -> str:
    size = path.stat().st_size
    if size != row["size"]:
        raise ValueError(f"size mismatch: {path.name}")
    sha256 = hashlib.sha256()
    git_blob = hashlib.sha1(f"blob {size}\0".encode())
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            sha256.update(chunk)
            git_blob.update(chunk)
    if row.get("lfs"):
        if sha256.hexdigest() != row["lfs"]["sha256"]:
            raise ValueError(f"LFS digest mismatch: {path.name}")
    elif git_blob.hexdigest() != row["blobId"]:
        raise ValueError(f"Git blob digest mismatch: {path.name}")
    return sha256.hexdigest()


def prepare(root: Path, role: str, spec: dict[str, str]) -> dict[str, Any]:
    repo, revision = spec["repo"], spec["revision"]
    if not re.fullmatch(r"Qwen/[A-Za-z0-9.-]+", repo) or not re.fullmatch(r"[a-f0-9]{40}", revision):
        raise ValueError("only exact Qwen public revisions allowed")
    role_root = root / role
    model_root = role_root / "model"
    model_root.mkdir(parents=True, exist_ok=True)
    metadata_path = role_root / "source.json"
    if metadata_path.exists():
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    else:
        with urllib.request.urlopen(
            f"https://huggingface.co/api/models/{repo}/revision/{revision}?blobs=true", timeout=45
        ) as response:
            metadata = json.load(response)
        if metadata["sha"] != revision or metadata["id"] != repo:
            raise ValueError("source revision mismatch")
        metadata_path.write_bytes((json.dumps(metadata, indent=2) + "\n").encode())
    if metadata["sha"] != revision or metadata["id"] != repo:
        raise ValueError("existing source revision mismatch")
    files = [row for row in metadata["siblings"] if eligible(row)]
    manifest = {}
    for row in files:
        name = row["rfilename"]
        target = model_root / name
        target.parent.mkdir(parents=True, exist_ok=True)
        if not target.exists():
            partial = target.with_name(target.name + ".partial")
            print(json.dumps({"role": role, "file": name, "bytes": row["size"], "state": "fetching"}), flush=True)
            url = f"https://huggingface.co/{repo}/resolve/{revision}/{name}?download=true"
            if row["size"] >= 64 * 1024 * 1024:
                download_ranges(url, partial, row["size"])
            else:
                subprocess.run([
                    "curl", "-q", "--fail", "--location", "--silent", "--show-error",
                    "--retry", "3", "--connect-timeout", "30", "--max-time", "180",
                    "--output", str(partial), url,
                ], check=True)
            digest = checked_digest(partial, row)
            partial.rename(target)
        else:
            digest = checked_digest(target, row)
        if target.suffix in LOAD_SUFFIXES:
            manifest[name] = digest
    manifest_path = role_root / "model_manifest.json"
    manifest_path.write_bytes((json.dumps(manifest, sort_keys=True, indent=2) + "\n").encode())
    result = {"repo": repo, "revision": revision, "files": len(files),
              "loadable_files": len(manifest), "bytes": sum(row["size"] for row in files),
              "model_manifest_sha256": hashlib.sha256(manifest_path.read_bytes()).hexdigest(),
              "source_metadata_sha256": hashlib.sha256(metadata_path.read_bytes()).hexdigest(),
              "weights_executed": False, "download_authentication": "none"}
    print(json.dumps({"role": role, "state": "verified", **result}), flush=True)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--lock", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    lock = json.loads(args.lock.read_text(encoding="utf-8"))
    if set(lock["models"]) != {"generator", "reranker"}:
        raise ValueError("unexpected model matrix")
    result = {role: prepare(args.output_dir, role, spec) for role, spec in lock["models"].items()}
    (args.output_dir / "asset_report.json").write_bytes((json.dumps(result, indent=2) + "\n").encode())


if __name__ == "__main__":
    main()
