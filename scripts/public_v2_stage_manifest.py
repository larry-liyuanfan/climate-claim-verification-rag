from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
from typing import Any


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def payload_summary(root: Path) -> dict[str, int | str]:
    manifest_path = root / "stage-manifest.json"
    aggregate = hashlib.sha256()
    file_count = 0
    total_bytes = 0
    for path in sorted(
        item for item in root.rglob("*") if item.is_file() and item != manifest_path
    ):
        relative = path.relative_to(root).as_posix()
        digest = file_sha256(path)
        size = path.stat().st_size
        aggregate.update(relative.encode("utf-8"))
        aggregate.update(b"\0")
        aggregate.update(str(size).encode("ascii"))
        aggregate.update(b"\0")
        aggregate.update(digest.encode("ascii"))
        aggregate.update(b"\n")
        file_count += 1
        total_bytes += size
    return {
        "file_count": file_count,
        "total_bytes": total_bytes,
        "payload_tree_sha256": aggregate.hexdigest(),
    }


def write_manifest(root: Path, stage_id: str) -> dict[str, Any]:
    manifest_path = root / "stage-manifest.json"
    if manifest_path.exists():
        raise FileExistsError(manifest_path)
    repository = Path(os.environ["REPO_DIR"])
    manifest: dict[str, Any] = {
        "schema_version": 1,
        "stage_id": stage_id,
        "git_commit": os.environ["CLIMATE_GIT_COMMIT"],
        "storage_config_sha256": file_sha256(
            repository / "configs" / "public_v2_storage.json"
        ),
        "slurm_job_id": os.environ["SLURM_JOB_ID"],
        **payload_summary(root),
    }
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return manifest


def verify_manifest(root: Path) -> dict[str, Any]:
    manifest_path = root / "stage-manifest.json"
    value = json.loads(manifest_path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError("stage manifest must be an object")
    actual = payload_summary(root)
    for key, actual_value in actual.items():
        if value.get(key) != actual_value:
            raise ValueError(f"stage archive {key} changed")
    return value


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Write or verify a stage manifest.")
    parser.add_argument("action", choices=("write", "verify"))
    parser.add_argument("root")
    parser.add_argument("--stage-id")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    root = Path(args.root)
    if args.action == "write":
        if not args.stage_id:
            raise ValueError("--stage-id is required for write")
        result = write_manifest(root, args.stage_id)
    else:
        result = verify_manifest(root)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
