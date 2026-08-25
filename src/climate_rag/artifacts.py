from __future__ import annotations

import hashlib
import os
import platform
import subprocess
import sys
from collections.abc import Mapping, Sequence
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

from .io import write_json, write_jsonl


def sha256_file(path: str | Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def git_sha(cwd: str | Path | None = None) -> str | None:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=cwd, text=True, stderr=subprocess.DEVNULL
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def build_manifest(
    *,
    command: str,
    arguments: Mapping[str, Any],
    inputs: Sequence[str | Path] = (),
    started_at: str | None = None,
    repository: str | Path | None = None,
) -> dict[str, Any]:
    finished_value = datetime.now(timezone.utc)
    if started_at:
        started_value = datetime.fromisoformat(started_at)
        if finished_value <= started_value:
            # Some Windows clocks expose coarser timestamp resolution than the
            # measured operation. Preserve strict manifest ordering deterministically.
            finished_value = started_value + timedelta(microseconds=1)
    finished = finished_value.isoformat()
    files: list[dict[str, Any]] = []
    for raw_path in inputs:
        path = Path(raw_path)
        if path.is_file():
            files.append(
                {
                    "path": str(path.resolve()),
                    "bytes": path.stat().st_size,
                    "sha256": sha256_file(path),
                }
            )
    return {
        "schema_version": 1,
        "command": command,
        "arguments": dict(arguments),
        "started_at_utc": started_at or finished,
        "finished_at_utc": finished,
        "git_sha": git_sha(repository),
        "inputs": files,
        "environment": {
            "python": sys.version,
            "platform": platform.platform(),
            "processor": platform.processor(),
            "slurm_job_id": os.getenv("SLURM_JOB_ID"),
            "cuda_visible_devices": os.getenv("CUDA_VISIBLE_DEVICES"),
        },
    }


def write_run_artifacts(
    output_dir: str | Path,
    *,
    command: str,
    arguments: Mapping[str, Any],
    metrics: Mapping[str, Any],
    started_at: str | None = None,
    inputs: Sequence[str | Path] = (),
    predictions: Sequence[Mapping[str, Any]] = (),
    error_cases: Sequence[Mapping[str, Any]] = (),
    notes: Sequence[str] = (),
    repository: str | Path | None = None,
) -> None:
    target = Path(output_dir)
    target.mkdir(parents=True, exist_ok=True)
    manifest = build_manifest(
        command=command,
        arguments=arguments,
        inputs=inputs,
        started_at=started_at,
        repository=repository,
    )
    write_json(target / "run_manifest.json", manifest)
    write_json(target / "metrics.json", dict(metrics))
    write_jsonl(target / "predictions.jsonl", predictions)
    write_jsonl(target / "error_cases.jsonl", error_cases)
    lines = [f"# Run report: {command}", "", "## Metrics", ""]
    lines.extend(f"- `{key}`: {value}" for key, value in sorted(metrics.items()))
    if notes:
        lines.extend(["", "## Notes", ""])
        lines.extend(f"- {note}" for note in notes)
    lines.extend(
        [
            "",
            "## Provenance",
            "",
            f"- Git commit: `{manifest['git_sha'] or 'unavailable'}`",
            f"- Finished (UTC): `{manifest['finished_at_utc']}`",
            "- See `run_manifest.json` for input hashes and environment metadata.",
            "",
        ]
    )
    (target / "report.md").write_text("\n".join(lines), encoding="utf-8")
