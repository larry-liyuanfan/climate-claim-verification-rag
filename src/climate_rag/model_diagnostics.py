"""Content-free public diagnostics; opt-in, bounded owner-only raw attachments."""
from __future__ import annotations

import hashlib
import os
import re
import stat
import uuid
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

PRIVATE_RESPONSE_LIMIT_BYTES = 32768
PRIVATE_TOTAL_LIMIT_BYTES = 1048576
SAFE_FIELDS = {"action", "reason", "evidence_assessment", "query", "label",
               "statements", "text", "evidence_id", "quote"}


def schema_error_locations(errors: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    """Never copy Pydantic input, message/context, or arbitrary extra field names."""
    return [{"loc": [x if isinstance(x, int) or x in SAFE_FIELDS else "<redacted>"
                     for x in error.get("loc", ())],
             "type": error["type"] if re.fullmatch(r"[a-z_]+", error["type"])
             else "validation_error"} for error in errors[:20]]


def validate_private_directory(root: Path) -> Path:
    if not root.is_absolute() or root.is_symlink() or not root.is_dir():
        raise ValueError("private response directory must be an existing absolute directory")
    resolved = root.resolve()
    if root != resolved:
        raise ValueError("private response directory must have no symlink/traversal components")
    if os.name == "posix":
        info = root.stat()
        if info.st_uid != getattr(os, "getuid")() or stat.S_IMODE(info.st_mode) & 0o077:
            raise ValueError("private response directory must be owner-only")
    return resolved


def response_diagnostics(
    raw: str, *, output_tokens: int, max_new_tokens: int,
    eos_observed: bool | None, generation_elapsed_ms: float,
    private_dir: Path | None = None,
) -> dict[str, Any]:
    payload = raw.encode("utf-8")
    result: dict[str, Any] = {
        "output_sha256": hashlib.sha256(payload).hexdigest(),
        "output_characters": len(raw), "output_bytes": len(payload),
        "output_tokens": output_tokens,
        "reached_max_new_tokens": output_tokens >= max_new_tokens,
        "eos_observed": eos_observed,
        "generation_elapsed_ms": generation_elapsed_ms,
        "private_attachment": "disabled",
    }
    if private_dir is not None:
        try:
            root = validate_private_directory(private_dir)
            existing = list(root.iterdir())
            if any(x.is_symlink() or not x.is_file() for x in existing):
                raise ValueError("private directory contains unexpected entries")
            used = sum(x.stat().st_size for x in existing)
            if len(payload) > PRIVATE_RESPONSE_LIMIT_BYTES:
                result["private_attachment"] = "skipped_per_response_limit"
            elif len(existing) >= 32 or used + len(payload) > PRIVATE_TOTAL_LIMIT_BYTES:
                result["private_attachment"] = "skipped_total_limit"
            else:
                # Exclusive creation; never append, overwrite, or follow a link.
                fd = os.open(root / (uuid.uuid4().hex + ".txt"),
                             os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
                with os.fdopen(fd, "wb") as stream:
                    stream.write(payload)
                result["private_attachment"] = "saved_owner_only"
        except (OSError, ValueError):
            # Telemetry/storage failure must not change a model decision or leak paths.
            result["private_attachment"] = "write_failed"
    return result
