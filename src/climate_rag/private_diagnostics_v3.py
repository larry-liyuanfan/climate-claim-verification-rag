"""Bounded, fail-closed private diagnostics for a serial offline provider.

Limits apply to the complete root, including earlier provider instances. This
is not a concurrent/shared-directory quota service. Receipts contain no text.
"""

from __future__ import annotations

import hashlib
import logging
import os
import uuid
from pathlib import Path
from typing import Any

from .model_diagnostics import validate_private_directory


class PrivateDiagnosticStore:
    def __init__(
        self, root: Path, *, max_files: int = 128, max_bytes: int = 1048576
    ) -> None:
        self.root = validate_private_directory(root)
        if max_files < 1 or max_bytes < 1:
            raise ValueError("private quota must be positive")
        self.max_files, self.max_bytes = max_files, max_bytes

    def sink(self, kind: str, limit: int) -> PrivateDiagnosticSink:
        if kind not in {"grammar", "response"} or limit < 1:
            raise ValueError("invalid private diagnostic sink")
        return PrivateDiagnosticSink(
            self, self.root / f"{uuid.uuid4().hex}-{kind}.txt", limit
        )

    def _usage(self) -> tuple[int, int]:
        validate_private_directory(self.root)
        files, size = 0, 0
        for item in self.root.rglob("*"):
            if item.is_symlink():
                raise ValueError("private tree must not contain symlinks")
            if item.is_file():
                files += 1
                size += item.stat().st_size
            elif not item.is_dir():
                raise ValueError("unexpected private entry")
        return files, size

    def append(self, path: Path, payload: bytes, *, created: bool) -> bytes:
        files, used = self._usage()
        if (not created and files >= self.max_files) or used >= self.max_bytes:
            return b""
        payload = payload[: self.max_bytes - used]
        if not payload:
            return b""
        flags = os.O_WRONLY | getattr(os, "O_NOFOLLOW", 0)
        flags |= os.O_APPEND if created else os.O_CREAT | os.O_EXCL
        fd = os.open(path, flags, 0o600)
        try:
            # os.write may be short. Only actual persisted bytes enter receipt.
            written = os.write(fd, payload)
            return payload[:written]
        finally:
            os.close(fd)


class PrivateDiagnosticSink:
    def __init__(self, store: PrivateDiagnosticStore, path: Path, limit: int) -> None:
        self.store, self.path, self.limit = store, path, limit
        self.attempted = self.stored = 0
        self.full_hash = hashlib.sha256()
        self.stored_hash = hashlib.sha256()
        self.io_failed = False

    def write(self, text: str) -> int:
        payload = text.encode("utf-8", errors="replace")
        already_truncated = self.attempted > self.stored
        self.attempted += len(payload)
        self.full_hash.update(payload)
        if self.io_failed or already_truncated:
            return len(text)
        try:
            saved = self.store.append(
                self.path,
                payload[: max(0, self.limit - self.stored)],
                created=self.stored > 0,
            )
            self.stored += len(saved)
            self.stored_hash.update(saved)
        except Exception:
            # No logging, stderr, path or exception-message fallback.
            self.io_failed = True
        return len(text)

    def receipt(self) -> dict[str, Any]:
        return {
            "attempted_bytes": self.attempted,
            "stored_bytes": self.stored,
            "dropped_bytes": self.attempted - self.stored,
            "truncated": self.attempted > self.stored,
            "sha256": self.full_hash.hexdigest(),
            "stored_prefix_sha256": self.stored_hash.hexdigest(),
            "io_failed": self.io_failed,
        }


class PrivateLogHandler(logging.Handler):
    """Never call logging.handleError: its default writes raw records to stderr."""

    def __init__(self, sink: PrivateDiagnosticSink) -> None:
        super().__init__(level=logging.ERROR)
        self.sink = sink

    def emit(self, record: logging.LogRecord) -> None:
        try:
            self.sink.write(self.format(record) + "\n")
        except Exception:
            self.sink.io_failed = True

    def handleError(self, record: logging.LogRecord) -> None:
        self.sink.io_failed = True
