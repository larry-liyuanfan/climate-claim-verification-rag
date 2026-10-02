"""Linux-only independent wall-clock supervisor; no model or ledger imports."""
from __future__ import annotations

import ctypes
import json
import os
from pathlib import Path
import signal
import sys
import time
from typing import Any, Callable


def children(pid: int) -> list[int]:
    try:
        return [int(p) for p in Path(f"/proc/{pid}/task/{pid}/children").read_text().split()]
    except FileNotFoundError:
        return []


def descendants() -> list[tuple[int, str]]:
    pending, found = children(os.getpid()), []
    while pending:
        pid = pending.pop()
        try:
            # starttime, field22; comm can contain whitespace or parentheses.
            identity = Path(f"/proc/{pid}/stat").read_text().rsplit(")", 1)[1].split()[19]
        except FileNotFoundError:
            continue
        found.append((pid, identity))
        pending.extend(children(pid))
    return found


def signal_owned(sig: int) -> None:
    for pid, identity in descendants():
        try:
            if Path(f"/proc/{pid}/stat").read_text().rsplit(")", 1)[1].split()[19] == identity:
                os.kill(pid, sig)
        except (FileNotFoundError, ProcessLookupError):
            pass


def reap(statuses: dict[int, int]) -> None:
    while True:
        try:
            pid, status = os.waitpid(-1, os.WNOHANG)
        except ChildProcessError:
            return
        if not pid:
            return
        statuses[pid] = status


def write_receipt(path: Path, value: dict[str, Any]) -> None:
    pending = path.with_name(path.name + ".pending")
    with pending.open("x", encoding="utf-8") as stream:
        os.chmod(pending, 0o600)
        json.dump(value, stream, sort_keys=True)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    # Atomic no-overwrite publication only after durable file contents. A
    # .pending file is never a result. Acceptance ALSO requires outer exit0;
    # filesystem publication and process exit cannot be a single transaction.
    os.link(pending, path)


def supervise(operation: Callable[[], dict[str, Any]], receipt: Path, *,
              total_seconds: float = 7200, started: float | None = None,
              cleanup_seconds: float = 20, term_seconds: float = 3,
              receipt_seconds: float = 3,
              identity: dict[str, str] | None = None,
              receipt_writer: Callable[[Path, dict[str, Any]], None] = write_receipt) -> dict[str, Any]:
    """Only a fresh, single-threaded CLI parent may supervise this one run.

    The subreaper owns descendants even if model workers start new sessions.
    A kernel-uninterruptible process cannot be guaranteed killed: report any
    unreaped identities as failure, never as success. This is not a sandbox.
    """
    if not sys.platform.startswith("linux") or children(os.getpid()):
        raise ValueError("fresh_linux_supervisor_required")
    if not 0 < term_seconds + receipt_seconds < cleanup_seconds < total_seconds <= 7200:
        raise ValueError("invalid_total_deadline")
    start = time.monotonic() if started is None else started
    hard = start + total_seconds
    soft = hard - cleanup_seconds
    if time.monotonic() >= soft or receipt.exists() or receipt.with_name(receipt.name + ".pending").exists():
        raise ValueError("expired_or_reused_deadline")
    libc = ctypes.CDLL(None, use_errno=True)
    previous = ctypes.c_int()
    if libc.prctl(37, ctypes.byref(previous), 0, 0, 0) != 0 or libc.prctl(36, 1, 0, 0, 0) != 0:
        raise ValueError("linux_subreaper_unavailable")
    interrupted: list[int] = []
    old_handlers = {sig: signal.getsignal(sig) for sig in (signal.SIGTERM, signal.SIGINT)}
    for sig in old_handlers:
        signal.signal(sig, lambda signum, frame: interrupted.append(signum))
    statuses: dict[int, int] = {}
    pid = os.fork()
    if pid == 0:
        for sig in old_handlers:
            signal.signal(sig, signal.SIG_DFL)
        os.setsid()
        code = 1
        try:
            code = 0 if operation().get("status") == "completed" else 1
        except BaseException as exc:
            print(f"cloud operator failed: {type(exc).__name__}: {exc}", file=sys.stderr, flush=True)
        finally:
            sys.stdout.flush()
            sys.stderr.flush()
            os._exit(code)
    try:
        while time.monotonic() < soft and not interrupted:
            reap(statuses)
            if pid in statuses:
                break
            time.sleep(0.02)
        timed_out = time.monotonic() >= soft
        outstanding = descendants()
        needs_cleanup = timed_out or bool(interrupted) or bool(outstanding)
        if needs_cleanup:
            signal_owned(signal.SIGTERM)
            grace = min(time.monotonic() + term_seconds, hard - receipt_seconds - 1)
            while time.monotonic() < grace and descendants():
                reap(statuses)
                time.sleep(0.02)
            while time.monotonic() < hard - receipt_seconds - 0.2 and descendants():
                # Stop potential fork producers before killing, then rescan
                # newly adopted descendants rather than relying on one PGID.
                signal_owned(signal.SIGSTOP)
                signal_owned(signal.SIGKILL)
                reap(statuses)
                time.sleep(0.02)
        reap(statuses)
        left = descendants()
        exit_code = os.waitstatus_to_exitcode(statuses[pid]) if pid in statuses else None
        success = not needs_cleanup and not left and exit_code == 0
        result: dict[str, Any] = {
            "identity": identity or {},
            "status": "timeout_unscored" if timed_out else "completed" if success else "failed_unscored",
            "timed_out": timed_out,
            "operator_returncode": exit_code, "interrupts": interrupted,
            "all_children_reaped": not left, "unreaped": left,
            "compute_ceiling_seconds": total_seconds, "cleanup_inside_ceiling": True,
            "elapsed_before_receipt_seconds": time.monotonic() - start,
            "raw_cost_and_trace_files_preserved": True,
            "partial_compact_is_not_success": not success,
        }
        # Slow open/serialization/fsync cannot stall the outer watchdog.
        writer = os.fork()
        if writer == 0:
            try:
                receipt_writer(receipt, result)
            except BaseException:
                os._exit(1)
            os._exit(0)
        write_end = min(time.monotonic() + receipt_seconds, hard - 0.1)
        while time.monotonic() < write_end:
            reap(statuses)
            if writer in statuses:
                break
            time.sleep(0.01)
        if writer not in statuses:
            signal_owned(signal.SIGKILL)
            while time.monotonic() < hard and descendants():
                reap(statuses)
                time.sleep(0.005)
        reap(statuses)
        result["deadline_receipt_written"] = statuses.get(writer) == 0
        if not result["deadline_receipt_written"] or descendants():
            result["status"] = "failed_deadline_receipt_or_reaping"
            result["all_children_reaped"] = not descendants()
        if interrupted:
            result["status"] = "interrupted_unscored"
        return result
    finally:
        while time.monotonic() < hard and descendants():
            signal_owned(signal.SIGKILL)
            reap(statuses)
            time.sleep(0.005)
        for sig, handler in old_handlers.items():
            signal.signal(sig, handler)
        libc.prctl(36, previous.value, 0, 0, 0)
