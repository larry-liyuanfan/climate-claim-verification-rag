import hashlib
import logging
import os
import stat

import pytest

from climate_rag.private_diagnostics_v3 import PrivateDiagnosticStore, PrivateLogHandler


def test_quota_persists_across_instances_and_logs_are_prefix_bounded(tmp_path):
    first = PrivateDiagnosticStore(tmp_path, max_files=2, max_bytes=10)
    sink = first.sink("grammar", 7)
    sink.write("0123456789")
    sink.write("tail")
    assert sink.path.read_bytes() == b"0123456"
    assert sink.receipt()["sha256"] == hashlib.sha256(b"0123456789tail").hexdigest()
    second = PrivateDiagnosticStore(tmp_path, max_files=2, max_bytes=10)
    other = second.sink("response", 100)
    other.write("ABCDE")
    assert other.path.read_bytes() == b"ABC"
    exhausted = second.sink("response", 100)
    exhausted.write("still hashed")
    assert exhausted.receipt()["stored_bytes"] == 0
    assert not exhausted.path.exists()
    assert sum(p.stat().st_size for p in tmp_path.iterdir()) == 10
    if os.name == "posix":
        assert all(stat.S_IMODE(p.stat().st_mode) == 0o600 for p in tmp_path.iterdir())


def test_file_count_and_prior_nested_files_count_toward_root_quota(tmp_path):
    nested = tmp_path / "old-attempt"
    nested.mkdir(mode=0o700)
    (nested / "fixture.txt").write_bytes(b"old")
    store = PrivateDiagnosticStore(tmp_path, max_files=1, max_bytes=50)
    sink = store.sink("response", 50)
    sink.write("new")
    assert sink.receipt()["truncated"] and not sink.receipt()["io_failed"]
    assert not sink.path.exists()


def test_handler_format_failure_never_uses_default_stderr(
    tmp_path, capsys, monkeypatch
):
    handler = PrivateLogHandler(PrivateDiagnosticStore(tmp_path).sink("grammar", 100))

    class BrokenMessage:
        def __str__(self):
            raise OSError("private-format-failure")

    monkeypatch.setattr(logging, "raiseExceptions", True)
    record = logging.LogRecord(
        "fixture", logging.ERROR, "", 1, BrokenMessage(), (), None
    )
    handler.handle(record)
    assert handler.sink.io_failed
    assert not capsys.readouterr().err


@pytest.mark.skipif(os.name != "posix", reason="POSIX owner and symlink invariant")
def test_symlink_in_diagnostic_root_fails_closed(tmp_path):
    (tmp_path / "link").symlink_to(tmp_path / "missing")
    sink = PrivateDiagnosticStore(tmp_path).sink("grammar", 100)
    sink.write("must not escape")
    assert sink.io_failed and sink.stored == 0
    assert not sink.path.exists()
