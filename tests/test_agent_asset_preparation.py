import hashlib
import importlib.util
from pathlib import Path

import pytest

spec = importlib.util.spec_from_file_location(
    "prepare_agent_assets", Path(__file__).resolve().parents[1] / "scripts/prepare_agent_assets.py")
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


@pytest.mark.parametrize("name", ["../model.json", "/model.json", "folder/../../x", "C:\\model.json"])
def test_remote_filename_escape_rejected(name):
    with pytest.raises(ValueError):
        module.eligible({"rfilename": name})


def test_never_fetch_executable_remote_code():
    assert not module.eligible({"rfilename": "modeling_qwen.py"})
    assert module.eligible({"rfilename": "1_LogitScore/config.json"})


def test_small_blob_and_lfs_verified_before_publish(tmp_path):
    path = tmp_path / "weight.partial"
    path.write_bytes(b"test")
    digest = hashlib.sha256(b"test").hexdigest()
    assert module.checked_digest(path, {"size": 4, "lfs": {"sha256": digest}}) == digest
    blob = hashlib.sha1(b"blob 4\0test").hexdigest()
    assert module.checked_digest(path, {"size": 4, "blobId": blob}) == digest
    with pytest.raises(ValueError, match="digest mismatch"):
        module.checked_digest(path, {"size": 4, "lfs": {"sha256": "0" * 64}})
    with pytest.raises(ValueError, match="size mismatch"):
        module.checked_digest(path, {"size": 5, "blobId": blob})


def test_resumed_range_appends_only_matching_response(tmp_path, monkeypatch):
    from types import SimpleNamespace

    path = tmp_path / "weights.partial"
    path.write_bytes(b"pre")

    def curl(args, **kwargs):
        assert args[args.index("--range") + 1] == "3-5"
        Path(args[args.index("--output") + 1]).write_bytes(b"fix")
        return SimpleNamespace(stdout="206")

    monkeypatch.setattr(module.subprocess, "run", curl)
    module.download_ranges("https://example.invalid/fixture", path, 6)
    assert path.read_bytes() == b"prefix"
    assert not list(tmp_path.glob("*.chunk-*"))


def test_ignored_range_never_appends(tmp_path, monkeypatch):
    from types import SimpleNamespace

    path = tmp_path / "weights.partial"
    path.write_bytes(b"pre")

    def curl(args, **kwargs):
        Path(args[args.index("--output") + 1]).write_bytes(b"bad")
        return SimpleNamespace(stdout="200")

    monkeypatch.setattr(module.subprocess, "run", curl)
    with pytest.raises(ValueError, match="range response"):
        module.download_ranges("https://example.invalid/fixture", path, 6)
    assert path.read_bytes() == b"pre"
