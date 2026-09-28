import importlib.util
import json
import tarfile
import zipfile
from pathlib import Path

import pytest

spec = importlib.util.spec_from_file_location(
    "package_agent_overlay", Path(__file__).resolve().parents[1] / "scripts/package_agent_overlay.py")
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def wheel(root, name="dep", version="1.0", python=">=3.10", requires=""):
    path = root / f"{name}-{version}-py3-none-any.whl"
    metadata = f"Name: {name}\nVersion: {version}\nRequires-Python: {python}\n"
    if requires:
        metadata += f"Requires-Dist: {requires}\n"
    with zipfile.ZipFile(path, "w") as archive:
        archive.writestr(f"{name}-{version}.dist-info/METADATA", metadata)
    return path


def test_lock_has_hashes_and_packaging_does_not_claim_imports(tmp_path):
    wheel(tmp_path)
    output = tmp_path / "bundle.tar.gz"
    report = module.package(tmp_path, output)
    assert report["wheel_count"] == 1
    assert report["linux_imports_executed"] is False
    with tarfile.open(output) as archive:
        assert b"dep==1.0 --hash=sha256:" in archive.extractfile("requirements.lock").read()
        manifest = json.load(archive.extractfile("wheelhouse_manifest.json"))
        assert manifest["dependency_metadata_checked"] is True
    with pytest.raises(ValueError, match="overwrite"):
        module.package(tmp_path, output)


def test_python_specific_missing_dependency_rejected(tmp_path):
    wheel(tmp_path, requires='missing>=1; python_version < "3.11"')
    with pytest.raises(ValueError, match="Linux dependency"):
        module.package(tmp_path, tmp_path / "bundle.tar.gz")


def test_newer_python_wheel_cannot_silently_enter_overlay(tmp_path):
    wheel(tmp_path, python=">=3.11")
    with pytest.raises(ValueError, match="Python requirement"):
        module.package(tmp_path, tmp_path / "bundle.tar.gz")
