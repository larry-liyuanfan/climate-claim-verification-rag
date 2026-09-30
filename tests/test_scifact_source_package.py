"""Synthetic Git/tar regressions: no real claims, gold, models or scheduling."""

import copy
import importlib.util
import io
import json
import os
import shutil
import subprocess
import sys
import tarfile
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("source_packager", ROOT / "scripts/package_scifact_source.py")
assert SPEC is not None and SPEC.loader is not None
PACK = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(PACK)


@pytest.fixture
def bash():
    if os.name == "nt":
        git = shutil.which("git")
        candidate = Path(git).resolve().parents[1] / "bin/bash.exe" if git else Path("missing")
        if candidate.is_file():
            return str(candidate)
    executable = shutil.which("bash")
    assert executable, "Bash is required: shell guard regressions must not silently skip"
    return executable


@pytest.fixture
def repo(tmp_path):
    PACK.git(tmp_path, "init")
    PACK.git(tmp_path, "config", "core.autocrlf", "true")
    files = {
        "SOURCE_REVISION": "$Format:%H$\n",
        ".gitattributes": "SOURCE_REVISION export-subst text eol=lf\n",
        ".gitignore": "data/\n",
        PACK.WRAPPER: (ROOT / PACK.WRAPPER).read_text(encoding="utf-8"),
        PACK.PACKAGER: (ROOT / PACK.PACKAGER).read_text(encoding="utf-8"),
    }
    for name, text in files.items():
        path = tmp_path / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding="utf-8", newline="\n")
    PACK.git(tmp_path, "add", "--", *files)
    PACK.git(tmp_path, "-c", "user.name=fixture", "-c", "user.email=fixture@example.invalid",
             "-c", "core.hooksPath=" + str(tmp_path / "absent-hooks"), "commit", "-m", "fixture")
    revision = PACK.git(tmp_path, "rev-parse", "HEAD").decode().strip()
    target = tmp_path / "data"
    target.mkdir()
    archive = target / "good.tar"
    archive.write_bytes(PACK.git(tmp_path, "-c", "core.autocrlf=false", "-c", "core.eol=lf",
                                "-c", "tar.umask=0022", "archive", "--format=tar", revision))
    return tmp_path, revision, archive


def mutate(archive, target, corruption):
    with tarfile.open(archive) as source:
        rows = [(copy.copy(m), source.extractfile(m).read() if m.isfile() else b"")
                for m in source.getmembers()]
    marker = next(row for row in rows if row[0].name == PACK.MARKER)
    if corruption.startswith("duplicate"):
        item, raw = copy.copy(marker[0]), marker[1]
        if corruption == "duplicate_40":
            raw = raw.rstrip(b"\n")
        rows.append((item, raw))
    elif corruption == "alias":
        item = copy.copy(marker[0])
        item.name = "./SOURCE_REVISION"
        rows.append((item, marker[1]))
    elif corruption in ("missing_marker", "missing_file", "missing_directory"):
        name = {"missing_marker": PACK.MARKER, "missing_file": ".gitignore",
                "missing_directory": "hpc"}[corruption]
        rows = [row for row in rows if row[0].name != name]
    elif corruption in ("crlf", "cr_only", "embedded_lf", "wrong_revision", "uppercase_revision"):
        raw = {"crlf": marker[1].replace(b"\n", b"\r\n"),
               "cr_only": marker[1].replace(b"\n", b"\r"),
               "embedded_lf": marker[1][:20] + b"\n" + marker[1][20:40],
               "wrong_revision": b"z" * 40 + b"\n",
               "uppercase_revision": marker[1].upper()}[corruption]
        rows = [(m, raw if m.name == PACK.MARKER else content) for m, content in rows]
    elif corruption == "mutated_blob":
        rows = [(m, b"tampered\n" if m.name == ".gitignore" else content) for m, content in rows]
    elif corruption == "wrong_mode":
        marker[0].mode = 0o755
    else:
        item = tarfile.TarInfo({"extra_file": "extra", "extra_directory": "extra/",
                               "absolute": "/escape", "parent": "../escape",
                               "backslash": "a\\b", "symlink": "link", "hardlink": "hard",
                               "fifo": "pipe"}[corruption])
        if corruption in ("symlink", "hardlink"):
            item.type = tarfile.SYMTYPE if corruption == "symlink" else tarfile.LNKTYPE
            item.linkname = PACK.MARKER
        elif corruption == "extra_directory":
            item.type = tarfile.DIRTYPE
        elif corruption == "fifo":
            item.type = tarfile.FIFOTYPE
        rows.append((item, b""))
    with tarfile.open(target, "w") as bundle:
        for item, raw in rows:
            item.size = len(raw) if item.isfile() else 0
            bundle.addfile(item, io.BytesIO(raw) if item.isfile() else None)


@pytest.mark.parametrize("corruption", [
    "duplicate_identical", "duplicate_40", "alias", "missing_marker", "missing_file",
    "missing_directory", "crlf", "cr_only", "embedded_lf", "wrong_revision", "uppercase_revision", "mutated_blob",
    "wrong_mode", "extra_file", "extra_directory", "absolute", "parent", "backslash",
    "symlink", "hardlink", "fifo",
])
def test_exact_archive_rejects_corruption(repo, corruption):
    root, revision, good = repo
    bad = good.with_name("bad.tar")
    mutate(good, bad, corruption)
    with pytest.raises(ValueError):
        PACK.validate_archive(root, revision, bad)


def test_good_archive_and_real_shell_guard(repo, bash):
    root, revision, archive = repo
    receipt = PACK.validate_archive(root, revision, archive)
    assert receipt["regular_files"] == 5
    assert receipt["source_revision_members"] == 1
    assert receipt["source_revision_bytes"] == 41
    PACK.run_shell_guard(archive, revision, root / PACK.WRAPPER, bash)


@pytest.mark.parametrize("corruption", [
    "duplicate_identical", "duplicate_40", "missing_marker", "crlf", "cr_only", "embedded_lf",
])
def test_real_shell_guard_rejects_marker_corruption(repo, bash, corruption):
    root, revision, good = repo
    bad = good.with_name("bad.tar")
    mutate(good, bad, corruption)
    with pytest.raises(subprocess.CalledProcessError):
        PACK.run_shell_guard(bad, revision, root / PACK.WRAPPER, bash)


def test_original_add_virtual_file_failure_reproduced(repo, bash):
    root, revision, good = repo
    bad = good.with_name("old-add-virtual.tar")
    bad.write_bytes(PACK.git(root, "-c", "core.autocrlf=false", "-c", "tar.umask=0022",
                            "archive", "--format=tar", "--add-virtual-file=SOURCE_REVISION:" + revision,
                            revision))
    env = dict(os.environ, CLIMATE_SOURCE_TAR=PACK.shell_path(bad), CLIMATE_SOURCE_GIT=revision)
    original = 'test "$(tar -xOf "$CLIMATE_SOURCE_TAR" SOURCE_REVISION | tr -d \'\\r\\n\')" = "$CLIMATE_SOURCE_GIT"'
    result = subprocess.run([bash, "-c", original], env=env, capture_output=True)
    assert result.returncode == 1
    with pytest.raises(ValueError, match="duplicate"):
        PACK.validate_archive(root, revision, bad)


def test_guard_checks_actual_wrapper_bytes(repo, bash):
    root, revision, archive = repo
    wrapper = root / PACK.WRAPPER
    wrapper.write_bytes(wrapper.read_bytes() + b"# changed wrapper\n")
    with pytest.raises(subprocess.CalledProcessError):
        PACK.run_shell_guard(archive, revision, wrapper, bash)


def test_tracked_cli_clean_checkout_reproducible_no_overwrite(repo, bash):
    root, revision, archive = repo
    common = [sys.executable, str(root / PACK.PACKAGER), "--commit", revision, "--bash", bash]
    receipts = []
    for name in ("package1", "package2"):
        output = archive.parent / name
        subprocess.run([*common, "--output", str(output)], check=True, capture_output=True)
        receipts.append(json.loads((output / "source-receipt.json").read_text()))
    assert receipts[0] == receipts[1]
    assert receipts[0]["shell_guard_passed"] and not receipts[0]["job_submitted"]
    repeat = subprocess.run([*common, "--output", str(output)], capture_output=True)
    assert repeat.returncode != 0
    (root / "new-untracked.py").write_text("pass\n")
    dirty = subprocess.run([*common, "--output", str(archive.parent / "no-output")], capture_output=True)
    assert dirty.returncode != 0 and not (archive.parent / "no-output").exists()


def test_attempt_directory_does_not_change_protocol_or_overwrite_r1(tmp_path):
    sys.path.insert(0, str(ROOT / "scripts"))
    try:
        from run_scifact_train_operator import ATTEMPT_ID, RELEASE, create_attempt_result
        original = tmp_path / "runs" / RELEASE
        original.mkdir(parents=True)
        (original / "preserved").write_bytes(b"original")
        env = {"CLIMATE_SCIFACT_TRAIN_RELEASE": RELEASE, "CLIMATE_SCIFACT_ATTEMPT_ID": ATTEMPT_ID,
               "CLIMATE_SCIFACT_INFRA_RETRY": "1"}
        result, identity = create_attempt_result(tmp_path, env)
        assert result == tmp_path / "runs" / ATTEMPT_ID
        assert identity == {"release_id": RELEASE, "attempt_id": ATTEMPT_ID, "infra_retry": 1}
        assert (original / "preserved").read_bytes() == b"original"
        with pytest.raises(FileExistsError):
            create_attempt_result(tmp_path, env)
        for key in env:
            with pytest.raises(ValueError, match="identity"):
                create_attempt_result(tmp_path, {**env, key: "changed"})
    finally:
        sys.path.pop(0)
