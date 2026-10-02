"""Stdlib-only owner-private container storage, separate from persistent assets.

Checks real POSIX metadata, not a chmod return value. This is not a sandbox or
protection against root/the same UID; raw artifacts must be recovered privately
before stopping the container. No remote access or automatic export is provided.
"""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import stat
import tempfile
from typing import Any

SINGLE_ROOT = "single-root-v1"
PRIVATE_POSIX = "runpod-private-posix-v1"


def is_split(value: dict[str, Any]) -> bool:
    return value.get("storage_contract") == PRIVATE_POSIX


def checked_directory(path: Path) -> dict[str, Any]:
    if os.name != "posix" or not path.is_absolute() or path.resolve() != path:
        raise ValueError("private_posix_absolute_nonsymlink_required")
    info = path.lstat()
    if not stat.S_ISDIR(info.st_mode) or info.st_uid != os.getuid() or stat.S_IMODE(info.st_mode) != 0o700:
        raise ValueError("private_directory_uid_mode")
    return {"path": str(path), "uid": info.st_uid, "mode": "0700", "device": info.st_dev}


def checked_file(path: Path) -> None:
    if path.resolve() != path:
        raise ValueError("private_file_symlink")
    info = path.lstat()
    if (not stat.S_ISREG(info.st_mode) or info.st_uid != os.getuid()
            or stat.S_IMODE(info.st_mode) != 0o600 or info.st_nlink != 1):
        raise ValueError("private_file_uid_mode_or_links")


def check_ancestors(root: Path) -> None:
    for parent in root.parents:
        info = parent.lstat()
        # Root-owned sticky /tmp is an acceptable ancestor, never the private root.
        sticky_root = info.st_uid == 0 and bool(info.st_mode & stat.S_ISVTX)
        if (not stat.S_ISDIR(info.st_mode) or info.st_uid not in (0, os.getuid())
                or (stat.S_IMODE(info.st_mode) & 0o022 and not sticky_root)):
            raise ValueError("private_ancestor_not_trusted")


def private_mount(root: Path, persistent: Path) -> dict[str, Any]:
    if root.stat().st_dev == persistent.stat().st_dev:
        raise ValueError("private_and_persistent_filesystems_not_separate")
    matches = []
    for line in Path("/proc/self/mountinfo").read_text().splitlines():
        before, after = line.split(" - ", 1)
        raw = before.split()[4]
        for escaped, literal in (("\\040", " "), ("\\011", "\t"), ("\\134", "\\")):
            raw = raw.replace(escaped, literal)
        path = Path(raw)
        if root.is_relative_to(path):
            matches.append((len(path.parts), path, after.split()[0]))
        elif path.is_relative_to(root):
            raise ValueError("private_nested_mount")
    if not matches:
        raise ValueError("private_mount_unobserved")
    _, mount, filesystem = max(matches, key=lambda item: item[0])
    if mount != Path("/") or filesystem not in {"overlay", "ext4", "xfs", "btrfs"}:
        raise ValueError("private_container_posix_mount_required")
    return {"mount": str(mount), "filesystem": filesystem, "device": root.stat().st_dev,
            "ephemeral_on_container_stop": True}


def required_parents(value: dict[str, Any]) -> list[Path]:
    root = Path(value["private_root"])
    return [root, root / "runs", root / "scoring", root / "runtime", root / "scratch"]


def permission_probe(root: Path) -> None:
    """A few non-sensitive bytes; observe chmod, file mode and receipt hardlink."""
    directory = Path(tempfile.mkdtemp(prefix=".permission-probe-", dir=root))
    path, link = directory / "probe", directory / "published"
    try:
        os.chmod(directory, 0o700)
        checked_directory(directory)
        fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o600)
        try:
            os.fchmod(fd, 0o600)
            os.write(fd, b"non-sensitive permission probe\n")
            os.fsync(fd)
        finally:
            os.close(fd)
        checked_file(path)
        os.link(path, link)
        if link.lstat().st_ino != path.lstat().st_ino:
            raise ValueError("private_atomic_receipt_link")
    finally:
        for item in (link, path):
            if item.exists() or item.is_symlink():
                item.unlink()
        directory.rmdir()


def check_output_tree(output: Path) -> None:
    def unreadable(error: OSError) -> None:
        raise error

    for directory, dirs, files in os.walk(output, onerror=unreadable, followlinks=False):
        checked_directory(Path(directory))
        for name in dirs:
            checked_directory(Path(directory) / name)
        for name in files:
            checked_file(Path(directory) / name)


def observe_storage(value: dict[str, Any], *, probe: bool = False,
                    require_gold: bool = False, require_output: bool = False) -> dict[str, Any]:
    if not is_split(value):
        return {"storage_contract": SINGLE_ROOT}
    root = Path(value["private_root"])
    directories = [checked_directory(path) for path in required_parents(value)]
    check_ancestors(root)
    mount = private_mount(root, Path(value["root"]))
    if any(row["device"] != mount["device"] for row in directories):
        raise ValueError("private_parent_submount")
    if require_gold:
        checked_file(Path(value["gold_path"]))
    if require_output:
        checked_directory(Path(value["output"]))
        check_output_tree(Path(value["output"]))
        checked_directory(Path(value["private_scratch"]))
    if probe:
        for parent in required_parents(value):
            permission_probe(parent)
    return {"storage_contract": PRIVATE_POSIX, "private_root": str(root),
            "directories": directories, "mount": mount,
            "raw_recovery_required_before_stop": True,
            "protection_against_same_uid_or_root": False}


def verify_storage(value: dict[str, Any], *, require_gold: bool = True,
                   require_output: bool = False) -> None:
    if not is_split(value):
        return
    path = Path(value["storage_receipt"])
    checked_file(path)
    raw = path.read_bytes()
    if hashlib.sha256(raw).hexdigest() != value["storage_receipt_sha256"]:
        raise ValueError("private_storage_receipt_hash")
    receipt = json.loads(raw)
    if (receipt.get("scope") != "private_posix_preflight_not_execution_authority"
            or receipt.get("source_git") != value["source_git"]
            or receipt.get("run_id") != value["run_id"]
            or receipt.get("provider_allocation_sha256") != value["provider_allocation_sha256"]
            or receipt.get("probe_passed") is not True
            or receipt.get("configuration") != observe_storage(value, require_gold=require_gold,
                                                                require_output=require_output)):
        raise ValueError("private_storage_receipt_or_configuration")


def private_environment(value: dict[str, Any]) -> dict[str, str]:
    """Scratch must be created/checked before use; inherited by all child stages."""
    if not is_split(value):
        return {}
    path = Path(value["private_scratch"])
    checked_directory(path)
    return {"TMPDIR": str(path), "TMP": str(path), "TEMP": str(path),
            "HF_HOME": str(path / "hf-cache"), "XDG_CACHE_HOME": str(path / "cache"),
            "HF_HUB_CACHE": str(path / "hf-cache/hub"),
            "HUGGINGFACE_HUB_CACHE": str(path / "hf-cache/hub"),
            "HF_ASSETS_CACHE": str(path / "hf-cache/assets"),
            "HUGGINGFACE_ASSETS_CACHE": str(path / "hf-cache/assets"),
            "HF_DATASETS_CACHE": str(path / "hf-cache/datasets"),
            "HF_MODULES_CACHE": str(path / "hf-cache/modules"),
            "TRANSFORMERS_CACHE": str(path / "hf-cache/transformers"),
            "PYTORCH_TRANSFORMERS_CACHE": str(path / "hf-cache/transformers"),
            "PYTORCH_PRETRAINED_BERT_CACHE": str(path / "hf-cache/transformers"),
            "TORCH_HOME": str(path / "torch"), "TORCHINDUCTOR_CACHE_DIR": str(path / "inductor"),
            "TORCH_EXTENSIONS_DIR": str(path / "torch-extensions"),
            "TRITON_CACHE_DIR": str(path / "triton"), "CUDA_CACHE_PATH": str(path / "cuda")}


def configure_private_environment(value: dict[str, Any], *, preflight: bool = False) -> None:
    if is_split(value):
        os.umask(0o077)
        # Import-only runtime preflight must not consume the exclusive run ID.
        config = {**value, "private_scratch": str(Path(value["private_root"]) / "scratch")} if preflight else value
        os.environ.update(private_environment(config))
        # Earlier import-only runtime checks may have cached the default /tmp.
        tempfile.tempdir = None
