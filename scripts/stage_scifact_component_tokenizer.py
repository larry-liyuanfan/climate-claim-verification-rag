"""Reuse frozen tokenizer dependencies in a unique CPU directory; never install."""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import tarfile
from pathlib import Path, PurePosixPath
from typing import Any

ROOT = Path("/data/gpfs/projects/punim2936/portfolio_20260903/climate-public-retrieval-v2")
RUNTIME_SHA = "2423a755979c426ecae5d17deead154393de712efd10e84884d8c87ff00d29f8"
INPUT_SHA = "c4907db623b644c9c883a807f07597f87746595da7fc6435714452304e50a44d"
TOKENIZER = "data/qwen3-4b-tokenizer-v3/"
TOKENIZER_NAMES = {"merges.txt", "tokenizer.json", "tokenizer_config.json", "vocab.json"}
# Explicit top-level module/distribution families, never wildcard all site-packages.
PACKAGES = {"transformers", "tokenizers", "huggingface_hub", "tqdm", "safetensors", "fsspec",
            "packaging", "requests", "urllib3", "certifi", "charset_normalizer", "idna", "regex",
            "yaml", "_yaml", "pyyaml", "filelock", "numpy", "numpy.libs", "jinja2", "markupsafe",
            "typing_extensions"}


def require(ok: bool) -> None:
    if not ok:
        raise ValueError("tokenizer_stage_contract_failed")


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def safe_name(member: tarfile.TarInfo) -> str:
    # The frozen virtualenv tar uses one leading ./ and a root directory entry.
    if member.name in {".", "./"}:
        require(member.isdir())
        return ""
    name = member.name[2:] if member.name.startswith("./") else member.name
    name = name.rstrip("/") if member.isdir() else name
    p = PurePosixPath(name)
    require(bool(name) and not p.is_absolute() and str(p) == name and
            not any(part in {"", ".", ".."} for part in p.parts) and
            ":" not in name and "\\" not in name)
    return name


def selected_package(top: str) -> bool:
    if top.endswith(".py"):
        return top[:-3].lower() in PACKAGES
    if top.endswith(".dist-info"):
        return top.split("-", 1)[0].lower() in PACKAGES
    return top.lower() in PACKAGES


def unpack(path: Path, target: Path, kind: str) -> dict[str, int]:
    require(not target.exists())
    with tarfile.open(path) as archive:
        all_members = archive.getmembers()
        names = [safe_name(m) for m in all_members]
        require(len(set(names)) == len(names))
        chosen: list[tuple[tarfile.TarInfo, str]] = []
        for m, name in zip(all_members, names, strict=True):
            if m.isdir():
                continue
            if kind == "runtime":
                prefix = "lib/python3.10/site-packages/"
                if not name.startswith(prefix):
                    continue
                relative = name[len(prefix):]
                if not selected_package(relative.split("/", 1)[0]):
                    continue
            elif kind == "tokenizer":
                if not name.startswith(TOKENIZER) or name[len(TOKENIZER):] not in TOKENIZER_NAMES:
                    continue
                relative = name[len(TOKENIZER):]
            else:
                relative = name
            # Never follow/extract links. Nonselected virtualenv bin symlinks
            # are ignored; a selected link fails the package before any write.
            require(m.isfile())
            require(not relative.endswith((".safetensors", ".bin", ".pt", ".pth")))
            chosen.append((m, relative))
        require(bool(chosen) and sum(m.size for m, _ in chosen) <= 450_000_000)
        if kind == "tokenizer":
            require({r for _, r in chosen} == TOKENIZER_NAMES)
        target.mkdir(mode=0o700)
        for m, relative in chosen:
            destination = target / relative
            require(destination.resolve().is_relative_to(target.resolve()))
            destination.parent.mkdir(parents=True, exist_ok=True)
            handle = archive.extractfile(m)
            require(handle is not None)
            assert handle is not None
            with destination.open("xb") as output:
                output.write(handle.read())
    return {"files": len(chosen), "bytes": sum(m.size for m, _ in chosen)}


def stage(source: Path, expected_sha: str, git: str) -> dict[str, Any]:
    require(bool(re.fullmatch("[0-9a-f]{40}", git)) and digest(source) == expected_sha)
    runtime = ROOT / "envs" / ("runtime-py310-" + RUNTIME_SHA + ".tar.gz")
    inputs = ROOT / "envs/scifact-semantic-inputs-c4907db6.tar"
    require(digest(runtime) == RUNTIME_SHA and digest(inputs) == INPUT_SHA)
    output = ROOT / "envs" / ("scifact-component-cpu-" + git[:12])
    output.mkdir(mode=0o700)
    stats = {"source": unpack(source, output / "source", "source"),
             "runtime": unpack(runtime, output / "tokenizer-site", "runtime"),
             "tokenizer": unpack(inputs, output / "tokenizer", "tokenizer")}
    require((output / "source/SOURCE_REVISION").read_text().strip() == git)
    require((output / "source/scripts/stage_scifact_component_tokenizer.py").read_bytes() == Path(__file__).read_bytes())
    result = {"source_git": git, "source_archive_sha256": expected_sha, "runtime_sha256": RUNTIME_SHA,
              "input_archive_sha256": INPUT_SHA, "unpacked": stats, "pip_or_download": False,
              "torch_or_weights_extracted": False, "model_calls": 0}
    with (output / "stage-receipt.json").open("x") as stream:
        json.dump(result, stream, sort_keys=True, indent=2)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-tar", type=Path, required=True)
    parser.add_argument("--source-sha", required=True)
    parser.add_argument("--source-git", required=True)
    args = parser.parse_args()
    print(json.dumps(stage(args.source_tar, args.source_sha, args.source_git), sort_keys=True))


if __name__ == "__main__":
    main()
