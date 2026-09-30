"""Assemble public evidence and frozen model inputs; benchmark gold is never included."""
from __future__ import annotations

import argparse
import hashlib
import io
import json
import tarfile
from pathlib import Path

from climate_rag.local_agent_model import verify_model_files

PUBLIC_CORPUS_SHA = "c14315aee9feecbbbbc3b0c7101d978b52b47cee9a613336304bcaa736460c71"
AUTHORED_SHA = "6ee90b8479192335fc543c3425c3fbbb89fea2d65ba9698fb54d86564bc3c3c2"
VALIDATION_SHA = "abfcb61e9e6a54641f025101a4011fa88e07e3697a36f0acd308fa8e86052674"


def digest(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(8 * 1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def json_bytes(value: object) -> bytes:
    return (json.dumps(value, sort_keys=True, indent=2) + "\n").encode()


def phase_arguments(phase: str, protocol_name: str, protocol_sha: str,
                    generator_manifest_sha: str, reranker_manifest_sha: str) -> tuple[list[str], bytes]:
    execution = json_bytes({"protocol_sha256": protocol_sha,
                            "model_manifest_sha256": generator_manifest_sha,
                            "reranker_manifest_sha256": reranker_manifest_sha,
                            "dense_file_hashes": None})
    args = ["--evidence", "{INPUT}/evidence.jsonl", "--protocol", "{INPUT}/" + protocol_name,
            "--expected-protocol-sha256", protocol_sha, "--phase", phase, "--provider", "local-qwen",
            "--model-dir", "{INPUT}/models/generator/model",
            "--model-manifest", "{INPUT}/models/generator/model_manifest.json",
            "--reranker-dir", "{INPUT}/models/reranker/model",
            "--reranker-manifest", "{INPUT}/models/reranker/model_manifest.json",
            "--execution-manifest", "{INPUT}/execution-" + phase + ".json",
            "--expected-execution-sha256", hashlib.sha256(execution).hexdigest()]
    return args, execution


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--models", type=Path, required=True)
    parser.add_argument("--evidence", type=Path, required=True)
    parser.add_argument("--authored-protocol", type=Path, required=True)
    parser.add_argument("--validation-protocol", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise ValueError("refuse to replace frozen bundle")
    for path, expected in ((args.evidence, PUBLIC_CORPUS_SHA), (args.authored_protocol, AUTHORED_SHA),
                           (args.validation_protocol, VALIDATION_SHA)):
        if digest(path) != expected:
            raise ValueError("unapproved evidence/protocol bytes")
    manifest_shas = {}
    for role in ("generator", "reranker"):
        manifest = args.models / role / "model_manifest.json"
        verify_model_files(args.models / role / "model", json.loads(manifest.read_text()))
        manifest_shas[role] = digest(manifest)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with tarfile.open(args.output, "x") as archive:
        for path, name in ((args.evidence, "evidence.jsonl"),
                           (args.authored_protocol, "authored-protocol.json"),
                           (args.validation_protocol, "validation-protocol.json")):
            archive.add(path, arcname=name)
        for path in sorted(args.models.rglob("*")):
            if path.is_symlink() or ".partial" in path.name:
                raise ValueError("incomplete or linked model inputs")
            if path.is_file():
                archive.add(path, arcname="models/" + path.relative_to(args.models).as_posix())
        for phase in ("pilot", "validation", "vnext"):
            protocol_name = "validation-protocol.json" if phase == "validation" else "authored-protocol.json"
            protocol_sha = VALIDATION_SHA if phase == "validation" else AUTHORED_SHA
            arguments, execution = phase_arguments(phase, protocol_name, protocol_sha,
                                                    manifest_shas["generator"], manifest_shas["reranker"])
            for name, data in (("args-" + phase + ".json", json_bytes(arguments)),
                               ("execution-" + phase + ".json", execution)):
                member = tarfile.TarInfo(name)
                member.size = len(data)
                archive.addfile(member, io.BytesIO(data))
    report = {"archive_sha256": digest(args.output), "bytes": args.output.stat().st_size,
              "corpus_sha256": PUBLIC_CORPUS_SHA, "authored_protocol_sha256": AUTHORED_SHA,
              "validation_protocol_sha256": VALIDATION_SHA, "model_manifests": manifest_shas,
              "dense_enabled": False, "first_stage": "BM25 common to all three routes",
              "gold_included": False, "real_model_executed": False}
    args.output.with_suffix(".manifest.json").write_bytes(json_bytes(report))
    print(json.dumps(report))


if __name__ == "__main__":
    main()
