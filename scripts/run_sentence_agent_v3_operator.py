"""Prepared v3 operator. Allocation/release is required; no scheduler calls here."""

from __future__ import annotations

import json
import os
import signal
import sys
import time
from pathlib import Path

from run_budget_agent_feedback_operator import runtime_environment
from run_budget_agent_full_operator import (
    ARCHIVES,
    ROOT,
    atomic_status,
    digest,
    execute,
    read_only_tree,
    safe_extract,
)

RELEASE = "climate-sentence-v3-20260930-pilot-r1"
CONFIG = "configs/agent_sentence_v3_pilot_20260930.json"


def main():
    if (
        not os.environ.get("SLURM_JOB_ID")
        or not os.environ.get("CUDA_VISIBLE_DEVICES")
        or os.environ.get("CLIMATE_V3_RELEASE") != RELEASE
    ):
        raise ValueError("separately released GPU allocation required")
    work = Path(os.environ["CLIMATE_V3_WORK"]).resolve()
    source = Path(__file__).resolve().parents[1]
    if source.parent != work or not work.name.startswith("climate-sentence-v3-"):
        raise ValueError("isolated scratch required")
    result = ROOT / "runs" / RELEASE
    result.mkdir(mode=0o700)  # no overwrite, retry or resume
    status = result / "operator-status.json"
    state = {
        "schema_version": "sentence-v3-development-operator",
        "release_id": RELEASE,
        "job_id": os.environ["SLURM_JOB_ID"],
        "status": "running",
        "stage": "source",
        "source_git": os.environ["CLIMATE_SOURCE_GIT"],
        "source_archive_sha256": os.environ["CLIMATE_SOURCE_SHA256"],
        "started_unix": time.time(),
        "input_extractions": 0,
        "gpu_memory_peak": None,
        "raw_response_exported": False,
    }

    def stage(name):
        state["stage"] = name
        atomic_status(status, state)

    def interrupted(number, frame):
        del frame
        state["signal"] = number
        raise InterruptedError("allocation signal")

    for number in (signal.SIGUSR1, signal.SIGTERM, signal.SIGINT):
        signal.signal(number, interrupted)
    try:
        if (source / "SOURCE_REVISION").read_text().strip() != state["source_git"]:
            raise ValueError("source revision mismatch")
        protocol_path = source / CONFIG
        protocol = json.loads(protocol_path.read_text())
        if (
            protocol["release_id"] != RELEASE
            or protocol["official_gold_available"]
            or len(protocol["tasks"]) != 6
            or len(protocol["routes"]) != 4
        ):
            raise ValueError("development-only protocol required")
        state["protocol_sha256"] = digest(protocol_path)
        if state["protocol_sha256"] != os.environ["CLIMATE_V3_PROTOCOL_SHA256"]:
            raise ValueError("unreleased protocol hash")
        state["archives"] = {}
        for name in ("input", "runtime", "overlay", "grammar"):
            if name == "grammar":
                archive = Path(os.environ["CLIMATE_V3_GRAMMAR_TAR"])
                expected = os.environ["CLIMATE_V3_GRAMMAR_SHA256"]
            else:
                filename, expected = ARCHIVES[name]
                archive = ROOT / "envs" / filename
            stage("hash_" + name)
            if (
                not archive.resolve().is_relative_to(ROOT / "envs")
                or digest(archive) != expected
            ):
                raise ValueError("archive scope/hash mismatch")
            stage("extract_" + name)
            size = safe_extract(archive, work / name, runtime=name == "runtime")
            state["archives"][name] = {"sha256": expected, "extracted_bytes": size}
            if name == "input":
                state["input_extractions"] += 1
        read_only_tree(work / "input")
        for name in ("overlay", "grammar"):
            stage("install_" + name)
            lock = (
                work / "overlay/requirements.lock"
                if name == "overlay"
                else source / "hpc/agent_v3_grammar.lock"
            )
            execute(
                [
                    sys.executable,
                    "-m",
                    "pip",
                    "install",
                    "--no-index",
                    "--no-deps",
                    "--require-hashes",
                    "--no-compile",
                    "--no-cache-dir",
                    "--disable-pip-version-check",
                    "--target",
                    str(work / "overlay-site"),
                    "--find-links",
                    str(work / name / "wheels"),
                    "-r",
                    str(lock),
                ],
                cwd=work,
                env=dict(os.environ),
                log=result / f"install-{name}.log",
            )
        env = runtime_environment(work, source)
        env.update(CLIMATE_V3_RELEASE=RELEASE, CLIMATE_SOURCE_GIT=state["source_git"])
        inputs = work / "input"
        arguments = json.loads((inputs / "args-pilot.json").read_text())
        options = dict(zip(arguments[::2], arguments[1::2], strict=True))
        command = [
            sys.executable,
            str(source / "scripts/run_sentence_agent_v3.py"),
            "--protocol",
            str(protocol_path),
            "--expected-protocol-sha256",
            state["protocol_sha256"],
            "--output-dir",
            str(result),
        ]
        for key in (
            "--evidence",
            "--model-dir",
            "--model-manifest",
            "--reranker-dir",
            "--reranker-manifest",
        ):
            value = options[key]
            if not value.startswith("{INPUT}/"):
                raise ValueError("unapproved input path")
            path = (inputs / value.removeprefix("{INPUT}/")).resolve()
            if not path.is_relative_to(inputs.resolve()) or not path.exists():
                raise ValueError("input path escape or missing")
            command.extend([key, str(path)])
        stage("runtime_preflight_then_pilot")
        execute(command, cwd=source, env=env, log=result / "inference.log")
        compact = json.loads((result / "compact.json").read_text())
        if (
            compact["source_git"] != state["source_git"]
            or compact["complete_slots"] != 24
        ):
            raise ValueError("matrix/source mismatch")
        state["status"] = state["stage"] = "complete"
        state["compact_sha256"] = digest(result / "compact.json")
        return 0
    except BaseException as exc:
        state["status"] = (
            "interrupted" if isinstance(exc, InterruptedError) else "failed"
        )
        state["error_type"] = type(exc).__name__
        raise
    finally:
        state["elapsed_seconds"] = time.time() - state["started_unix"]
        state["file_receipts"] = {
            p.relative_to(result).as_posix(): {
                "sha256": digest(p),
                "bytes": p.stat().st_size,
            }
            for p in result.rglob("*")
            if p.is_file() and p != status
        }
        atomic_status(status, state)


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as error:
        print("v3 operator failed: " + type(error).__name__, file=sys.stderr)
        raise SystemExit(1) from None
