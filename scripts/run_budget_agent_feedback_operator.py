"""Single released development pilot: node-local preflight then model replay, no gold."""
from __future__ import annotations

import json
import os
import signal
import sys
import time
from pathlib import Path

from run_budget_agent_full_operator import (
    ARCHIVES, ROOT, atomic_status, digest, execute, read_only_tree, safe_extract,
)

RELEASE = "climate-feedback-v2-20260930-pilot-r1"
CONFIG = "configs/budget_agent_feedback_pilot_20260930.json"


def runtime_environment(work, source):
    env = {k: v for k, v in os.environ.items() if not k.startswith("CLIMATE_")}
    env.update(PYTHONDONTWRITEBYTECODE="1", PYTHONNOUSERSITE="1", HF_HUB_OFFLINE="1",
               TRANSFORMERS_OFFLINE="1", HF_HUB_DISABLE_TELEMETRY="1",
               HF_HOME=str(work / "cache"), XDG_CACHE_HOME=str(work / "cache"))
    env["PYTHONPATH"] = os.pathsep.join((str(work / "overlay-site"),
        str(work / "runtime/lib/python3.10/site-packages"), str(source / "src"), os.environ.get("PYTHONPATH", "")))
    return env


def main():
    if (not os.environ.get("SLURM_JOB_ID") or not os.environ.get("CUDA_VISIBLE_DEVICES")
            or os.environ.get("CLIMATE_FEEDBACK_RELEASE") != RELEASE):
        raise ValueError("released allocation required")
    work = Path(os.environ["CLIMATE_FEEDBACK_WORK"]).resolve()
    source = Path(__file__).resolve().parents[1]
    if source.parent != work or not work.name.startswith("climate-feedback-"):
        raise ValueError("isolated scratch required")
    result = ROOT / "runs" / RELEASE
    result.mkdir(mode=0o700)  # consumed once, never automatic retry/resume
    status = result / "operator-status.json"
    state = {"schema_version": "feedback-pilot-operator-v2", "release_id": RELEASE,
             "job_id": os.environ["SLURM_JOB_ID"], "status": "running", "stage": "source",
             "source_git": os.environ["CLIMATE_SOURCE_GIT"],
             "source_archive_sha256": os.environ["CLIMATE_SOURCE_SHA256"],
             "started_unix": time.time(), "input_extractions": 0,
             "gpu_memory_peak": None, "raw_response_exported": False}

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
        stage("source")
        if (source / "SOURCE_REVISION").read_text().strip() != state["source_git"]:
            raise ValueError("source revision mismatch")
        protocol_path = source / CONFIG
        protocol = json.loads(protocol_path.read_text())
        if (protocol["official_gold_available"] or len(protocol["pilot"]) != 6
                or protocol["budget"]["controller_protocol"] != "feedback-v2"):
            raise ValueError("development-only protocol required")
        state["protocol_sha256"] = digest(protocol_path)
        state["archives"] = {}
        for name in ("input", "runtime", "overlay"):
            filename, expected = ARCHIVES[name]
            archive = ROOT / "envs" / filename
            stage("hash_" + name)
            if not archive.resolve().is_relative_to(ROOT / "envs") or digest(archive) != expected:
                raise ValueError("archive scope/hash mismatch")
            stage("extract_" + name)
            size = safe_extract(archive, work / name, runtime=name == "runtime")
            state["archives"][name] = {"sha256": expected, "extracted_bytes": size}
            if name == "input":
                state["input_extractions"] += 1
        read_only_tree(work / "input")
        stage("overlay")
        execute([sys.executable, "-m", "pip", "install", "--no-index", "--no-deps", "--require-hashes",
                 "--no-compile", "--no-cache-dir", "--disable-pip-version-check", "--target", str(work / "overlay-site"),
                 "--find-links", str(work / "overlay/wheels"), "-r", str(work / "overlay/requirements.lock")],
                cwd=work, env=dict(os.environ), log=result / "runtime-install.log")
        env = runtime_environment(work, source)
        # Paths only; no original benchmark gold, test labels or claims are read.
        inputs = work / "input"
        original_args = json.loads((inputs / "args-pilot.json").read_text())
        options = dict(zip(original_args[::2], original_args[1::2], strict=True))
        paths = {}
        for key in ("--evidence", "--model-dir", "--model-manifest", "--reranker-dir", "--reranker-manifest"):
            value = options[key]
            if not value.startswith("{INPUT}/"):
                raise ValueError("unapproved input path")
            path = (inputs / value.removeprefix("{INPUT}/")).resolve()
            if not path.is_relative_to(inputs.resolve()) or not path.exists():
                raise ValueError("input path escape or missing")
            paths[key] = path
        manifest = {"protocol_sha256": state["protocol_sha256"],
                    "model_manifest_sha256": digest(paths["--model-manifest"]),
                    "reranker_manifest_sha256": digest(paths["--reranker-manifest"]),
                    "dense_file_hashes": None}
        manifest_path = result / "execution-manifest.json"
        with manifest_path.open("x") as stream:
            json.dump(manifest, stream, indent=2, sort_keys=True)
        command = [sys.executable, str(source / "scripts/run_budget_agent.py"),
                   "--provider", "local-qwen", "--phase", "pilot", "--protocol", str(protocol_path),
                   "--expected-protocol-sha256", state["protocol_sha256"],
                   "--execution-manifest", str(manifest_path), "--expected-execution-sha256", digest(manifest_path)]
        for key, path in paths.items():
            command.extend([key, str(path)])
        stage("preflight")
        execute([*command, "--preflight-only", "--output", str(result / "preflight.json")],
                cwd=source, env=env, log=result / "preflight.log")
        state["preflight"] = "passed_no_generation"
        private = result / "private-responses"
        private.mkdir(mode=0o700)
        env["CLIMATE_PRIVATE_RESPONSE_DIR"] = str(private)
        stage("pilot")
        execute([*command, "--output", str(result / "run.json")], cwd=source, env=env, log=result / "inference.log")
        stage("audit")
        execute([sys.executable, str(source / "scripts/score_budget_agent_feedback.py"),
                 "--run", str(result / "run.json"), "--protocol", str(protocol_path),
                 "--expected-protocol-sha256", state["protocol_sha256"],
                 "--output", str(result / "compact.json")], cwd=source, env=env, log=result / "audit.log")
        compact = json.loads((result / "compact.json").read_text())
        if compact["source_git"] != state["source_git"] or compact["complete_slots"] != 18:
            raise ValueError("pilot identity/matrix mismatch")
        state["status"] = state["stage"] = "complete"
        state["compact_sha256"] = digest(result / "compact.json")
        return 0
    except BaseException as exc:
        state["status"] = "interrupted" if isinstance(exc, InterruptedError) else "failed"
        state["error_type"] = type(exc).__name__
        raise
    finally:
        state["elapsed_seconds"] = time.time() - state["started_unix"]
        state["file_receipts"] = {p.relative_to(result).as_posix(): {"sha256": digest(p), "bytes": p.stat().st_size}
                                  for p in result.rglob("*") if p.is_file() and p != status}
        atomic_status(status, state)


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as error:
        print("Feedback pilot failed: " + type(error).__name__, file=sys.stderr)
        raise SystemExit(1) from None
