"""Explicit post-observation continuation operator; no automatic retry."""
from __future__ import annotations

import json
import os
import signal
import sys
import time
from pathlib import Path
from types import FrameType
from collections.abc import Callable
from typing import Any, cast

from climate_rag.component_execution import PROTOCOL_SHA, SLOTS_SHA, TARGETS_SHA, durable
from climate_rag.component_continuation import RELEASE, PREVIOUS_RELEASE, load_carried, policy_identity
from run_scifact_component_operator import inference_exit, generator_only
from climate_rag.scifact_component_contract import require
from climate_rag.scifact_semantic_contract import GRAMMAR_SHA, checked
from run_budget_agent_full_operator import ARCHIVES, ROOT, atomic_status, digest, execute, read_only_tree, safe_extract
from run_sentence_agent_v3_operator import v3_runtime_environment

PREP = ROOT / "posthoc/scifact-component-preparation-426ff7343fb3"


def main() -> None:
    env0 = dict(os.environ)
    require(os.name == "posix" and bool(env0.get("SLURM_JOB_ID")) and bool(env0.get("CUDA_VISIBLE_DEVICES"))
            and env0.get("CLIMATE_COMPONENT_RELEASE") == RELEASE, "separate_gpu_release_required")
    source = Path(__file__).resolve().parents[1]
    work = Path(env0["CLIMATE_COMPONENT_WORK"]).resolve()
    require(source.parent == work and work.name.startswith("climate-component-"), "component_scratch")
    require((source / "SOURCE_REVISION").read_text().strip() == env0["CLIMATE_SOURCE_GIT"], "source_identity")
    checked(PREP / "protocol.json", PROTOCOL_SHA)
    frozen_input = checked(PREP / "inference/slots.json", SLOTS_SHA)
    load_carried(ROOT / "runs" / PREVIOUS_RELEASE)
    # The single fixed root reserves the whole release BEFORE any model loading.
    # It must not be deleted or renamed to obtain another 37 attempts.
    result = ROOT / "runs" / RELEASE
    result.mkdir(mode=0o700)
    status = result / "operator-status.json"
    state: dict[str, Any] = {"release": RELEASE, "status": "running", "source_git": env0["CLIMATE_SOURCE_GIT"],
             "source_archive_sha256": env0["CLIMATE_SOURCE_SHA256"], "preparation_protocol_sha256": PROTOCOL_SHA,
             "job_id": env0["SLURM_JOB_ID"], "targets_read": False, "started_unix": time.time(),
             "preflight_policy": policy_identity()}
    atomic_status(status, state)

    def interrupted(number: int, frame: FrameType | None) -> None:
        raise InterruptedError("component_allocation_interrupted")
    for sig in (getattr(signal, "SIGUSR1", signal.SIGTERM), signal.SIGTERM, signal.SIGINT):
        signal.signal(sig, interrupted)
    try:
        inputs = work / "component-input"
        inputs.mkdir(mode=0o700)
        (inputs / "slots.json").write_bytes(frozen_input)
        read_only_tree(inputs)
        for name in ("input", "runtime", "overlay", "grammar"):
            filename, expected = (("agent-v3-grammar-" + GRAMMAR_SHA + ".tar", GRAMMAR_SHA)
                                  if name == "grammar" else ARCHIVES[name])
            archive = ROOT / "envs" / filename
            require(digest(archive) == expected, "asset_hash")
            if name == "input":
                generator_only(archive, work / name, expected)
            else:
                safe_extract(archive, work / name, runtime=name == "runtime")
        read_only_tree(work / "input")
        # FUTURE execution only: reuse pinned existing wheels, never network/pip resolution.
        for name in ("overlay", "grammar"):
            lock = work / "overlay/requirements.lock" if name == "overlay" else source / "hpc/agent_v3_grammar.lock"
            execute([sys.executable, "-m", "pip", "install", "--no-index", "--no-deps", "--require-hashes",
                "--no-compile", "--no-cache-dir", "--disable-pip-version-check", "--target", str(work / "overlay-site"),
                "--find-links", str(work / name / "wheels"), "-r", str(lock)], cwd=work, env=env0, log=result / f"install-{name}.log")
        environment = cast(Callable[[Path, Path], dict[str, str]], v3_runtime_environment)
        env = environment(work, source)
        env.update(PYTHONHASHSEED="0", CLIMATE_COMPONENT_RELEASE=RELEASE,
                   CLIMATE_SOURCE_GIT=env0["CLIMATE_SOURCE_GIT"], CLIMATE_SOURCE_SHA256=env0["CLIMATE_SOURCE_SHA256"])
        command = [sys.executable, str(source / "scripts/run_scifact_component_continuation.py"), "--slots", str(inputs / "slots.json"),
            "--model-dir", str(work / "input/models/generator/model"),
            "--model-manifest", str(work / "input/models/generator/model_manifest.json")]
        state["stage"] = "inference"
        atomic_status(status, state)
        exit_proof = inference_exit(command, cwd=source, env=env, log=result / "inference.log")
        identity = result / "worker-identity.json"
        if not identity.exists():
            raise RuntimeError("worker_identity_missing_no_model_retry")
        durable(result / "inference-exited.json", {"release": RELEASE, **exit_proof,
                "worker_identity_sha256": digest(identity)})
        # No target path, target bytes or scoring module passed into inference.
        scoring = work / "component-scoring"
        scoring.mkdir(mode=0o700)
        (scoring / "targets.json").write_bytes(checked(PREP / "scoring/targets.json", TARGETS_SHA))
        state.update(stage="scoring_after_exit", targets_read=True)
        atomic_status(status, state)
        execute([sys.executable, str(source / "scripts/score_scifact_component_continuation.py"), "--result", str(result),
            "--slots", str(inputs / "slots.json"), "--targets", str(scoring / "targets.json"),
            "--exit-sha", digest(result / "inference-exited.json")], cwd=source, env=env, log=result / "score.log")
        compact = json.loads((result / "compact.json").read_bytes())
        state.update(status=compact["status"], compact_sha256=digest(result / "compact.json"))
    except BaseException as exc:
        state.update(status="failed_no_automatic_retry", failure_type=type(exc).__name__)
        raise
    finally:
        state["elapsed_seconds"] = time.time()-state["started_unix"]
        atomic_status(status, state)


if __name__ == "__main__":
    main()
