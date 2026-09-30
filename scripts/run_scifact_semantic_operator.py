"""Serial semantic pair operator; gold is extracted after both inference exits.

Dataflow isolation, not an OS sandbox against the shared account. No submission.
"""
from __future__ import annotations

import json
import os
import shutil
import signal
import sys
import tarfile
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any, cast

from climate_rag.scifact_semantic_contract import (
    GRAMMAR_SHA, INFERENCE_NAMES, POLICIES, RELEASE, SCORING_NAMES, checked, load_inference,
)
from climate_rag.scifact_semantic_receipts import previous_arm, read_completed_run
from run_budget_agent_full_operator import (
    ARCHIVES, ROOT, atomic_status, digest, execute, read_only_tree, safe_extract,
)
from run_sentence_agent_v3_operator import v3_runtime_environment
from run_scifact_bounded_operator import extract_exact

extract_checked = cast(Callable[[Path, Path, str, frozenset[str]], int], extract_exact)
isolated_environment = cast(Callable[[Path, Path], dict[str, str]], v3_runtime_environment)


def model_assets_only(archive: Path, target: Path, expected_sha: str) -> int:
    if target.exists() or digest(archive) != expected_sha:
        raise ValueError("semantic_model_archive_identity")
    with tarfile.open(archive) as bundle:
        members = [m for m in bundle.getmembers() if m.name.startswith(
            ("models/generator/model/", "models/reranker/model/"))
            or m.name in {"models/generator/model_manifest.json", "models/reranker/model_manifest.json"}]
        resolved = [(target / m.name).resolve() for m in members]
        if (not members or len(set(resolved)) != len(resolved)
                or any(not m.isfile() or not p.is_relative_to(target.resolve())
                       for m, p in zip(members, resolved, strict=True))):
            raise ValueError("semantic_model_member_allowlist")
        size = sum(m.size for m in members)
        if shutil.disk_usage(target.parent).free < size + 1024**3:
            raise ValueError("semantic_model_scratch_capacity")
        target.mkdir(mode=0o700)
        bundle.extractall(target, members=members)
    return size


def main() -> None:
    env0 = dict(os.environ)
    policy = env0.get("CLIMATE_SEMANTIC_POLICY", "")
    if (not env0.get("SLURM_JOB_ID") or not env0.get("CUDA_VISIBLE_DEVICES")
            or env0.get("CLIMATE_SEMANTIC_RELEASE") != RELEASE or policy not in POLICIES):
        raise ValueError("semantic_separate_gpu_release_required")
    work = Path(env0["CLIMATE_SEMANTIC_WORK"]).resolve()
    source = Path(__file__).resolve().parents[1]
    if source.parent != work or not work.name.startswith("climate-semantic-pair-"):
        raise ValueError("semantic_scratch_scope")
    if (source / "SOURCE_REVISION").read_text().strip() != env0["CLIMATE_SOURCE_GIT"]:
        raise ValueError("semantic_source_revision")
    receipt_path = Path(env0["CLIMATE_SEMANTIC_BUNDLE_RECEIPT"])
    receipt = json.loads(checked(receipt_path, env0["CLIMATE_SEMANTIC_BUNDLE_RECEIPT_SHA"]))
    if (not receipt_path.resolve().is_relative_to(ROOT / "envs")
            or receipt["execution_source_git"] != env0["CLIMATE_SOURCE_GIT"]
            or receipt["release_id"] != RELEASE or env0["CLIMATE_GRAMMAR_SHA256"] != GRAMMAR_SHA):
        raise ValueError("semantic_bundle_release_identity")
    for name in ("inference", "scoring"):
        path = Path(env0[f"CLIMATE_{name.upper()}_TAR"])
        expected = receipt["bundles"][name]["sha256"]
        if (not path.resolve().is_relative_to(ROOT / "envs")
                or expected != env0[f"CLIMATE_{name.upper()}_SHA256"] or digest(path) != expected):
            raise ValueError("semantic_bundle_scope_or_hash")
    protocol_sha = receipt["protocol_sha256"]
    extract_checked(Path(env0["CLIMATE_INFERENCE_TAR"]), work / "semantic-inference",
                  receipt["bundles"]["inference"]["sha256"], INFERENCE_NAMES)
    read_only_tree(work / "semantic-inference")
    protocol, _, _ = load_inference(work / "semantic-inference", protocol_sha, env0["CLIMATE_SOURCE_GIT"])
    predecessor = (previous_arm(ROOT, protocol, protocol_sha, env0["CLIMATE_SOURCE_SHA256"],
                               env0["CLIMATE_INFERENCE_SHA256"]) if policy == POLICIES[1] else None)
    result = ROOT / "runs" / (RELEASE + "-" + policy)
    result.mkdir(mode=0o700)
    status = result / "operator-status.json"
    state: dict[str, Any] = {"release_id": RELEASE, "arm": policy, "source_git": env0["CLIMATE_SOURCE_GIT"],
        "source_archive_sha256": env0["CLIMATE_SOURCE_SHA256"], "protocol_sha256": protocol_sha,
        "job_id": env0["SLURM_JOB_ID"], "status": "running", "started_unix": time.time(),
        "train_gold_extracted": False, "official_dev_read": False,
        "accepted_predecessor_run_sha256": predecessor, "archives": {}}

    def stage(name: str) -> None:
        state["stage"] = name
        atomic_status(status, state)

    def interrupted(number: int, frame: Any) -> None:
        raise InterruptedError(f"signal {number}")

    for sig in (getattr(signal, "SIGUSR1", signal.SIGTERM), signal.SIGTERM, signal.SIGINT):
        signal.signal(sig, interrupted)
    try:
        for name in ("input", "runtime", "overlay", "grammar"):
            if name == "grammar":
                path, expected = Path(env0["CLIMATE_GRAMMAR_TAR"]), GRAMMAR_SHA
            else:
                filename, expected = ARCHIVES[name]
                path = ROOT / "envs" / filename
            stage("extract_" + name)
            if not path.resolve().is_relative_to(ROOT / "envs") or digest(path) != expected:
                raise ValueError("semantic_asset_scope_hash")
            size = (model_assets_only(path, work / name, expected) if name == "input"
                    else safe_extract(path, work / name, runtime=name == "runtime"))
            state["archives"][name] = {"sha256": expected, "bytes": size}
        read_only_tree(work / "input")
        for name in ("overlay", "grammar"):
            stage("install_" + name)
            lock = work / "overlay/requirements.lock" if name == "overlay" else source / "hpc/agent_v3_grammar.lock"
            execute([sys.executable, "-m", "pip", "install", "--no-index", "--no-deps", "--require-hashes",
                "--no-compile", "--no-cache-dir", "--disable-pip-version-check", "--target", str(work / "overlay-site"),
                "--find-links", str(work / name / "wheels"), "-r", str(lock)],
                cwd=work, env=env0, log=result / f"install-{name}.log")
        child_env = isolated_environment(work, source)
        child_env.update(PYTHONHASHSEED="0", CLIMATE_SEMANTIC_RELEASE=RELEASE,
            CLIMATE_SOURCE_GIT=env0["CLIMATE_SOURCE_GIT"], CLIMATE_SOURCE_SHA256=env0["CLIMATE_SOURCE_SHA256"],
            CLIMATE_INFERENCE_SHA256=env0["CLIMATE_INFERENCE_SHA256"])
        if any(k.startswith("CLIMATE_") and k not in {"CLIMATE_SEMANTIC_RELEASE", "CLIMATE_SOURCE_GIT",
               "CLIMATE_SOURCE_SHA256", "CLIMATE_INFERENCE_SHA256"} for k in child_env):
            raise ValueError("semantic_private_environment_leak")
        command = [sys.executable, str(source / "scripts/run_scifact_semantic_arm.py"),
            "--inference-dir", str(work / "semantic-inference"), "--protocol-sha", protocol_sha,
            "--policy", policy, "--output-dir", str(result / "inference")]
        for role, arg in (("generator", "model"), ("reranker", "reranker")):
            command.extend(["--" + arg + "-dir", str(work / "input/models" / role / "model"),
                "--" + arg + "-manifest", str(work / "input/models" / role / "model_manifest.json")])
        stage("synthetic_preflight_then_train_inference")
        execute(command, cwd=source, env=child_env, log=result / "inference.log")
        run_path = result / "inference/run.json"
        read_completed_run(run_path, protocol, protocol_sha, env0["CLIMATE_SOURCE_SHA256"],
                           env0["CLIMATE_INFERENCE_SHA256"], policy)
        state["inference_sha256"] = digest(run_path)
        if policy == POLICIES[1]:
            if previous_arm(ROOT, protocol, protocol_sha, env0["CLIMATE_SOURCE_SHA256"],
                            env0["CLIMATE_INFERENCE_SHA256"]) != predecessor:
                raise ValueError("semantic_predecessor_changed")
            stage("extract_scoring_after_both_inference_exits")
            extract_checked(Path(env0["CLIMATE_SCORING_TAR"]), work / "semantic-scoring",
                          env0["CLIMATE_SCORING_SHA256"], SCORING_NAMES)
            state["train_gold_extracted"] = True
            stage("score_semantic_pair")
            execute([sys.executable, str(source / "scripts/score_scifact_semantic_pair.py"),
                "--first-run", str(ROOT / "runs" / (RELEASE + "-" + POLICIES[0]) / "inference/run.json"),
                "--second-run", str(run_path), "--inference-dir", str(work / "semantic-inference"),
                "--scoring-dir", str(work / "semantic-scoring"), "--protocol-sha", protocol_sha,
                "--manifest-sha", receipt["scoring_manifest_sha256"], "--source-sha", env0["CLIMATE_SOURCE_SHA256"],
                "--inference-sha", env0["CLIMATE_INFERENCE_SHA256"], "--source-git", env0["CLIMATE_SOURCE_GIT"],
                "--output", str(result / "score.json")], cwd=source, env=child_env, log=result / "score.log")
            state["score_sha256"] = digest(result / "score.json")
        state["status"] = "complete"
        stage("complete")
    except BaseException as exc:
        state.update(status="failed", exception_type=type(exc).__name__)
        atomic_status(status, state)
        raise
    finally:
        state["elapsed_seconds"] = time.time() - state["started_unix"]
        atomic_status(status, state)


if __name__ == "__main__":
    main()
