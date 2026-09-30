"""Prepared serial pair operator. Both inference children exit before gold extraction."""

import json
import os
import signal
import sys
import tarfile
import time
from pathlib import Path

from run_budget_agent_full_operator import (
    ARCHIVES, ROOT, atomic_status, digest, execute, read_only_tree, safe_extract,
)
from run_sentence_agent_v3_operator import v3_runtime_environment

RELEASE = "climate-scifact-bounded-gap-pair-20260930-v1"
ARMS = ("format_repaired", "format_repaired_gap")
PAIR_SHA = "4cc5033928a0833ee99ad64ecd5a7b10c7a1fde7ac4f219d6334d2349d9daaf6"


def validate_external_assets(environ, source):
    path = source / "docs/protocols/scifact-bounded-gap-pair-20260930.json"
    if digest(path) != PAIR_SHA:
        raise ValueError("paired preregistration hash mismatch")
    pair = json.loads(path.read_text())
    expected = {
        "CLIMATE_INFERENCE_SHA256": pair["inference_tar_sha256"],
        "CLIMATE_SCORING_SHA256": pair["scoring_tar_sha256"],
        "CLIMATE_DIAGNOSTIC_PROTOCOL_SHA256": pair["parent_protocol_sha256"],
        "CLIMATE_GRAMMAR_SHA256": "36ed51e0a9a71ded058cc0e7290a78a7f5f638b91bff517d49ad6ea83c70e410",
    }
    if any(environ.get(key) != value for key, value in expected.items()):
        raise ValueError("external assets differ from frozen pair")


def create_attempt_result(root, environ):
    arm = environ.get("CLIMATE_SCIFACT_ARM")
    if environ.get("CLIMATE_SCIFACT_PAIR_RELEASE") != RELEASE or arm not in ARMS:
        raise ValueError("unreleased paired arm")
    result = root / "runs" / (RELEASE + "-" + arm)
    if arm == ARMS[1]:
        previous = root / "runs" / (RELEASE + "-" + ARMS[0])
        state = json.loads((previous / "operator-status.json").read_text())
        if (state["status"] != "complete" or state["source_git"] != environ["CLIMATE_SOURCE_GIT"]
                or state["source_archive_sha256"] != environ["CLIMATE_SOURCE_SHA256"]
                or state["pair_protocol_sha256"] != PAIR_SHA
                or state["inference_sha256"] != digest(previous / "inference/run.json")):
            raise ValueError("previous arm incomplete or different frozen source/protocol")
    result.mkdir(mode=0o700)
    return result, {"release_id": RELEASE, "arm": arm, "pair_protocol_sha256": PAIR_SHA}


def extract_exact(archive, target, expected_sha, names):
    if digest(archive) != expected_sha:
        raise ValueError("diagnostic bundle SHA mismatch")
    with tarfile.open(archive) as bundle:
        members = bundle.getmembers()
        if (len(members) != len(names) or {m.name for m in members} != set(names)
                or not all(m.isfile() for m in members)):
            raise ValueError("diagnostic bundle is not the exact file allowlist")
    return safe_extract(archive, target)


def main():
    if (not os.environ.get("SLURM_JOB_ID") or not os.environ.get("CUDA_VISIBLE_DEVICES")
            or os.environ.get("CLIMATE_SCIFACT_PAIR_RELEASE") != RELEASE):
        raise ValueError("root-released GPU allocation required")
    work = Path(os.environ["CLIMATE_SCIFACT_PAIR_WORK"]).resolve()
    source = Path(__file__).resolve().parents[1]
    if source.parent != work or not work.name.startswith("climate-scifact-pair-"):
        raise ValueError("isolated scratch required")
    if (source / "SOURCE_REVISION").read_text().strip() != os.environ["CLIMATE_SOURCE_GIT"]:
        raise ValueError("source revision mismatch")
    validate_external_assets(os.environ, source)  # before creating a run or extracting gold
    result, identity = create_attempt_result(ROOT, os.environ)
    status_path = result / "operator-status.json"
    state = {**identity, "source_git": os.environ["CLIMATE_SOURCE_GIT"],
             "source_archive_sha256": os.environ["CLIMATE_SOURCE_SHA256"],
             "job_id": os.environ["SLURM_JOB_ID"], "status": "running",
             "started_unix": time.time(), "train_gold_extracted": False,
             "official_dev_read": False, "archives": {}}

    def stage(name):
        state["stage"] = name
        atomic_status(status_path, state)

    def interrupted(number, frame):
        raise InterruptedError(f"signal {number}")

    for number in (signal.SIGUSR1, signal.SIGTERM, signal.SIGINT):
        signal.signal(number, interrupted)
    try:
        for name in ("input", "runtime", "overlay", "grammar"):
            if name == "grammar":
                path = Path(os.environ["CLIMATE_GRAMMAR_TAR"])
                expected = os.environ["CLIMATE_GRAMMAR_SHA256"]
            else:
                filename, expected = ARCHIVES[name]
                path = ROOT / "envs" / filename
            stage("extract_" + name)
            if not path.resolve().is_relative_to(ROOT / "envs") or digest(path) != expected:
                raise ValueError("asset scope/hash mismatch")
            state["archives"][name] = {"sha256": expected,
                "bytes": safe_extract(path, work / name, runtime=name == "runtime")}
        read_only_tree(work / "input")
        for name in ("overlay", "grammar"):
            stage("install_" + name)
            lock = work / "overlay/requirements.lock" if name == "overlay" else source / "hpc/agent_v3_grammar.lock"
            execute([sys.executable, "-m", "pip", "install", "--no-index", "--no-deps",
                     "--require-hashes", "--no-compile", "--no-cache-dir", "--disable-pip-version-check",
                     "--target", str(work / "overlay-site"), "--find-links", str(work / name / "wheels"),
                     "-r", str(lock)], cwd=work, env=dict(os.environ), log=result / f"install-{name}.log")
        for name in ("inference", "scoring"):
            path = Path(os.environ[f"CLIMATE_{name.upper()}_TAR"])
            if not path.resolve().is_relative_to(ROOT / "envs"):
                raise ValueError("diagnostic archive outside project envs")
        stage("extract_inference_only")
        extract_exact(Path(os.environ["CLIMATE_INFERENCE_TAR"]), work / "diagnostic-inference",
                      os.environ["CLIMATE_INFERENCE_SHA256"], {"claims.jsonl", "corpus.jsonl", "protocol.json"})
        read_only_tree(work / "diagnostic-inference")
        env = v3_runtime_environment(work, source)
        env.update(PYTHONHASHSEED="0", CLIMATE_SCIFACT_PAIR_RELEASE=RELEASE,
                   CLIMATE_SOURCE_GIT=state["source_git"])
        if digest(source / "docs/protocols/scifact-bounded-gap-pair-20260930.json") != PAIR_SHA:
            raise ValueError("paired protocol not frozen")
        # Scoring archive path/env is absent from the child. This is dataflow
        # isolation in allocated scratch, not an OS sandbox against the shared account.
        options_list = json.loads((work / "input/args-pilot.json").read_text())
        options = dict(zip(options_list[::2], options_list[1::2], strict=True))
        command = [sys.executable, str(source / "scripts/run_scifact_bounded_arm.py"),
                   "--inference-dir", str(work / "diagnostic-inference"),
                   "--expected-protocol-sha256", os.environ["CLIMATE_DIAGNOSTIC_PROTOCOL_SHA256"],
                   "--output-dir", str(result / "inference"),
                   "--arm", state["arm"],
                   "--pair-protocol", str(source / "docs/protocols/scifact-bounded-gap-pair-20260930.json")]
        for key in ("--model-dir", "--model-manifest", "--reranker-dir", "--reranker-manifest"):
            value = options[key]
            if not value.startswith("{INPUT}/"):
                raise ValueError("unapproved model asset path")
            path = (work / "input" / value.removeprefix("{INPUT}/")).resolve()
            if not path.is_relative_to(work / "input") or not path.exists():
                raise ValueError("model asset escape/missing")
            command.extend([key, str(path)])
        stage("synthetic_preflight_then_train_inference")
        execute(command, cwd=source, env=env, log=result / "inference.log")
        state["inference_sha256"] = digest(result / "inference/run.json")
        if state["arm"] == ARMS[1]:
            stage("extract_scoring_after_both_inference_exits")
            extract_exact(Path(os.environ["CLIMATE_SCORING_TAR"]), work / "diagnostic-scoring",
                          os.environ["CLIMATE_SCORING_SHA256"], {"gold.jsonl", "selected-strata.json", "manifest.json"})
            state["train_gold_extracted"] = True
            scoring = work / "diagnostic-scoring"
            previous = ROOT / "runs" / (RELEASE + "-" + ARMS[0])
            stage("score_paired_train_point_diagnostic")
            execute([sys.executable, str(source / "scripts/score_scifact_bounded_pair.py"),
                     "--first-run", str(previous / "inference/run.json"),
                     "--second-run", str(result / "inference/run.json"),
                     "--corpus", str(work / "diagnostic-inference/corpus.jsonl"),
                     "--gold", str(scoring / "gold.jsonl"),
                     "--selection", str(scoring / "selected-strata.json"),
                     "--manifest", str(scoring / "manifest.json"),
                     "--r2-run", str(ROOT / "runs/climate-scifact-train-diagnostic-20260930-r2/inference/run.json"),
                     "--output", str(result / "score.json")], cwd=source, env=env, log=result / "score.log")
            state["score_sha256"] = digest(result / "score.json")
        state["status"] = "complete"
        stage("complete")
    except BaseException as exc:
        state.update(status="failed", exception_type=type(exc).__name__)
        atomic_status(status_path, state)
        raise
    finally:
        state["elapsed_seconds"] = time.time() - state["started_unix"]
        atomic_status(status_path, state)


if __name__ == "__main__":
    main()
