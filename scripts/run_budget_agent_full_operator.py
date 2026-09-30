"""Preparation-only full operator. Requires a separate release and GPU allocation.

Frozen single-phase runners share disk assets, NOT resident model objects.
All generated text, row records, logs and scores remain in the private run root.
"""
from __future__ import annotations

import hashlib
import importlib.util
import json
import os
import re
import shutil
import signal
import subprocess
import sys
import tarfile
import tempfile
import time
from pathlib import Path, PurePosixPath

ROOT = Path("/data/gpfs/projects/punim2936/portfolio_20260903/climate-public-retrieval-v2")
INFERENCE = "72eaa90567e3603a3b940edffbb21b684bf1d1ba"
SCORER = "208ff931badff270cbbc9593c8ab54f5c79aec9e"
GOLD_SHA = "d2dd28422bffacf87ded2153b3bfac4ca9e1edc903e1d4e7a88d3a40f5d2fd8b"
SELECTION_SHA = "988b6682034a70966c8bbad5ff3c42202933fe85791b97a5f39e4a1b68071bc6"
PROMPT_SHA = "6b0163a4ef9da624e35dffb51fff34913ccb8a90db395c8dfec68bbe409febaa"
SCHEMA_SHA = "9743f8ba5fcf4364ebc71732ff150d879bd9cd74c95c91a0acf3b913af4535dc"
CORPUS_SHA = "c14315aee9feecbbbbc3b0c7101d978b52b47cee9a613336304bcaa736460c71"
MODEL_SHA = "d1dd9783afdf4e0fbd21eee824834d71b86982f5a5d5f6f371fe07f2f76f3cf6"
RERANKER_SHA = "de1d4ac39101816774439e68881e2308c5e5f1bd94d0b0dc4c492a56c2681052"
PHASES = ("validation", "vnext")
PROTOCOLS = {
    "validation": ("validation-protocol.json", "abfcb61e9e6a54641f025101a4011fa88e07e3697a36f0acd308fa8e86052674", 96),
    "vnext": ("authored-protocol.json", "6ee90b8479192335fc543c3425c3fbbb89fea2d65ba9698fb54d86564bc3c3c2", 24),
}
ARCHIVES = {
    "inference": (f"budget-agent-protocol-source-{INFERENCE}.tar", "ad533a3b14a81ab658049d6261cfc22848ebd792f3333e9f635d327d034d4cb1"),
    "scorer": (f"budget-agent-scorer-source-{SCORER}.tar", "7498d367565eeb59bb7cf724c33f835608a0c9f01da3768d7308700a380c00a1"),
    "input": ("budget-agent-inputs-20260929-563738f0be1f7bf7b99b8de20bcdeab552f7e893166dec260b8f7f1e7951c3c1.tar", "563738f0be1f7bf7b99b8de20bcdeab552f7e893166dec260b8f7f1e7951c3c1"),
    "runtime": ("runtime-py310-2423a755979c426ecae5d17deead154393de712efd10e84884d8c87ff00d29f8.tar.gz", "2423a755979c426ecae5d17deead154393de712efd10e84884d8c87ff00d29f8"),
    "overlay": ("budget-agent-linux-overlay-20260929-e7223ac2f4c08a736e763d485a47dcb088ce62c28fb9f1da556661a3f71da0ce.tar.gz", "e7223ac2f4c08a736e763d485a47dcb088ce62c28fb9f1da556661a3f71da0ce"),
}


def digest(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


SAFE_ERROR_CODES = frozenset({
    "allocation_required", "separate_full_release_required", "scratch_scope", "scratch_name",
    "result_scope", "frozen_gold_missing_or_mismatch", "archive_scope", "archive_sha_mismatch",
    "extraction_destination_exists", "empty_archive", "runtime_site_packages_absent",
    "archive_traversal", "archive_link_or_special", "archive_duplicate", "scratch_capacity",
    "exported_git_mismatch", "selection_sha_mismatch", "protocol_sha_mismatch", "child_failed",
    "bundle_loader_missing", "argument_phase_mismatch", "unapproved_argument", "phase_mismatch",
    "inference_identity", "dirty_inference", "provider_mismatch", "protocol_mismatch",
    "scorer_identity", "run_sha_mismatch", "matrix_incomplete", "prompt_identity",
    "schema_identity", "corpus_identity", "model_identity", "dense_disabled", "gold_identity",
    "selection_identity",
})


class OperatorError(ValueError):
    """Only allowlisted diagnostic codes may leave private logs."""


def error_code(error: BaseException) -> str:
    if (isinstance(error, OperatorError) and error.args and isinstance(error.args[0], str)
            and error.args[0] in SAFE_ERROR_CODES):
        return error.args[0]
    if isinstance(error, InterruptedError):
        return "allocation_interrupted"
    return "unexpected_error"


def require(condition: bool, code: str) -> None:
    if not condition:
        raise OperatorError(code)


def atomic_status(path: Path, state: dict) -> None:
    # Only this operator's reserved directory; never another job's state.
    with tempfile.NamedTemporaryFile(mode="w", dir=path.parent, delete=False) as stream:
        json.dump(state, stream, indent=2, sort_keys=True)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
        temporary = Path(stream.name)
    temporary.replace(path)


def safe_extract(archive: Path, target: Path, *, runtime: bool = False) -> int:
    """One extraction per archive; reject links/traversal/duplicates, precheck capacity."""
    require(not target.exists(), "extraction_destination_exists")
    with tarfile.open(archive) as bundle:
        members = bundle.getmembers()
        if runtime:
            members = [m for m in members if PurePosixPath(m.name).as_posix().startswith(
                "lib/python3.10/site-packages/")]
        require(bool(members), "runtime_site_packages_absent" if runtime else "empty_archive")
        names = set()
        for member in members:
            require(".." not in PurePosixPath(member.name).parts, "archive_traversal")
            path = (target / member.name).resolve()
            require(path.is_relative_to(target.resolve()), "archive_traversal")
            require(member.isfile() or member.isdir(), "archive_link_or_special")
            require(path not in names, "archive_duplicate")
            names.add(path)
        size = sum(m.size for m in members if m.isfile())
        require(shutil.disk_usage(target.parent).free >= size + 1024**3, "scratch_capacity")
        target.mkdir(mode=0o700)
        bundle.extractall(target, members=members)
    return size


def read_only_tree(root: Path) -> None:
    for path in root.rglob("*"):
        path.chmod(0o555 if path.is_dir() else 0o444)
    root.chmod(0o555)


def execute(command: list[str], *, cwd: Path, env: dict, log: Path) -> None:
    """Signal cleanup targets only the child process group created here."""
    with log.open("x") as stream:
        child = subprocess.Popen(command, cwd=cwd, env=env, stdout=stream,
                                 stderr=subprocess.STDOUT, start_new_session=True)
        try:
            status = child.wait()
        except BaseException:
            if child.poll() is None:
                os.killpg(child.pid, signal.SIGTERM)
                try:
                    child.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    os.killpg(child.pid, signal.SIGKILL)
                    child.wait()
            raise
    require(status == 0, "child_failed")


def run_phases(state: dict, status_path: Path, infer, score, validate) -> None:
    """Injectable callbacks allow synthetic failure/scratch-reuse tests, no model calls."""
    state["status"] = "running"
    atomic_status(status_path, state)
    for phase in PHASES:
        row = state["phases"][phase]
        row["status"] = "inference_started"
        state["stage"] = phase + "_inference"
        atomic_status(status_path, state)
        infer(phase)
        row["status"] = "scoring_started"
        state["stage"] = phase + "_scoring"
        atomic_status(status_path, state)
        score(phase)
        state["stage"] = phase + "_verify"
        atomic_status(status_path, state)
        validate(phase)
        row["status"] = "complete"
        atomic_status(status_path, state)
    state["status"] = "complete"
    state["stage"] = "complete"
    atomic_status(status_path, state)


def phase_commands(work: Path, result: Path, gold: Path, arguments: dict) -> dict:
    commands = {}
    for phase in PHASES:
        inference_args = arguments[phase]
        require(inference_args[inference_args.index("--phase") + 1] == phase, "argument_phase_mismatch")
        require("--gold" not in inference_args and "--output" not in inference_args, "unapproved_argument")
        command = [sys.executable, str(work / "scorer/scripts/score_budget_agent.py"), "--run", str(result / phase / "run.json"),
                   "--protocol", str(work / "input" / PROTOCOLS[phase][0]),
                   "--expected-protocol-sha256", PROTOCOLS[phase][1], "--phase", phase,
                   "--output", str(result / phase / "score.json")]
        if phase == "validation":
            command += ["--gold", str(gold)]
        commands[phase] = {
            "infer": [sys.executable, str(work / "inference/scripts/run_budget_agent.py"), *inference_args,
                      "--output", str(result / phase / "run.json")], "score": command}
    return commands


def verify_phase(directory: Path, phase: str) -> dict:
    run_path, score_path = directory / "run.json", directory / "score.json"
    run = json.loads(run_path.read_text())
    score = json.loads(score_path.read_text())
    receipt = json.loads((directory / "run.consumed.json").read_text())
    protocol_sha, slots = PROTOCOLS[phase][1:]
    require(run["phase"] == score["phase"] == receipt["phase"] == phase, "phase_mismatch")
    require(run["code_sha"] == score["inference_source_git_sha"] == INFERENCE, "inference_identity")
    require(run["working_tree_dirty"] is False, "dirty_inference")
    require(receipt["provider"] == "local-qwen", "provider_mismatch")
    require(run["protocol_sha256"] == score["protocol_sha256"] == receipt["protocol_sha256"] == protocol_sha,
            "protocol_mismatch")
    require(score["scorer_identity"]["git_sha"] == SCORER, "scorer_identity")
    require(score["run_sha256"] == digest(run_path), "run_sha_mismatch")
    require(score["matrix"] == {"expected_slots": slots, "actual_slots": slots,
                               "task_denominator_per_strategy": slots // 3, "validated": True}, "matrix_incomplete")
    require(run["prompt_identity"]["system_prompt_sha256"] == PROMPT_SHA, "prompt_identity")
    require(run["prompt_identity"]["schema_json_sha256"] == SCHEMA_SHA, "schema_identity")
    require(run["corpus_sha256"] == CORPUS_SHA and run["document_count"] == 5240, "corpus_identity")
    require(run["model_sha256"] == MODEL_SHA and run["reranker_sha256"] == RERANKER_SHA, "model_identity")
    require(run["dense_enabled"] is False and run["dense_file_hashes"] is None, "dense_disabled")
    require(score["gold_sha256"] == (GOLD_SHA if phase == "validation" else None), "gold_identity")
    require(score["selection_manifest_sha256"] == (SELECTION_SHA if phase == "validation" else None), "selection_identity")
    return {p.name: {"sha256": digest(p), "bytes": p.stat().st_size}
            for p in (run_path, score_path, directory / "run.consumed.json")}


def main() -> int:
    require(bool(os.environ.get("SLURM_JOB_ID")) and bool(os.environ.get("CUDA_VISIBLE_DEVICES")), "allocation_required")
    release = os.environ.get("CLIMATE_FULL_RELEASE_ID", "")
    require(bool(re.fullmatch(r"climate-full-72eaa90-[a-z0-9-]{6,64}", release)), "separate_full_release_required")
    work = Path(os.environ["CLIMATE_FULL_TASK_ROOT"])
    require(work.resolve().parent == Path(os.environ["CLIMATE_FULL_SCRATCH_PARENT"]).resolve(), "scratch_scope")
    require(work.name.startswith("climate-agent-full-"), "scratch_name")
    result = ROOT / "runs" / release
    require(result.resolve().parent == (ROOT / "runs").resolve(), "result_scope")
    # Atomic no-reuse reservation BEFORE extracting or loading anything.
    result.mkdir(mode=0o700)
    os.umask(0o077)
    status_path = result / "operator-status.json"
    state = {"status": "preparing", "stage": "reserve_result", "job_id": os.environ["SLURM_JOB_ID"], "release_id": release,
             "inference_git_sha": INFERENCE, "scorer_git_sha": SCORER,
             "operator_git_sha": os.environ["CLIMATE_OPERATOR_GIT"],
             "operator_archive_sha256": os.environ["CLIMATE_OPERATOR_SHA256"],
             "operator_script_sha256": digest(Path(__file__)),
             "wrapper_sha256": os.environ["CLIMATE_FULL_WRAPPER_SHA256"],
             "phases": {phase: {"status": "not_started"} for phase in PHASES},
             "input_extractions": 0, "disk_assets_shared": True, "resident_models_shared": False,
             "gpu_memory_peak": None, "raw_outputs_exported": False, "started_unix": time.time()}
    atomic_status(status_path, state)

    def interrupted(number, frame):
        del frame
        state["signal"] = number
        raise InterruptedError("allocation_signal")

    for number in (signal.SIGUSR1, signal.SIGTERM, signal.SIGINT):
        signal.signal(number, interrupted)

    def stage(name):
        state["stage"] = name
        atomic_status(status_path, state)

    try:
        stage("verify_gold")
        gold = ROOT / "envs" / f"budget-agent-validation-gold-{GOLD_SHA}.json"
        require(gold.is_file() and digest(gold) == GOLD_SHA, "frozen_gold_missing_or_mismatch")
        state["gold_sha256"] = GOLD_SHA
        state["archives"] = {}
        state["extracted_bytes"] = {}
        for name, (filename, expected) in ARCHIVES.items():
            stage("hash_" + name)
            archive = ROOT / "envs" / filename
            require(archive.resolve().is_relative_to(ROOT / "envs"), "archive_scope")
            require(digest(archive) == expected, "archive_sha_mismatch")
            state["archives"][name] = {"sha256": expected, "bytes": archive.stat().st_size}
            stage("extract_" + name)
            state["extracted_bytes"][name] = safe_extract(archive, work / name, runtime=name == "runtime")
            if name == "input":
                state["input_extractions"] += 1
            atomic_status(status_path, state)
        stage("verify_sources_and_inputs")
        for name, revision in (("inference", INFERENCE), ("scorer", SCORER)):
            require((work / name / "SOURCE_REVISION").read_text().strip() == revision, "exported_git_mismatch")
        selection = work / "scorer/docs/verified-runs/budget-agent-validation-selection-20260929.json"
        require(digest(selection) == SELECTION_SHA, "selection_sha_mismatch")
        state["selection_sha256"] = SELECTION_SHA
        for phase, (filename, expected, _) in PROTOCOLS.items():
            require(digest(work / "input" / filename) == expected, "protocol_sha_mismatch")
        state["input_members"] = {p.name: {"sha256": digest(p), "bytes": p.stat().st_size}
                                  for p in (work / "input").iterdir() if p.is_file()}
        state["model_manifest_files"] = {
            role: digest(work / f"input/models/{role}/model_manifest.json")
            for role in ("generator", "reranker")}
        read_only_tree(work / "input")
        for name in ("inference", "scorer"):
            read_only_tree(work / name)
        stage("install_overlay")
        execute([sys.executable, "-m", "pip", "install", "--no-index", "--no-deps", "--require-hashes",
                 "--no-compile", "--no-cache-dir", "--disable-pip-version-check", "--target", str(work / "overlay-site"),
                 "--find-links", str(work / "overlay/wheels"), "-r", str(work / "overlay/requirements.lock")],
                cwd=work, env=dict(os.environ), log=result / "runtime-install.log")
        env = dict(os.environ)
        for key in list(env):
            if key.startswith("CLIMATE_"):
                env.pop(key)
        env.update(PYTHONDONTWRITEBYTECODE="1", PYTHONNOUSERSITE="1", HF_HUB_OFFLINE="1",
                   TRANSFORMERS_OFFLINE="1", HF_HUB_DISABLE_TELEMETRY="1",
                   HF_HOME=str(work / "cache"), XDG_CACHE_HOME=str(work / "cache"))
        module_path = os.environ.get("PYTHONPATH", "")

        def environment(source: str) -> dict:
            return {**env, "PYTHONPATH": os.pathsep.join((str(work / "overlay-site"),
                    str(work / "runtime/lib/python3.10/site-packages"), str(work / source / "src"), module_path))}

        stage("verify_dependencies")
        execute([sys.executable, "-c", "import importlib.metadata; assert importlib.metadata.version('langchain-core') == '1.6.5'"],
                cwd=work, env=environment("inference"), log=result / "dependency-check.log")
        stage("resolve_frozen_arguments")
        spec = importlib.util.spec_from_file_location("frozen_bundle", work / "inference/src/climate_rag/runtime_bundle.py")
        require(spec is not None and spec.loader is not None, "bundle_loader_missing")
        bundle_module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(bundle_module)
        arguments = {}
        for phase in PHASES:
            arguments[phase] = bundle_module.resolve_bundle_args(
                json.loads((work / f"input/args-{phase}.json").read_text()), work / "input")
            require(arguments[phase][arguments[phase].index("--phase") + 1] == phase, "argument_phase_mismatch")
            directory = result / phase
            directory.mkdir(mode=0o700)
            (directory / "private-responses").mkdir(mode=0o700)
        commands = phase_commands(work, result, gold, arguments)

        def infer(phase):
            directory = result / phase
            child_env = environment("inference")
            child_env["CLIMATE_PRIVATE_RESPONSE_DIR"] = str(directory / "private-responses")
            execute(commands[phase]["infer"], cwd=work / "inference", env=child_env,
                    log=directory / "inference.log")

        def score(phase):
            directory = result / phase
            execute(commands[phase]["score"], cwd=work / "scorer", env=environment("scorer"), log=directory / "scoring.log")

        def validate(phase):
            state["phases"][phase]["outputs"] = verify_phase(result / phase, phase)

        run_phases(state, status_path, infer, score, validate)
        return 0
    except BaseException as error:
        state["status"] = "interrupted" if isinstance(error, InterruptedError) else "failed"
        state["error_type"] = type(error).__name__  # never raw model text / exception input
        state["error_code"] = error_code(error)
        for row in state["phases"].values():
            if row["status"] in {"inference_started", "scoring_started"}:
                row["status"] = "incomplete"
        raise
    finally:
        # Completed/partial phase files persist even on signal. Hard SIGKILL can leave
        # an honest started/incomplete state, never a synthetic completed matrix.
        state["finished_unix"] = time.time()
        state["elapsed_seconds"] = state["finished_unix"] - state["started_unix"]
        state["file_receipts"] = {
            p.relative_to(result).as_posix(): {"sha256": digest(p), "bytes": p.stat().st_size}
            for p in result.rglob("*") if p.is_file() and p != status_path}
        atomic_status(status_path, state)


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as error:
        print("Full operator failed: " + type(error).__name__ + " [" + error_code(error) + "]", file=sys.stderr)
        raise SystemExit(1) from None
