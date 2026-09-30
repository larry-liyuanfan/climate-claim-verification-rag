"""Bounded single inference child, then exit/reap proof, then existing gold scorer."""
from __future__ import annotations

import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import tarfile
import time
from typing import Any, Callable, cast

from climate_rag.scifact_adapter_regression import (
    ADAPTER_SHA, INFERENCE_SHA, SCORING_SHA, TRAINING_SHA, checkpoint_metadata, policy_identity, require,
)
from climate_rag.scifact_semantic_contract import INFERENCE_NAMES, SCORING_NAMES, write_once
from run_budget_agent_full_operator import ARCHIVES, read_only_tree
from run_scifact_bounded_operator import extract_exact
from run_scifact_grounding_train_operator import ROOT, runtime_check, sha

extract_bound = cast(Callable[[Path, Path, str, Any], Any], extract_exact)

OUTPUT = ROOT / 'runs/scifact-adapter-bare-regression-20261001-v1'
ADAPTER = ROOT / 'runs/scifact-grounding-training-20261001-v1'
BUNDLES = ROOT / 'envs/scifact-semantic-bundles-99cd9ff'
RESOURCE = {'gpu': 'A100:1', 'cpus': 8, 'host_ram_gib': 32, 'scratch_gib': 30, 'slurm_seconds': 7200}


def validate_release(release: dict[str, Any], source: Path, *, draft: bool = False) -> None:
    require(release['authorization'] == ('DRAFT_NOT_AUTHORIZED' if draft else 'coordinator_exact_hash_release'),
            'explicit_exact_release_required')
    require(release['purpose'] == 'old12_four_route_regression' and release['output'] == str(OUTPUT)
            and release['policy'] == policy_identity(source)
            and release['adapter_training_sha256'] == TRAINING_SHA and release['adapter_model_sha256'] == ADAPTER_SHA
            and release['inference_archive_sha256'] == INFERENCE_SHA and release['scoring_archive_sha256'] == SCORING_SHA
            and release['model_archive_sha256'] == ARCHIVES['input'][1]
            and release['max_worker_seconds'] == 6300 and release['max_operator_seconds'] == 6900
            and release['scoring_timeout_seconds'] == 180 and release['resource_cap'] == RESOURCE
            and release['training_authorized'] is False and release['validation_authorized'] is False
            and release['official_dev_test_read'] is False and release['no_automatic_retry'] is True,
            'frozen_regression_contract')


def extract_models(archive: Path, target: Path, digest: str) -> int:
    require(not target.exists() and sha(archive) == digest, 'model_archive')
    with tarfile.open(archive) as bundle:
        members = [m for m in bundle.getmembers() if any(m.name.startswith(f'models/{kind}/model/')
            or m.name == f'models/{kind}/model_manifest.json' for kind in ('generator', 'reranker'))]
        paths = [(target / m.name).resolve() for m in members]
        require(bool(members) and len(paths) == len(set(paths))
                and all(m.isfile() and p.is_relative_to(target.resolve()) for m, p in zip(members, paths, strict=True)),
                'model_member_allowlist')
        size = sum(m.size for m in members)
        require(shutil.disk_usage(target.parent).free >= size + 1024**3, 'model_scratch_capacity')
        target.mkdir(mode=0o700)
        bundle.extractall(target, members=members)
    return size


def bounded_child(command: list[str], log: Path, timeout: float) -> dict[str, Any]:
    require(timeout > 0, 'no_remaining_inference_time')
    started = time.time()
    timed_out = False
    interrupted: str | None = None
    with log.open('xb') as stream:
        child = subprocess.Popen(command, stdout=stream, stderr=stream, start_new_session=True)
        try:
            code = child.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            timed_out = True
            try:
                getattr(os, 'killpg')(child.pid, getattr(signal, 'SIGKILL'))
            except ProcessLookupError:
                pass
            code = child.wait()
        except BaseException as exc:
            # Only this process's isolated child group; never a Slurm/user-wide kill.
            interrupted = type(exc).__name__
            if child.poll() is None:
                try:
                    getattr(os, 'killpg')(child.pid, getattr(signal, 'SIGKILL'))
                except ProcessLookupError:
                    pass
            code = child.wait()
    return {'child_reaped': True, 'returncode': code, 'timed_out': timed_out,
            'parent_wait_interrupted': interrupted,
            'started_unix': started, 'ended_unix': time.time(), 'hard_timeout_seconds': timeout}


def main() -> None:
    began = time.time()
    require(os.name == 'posix' and bool(os.environ.get('SLURM_JOB_ID'))
            and bool(os.environ.get('CUDA_VISIBLE_DEVICES')), 'allocated_gpu_required')
    def interrupted(number: int, frame: Any) -> None:
        raise InterruptedError(f'allocated_operator_signal_{number}')
    for number in (signal.SIGTERM, signal.SIGINT):
        signal.signal(number, interrupted)
    source = Path(__file__).resolve().parents[1]
    work = Path(os.environ['CLIMATE_GROUNDING_WORK']).resolve()
    require(source.parent == work and work.name.startswith('climate-grounding-'), 'private_scratch')
    release_path = Path(os.environ['CLIMATE_GROUNDING_RELEASE_FILE'])
    release_sha = os.environ['CLIMATE_GROUNDING_RELEASE_SHA']
    require(sha(release_path) == release_sha, 'release_hash')
    release = json.loads(release_path.read_bytes())
    validate_release(release, source)
    require((source / 'SOURCE_REVISION').read_text().strip() == release['source_git']
            == os.environ['CLIMATE_SOURCE_GIT'], 'exact_source')
    require(release['source_archive_sha256'] == os.environ['CLIMATE_SOURCE_SHA256']
            and sha(source / 'hpc/scifact_adapter_regression.sbatch') == release['wrapper_sha256'], 'wrapper_identity')
    require(not OUTPUT.exists(), 'fixed_unused_output')
    OUTPUT.mkdir(mode=0o700)
    allocation = OUTPUT / 'allocation'
    allocation.mkdir(mode=0o700)
    write_once(allocation / 'started.json', {'source_git': release['source_git'], 'release_sha256': release_sha,
        'job_id': os.environ['SLURM_JOB_ID'], 'started_unix': began, 'training': False, 'validation': False})
    write_once(allocation / 'runtime-observed.json', runtime_check(release))
    checkpoint_metadata(ADAPTER)
    extract_models(ROOT / 'envs' / ARCHIVES['input'][0], work / 'input', release['model_archive_sha256'])
    read_only_tree(work / 'input')
    extract_bound(BUNDLES / 'inference.tar', work / 'inference-input', INFERENCE_SHA, INFERENCE_NAMES)
    read_only_tree(work / 'inference-input')
    command = [sys.executable, str(source / 'scripts/run_scifact_adapter_regression.py'),
        '--inference-dir', str(work / 'inference-input'), '--adapter', str(ADAPTER),
        '--output', str(OUTPUT / 'inference'), '--release', str(release_path), '--release-sha', release_sha]
    for kind in ('model', 'reranker'):
        directory = work / 'input/models' / ('generator' if kind == 'model' else kind)
        command.extend(['--' + kind + '-dir', str(directory / 'model'), '--' + kind + '-manifest', str(directory / 'model_manifest.json')])
    remaining = release['max_operator_seconds'] - (time.time() - began) - release['scoring_timeout_seconds']
    exit_proof = bounded_child(command, allocation / 'worker.log', min(release['max_worker_seconds'], remaining))
    completed = OUTPUT / 'inference/run.json'
    exit_proof.update(release_sha256=release_sha, run_sha256=sha(completed) if completed.exists() else None)
    write_once(allocation / 'worker-exit.json', exit_proof)
    # Only now may scoring inputs be extracted. A partial inference gets costs,
    # but never loads gold or creates a quality gate.
    if exit_proof['returncode'] == 0 and not exit_proof['parent_wait_interrupted'] and completed.is_file():
        extract_bound(BUNDLES / 'scoring.tar', work / 'scoring', SCORING_SHA, SCORING_NAMES)
        read_only_tree(work / 'scoring')
    score_command = [sys.executable, str(source / 'scripts/score_scifact_adapter_regression.py'),
        '--output', str(OUTPUT), '--inference-dir', str(work / 'inference-input'), '--scoring-dir', str(work / 'scoring'),
        '--release', str(release_path), '--release-sha', release_sha]
    with (allocation / 'score.log').open('xb') as stream:
        scored = subprocess.run(score_command, stdout=stream, stderr=stream, check=False,
                                timeout=release['scoring_timeout_seconds'])
    write_once(allocation / 'ended.json', {'inference_returncode': exit_proof['returncode'],
        'score_returncode': scored.returncode, 'ended_unix': time.time(), 'no_automatic_retry': True,
        'validation_launched': False, 'further_execution_authorized': False})
    raise SystemExit(exit_proof['returncode'] or scored.returncode)


if __name__ == '__main__':
    main()
