"""One isolated worker, reaped exit, durable physical costs, then separate scorer."""
from __future__ import annotations

import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
from typing import Any

from climate_rag.scifact_adapter_regression import INFERENCE_SHA, SCORING_SHA, checkpoint_metadata, require
from climate_rag.scifact_evidence_note import CAPS, NOTE_BOUNDARY, NOTE_SCHEMA, NOTE_SYSTEM, VERSION
from climate_rag.scifact_mixed_checkpoint import binding_identity
from climate_rag.scifact_mixed_launch import ROOT, ARCHIVE_NAME, ARCHIVE_SHA
from climate_rag.scifact_semantic_contract import INFERENCE_NAMES, SCORING_NAMES, checked, encoded, sha
from climate_rag.scifact_read_continuation import ordered_write as write_once
from run_budget_agent_full_operator import read_only_tree
from run_scifact_adapter_regression_operator import BUNDLES, extract_bound
from run_scifact_component_operator import generator_only
from run_scifact_grounding_train_operator import runtime_check

CONTROL = ROOT / 'runs/scifact-mixed-four-route-6f764f824125'
STATES = ROOT / 'posthoc/scifact-selected-read-opportunity-dcbe368c1c02/private/states-before-gold.json'
STATES_SHA = '5a49b53aa6ab7790231b224b5966f3e8e6f2fff1856219c1c177af16ded714b8'
CONTROL_HASHES = {
    'inference/run.json': '9680efeee449a7d444d6d3c8fa450ce01a3049533c01753b6dd68da42bf92575',
    'allocation/worker-exit.json': '07b39c67c65fd3193fde17a2b0c6669035d87bc89a514e85b8364ed6a0fb7959',
    'cost-before-quality.json': '649a1aac7b71570503b1a522251d017de0e12cae764ab3e295e32c353ba96ddb',
    'score.json': 'ff103f1db57b55c30f32121e1ef639e32b68eb6b9140f64c0a07ef5015fcbc23'}
WRAPPER = 'hpc/scifact_evidence_note.sbatch'
RESOURCE = {'gpu': 'A100:1', 'cpus': 8, 'host_ram_gib': 32, 'scratch_gib': 30, 'slurm_seconds': 4200}


def policy(source: Path, binding: dict[str, Any]) -> dict[str, Any]:
    names = ('src/climate_rag/scifact_evidence_note.py', 'src/climate_rag/local_scifact_provider.py',
        'src/climate_rag/local_bounded_scifact_provider.py', 'src/climate_rag/scifact_terminal.py',
        'src/climate_rag/scifact_bounded_runtime.py', 'src/climate_rag/scifact_bounded_agent.py',
        'scripts/run_scifact_evidence_note.py', 'scripts/run_scifact_evidence_note_operator.py',
        'scripts/score_scifact_evidence_note.py')
    return {'version': VERSION, 'caps': CAPS, 'adapter_active': True, **binding_identity(binding),
        'scope': 'all_old12_exposed_TRAIN_fixed_rerank_first_frames_not_gate',
        'note_system_sha256': sha(NOTE_SYSTEM.encode()), 'note_schema_sha256': sha(encoded(NOTE_SCHEMA)),
        'terminal_boundary_sha256': sha(NOTE_BOUNDARY.encode()),
        'implementation_sha256': {n: sha((source / n).read_bytes()) for n in names},
        'greedy': True, 'thinking': False, 'truncate_or_repack': False, 'execute_tools': False,
        'budget_interpretation': '512_note_plus_512_terminal_not_equal_to_old512',
        'sealed_splits': ['validation12', 'dev300', 'retired_test']}


def validate_release(release: dict[str, Any], source: Path) -> None:
    require(release['authorization'] == 'coordinator_exact_hash_release'
        and release['purpose'] == VERSION and release['policy'] == policy(source, release['checkpoint_binding'])
        and release['control'] == CONTROL_HASHES and release['states_sha256'] == STATES_SHA
        and release['inference_archive_sha256'] == INFERENCE_SHA and release['scoring_archive_sha256'] == SCORING_SHA
        and release['model_archive_sha256'] == ARCHIVE_SHA and release['resource_cap'] == RESOURCE
        and release['max_worker_seconds'] == 3300 and release['max_operator_seconds'] == 3900
        and release['scoring_timeout_seconds'] == 180
        and release['output'] == (ROOT / 'runs' / ('scifact-evidence-note-' + release['source_git'][:12])).as_posix()
        and release['training_authorized'] is False and release['validation_authorized'] is False
        and release['no_automatic_retry'] is True, 'exact_note_release_only')


def bounded_worker(command: list[str], allocation: Path, inference: Path, seconds: float) -> dict[str, Any]:
    """Only our isolated child group is killed; reservation watchdog has no retries."""
    require(os.name == 'posix' and seconds > 0, 'POSIX_positive_timeout_required')
    child: subprocess.Popen[bytes] | None = None
    code: int | None = None
    error: str | None = None
    launching, pending = True, False
    started = time.time()
    handlers: dict[signal.Signals, Any] = {}
    def stop(number: int, frame: Any) -> None:
        nonlocal pending
        if launching:
            pending = True
        else:
            raise InterruptedError('operator_signal')
    for sig in (signal.SIGTERM, signal.SIGINT):
        handlers[sig] = signal.signal(sig, stop)
    try:
        with (allocation / 'worker.log').open('xb') as stream:
            if pending:
                raise InterruptedError('launch_cancelled')
            child = subprocess.Popen(command, stdout=stream, stderr=stream, start_new_session=True)
            launching = False
            if pending:
                raise InterruptedError('launch_interrupted')
            while child.poll() is None:
                if time.time() - started >= seconds:
                    raise TimeoutError('worker_deadline')
                for reservation in inference.glob('case-*/*/reserved.json'):
                    if not (reservation.parent / 'finished.json').exists():
                        # A concurrent exclusive write may still be incomplete.
                        try:
                            stamp = json.loads(reservation.read_bytes())['started_unix']
                        except json.JSONDecodeError:
                            stamp = reservation.stat().st_mtime
                        if time.time() - stamp >= 120:
                            raise TimeoutError('stage_deadline')
                time.sleep(0.1)
            code = child.wait()
    except BaseException as exc:
        error = type(exc).__name__
    finally:
        for sig in handlers:
            signal.signal(sig, signal.SIG_IGN)
        try:
            if child is not None:
                if child.poll() is None:
                    try:
                        getattr(os, 'killpg')(child.pid, signal.SIGKILL)
                    except ProcessLookupError:
                        pass
                code = child.wait()
        finally:
            for sig, handler in handlers.items():
                signal.signal(sig, handler)
    return {'child_started': child is not None, 'child_reaped': child is not None and code is not None,
        'returncode': code, 'interrupted': error, 'started_unix': started, 'ended_unix': time.time(),
        'watchdog_poll_seconds': 0.1, 'max_stage_seconds': 120, 'automatic_retry': False}


def main() -> None:
    began = time.time()
    require(os.name == 'posix' and bool(os.environ.get('SLURM_JOB_ID'))
            and bool(os.environ.get('CUDA_VISIBLE_DEVICES')), 'allocated_gpu_required')
    source = Path(__file__).resolve().parents[1]
    work = Path(os.environ['CLIMATE_GROUNDING_WORK']).resolve()
    require(source.parent == work and work.name.startswith('climate-grounding-'), 'private_scratch')
    release_path = Path(os.environ['CLIMATE_GROUNDING_RELEASE_FILE'])
    release_sha = os.environ['CLIMATE_GROUNDING_RELEASE_SHA']
    release = json.loads(checked(release_path, release_sha))
    validate_release(release, source)
    require((source / 'SOURCE_REVISION').read_text().strip() == release['source_git'] == os.environ['CLIMATE_SOURCE_GIT']
        and release['source_archive_sha256'] == os.environ['CLIMATE_SOURCE_SHA256']
        and sha((source / WRAPPER).read_bytes()) == release['wrapper_sha256'], 'exact_source_wrapper')
    output = Path(release['output'])
    output.mkdir(mode=0o700)
    allocation = output / 'allocation'
    allocation.mkdir(mode=0o700)
    write_once(allocation / 'started.json', {'release_sha256': release_sha, 'source_git': release['source_git'],
        'job_id': os.environ['SLURM_JOB_ID'], 'started_unix': began})
    write_once(allocation / 'runtime-observed.json', runtime_check(release))
    checkpoint_metadata(Path(release['checkpoint_binding']['directory']), release['checkpoint_binding'])
    # Only old inference/state metadata is read before the new worker exits.
    checked(CONTROL / 'inference/run.json', CONTROL_HASHES['inference/run.json'])
    checked(STATES, STATES_SHA)
    generator_only(ROOT / 'envs' / ARCHIVE_NAME, work / 'input', ARCHIVE_SHA)
    read_only_tree(work / 'input')
    extract_bound(BUNDLES / 'inference.tar', work / 'inference-input', INFERENCE_SHA, INFERENCE_NAMES)
    read_only_tree(work / 'inference-input')
    common = ['--release', str(release_path), '--release-sha', release_sha, '--inference-dir', str(work / 'inference-input')]
    command = [sys.executable, str(source / 'scripts/run_scifact_evidence_note.py'), *common,
        '--model-dir', str(work / 'input/models/generator/model'),
        '--model-manifest', str(work / 'input/models/generator/model_manifest.json'), '--output', str(output / 'inference')]
    remaining = release['max_operator_seconds'] - (time.time() - began) - 180
    proof = bounded_worker(command, allocation, output / 'inference', min(remaining, release['max_worker_seconds']))
    run = output / 'inference/run.json'
    proof.update(release_sha256=release_sha, run_sha256=sha(run.read_bytes()) if run.exists() else None)
    write_once(allocation / 'worker-exit.json', proof)
    from score_scifact_evidence_note import collect_costs
    costs = collect_costs(output, release, release_sha)
    # Durable cost identity precedes even extracting scoring inputs.
    write_once(output / 'cost-before-quality.json', costs)
    score_code = 2
    if costs['quality_ready']:
        extract_bound(BUNDLES / 'scoring.tar', work / 'scoring', SCORING_SHA, SCORING_NAMES)
        read_only_tree(work / 'scoring')
        with (allocation / 'score.log').open('xb') as stream:
            scored = subprocess.run([sys.executable, str(source / 'scripts/score_scifact_evidence_note.py'),
                *common, '--output', str(output), '--scoring-dir', str(work / 'scoring')],
                stdout=stream, stderr=stream, timeout=180, check=False)
        score_code = scored.returncode
    write_once(allocation / 'ended.json', {'inference_returncode': proof['returncode'], 'score_returncode': score_code,
        'ended_unix': time.time(), 'validation_launched': False, 'further_execution_authorized': False})
    raise SystemExit(proof['returncode'] or score_code)


if __name__ == '__main__':
    main()
