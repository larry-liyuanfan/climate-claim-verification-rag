"""Exactly one tune-only pair; CPU preparation is not execution authority."""
from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import sys
import time
from typing import Any

from climate_rag.component_execution import durable
from run_scifact_grounding_train_operator import ROOT, require, runtime_check, sha

TRAINING_SHA = 'f5e6a865ec09cb67a520e36ba4646fb297a381383d4d0357208488ef6e9168a0'
ADAPTER_SHA = 'dd1974a26549b3337824252a9c38773873acce72427f92e656835bf9d71935d4'
DATA_SHA = '28de2c5d1d531aabdb757ccb45db0b91232a54ac7089ac7dc5ebf42e65ba3b2f'
CONFIG_SHA = '0e63b57b59c5dcf322dcfd620b881edc30b7e71a3ea61bcc9a3def869bcb85f2'


def fixed_tune_release(release: dict[str, Any]) -> None:
    require(release['authorization'] == 'coordinator_exact_hash_release'
            and release['purpose'] == 'evaluate_tune' and release['partition'] == 'tune'
            and release['max_runtime_seconds'] == 1500 and release['max_total_calls'] == 24
            and release['training_authorized'] is False and release['validation_authorized'] is False,
            'tune_only_release_required')
    require(release['adapter_training_sha256'] == TRAINING_SHA
            and release['adapter_model_sha256'] == ADAPTER_SHA
            and release['data_manifest_sha256'] == DATA_SHA
            and release['config_sha256'] == CONFIG_SHA, 'frozen_tune_identity')


def score_after_exit(source: Path, output: Path, bundle: Path, release_sha: str,
                     process_returncode: int, allocation: Path) -> int:
    """Never launch scorer before both inference processes have exited/reaped."""
    proof = output.with_name(output.name + '-execution') / 'worker-exit.json'
    exit_proof = json.loads(proof.read_bytes())
    require(exit_proof['child_reaped'] is True and exit_proof['release_sha256'] == release_sha
            and exit_proof['data_manifest_sha256'] == DATA_SHA
            and process_returncode == exit_proof['returncode'] % 256, 'paired_inference_exit_required')
    terminated = output / 'inference-terminated.json'
    missing_summary = any(list((output / arm).glob('slot-*/started.json'))
        and not (output / arm / 'complete.json').is_file() for arm in ('base', 'adapted'))
    if not terminated.is_file() or missing_summary:
        physical = {p.relative_to(output).as_posix(): sha(p)
                    for p in output.rglob('*') if p.is_file()}
        reservations = sum(len(list((output / arm).glob('slot-*/started.json')))
                           for arm in ('base', 'adapted'))
        require(0 <= reservations <= 24, 'interrupted_reservation_cap')
        durable(allocation / 'cost-audit-pending.json', {
            'planned_slots': 24, 'durable_reservations': reservations,
            'unreserved_slots': 24 - reservations, 'cost_audit_pending': True,
            'known_cost_totals': None, 'unknown_cost': True,
            'missing_termination': not terminated.is_file(), 'missing_arm_summary': missing_summary,
            'worker_exit_sha256': sha(proof), 'physical_files_sha256': physical,
            'quality_scored': False, 'validation_gate_open': False, 'retry_authorized': False})
        return 2
    require(sha(terminated) == exit_proof['terminated_sha256'], 'termination_hash_changed')
    durable(allocation / 'before-scoring.json', {'inference_parent_exited': True,
        'child_reaped': True, 'worker_exit_sha256': sha(proof), 'termination_sha256': sha(terminated),
        'inference_returncode': process_returncode, 'partition': 'tune', 'started_unix': time.time()})
    command = [sys.executable, str(source / 'scripts/run_scifact_grounding_candidate.py'),
        'score', '--partition', 'tune', '--bundle', str(bundle), '--data-sha', DATA_SHA,
        '--output', str(output)]
    with (allocation / 'score.log').open('xb') as log:
        scored = subprocess.run(command, stdout=log, stderr=log, check=False, timeout=150)
    return scored.returncode


def main() -> None:
    require(os.name == 'posix' and bool(os.environ.get('SLURM_JOB_ID'))
            and bool(os.environ.get('CUDA_VISIBLE_DEVICES')), 'allocated_gpu_required')
    source = Path(__file__).resolve().parents[1]
    work = Path(os.environ['CLIMATE_GROUNDING_WORK']).resolve()
    require(source.parent == work and work.name.startswith('climate-grounding-'), 'private_scratch')
    release_path = Path(os.environ['CLIMATE_GROUNDING_RELEASE_FILE'])
    release_sha = os.environ['CLIMATE_GROUNDING_RELEASE_SHA']
    require(sha(release_path) == release_sha, 'release_hash')
    release = json.loads(release_path.read_bytes())
    fixed_tune_release(release)
    require((source / 'SOURCE_REVISION').read_text().strip() == release['source_git']
            == os.environ['CLIMATE_SOURCE_GIT'], 'source_identity')
    require(release['source_archive_sha256'] == os.environ['CLIMATE_SOURCE_SHA256']
            and sha(source / 'hpc/scifact_grounding_tune.sbatch') == release['wrapper_sha256'],
            'source_wrapper_identity')
    output = ROOT / 'runs/scifact-grounding-tune-20261001-v1'
    require(release['output'] == str(output) and not output.exists(), 'fixed_unused_output')
    allocation = output.with_name(output.name + '-allocation')
    allocation.mkdir(mode=0o700)
    observed = runtime_check(release)
    adapter = ROOT / 'runs/scifact-grounding-training-20261001-v1'
    require(sha(adapter / 'complete.json') == TRAINING_SHA
            and sha(adapter / 'final/adapter_model.safetensors') == ADAPTER_SHA,
            'fixed_checkpoint_changed')
    durable(allocation / 'runtime-observed.json', observed)
    durable(allocation / 'started.json', {'source_git': release['source_git'],
        'release_sha256': release_sha, 'job_id': os.environ['SLURM_JOB_ID'],
        'started_unix': time.time(), 'partition': 'tune', 'max_calls': 24,
        'training': False, 'validation': False, 'warmup_calls': 0})
    from run_budget_agent_full_operator import ARCHIVES
    from run_scifact_component_operator import generator_only
    filename, digest = ARCHIVES['input']
    require(release['model_archive_sha256'] == digest, 'frozen_model_archive')
    generator_only(ROOT / 'envs' / filename, work / 'input', digest)
    bundle = ROOT / 'posthoc/scifact-grounding-candidate-40d84a377bd1'
    command = [sys.executable, str(source / 'scripts/run_scifact_grounding_candidate.py'),
        'evaluate', '--partition', 'tune', '--bundle', str(bundle), '--data-sha', DATA_SHA,
        '--output', str(output), '--adapter', str(adapter),
        '--model', str(work / 'input/models/generator/model'),
        '--model-manifest', str(work / 'input/models/generator/model_manifest.json'),
        '--release', str(release_path), '--release-sha', release_sha]
    # run() reaps this parent; its own durable proof attests its inference child.
    completed = subprocess.run(command, check=False)
    durable(allocation / 'inference-parent-exited.json', {'returncode': completed.returncode,
        'reaped': True, 'ended_unix': time.time()})
    # Even a nonzero partial inference can receive a cost audit, never a replay.
    score_code = score_after_exit(source, output, bundle, release_sha, completed.returncode, allocation)
    durable(allocation / 'ended.json', {'inference_returncode': completed.returncode,
        'score_returncode': score_code, 'ended_unix': time.time(), 'no_automatic_retry': True,
        'validation_launched': False})
    raise SystemExit(completed.returncode or score_code)


if __name__ == '__main__':
    main()
