"""Reuse one bounded child plus exit-before-score; a DRAFT can never execute."""
from __future__ import annotations

import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
from typing import Any

from climate_rag.scifact_adapter_regression import ADAPTER_SHA, INFERENCE_SHA, SCORING_SHA, TRAINING_SHA, require
from climate_rag.scifact_read_continuation import policy
from climate_rag.scifact_semantic_contract import INFERENCE_NAMES, SCORING_NAMES, write_once
from run_budget_agent_full_operator import ARCHIVES, read_only_tree
from run_scifact_adapter_regression_operator import ADAPTER, BUNDLES, bounded_child, extract_bound, extract_models
from run_scifact_grounding_train_operator import ROOT, runtime_check, sha

OUTPUT = ROOT/'runs/scifact-read-conditional-20261001-v1'
RESOURCE = {'gpu':'A100:1','cpus':8,'host_ram_gib':32,'scratch_gib':30,'slurm_seconds':900}


def validate_release(release: dict[str, Any], source: Path, *, draft: bool = False) -> None:
    require(release['authorization'] == ('DRAFT_NOT_AUTHORIZED' if draft else 'coordinator_exact_hash_release'), 'exact_release_required')
    require(release['purpose'] == 'three_frozen_read_conditional_calls' and release['output'] == str(OUTPUT)
        and release['policy'] == policy(source) and release['resource_cap'] == RESOURCE
        and release['max_worker_seconds'] == 480 and release['max_operator_seconds'] == 870
        and release['scoring_timeout_seconds'] == 90 and release['no_automatic_retry'] is True
        and release['training_authorized'] is False and release['validation_authorized'] is False
        and release['official_dev_test_read'] is False and release['adapter_model_sha256'] == ADAPTER_SHA
        and release['adapter_training_sha256'] == TRAINING_SHA
        and release['inference_archive_sha256'] == INFERENCE_SHA and release['scoring_archive_sha256'] == SCORING_SHA
        and release['model_archive_sha256'] == ARCHIVES['input'][1], 'frozen_conditional_contract')
    bundle = Path(release['conditional_inputs'])
    require(bundle.parent.parent == ROOT/'posthoc' and bundle.parent.name.startswith('scifact-read-continuation-')
        and bundle.name == 'conditional-inputs.json'
        and all(len(release[k]) == 64 for k in ('conditional_inputs_sha256','preparation_compact_sha256','private_membership_sha256')),
        'fixed_preparation_bundle')


def main() -> None:
    began = time.time()
    require(os.name == 'posix' and bool(os.environ.get('SLURM_JOB_ID'))
            and bool(os.environ.get('CUDA_VISIBLE_DEVICES')), 'allocated_gpu_only')
    def interrupted(number: int, frame: Any) -> None:
        raise InterruptedError(f'conditional_operator_signal_{number}')
    for number in (signal.SIGTERM, signal.SIGINT):
        signal.signal(number, interrupted)
    source = Path(__file__).resolve().parents[1]
    work = Path(os.environ['CLIMATE_GROUNDING_WORK']).resolve()
    require(source.parent == work and work.name.startswith('climate-grounding-'), 'scratch_source')
    release_path = Path(os.environ['CLIMATE_GROUNDING_RELEASE_FILE'])
    release_sha = os.environ['CLIMATE_GROUNDING_RELEASE_SHA']
    require(sha(release_path) == release_sha, 'release_sha')
    release = json.loads(release_path.read_bytes())
    validate_release(release, source)
    require((source/'SOURCE_REVISION').read_text().strip() == release['source_git'] == os.environ['CLIMATE_SOURCE_GIT']
        and release['source_archive_sha256'] == os.environ['CLIMATE_SOURCE_SHA256']
        and sha(source/'hpc/scifact_read_continuation.sbatch') == release['wrapper_sha256'], 'execution_binding')
    bundle = Path(release['conditional_inputs'])
    require(sha(bundle) == release['conditional_inputs_sha256']
        and sha(bundle.parent/'compact.json') == release['preparation_compact_sha256'], 'prepared_state_binding')
    OUTPUT.mkdir(mode=0o700)
    allocation = OUTPUT/'allocation'
    allocation.mkdir(mode=0o700)
    write_once(allocation/'started.json', {'job_id':os.environ['SLURM_JOB_ID'], 'release_sha256':release_sha,
        'source_git':release['source_git'], 'started_unix':began})
    write_once(allocation/'runtime-observed.json', runtime_check(release))
    # Reuse the audited archive extractor; reranker files may be staged but never loaded/called.
    extract_models(ROOT/'envs'/ARCHIVES['input'][0], work/'input', release['model_archive_sha256'])
    read_only_tree(work/'input')
    extract_bound(BUNDLES/'inference.tar', work/'inference-input', INFERENCE_SHA, INFERENCE_NAMES)
    model = work/'input/models/generator'
    command = [sys.executable, str(source/'scripts/run_scifact_read_continuation.py'),
        '--release',str(release_path),'--release-sha',release_sha,'--inference-dir',str(work/'inference-input'),
        '--adapter',str(ADAPTER),'--model-dir',str(model/'model'),'--model-manifest',str(model/'model_manifest.json'),
        '--output',str(OUTPUT/'inference')]
    remaining = 870 - (time.time()-began) - 90
    proof = bounded_child(command, allocation/'worker.log', min(480, remaining))
    run = OUTPUT/'inference/run.json'
    proof.update(release_sha256=release_sha, run_sha256=sha(run) if run.is_file() else None)
    write_once(allocation/'worker-exit.json', proof)
    if proof['returncode'] == 0 and not proof['parent_wait_interrupted'] and run.is_file():
        extract_bound(BUNDLES/'scoring.tar', work/'scoring', SCORING_SHA, SCORING_NAMES)
        read_only_tree(work/'scoring')
    score = [sys.executable,str(source/'scripts/score_scifact_read_continuation.py'),
        '--release',str(release_path),'--release-sha',release_sha,'--inference-dir',str(work/'inference-input'),
        '--scoring-dir',str(work/'scoring')]
    with (allocation/'score.log').open('xb') as stream:
        scored = subprocess.run(score, stdout=stream, stderr=stream, check=False, timeout=90)
    write_once(allocation/'ended.json', {'inference_returncode':proof['returncode'], 'score_returncode':scored.returncode,
        'ended_unix':time.time(), 'further_execution_authorized':False})
    raise SystemExit(proof['returncode'] or scored.returncode)


if __name__ == '__main__':
    main()
