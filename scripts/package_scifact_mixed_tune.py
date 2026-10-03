"""Freeze one tune12 candidate from the accepted checkpoint; never execute it."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import tarfile
from typing import Any

from climate_rag.scifact_mixed_checkpoint import binding_identity
from climate_rag.scifact_mixed_launch import (
    ARCHIVE_SHA, PYTHON_EXECUTABLE, RESOURCE_CAP, ROOT,
    RUNTIME_FILES_SHA, RUNTIME_RECEIPT_SHA,
)
from climate_rag.scifact_read_continuation import ordered_write
from climate_rag.scifact_semantic_contract import sha
from climate_rag.scifact_state_supervision import require
from package_scifact_source import package, run_shell_guard
from run_scifact_grounding_tune_operator import CONFIG_SHA, DATA_SHA, fixed_tune_release

WRAPPER = 'hpc/scifact_grounding_tune.sbatch'
CLOSEOUT = 'docs/verified-runs/scifact-mixed-training-closeout-31858295.json'


def release_candidate(revision: str, receipt: dict[str, Any], closeout_raw: bytes,
                      wrapper_sha: str) -> dict[str, Any]:
    closeout = json.loads(closeout_raw)
    require(closeout['status'] == 'training_accepted_evaluation_not_released'
        and closeout['job_id'] == '31858295' and closeout['slurm_state'] == 'COMPLETED'
        and closeout['exit_code'] == '0:0' and closeout['independent_coordinator_acceptance'] is True
        and closeout['denominators'] == {'claims':144,'decision_records':223,'optimizer_updates':36},
        'accepted_training_required_for_candidate')
    binding = closeout['checkpoint_binding']
    identity = binding_identity(binding)
    release = {'authorization':'coordinator_exact_hash_release','purpose':'evaluate_tune',
        'requires_separate_coordinator_submission_authorization':True,
        'partition':'tune','max_runtime_seconds':1500,'max_total_calls':24,
        'training_authorized':False,'validation_authorized':False,'next_validation_calls':0,
        'checkpoint_binding':binding,'adapter_training_sha256':identity['adapter_training_sha256'],
        'adapter_model_sha256':identity['adapter_model_sha256'],
        'data_manifest_sha256':DATA_SHA,'config_sha256':CONFIG_SHA,
        'source_git':revision,'source_archive_sha256':receipt['source_archive_sha256'],
        'source_archive_bytes':receipt['source_archive_bytes'],
        'output':(ROOT/'runs'/('scifact-mixed-tune-'+revision[:12])).as_posix(),
        'training_closeout_path':CLOSEOUT,'training_closeout_sha256':sha(closeout_raw),
        'wrapper_path':WRAPPER,'wrapper_sha256':wrapper_sha,'model_archive_sha256':ARCHIVE_SHA,
        'runtime_receipt_sha256':RUNTIME_RECEIPT_SHA,'runtime_files_sha256':RUNTIME_FILES_SHA,
        'python_executable':PYTHON_EXECUTABLE,'resource_cap':RESOURCE_CAP,
        'budget_boundary':'base12_then_adapted12_same_inputs_greedy_no_extra_warmup',
        'actual_PEFT_reload_required':True,'scoring_order':'physical_wire_costs_then_gold',
        'official_dev_test_read':False,'validation12_read':False,'automatic_retry':False,
        'further_execution_authorized':False,'job_submitted':False}
    fixed_tune_release(release)
    return release


def freeze(repo: Path, revision: str, output: Path, bash: str) -> dict[str, Any]:
    receipt = package(repo,revision,output,bash)
    archive = output/'source.tar'
    with tarfile.open(archive) as bundle:
        def content(name: str) -> bytes:
            stream = bundle.extractfile(name)
            if stream is None:
                raise ValueError('source_member_missing:'+name)
            return stream.read()
        raw, closeout = content(WRAPPER), content(CLOSEOUT)
    wrapper = output/'mixed-tune.sbatch'
    with wrapper.open('xb') as stream:
        stream.write(raw)
    run_shell_guard(archive,revision,wrapper,bash)
    wrong = output/'wrong-tune-wrapper.sbatch'
    with wrong.open('xb') as stream:
        stream.write(raw+b'# synthetic mismatch\n')
    empty = output/'wrong-tune-archive.tar'
    with tarfile.open(empty,'w'):
        pass
    rejected = []
    for name,tar,commit,shell in (('revision',archive,'0'*40,wrapper),
            ('wrapper',archive,revision,wrong),('archive',empty,revision,wrapper)):
        try:
            run_shell_guard(tar,commit,shell,bash)
        except subprocess.CalledProcessError:
            rejected.append(name)
        else:
            raise ValueError('actual_tune_guard_accepted_mismatch:'+name)
    release = release_candidate(revision,receipt,closeout,sha(raw))
    path = output/'mixed-tune-release-candidate.json'
    ordered_write(path,release)
    result = {'source_git':revision,'source_archive_sha256':receipt['source_archive_sha256'],
        'source_archive_bytes':receipt['source_archive_bytes'],'wrapper_sha256':sha(raw),
        'release_candidate_sha256':sha(path.read_bytes()),
        'checkpoint_binding_sha256':binding_identity(release['checkpoint_binding'])['binding_sha256'],
        'actual_tune_wrapper_guard_passed':True,'rejected_mismatches':rejected,
        'job_submitted':False,'model_calls':0,'gold_read':False}
    ordered_write(output/'mixed-tune-source-receipt.json',result)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--repo',type=Path,required=True)
    parser.add_argument('--commit',required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--bash',required=True)
    args = parser.parse_args()
    print(freeze(args.repo,args.commit,args.output,args.bash))


if __name__ == '__main__':
    main()
