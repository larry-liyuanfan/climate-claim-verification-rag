"""Build an exact CPU-reviewed candidate. Never submit or run inference."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import tarfile
from typing import Any

from climate_rag.scifact_adapter_regression import INFERENCE_SHA, SCORING_SHA
from climate_rag.scifact_evidence_note import VERSION
from climate_rag.scifact_mixed_launch import (
    ARCHIVE_SHA, PYTHON_EXECUTABLE, ROOT, RUNTIME_FILES_SHA, RUNTIME_RECEIPT_SHA,
)
from climate_rag.scifact_read_continuation import ordered_write
from climate_rag.scifact_semantic_contract import sha
from package_scifact_source import package, run_shell_guard
from run_scifact_evidence_note_operator import CONTROL_HASHES, RESOURCE, STATES_SHA, WRAPPER, policy, validate_release


def freeze(repo: Path, revision: str, output: Path, bash: str) -> dict[str, Any]:
    receipt = package(repo, revision, output, bash)
    archive = output / 'source.tar'
    with tarfile.open(archive) as bundle:
        def content(name: str) -> bytes:
            stream = bundle.extractfile(name)
            if stream is None:
                raise ValueError('source_member_missing')
            return stream.read()
        raw = content(WRAPPER)
        binding = json.loads(content('docs/verified-runs/scifact-mixed-training-closeout-31858295.json'))['checkpoint_binding']
    wrapper = output / 'evidence-note.sbatch'
    with wrapper.open('xb') as stream:
        stream.write(raw)
    run_shell_guard(archive, revision, wrapper, bash)
    wrong = output / 'wrong-note-wrapper.sbatch'
    with wrong.open('xb') as stream:
        stream.write(raw + b'# synthetic mismatch\n')
    empty = output / 'wrong-note-archive.tar'
    with tarfile.open(empty, 'w'):
        pass
    rejected = []
    for name, tar, commit, shell in (('revision', archive, '0' * 40, wrapper),
            ('wrapper', archive, revision, wrong), ('archive', empty, revision, wrapper)):
        try:
            run_shell_guard(tar, commit, shell, bash)
        except subprocess.CalledProcessError:
            rejected.append(name)
        else:
            raise ValueError('note_shell_guard_accepted_mismatch:' + name)
    release = {'authorization': 'coordinator_exact_hash_release', 'purpose': VERSION,
        'requires_separate_coordinator_submission_authorization': True,
        'source_git': revision, 'source_archive_sha256': receipt['source_archive_sha256'],
        'source_archive_bytes': receipt['source_archive_bytes'], 'wrapper_path': WRAPPER, 'wrapper_sha256': sha(raw),
        'output': (ROOT / 'runs' / ('scifact-evidence-note-' + revision[:12])).as_posix(),
        'policy': policy(repo, binding), 'checkpoint_binding': binding,
        'control': CONTROL_HASHES, 'states_sha256': STATES_SHA,
        'inference_archive_sha256': INFERENCE_SHA, 'scoring_archive_sha256': SCORING_SHA,
        'model_archive_sha256': ARCHIVE_SHA, 'runtime_receipt_sha256': RUNTIME_RECEIPT_SHA,
        'runtime_files_sha256': RUNTIME_FILES_SHA, 'python_executable': PYTHON_EXECUTABLE,
        'resource_cap': RESOURCE, 'max_worker_seconds': 3300, 'max_operator_seconds': 3900, 'scoring_timeout_seconds': 180,
        'training_authorized': False, 'validation_authorized': False, 'no_automatic_retry': True,
        'walltime_basis': '31871386 elapsed189s incl reranker load; 24x120s hard stage ceilings + loading/audit margin, no reranker load now',
        'job_submitted': False, 'further_execution_authorized': False}
    validate_release(release, repo)
    path = output / 'evidence-note-release-candidate.json'
    ordered_write(path, release)
    result = {'source_git': revision, 'source_archive_sha256': receipt['source_archive_sha256'],
        'source_archive_bytes': receipt['source_archive_bytes'], 'wrapper_sha256': sha(raw),
        'release_candidate_sha256': sha(path.read_bytes()), 'actual_note_wrapper_guard_passed': True,
        'rejected_mismatches': rejected, 'job_submitted': False, 'model_calls': 0, 'gold_read': False}
    ordered_write(output / 'evidence-note-source-receipt.json', result)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('repo', 'output'):
        parser.add_argument('--' + name, type=Path, required=True)
    parser.add_argument('--commit', required=True)
    parser.add_argument('--bash', required=True)
    args = parser.parse_args()
    print(json.dumps(freeze(args.repo, args.commit, args.output, args.bash)))


if __name__ == '__main__':
    main()
