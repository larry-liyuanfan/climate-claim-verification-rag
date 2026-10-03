"""Freeze a GPU release candidate; no upload, allocation or model access."""
from __future__ import annotations

import argparse
from pathlib import Path
import subprocess
import tarfile
from typing import Any

from climate_rag.scifact_mixed_launch import (
    ARCHIVE_NAME, ARCHIVE_SHA, MANIFEST_RELATIVE, MODEL_RELATIVE, PYTHON_EXECUTABLE,
    RESOURCE_CAP, ROOT, RUNTIME_FILES_SHA, RUNTIME_RECEIPT_SHA, WRAPPER,
    executable_names, preparation_fields,
)
from climate_rag.scifact_mixed_training import VERSION, config_sha
from climate_rag.scifact_read_continuation import ordered_write
from climate_rag.scifact_semantic_contract import MODEL_SHA, sha
from package_scifact_source import package, run_shell_guard


def freeze(repo: Path, revision: str, output: Path, bash: str, preparation: Path) -> dict[str, Any]:
    inputs = preparation_fields(preparation.read_bytes())  # Fail before producing a candidate on failed/unaccepted CPU data.
    receipt = package(repo, revision, output, bash)
    archive = output/'source.tar'
    with tarfile.open(archive) as bundle:
        def content(name: str) -> bytes:
            stream = bundle.extractfile(name)
            if stream is None:
                raise ValueError('source_member_missing:'+name)
            return stream.read()
        raw = content(WRAPPER)
        executable = {name: sha(content(name)) for name in sorted(executable_names(bundle.getnames()))}
    wrapper = output/'mixed-train.sbatch'
    with wrapper.open('xb') as stream:
        stream.write(raw)
    run_shell_guard(archive, revision, wrapper, bash)
    wrong = output/'wrong-mixed-wrapper.sbatch'
    with wrong.open('xb') as stream:
        stream.write(raw+b'# synthetic mismatch\n')
    empty = output/'wrong-mixed-archive.tar'
    with tarfile.open(empty, 'w'):
        pass
    rejected = []
    for name, tar, commit, shell in (('revision', archive, '0'*40, wrapper),
            ('wrapper', archive, revision, wrong), ('archive', empty, revision, wrapper)):
        try:
            run_shell_guard(tar, commit, shell, bash)
        except subprocess.CalledProcessError:
            rejected.append(name)
        else:
            raise ValueError('actual_mixed_GPU_guard_accepted_mismatch:'+name)
    release = {'authorization': 'coordinator_exact_hash_release', 'purpose': VERSION,
        'requires_separate_coordinator_submission_authorization': True,
        'source_git': revision, 'source_archive_sha256': receipt['source_archive_sha256'],
        'source_archive_bytes': receipt['source_archive_bytes'], 'wrapper_path': WRAPPER, 'wrapper_sha256': sha(raw),
        'source_files_sha256': executable, 'config_sha256': config_sha(), **inputs,
        'model_archive_name': ARCHIVE_NAME, 'model_archive_sha256': ARCHIVE_SHA,
        'model_relative_directory': MODEL_RELATIVE, 'model_relative_manifest': MANIFEST_RELATIVE,
        'model_manifest_sha256': MODEL_SHA, 'runtime_receipt_sha256': RUNTIME_RECEIPT_SHA,
        'runtime_files_sha256': RUNTIME_FILES_SHA, 'python_executable': PYTHON_EXECUTABLE,
        'output': (ROOT/'runs'/('scifact-mixed-training-'+revision[:12])).as_posix(),
        'max_runtime_seconds': 1500, 'resource_cap': RESOURCE_CAP, 'automatic_retry': False,
        'actual_mixed_GPU_wrapper_guard_passed': True, 'rejected_mismatches': rejected,
        'generic_legacy_guard_is_not_mixed_GPU_evidence': True, 'job_submitted': False}
    ordered_write(output/'mixed-train-release-candidate.json', release)
    return {'source_git': revision, 'source_archive_sha256': receipt['source_archive_sha256'],
        'source_archive_bytes': receipt['source_archive_bytes'], 'wrapper_sha256': sha(raw),
        'release_candidate_sha256': sha((output/'mixed-train-release-candidate.json').read_bytes()),
        'executable_files': len(executable), 'preparation_receipt_sha256': inputs['preparation_receipt_sha256'],
        'actual_mixed_GPU_wrapper_guard_passed': True, 'rejected_mismatches': rejected,
        'resource_cap': RESOURCE_CAP, 'job_submitted': False}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--repo', type=Path, required=True)
    parser.add_argument('--commit', required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--bash', required=True)
    parser.add_argument('--preparation-receipt', type=Path, required=True)
    args = parser.parse_args()
    print(freeze(args.repo, args.commit, args.output, args.bash, args.preparation_receipt))


if __name__ == '__main__':
    main()
