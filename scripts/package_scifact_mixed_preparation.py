"""Freeze actual mixed-CPU wrapper and check its guard; no upload/submission."""
from __future__ import annotations

import argparse
from pathlib import Path
import subprocess
import tarfile
from typing import Any

from climate_rag.scifact_mixed_inputs import SCOPE, input_config_sha
from climate_rag.scifact_mixed_preparation import PINS, VERSION, preparation_config_sha
from climate_rag.scifact_read_continuation import ordered_write
from climate_rag.scifact_semantic_contract import CORPUS_SHA, TOKENIZER_SHA, sha
from package_scifact_source import package, run_shell_guard

WRAPPER = 'hpc/scifact_mixed_prepare_cpu.sbatch'


def freeze(repo: Path, revision: str, output: Path, bash: str) -> dict[str, Any]:
    receipt = package(repo, revision, output, bash)
    archive = output/'source.tar'
    with tarfile.open(archive) as bundle:
        stream = bundle.extractfile(WRAPPER)
        if stream is None:
            raise ValueError('mixed_wrapper_missing')
        raw = stream.read()
    wrapper = output/'mixed-prepare.sbatch'
    with wrapper.open('xb') as stream:
        stream.write(raw)
    run_shell_guard(archive, revision, wrapper, bash)
    wrong = output/'wrong-wrapper.sbatch'
    with wrong.open('xb') as stream:
        stream.write(raw+b'# synthetic tamper\n')
    empty = output/'wrong-archive.tar'
    with tarfile.open(empty, 'w'):
        pass
    negatives = []
    for name, tar, commit, shell in (('revision', archive, '0'*40, wrapper),
            ('wrapper', archive, revision, wrong), ('archive', empty, revision, wrapper)):
        try:
            run_shell_guard(tar, commit, shell, bash)
        except subprocess.CalledProcessError:
            negatives.append(name)
        else:
            raise ValueError('mixed_negative_guard_accepted:'+name)
    release = {'version': VERSION, 'scope': SCOPE, 'source_git': revision,
        'source_archive_sha256': receipt['source_archive_sha256'], 'source_archive_bytes': receipt['source_archive_bytes'],
        'wrapper_path': WRAPPER, 'wrapper_sha256': sha(raw), 'input_sha256': PINS,
        'config_sha256': preparation_config_sha(), 'input_config_sha256': input_config_sha(),
        'corpus_sha256': CORPUS_SHA, 'tokenizer_sha256': TOKENIZER_SHA,
        'actual_mixed_wrapper_guard_passed': True, 'rejected_mismatches': negatives,
        'legacy_guard_is_not_mixed_evidence': True, 'job_submitted': False,
        'training_authorized': False, 'model_calls': 0, 'optimizer_steps': 0, 'automatic_retry': False,
        'resource_request': {'cpus': 1, 'memory_GiB': 4, 'minutes': 15, 'gpus': 0}}
    ordered_write(output/'mixed-release.json', release)
    return {**release, 'release_sha256': sha((output/'mixed-release.json').read_bytes())}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--repo', type=Path, required=True)
    parser.add_argument('--commit', required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--bash', required=True)
    args = parser.parse_args()
    print(freeze(args.repo, args.commit, args.output, args.bash))


if __name__ == '__main__':
    main()
