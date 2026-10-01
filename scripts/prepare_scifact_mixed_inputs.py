"""One hash-released CPU preparation; no gold, teacher, retrieval or model calls."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import signal
from typing import Any

from climate_rag.scifact_mixed_inputs import SCOPE, input_config_sha
from climate_rag.scifact_mixed_preparation import PINS, VERSION, preparation_config_sha, prepare_bundle
from climate_rag.scifact_semantic_contract import CORPUS_SHA, TOKENIZER_SHA, checked
from climate_rag.scifact_state_supervision import require
from audit_scifact_candidate_pool_metadata import ROOT
from prepare_scifact_nei47 import load_assets, no_torch
from prepare_scifact_semantic_pair import verify_source_tree

WRAPPER = 'hpc/scifact_mixed_prepare_cpu.sbatch'


def prepare(args: Any) -> dict[str, Any]:
    source = Path(__file__).resolve().parents[1]
    verify_source_tree(source, args.source_archive, args.source_sha, args.source_git)
    checked(source/WRAPPER, args.wrapper_sha)
    release = json.loads(checked(args.release_file, args.release_sha))
    require(release['version'] == VERSION and release['scope'] == SCOPE
            and release['source_git'] == args.source_git and release['source_archive_sha256'] == args.source_sha
            and release['wrapper_path'] == WRAPPER and release['wrapper_sha256'] == args.wrapper_sha
            and release['input_sha256'] == PINS and release['config_sha256'] == preparation_config_sha()
            and release['input_config_sha256'] == input_config_sha()
            and release['corpus_sha256'] == CORPUS_SHA and release['tokenizer_sha256'] == TOKENIZER_SHA
            and release['resource_request'] == {'cpus': 1, 'memory_GiB': 4, 'minutes': 15, 'gpus': 0}
            and release['training_authorized'] is False and release['job_submitted'] is False,
            'exact_CPU_preparation_release_required')
    release['release_sha256'] = args.release_sha
    out = ROOT/'posthoc'/('scifact-mixed-program-'+args.source_git[:12])
    def read_raw(name: str) -> bytes:
        path = ROOT/name
        require(path.resolve().is_relative_to(ROOT) and not path.is_symlink(), 'private_source_regular_file')
        return path.read_bytes()
    no_torch()
    return prepare_bundle(out, release, read_raw, load_assets, final_check=no_torch)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('source-git', 'source-sha', 'wrapper-sha', 'release-sha'):
        parser.add_argument('--'+name, required=True)
    for name in ('source-archive', 'release-file'):
        parser.add_argument('--'+name, type=Path, required=True)
    args = parser.parse_args()
    require(os.name == 'posix' and os.environ.get('USE_TORCH') == '0'
            and os.environ.get('HF_HUB_OFFLINE') == '1' and os.environ.get('PYTHONHASHSEED') == '0'
            and os.environ.get('SLURM_CPUS_PER_TASK') == '1' and bool(os.environ.get('SLURM_JOB_ID'))
            and not os.environ.get('CUDA_VISIBLE_DEVICES'), 'offline_one_CPU_no_GPU_allocation')
    import resource
    resource.setrlimit(resource.RLIMIT_CPU, (890, 890))
    resource.setrlimit(resource.RLIMIT_AS, (4*1024**3, 4*1024**3))
    def timeout(signum: int, frame: Any) -> None:
        raise KeyboardInterrupt('CPU_preparation_deadline_or_termination')
    signal.signal(signal.SIGALRM, timeout)
    signal.signal(signal.SIGTERM, timeout)
    signal.alarm(885)
    result = prepare(args)
    print(json.dumps(result), flush=True)  # Counts/hashes/resources only, never examples.
    if result['status'] != 'complete':
        raise SystemExit(2)


if __name__ == '__main__':
    main()
