"""Allocated CPU-only NEI47 preparation; fixed exposed inputs, no model or training."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import signal
import sys
from typing import Any

from climate_rag.scifact_grounding import Abstract, parse_abstract
from climate_rag.scifact_nei_preparation import CONFIG, INPUT_SHA, PARENT, VERSION, prepare_reserved
from climate_rag.scifact_semantic_contract import CORPUS_SHA, TOKENIZER_SHA, checked, encoded, sha
from climate_rag.scifact_state_supervision import require
from audit_scifact_candidate_pool_metadata import OLD, ROOT
from prepare_scifact_semantic_pair import verify_source_tree

WRAPPER = 'hpc/scifact_nei47_cpu.sbatch'


def no_torch() -> None:
    require('torch' not in sys.modules, 'CPU_no_torch')


def load_metadata() -> dict[str, Any]:
    objects = {name: json.loads(checked(ROOT/PARENT/name, INPUT_SHA[name])) for name in
               ('compact.json', 'selection-before-content.json', 'component-reservations.json', 'claim-reports.json')}
    compact = objects['compact.json']
    require(compact['source_git'] == '09f3d9e72216dd6d61adff27c10179176a41656c'
            and all(compact['private_file_sha256'][n] == h for n, h in INPUT_SHA.items() if n != 'compact.json'),
            'frozen_parent_source_and_inputs')
    return {'compact': compact, 'selection': objects['selection-before-content.json'],
            'reservation': objects['component-reservations.json'], 'reports': objects['claim-reports.json']}


def load_assets() -> tuple[dict[int, Abstract], Any]:
    require('torch' not in sys.modules, 'CPU_no_torch')
    docs = [parse_abstract(json.loads(line)) for line in checked(ROOT/OLD/'inference/corpus.jsonl', CORPUS_SHA).splitlines()]
    corpus = {d.doc_id: d for d in docs}
    require(len(docs) == len(corpus) == 5183, 'unique_complete_public5183')
    token_dir = ROOT/'posthoc/scifact-read-continuation-fc4ffd61a761/tokenizer'
    for name, digest in TOKENIZER_SHA.items():
        checked(token_dir/name, digest)
    from transformers.models.auto.tokenization_auto import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(token_dir, local_files_only=True, trust_remote_code=False)
    require('torch' not in sys.modules, 'CPU_no_torch')
    return corpus, tokenizer


def prepare(source_git: str, archive: Path, archive_sha: str, wrapper_sha: str,
            release_file: Path, release_sha: str) -> dict[str, Any]:
    source = Path(__file__).resolve().parents[1]
    verify_source_tree(source, archive, archive_sha, source_git)
    checked(source/WRAPPER, wrapper_sha)
    release = json.loads(checked(release_file, release_sha))
    require(release['version'] == VERSION and release['source_git'] == source_git
            and release['source_archive_sha256'] == archive_sha and release['wrapper_sha256'] == wrapper_sha
            and release['wrapper_path'] == WRAPPER and release['input_sha256'] == INPUT_SHA
            and release['config_sha256'] == sha(encoded(CONFIG)) and release['scope'] == 'exposed_train_NEI47'
            and release['corpus_sha256'] == CORPUS_SHA and release['tokenizer_sha256'] == TOKENIZER_SHA
            and release['training_authorized'] is False and release['job_submitted'] is False,
            'exact_NEI47_release_contract')
    release['release_sha256'] = release_sha
    out = ROOT/'posthoc'/('scifact-nei47-program-'+source_git[:12])
    result = prepare_reserved(out, release, load_metadata,
        lambda: checked(ROOT/PARENT/'complete-selected-fit-gold.json', INPUT_SHA['complete-selected-fit-gold.json']),
        INPUT_SHA['complete-selected-fit-gold.json'], load_assets, final_check=no_torch)
    return result


def main() -> None:
    import resource
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('source-git', 'source-sha', 'wrapper-sha', 'release-sha'):
        parser.add_argument('--'+name, required=True)
    for name in ('source-archive', 'release-file'):
        parser.add_argument('--'+name, type=Path, required=True)
    args = parser.parse_args()
    require(os.name == 'posix' and os.environ.get('PYTHONHASHSEED') == '0'
            and os.environ.get('USE_TORCH') == '0' and os.environ.get('HF_HUB_OFFLINE') == '1'
            and os.environ.get('SLURM_CPUS_PER_TASK') == '1' and bool(os.environ.get('SLURM_JOB_ID')),
            'offline_one_CPU_allocation_required')
    resource.setrlimit(resource.RLIMIT_CPU, (890, 890))
    resource.setrlimit(resource.RLIMIT_AS, (4*1024**3, 4*1024**3))
    def timeout(signum: int, frame: Any) -> None:
        # Bypass controller/tool `except Exception` repair paths, stop the whole run.
        raise KeyboardInterrupt('allocated_CPU_deadline_or_termination')
    signal.signal(signal.SIGALRM, timeout)
    signal.signal(signal.SIGTERM, timeout)
    signal.alarm(885)
    result = prepare(args.source_git, args.source_archive, args.source_sha, args.wrapper_sha,
                     args.release_file, args.release_sha)
    print(json.dumps(result, sort_keys=True), flush=True)
    if not result['whole_cohort_data_ready']:
        raise SystemExit(2)


if __name__ == '__main__':
    main()
