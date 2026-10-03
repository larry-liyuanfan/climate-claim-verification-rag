"""Once-only metadata selection, then original gold-free retrieval; no model."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import tarfile
import time
from typing import Any

from climate_rag.scifact_grounding import parse_abstract
from climate_rag.scifact_prospective_inputs import (
    PROTOCOL, SCOPE, freeze_selection, prepare_frame, select_metadata, validate_frames,
)
from climate_rag.scifact_read_continuation import ordered_write
from climate_rag.scifact_retrieval import SciFactBM25
from climate_rag.scifact_semantic_contract import CORPUS_SHA, TOKENIZER_SHA, checked, sha, write_once
from climate_rag.scifact_utility_contract import identity
from preflight_scifact_evidence_commit import probe
from prepare_scifact_natural import QUERY_SHA
from prepare_scifact_semantic_pair import verify_source_tree
from prepare_scifact_state_supervision import member
from prepare_scifact_utility8 import ARCHIVE, ARCHIVE_SHA, CORPUS, MANIFEST_SHA, PREP, select_queries
from run_scifact_grounding_train_operator import ROOT, require
from scifact_evidence_input import PREPARED, release_inputs

SUP = 'posthoc/scifact-supplemental-fit-09f3d9e72216/'
MIX = 'posthoc/scifact-mixed-program-cb1e76462537/roster.json'
NAT = 'posthoc/scifact-natural-fit24-v1-20261001/selection.json'
NEI = 'posthoc/scifact-nei47-program-782892d57365/NEI-selection.json'
UTIL = 'runs/scifact-utility8-20261001-v1/prepared/selection.json'
OLD = 'posthoc/scifact-semantic-preparation-779e49883570/private/selected-before-probe.json'
METADATA = {
    SUP+'selection-before-content.json': '72d0012fd12bf76396fb88c5f0d9581ef8d154996608e5fd167ea8f23e0a1355',
    SUP+'component-reservations.json': '5afb9f5c5843ac083ea046b2276a0bdf117a8046beaa43190b469ae3409c56d2',
    MIX: '4fdae46c4c41d51f4e1c120b95881316577092bc6004b6d03b836da41d44d349',
    NAT: '107c50a8ecc1fd4d1b352cdc7e2f86b2997d6abc027d5afeb176bab1d09f7acb',
    NEI: 'a55441fa2a9bace07b307ccf2c0b622a033323a8923e2fb3e047a782e32c666a',
    UTIL: 'b1746e7931e02e5767e955966f423ee70ab799d7e34b2be4e99f88a5010e810c',
    OLD: '118d57233b1205f294d7090b6820d658f447ba09263fe8179bdf8e66497338a8',
}
TOKENIZER = ROOT/'posthoc/scifact-read-continuation-fc4ffd61a761/tokenizer'
WRAPPER = 'hpc/scifact_evidence_prospective_cpu.sbatch'


def metadata_selection(root: Path = ROOT) -> Any:
    data = {p: json.loads(checked(root/p, digest)) for p, digest in METADATA.items()}
    original, reservation = data[SUP+'selection-before-content.json'], data[SUP+'component-reservations.json']
    excluded = set(reservation['ordered_components'])
    excluded.update(data[MIX]['payload']['components'].values())
    excluded.update(data[MIX]['payload']['excluded_components'])
    for name in (NAT, UTIL):
        excluded.update(r['component'] for r in data[name]['selected'])
    excluded.update(data[NAT]['excluded_components'])
    excluded.update(data[NEI]['selected_components'])
    excluded.update(data[NEI]['original_ordered_components'])
    excluded.update(r['component'] for r in data[OLD]['selection'])
    remaining = set(original['ordered_candidate_components']) - excluded
    pool = [{'id': int(i), 'component': c} for i, c in original['candidate_id_to_component'].items() if c in remaining]
    # Exact file pins bind the mapping, not merely the two marginal sets.
    return select_metadata(pool, METADATA)


def prepare(source_git: str, archive: Path, archive_sha: str, wrapper_sha: str) -> dict[str, Any]:
    started = time.perf_counter()
    source = Path(__file__).resolve().parents[1]
    verify_source_tree(source, archive, archive_sha, source_git)
    checked(source/WRAPPER, wrapper_sha)
    require(os.name == 'posix' and bool(os.environ.get('SLURM_JOB_ID'))
            and not os.environ.get('CUDA_VISIBLE_DEVICES') and os.environ.get('USE_TORCH') == '0', 'CPU_only_allocation')
    import resource
    selection = metadata_selection()
    frozen = freeze_selection(PREPARED, selection)  # durable before any selected query read
    print(json.dumps({'selection_frozen': True, 'claims': 24, 'selection_sha256': frozen['selection.json']}), flush=True)
    bundle = ROOT/'envs'/ARCHIVE
    checked(bundle, ARCHIVE_SHA)
    ids = [r['id'] for r in selection['selected']]
    with tarfile.open(bundle) as tar:
        with member(tar, PREP+'/preparation-manifest.json') as stream:
            raw = stream.read()
        require(sha(raw) == MANIFEST_SHA and json.loads(raw)['output_file_sha256'][
            'inference/claims_train_eligible.jsonl'] == QUERY_SHA, 'gold_free_query_member')
        with member(tar, PREP+'/inference/claims_train_eligible.jsonl') as stream:
            claims = select_queries(stream, QUERY_SHA, ids)
    began = time.perf_counter()
    raw_corpus = checked(ROOT/CORPUS, CORPUS_SHA)
    docs = [parse_abstract(json.loads(line)) for line in raw_corpus.splitlines()]
    corpus = {d.doc_id: d for d in docs}
    require(len(docs) == len(corpus) == 5183, 'full_original_corpus')
    retrieve = SciFactBM25(corpus)
    build_seconds = time.perf_counter()-began
    for name, digest in TOKENIZER_SHA.items():
        checked(TOKENIZER/name, digest)
    import transformers
    tokenizer = getattr(transformers, 'AutoTokenizer').from_pretrained(TOKENIZER, local_files_only=True)
    pairs = [prepare_frame(c, retrieve, tokenizer) for c in claims]
    frames, rows = [p[0] for p in pairs], [p[1] for p in pairs]
    validate_frames(claims, frames, corpus, tokenizer)
    probes = [probe(frame, tokenizer) for frame in frames]
    inference = PREPARED/'inference'
    inference.mkdir(mode=0o700)
    ordered_write(inference/'claims.json', claims)
    ordered_write(inference/'initial-frames.json', frames)  # preserves alias/visible insertion order
    with (inference/'corpus.jsonl').open('xb') as stream:
        stream.write(raw_corpus)
    ordered_write(PREPARED/'private-preparation-timing.json', rows)
    ordered_write(PREPARED/'private-prompt-probes.json', probes)
    receipt = {'protocol': PROTOCOL, 'scope': SCOPE, 'source_git': source_git,
        'source_archive_sha256': archive_sha, 'cpu_wrapper_sha256': wrapper_sha,
        'selection_sha256': frozen['selection.json'], 'component_reservations_sha256': frozen['component-reservations.json'],
        'claims_sha256': sha((inference/'claims.json').read_bytes()),
        'frames_sha256': sha((inference/'initial-frames.json').read_bytes()),
        'corpus_sha256': CORPUS_SHA, 'tokenizer_sha256': TOKENIZER_SHA,
        'ordered_ids_sha256': identity(ids), 'query_member_sha256': QUERY_SHA,
        'model_calls': 0, 'official_gold_decoded': False, 'protected_split_read': False, 'reselection': False,
        'shared_preparation_cost': {'corpus_load_and_BM25_build_seconds': build_seconds,
            'retrieval_seconds': sum(r['elapsed_seconds'] for r in rows),
            'retrieval_and_packing_seconds': sum(r['preparation_elapsed_seconds'] for r in rows),
            'preparation_total_seconds': time.perf_counter()-started,
            'process_maxrss_kib': resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
            'initial_retrieval_calls': 24, 'shared_among_arms': 3,
            'meaning': 'once_only_offline_preparation_not_online_latency_or_free_retrieval'},
        'prompt_probe': {'maximum_observed_tokens': max(p['maximum_tokens'] for p in probes),
            'overflow_count': sum(p['overflow'] for p in probes),
            'scope': 'first_two_refs_fixed_synthetic_feedback_not_all_reachable_states'},
        'physical_prefix_created': False, 'planned_slots': 72, 'max_future_model_calls': 240}
    write_once(PREPARED/'preparation.json', receipt)
    fields = release_inputs(PREPARED)
    compact = {**receipt, **fields, 'selected_claim_count':24, 'selected_component_count':24,
        'pool_components':selection['pool_components'], 'pool_ids':selection['pool_ids'],
        'pool_id_component_sha256':selection['pool_id_component_sha256'],
        'future_gpu_authorized': False, 'job_id':os.environ['SLURM_JOB_ID']}
    write_once(PREPARED/'compact.json', compact)
    print(json.dumps(compact), flush=True)
    return compact


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('source-git', 'source-sha', 'wrapper-sha'):
        parser.add_argument('--'+name, required=True)
    parser.add_argument('--source-archive', type=Path, required=True)
    args = parser.parse_args()
    prepare(args.source_git, args.source_archive, args.source_sha, args.wrapper_sha)


if __name__ == '__main__':
    main()
