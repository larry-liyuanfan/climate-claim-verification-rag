"""Single approved CPU comparison: same 48 FIT, full shared public corpus."""
from __future__ import annotations

import argparse
from collections import Counter
import json
import os
from pathlib import Path
import signal
import sys
import time
from typing import Any

from climate_rag.scifact_grounding import parse_gold
from climate_rag.scifact_read_continuation import ordered_write
from climate_rag.scifact_retrieval import SciFactBM25
from climate_rag.scifact_semantic_contract import TOKENIZER_SHA, checked, encoded, sha, write_once
from climate_rag.scifact_shared_supervision import (
    CONFIG, VERSION, build_shared_claim, config_sha, document_partitions, layer_inventory,
    matched_fullpool_state, paired_summary, provenance, read_transition_summary,
)
from climate_rag.scifact_state_supervision import (
    candidates_from_rows, require, tokenize_target, validate_tokenized, validate_weights,
)
from climate_rag.scifact_terminal import source_from_abstract
from prepare_scifact_semantic_pair import verify_source_tree
from prepare_scifact_state_supervision import ROOT, OLD, narrow_inputs, validate_old_rows

BASE = ROOT/'posthoc/scifact-state-supervision-fit-v2-9bf5fdad2ec9'
BASE_SHA = '3cfc8acffcb001488ce841310349fa05a806a2216d66336afe331d9cd04e1bde'
GOLD_SHA = '8daf2e74e74ec7f656be7c0b8234d7e36b7581f969c35b20a52554c5fb78e72f'
REPORTS_SHA = '3914f74b68e3116b70b93df0972055a33c49f7394452fc9bdbf20f2bc5cf625c'
OLD_STATES = ROOT/'posthoc/scifact-selected-read-opportunity-dcbe368c1c02/private/states-before-gold.json'
STATES_SHA = '5a49b53aa6ab7790231b224b5966f3e8e6f2fff1856219c1c177af16ded714b8'


def prepare(source_git: str, source_archive: Path, source_sha: str, wrapper_sha: str) -> dict[str, Any]:
    started = time.perf_counter()
    source = Path(__file__).resolve().parents[1]
    verify_source_tree(source, source_archive, source_sha, source_git)
    checked(source/'hpc/scifact_shared_supervision_cpu.sbatch', wrapper_sha)
    baseline = json.loads(checked(BASE/'compact.json', BASE_SHA))
    require(baseline['private_file_sha256']['complete-fit-gold.json'] == GOLD_SHA
            and baseline['private_file_sha256']['claim-reports.json'] == REPORTS_SHA, 'baseline_file_bindings')
    fit_ids, old_rows, families, corpus = narrow_inputs(ROOT/OLD)
    require(len(corpus) == 5183 and sha(encoded(fit_ids)) == baseline['fit_ids_sha256'], 'same_fit_full_corpus')
    # Reuse the already verified complete 48 only; no archive/non-FIT gold scan.
    saved_gold = json.loads(checked(BASE/'complete-fit-gold.json', GOLD_SHA))
    require([r['id'] for r in saved_gold] == fit_ids, 'exact_complete_fit_gold_order')
    claims = {g.claim_id: g for g in (parse_gold(r, corpus) for r in saved_gold)}
    components = validate_old_rows(old_rows, claims, families, corpus)
    require(sha(encoded(components)) == baseline['fit_components_sha256'], 'same_claim_components')
    old_reports = json.loads(checked(BASE/'claim-reports.json', REPORTS_SHA))
    token_dir = ROOT/'posthoc/scifact-read-continuation-fc4ffd61a761/tokenizer'
    for name, digest in TOKENIZER_SHA.items():
        checked(token_dir/name, digest)
    require('torch' not in sys.modules, 'cpu_no_torch')
    from transformers.models.auto.tokenization_auto import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(token_dir, local_files_only=True, trust_remote_code=False)
    retrieve = SciFactBM25(corpus)
    out = ROOT/'posthoc'/('scifact-shared-corpus-fit-'+source_git[:12])
    out.mkdir(mode=0o700)
    records, reports, frames = [], [], []
    for number, i in enumerate(fit_ids, 1):
        require(time.perf_counter()-started < 580, 'cpu_preparation_deadline')
        candidates = list(retrieve(claims[i].claim, 20))
        built = build_shared_claim(claims[i], components[i], candidates, corpus, tokenizer)
        if len(built['captured_frames']) == 2:
            before, after = [f['observation'] for f in built['captured_frames']]
            require(built['report']['program_teacher_reads'] == 1
                    and after['feedback'] == 'tool_completed: inspect current_citable; no semantic conclusion implied'
                    and after['remaining_calls'] == before['remaining_calls']-1
                    and after['remaining_tools'] == before['remaining_tools']-1, 'actual_controller_read_transition')
        records.extend(built['records'])
        reports.append(built['report'])
        frames.append({'claim_id': i, 'candidates': [{'doc_id': int(s.source_id), 'source_sha256': s.text_sha256} for s in candidates],
                       'frames': built['captured_frames'], 'capture_counts': {k: built['report'][k] for k in
                           ('program_teacher_reads', 'scripted_responses', 'model_calls', 'model_tool_executions')}})
        if number % 8 == 0:
            print(json.dumps({'progress': {'claims_finished': number, 'records': len(records),
                                           'elapsed_seconds': time.perf_counter()-started}}), flush=True)
    require(len(records) <= 192 and 'torch' not in sys.modules, 'no_model_record_bound')
    ordered_write(out/'fit-records.json', records)
    ordered_write(out/'captured-frames.json', frames)
    write_once(out/'claim-reports.json', reports)
    # Same exact complete-gold identity remains private for downstream provenance.
    write_once(out/'complete-fit-gold.json', saved_gold)
    require(sha((out/'complete-fit-gold.json').read_bytes()) == GOLD_SHA, 'gold_bytes_preserved')
    restored = json.loads((out/'fit-records.json').read_bytes())
    for row in restored:
        require(row['data_provenance'] == provenance(), 'shared_record_protocol')
        validate_tokenized(row, tokenize_target(tokenizer, row['frame'], row['target'], candidates_from_rows(row['candidates'], corpus)))
    ready = all(r['training_ready'] for r in reports)
    if ready:
        validate_weights(restored, fit_ids)
    indexed = [{'doc_id': i, 'source_sha256': source_from_abstract(corpus[i]).text_sha256} for i in sorted(corpus)]
    exposure_private, exposure = layer_inventory(restored, frames, indexed, document_partitions(families), corpus)
    write_once(out/'private-three-layer-provenance.json', exposure_private)
    comparison = paired_summary(old_reports, reports)
    matched = matched_fullpool_state(json.loads(checked(OLD_STATES, STATES_SHA)), frames, records, tokenizer)
    import resource
    usage = resource.getrusage(resource.RUSAGE_SELF)
    compact = {'version': VERSION, 'config': CONFIG, 'config_sha256': config_sha(),
        'source_git': source_git, 'source_archive_sha256': source_sha, 'cpu_wrapper_sha256': wrapper_sha,
        'baseline_compact_sha256': BASE_SHA, 'reused_complete_fit_gold_sha256': GOLD_SHA,
        'original_train_member_sha256': baseline['train_member_sha256'],
        'fit_ids_sha256': baseline['fit_ids_sha256'], 'fit_components_sha256': baseline['fit_components_sha256'],
        'input_file_sha256': {str(BASE/'claim-reports.json'): REPORTS_SHA, str(OLD_STATES): STATES_SHA,
                              **{str(ROOT/OLD/n): h for n, h in baseline['allowlisted_file_sha256'].items()}},
        'tokenizer_sha256': TOKENIZER_SHA, 'same_fit_claims': 48, 'candidate_pool_documents': len(corpus),
        'decision_records': len(records), 'action_counts': dict(Counter(r['target']['action'] for r in records)),
        'initial_complete_witness_claims': sum(r['initial_complete'] for r in reports),
        'post_read_complete_witness_claims': sum(r['post_read_complete'] is True for r in reports),
        'program_teacher_reads': sum(r['program_teacher_reads'] for r in reports),
        'scripted_responses': sum(r['scripted_responses'] for r in reports),
        'read_transitions': read_transition_summary(frames, reports, records),
        'whole_gold_covered_target_claims': sum(r['whole_gold_covered_by_target'] for r in reports),
        'claims_with_gaps': sum(bool(r['gaps']) for r in reports),
        'gap_categories': dict(Counter(g for r in reports for g in r['gaps'])),
        'retained_claim_masses': dict(Counter(str(r['retained_weight_numerator'])+'/'+str(r['retained_weight_denominator']) for r in reports)),
        'joint_tokens_max': max((r['packing']['input_tokens']+r['packing']['target_tokens'] for r in records), default=0),
        'sequence_tokens_total': sum(r['packing']['input_tokens']+r['packing']['target_tokens'] for r in records),
        'baseline_sequence_tokens_total': baseline['supervised_input_tokens_sum']+baseline['supervised_target_tokens_sum'],
        'paired_comparison': comparison, 'predeclared_matched_case': matched, 'three_layer_provenance': exposure,
        'supervision_data_ready': ready, 'training_authorized': False, 'model_calls': 0, 'optimizer_steps': 0,
        'model_tool_success_claimed': False, 'source_family_unseen_claimed': False,
        'protected_claim_text_gold_read': False, 'archive_non_fit_gold_scan_performed': False,
        'scope': 'same_48_previously_exposed_TRAIN_program_teacher_shared_public_corpus_not_test_or_Agent_gain',
        'elapsed_seconds': time.perf_counter()-started, 'cpu_seconds': usage.ru_utime+usage.ru_stime,
        'maxrss_kib': usage.ru_maxrss,
        'private_file_sha256': {p.name: sha(p.read_bytes()) for p in out.iterdir() if p.is_file()}}
    write_once(out/'compact.json', compact)
    print(json.dumps(compact, sort_keys=True), flush=True)
    return compact


def main() -> None:
    import resource
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source-git', required=True)
    parser.add_argument('--source-archive', type=Path, required=True)
    parser.add_argument('--source-sha', required=True)
    parser.add_argument('--wrapper-sha', required=True)
    args = parser.parse_args()
    require(os.name == 'posix' and os.environ.get('PYTHONHASHSEED') == '0'
            and os.environ.get('USE_TORCH') == '0' and bool(os.environ.get('SLURM_JOB_ID')), 'cpu_allocation_required')
    resource.setrlimit(resource.RLIMIT_CPU, (600, 600))
    resource.setrlimit(resource.RLIMIT_AS, (4*1024**3, 4*1024**3))
    signal.alarm(600)
    prepare(args.source_git, args.source_archive, args.source_sha, args.wrapper_sha)


if __name__ == '__main__':
    main()
