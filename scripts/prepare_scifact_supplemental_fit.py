"""One CPU-only, prospectively selected supplemental FIT preparation."""
from __future__ import annotations

import argparse
from collections import Counter
import json
import os
from pathlib import Path
import signal
import sys
import tarfile
import time
from typing import Any

from climate_rag.scifact_grounding import parse_abstract
from climate_rag.scifact_read_continuation import ordered_write
from climate_rag.scifact_retrieval import SciFactBM25
from climate_rag.scifact_semantic_contract import CORPUS_SHA, TOKENIZER_SHA, checked, encoded, sha, write_once
from climate_rag.scifact_shared_supervision import document_partitions, layer_inventory, read_transition_summary
from climate_rag.scifact_state_supervision import candidates_from_rows, require, tokenize_target, validate_tokenized, validate_weights
from climate_rag.scifact_supplemental_fit import (
    CONFIG, VERSION, action_mass, config_sha, data_provenance, freeze_selection,
    load_reserved_claims, prepare_selected_claim, select_metadata,
)
from climate_rag.scifact_terminal import source_from_abstract
from audit_scifact_candidate_pool_metadata import ARCHIVE, FILES, MEMBERS, OLD, PREP, ROOT, Inputs
from prepare_scifact_semantic_pair import verify_source_tree
from prepare_scifact_state_supervision import member

CONTROL = 'posthoc/scifact-shared-corpus-fit-6fd92eb0ac91'
CONTROL_SHA = '27ad7cec3bb9aa1d9c12cd8618d9f9546956ed8b171e0483950333a3959c686d'
CONTROL_ROWS_SHA = '42e18a523026e617edeb0a1597e734e6d9431f324849caa487c8f798d51a58fa'
TRAIN_SHA = '9e07864a797dd515391fa1c3f3573249b70399f95ff6c740bf536ad11c8aa6ca'
REMAINING_SHA = 'c4d37c0dc6419ade25f130184f68a4d5d3b10f385d56d1726d11f964c48ada85'
ELIGIBLE_SHA = 'e44bd95cf584e095d3ff3dbb2c911b7a8ce1e16af2f7b3548c4e970e540aede5'


def metadata_selection(inputs: Inputs, source_identity: dict[str, str]) -> tuple[dict[str, Any], dict[str, Any], list[int]]:
    original, assignment = inputs.metadata_members()
    manifest = inputs.read(OLD+'/manifest.json')
    require(original['output_file_sha256']['private-group-assignment.json'] == MEMBERS['private-group-assignment.json']
            == manifest['assignment_sha256'] and original['output_file_sha256']['gold/claims_train.jsonl'] == TRAIN_SHA,
            'original_metadata_and_train_identity')
    names = ('selection-before-packing.json', 'families.json', 'ledger.json')
    split, families, ledger = [inputs.read(OLD+'/private/'+name) for name in names]
    require(all(manifest['files']['private/'+name] == FILES[OLD+'/private/'+name] for name in names)
            and families['corpus_sha256'] == CORPUS_SHA and not families['claim_grouping_recomputed']
            and manifest['files']['inference/corpus.jsonl'] == CORPUS_SHA, 'old_metadata_bindings')
    require(len(assignment['eligible_train_ids']) == 531 and ledger['gold_preparation_seen_count'] == 531
            and [len(split[p]) for p in ('fit', 'tune', 'validation')] == [48, 12, 12], 'frozen_original_counts')
    train = inputs.read('runs/scifact-grounding-training-20261001-v1/complete.json')
    require(train['data_manifest_sha256'] == FILES[OLD+'/manifest.json'] and train['records_seen_once'] == 96,
            'old_training_receipt_binding')
    later = {'later_original_training': set(split['fit'])}
    for arm in ('base', 'adapted'):
        ids = inputs.reservations('runs/scifact-grounding-tune-20261001-v1/'+arm+'/complete.json', 12, True)
        require(ids == set(split['tune']), 'tune_physical_reservations')
        later['later_tune_'+arm] = ids
    later['later_regression'] = inputs.reservations('runs/scifact-adapter-bare-regression-20261001-v1/cost-before-quality.json', 48)
    later['later_conditional'] = inputs.reservations('runs/scifact-read-conditional-20261001-v1/cost-before-quality.json', 3)
    require(len(later['later_regression']) == 12 and len(later['later_conditional']) == 3
            and later['later_conditional'] <= later['later_regression'], 'later_reservation_identity')
    control = json.loads(checked(ROOT/CONTROL/'compact.json', CONTROL_SHA))
    require(control['fit_ids_sha256'] == sha(encoded(split['fit']))
            and control['private_file_sha256']['fit-records.json'] == CONTROL_ROWS_SHA
            and control['same_fit_claims'] == 48, 'unchanged_original_shared_control')
    inputs.reads[CONTROL+'/compact.json'] = CONTROL_SHA
    selection = select_metadata(assignment, families, ledger, split, later, inputs.reads)
    require(selection['remaining_components'] == {'count': 160, 'sha256': REMAINING_SHA}
            and selection['eligible_ids_in_remaining_components'] == {'count': 256, 'sha256': ELIGIBLE_SHA},
            'conditional_eligibility_snapshot_drift')
    require(len(selection['selected_ordered_ids']) == 96, 'frozen_selection_size')
    selection.update(source_identity=source_identity, original_fit_ids_sha256=control['fit_ids_sha256'],
        shared_public_corpus_sha256=CORPUS_SHA, original_shared_control_sha256=CONTROL_SHA,
        actual_selected_claims=96, actual_selected_components=96)
    return selection, families, split['fit']


def prepare(source_git: str, archive: Path, archive_sha: str, wrapper_sha: str) -> dict[str, Any]:
    started = time.perf_counter()
    source = Path(__file__).resolve().parents[1]
    verify_source_tree(source, archive, archive_sha, source_git)
    checked(source/'hpc/scifact_supplemental_fit_cpu.sbatch', wrapper_sha)
    identity = {'source_git': source_git, 'source_archive_sha256': archive_sha, 'cpu_wrapper_sha256': wrapper_sha}
    # No caller-supplied salt, cohort size, replacement flag or arbitrary path.
    selection, families, original_fit_ids = metadata_selection(Inputs(ROOT), identity)
    out = ROOT/'posthoc'/('scifact-supplemental-fit-'+source_git[:12])
    out.mkdir(mode=0o700)
    frozen = freeze_selection(out, selection)
    require(all(sha((out/name).read_bytes()) == digest for name, digest in frozen.items()), 'durable_selection_snapshot')
    print(json.dumps({'selection_frozen': True, 'selected_claims': len(selection['selected_ordered_ids']),
                      'selection_sha256': frozen['selection-before-content.json']}), flush=True)
    # Only public corpus is decoded now; no old/tune/dev/validation question rows.
    raw_corpus = checked(ROOT/OLD/'inference/corpus.jsonl', CORPUS_SHA)
    docs = [parse_abstract(json.loads(line)) for line in raw_corpus.splitlines()]
    corpus = {d.doc_id: d for d in docs}
    require(len(docs) == len(corpus) == 5183, 'unique_complete_shared_public_corpus')
    with tarfile.open(ROOT/ARCHIVE) as bundle:
        claims = load_reserved_claims(lambda: member(bundle, PREP+'/gold/claims_train.jsonl'), TRAIN_SHA, out, frozen, corpus)
    ids = selection['selected_ordered_ids']
    components = dict(zip(ids, selection['selected_ordered_components'], strict=True))
    require(list(claims) == ids, 'selected_order_preserved')
    for i, claim in claims.items():
        for doc in set(claim.evidence) | set(claim.cited_doc_ids):
            family = str(families['document_family'][str(doc)])
            require(families['family_component'][family] == components[i], 'selected_gold_component_binding')
    write_once(out/'complete-selected-fit-gold.json', [claims[i].gold_row() for i in ids])
    token_dir = ROOT/'posthoc/scifact-read-continuation-fc4ffd61a761/tokenizer'
    for name, digest in TOKENIZER_SHA.items():
        checked(token_dir/name, digest)
    require('torch' not in sys.modules, 'cpu_no_torch')
    from transformers.models.auto.tokenization_auto import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(token_dir, local_files_only=True, trust_remote_code=False)
    retrieve = SciFactBM25(corpus)
    records, reports, frames = [], [], []
    for number, i in enumerate(ids, 1):
        require(time.perf_counter()-started < 875, 'cpu_preparation_deadline')
        candidates = list(retrieve(claims[i].claim, 20)) if claims[i].evidence else []
        built = prepare_selected_claim(claims[i], components[i], candidates, corpus, tokenizer, frozen)
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
        if number % 12 == 0:
            print(json.dumps({'progress': {'selected_finished': number, 'records': len(records),
                                          'elapsed_seconds': time.perf_counter()-started}}), flush=True)
    require(len(records) <= len(ids)*4 and 'torch' not in sys.modules, 'no_model_record_bound')
    ordered_write(out/'fit-records.json', records)
    ordered_write(out/'captured-frames.json', frames)
    write_once(out/'claim-reports.json', reports)
    restored = json.loads((out/'fit-records.json').read_bytes())
    for row in restored:
        require(row['data_provenance'] == data_provenance(frozen), 'supplemental_record_protocol')
        validate_tokenized(row, tokenize_target(tokenizer, row['frame'], row['target'], candidates_from_rows(row['candidates'], corpus)))
    ready_ids = [r['claim_id'] for r in reports if r['training_ready']]
    if ready_ids:
        validate_weights([r for r in restored if r['claim_id'] in ready_ids], ready_ids)
    indexed = [{'doc_id': i, 'source_sha256': source_from_abstract(corpus[i]).text_sha256} for i in sorted(corpus)]
    private, exposure = layer_inventory(restored, frames, indexed, document_partitions(families), corpus,
                                        expected_provenance=data_provenance(frozen))
    write_once(out/'private-four-layer-provenance.json', private)
    # Read-only preserved control. No re-retrieval or new preparation for its 48.
    control = json.loads(checked(ROOT/CONTROL/'compact.json', CONTROL_SHA))
    old_records = json.loads(checked(ROOT/CONTROL/'fit-records.json', CONTROL_ROWS_SHA))
    import resource
    usage = resource.getrusage(resource.RUSAGE_SELF)
    compact = {'version': VERSION, 'config': CONFIG, 'config_sha256': config_sha(), **identity,
        'selection_file_sha256': frozen, 'input_sha256': selection['input_sha256'],
        'conditional_remaining_components': selection['remaining_components'],
        'conditional_eligible_ids': selection['eligible_ids_in_remaining_components'],
        'train_member_sha256': TRAIN_SHA, 'tokenizer_sha256': TOKENIZER_SHA,
        'shared_corpus_sha256': CORPUS_SHA, 'candidate_pool_documents': len(corpus),
        'actual_selected_claims': len(ids), 'selected_ids_sha256': sha(encoded(ids)),
        'selected_components_sha256': sha(encoded(selection['selected_ordered_components'])),
        'preparation_status': dict(Counter(r['status'] for r in reports)),
        'official_label_scope': dict(Counter(r['official_label_scope'] for r in reports)),
        'teacher_supported_claims': sum(r['status'] != 'not_applicable' for r in reports),
        'data_ready_claims': len(ready_ids), 'ready_claim_ids_sha256': sha(encoded(ready_ids)),
        'whole_cohort_data_ready': len(ready_ids) == len(ids),
        'initial_complete_claims': sum(r['initial_complete'] for r in reports),
        'post_read_complete_claims': sum(r['post_read_complete'] is True for r in reports),
        'context_abstain_claims': len({r['claim_id'] for r in records if r['target']['action'] == 'abstain'}),
        'whole_gold_covered_target_claims': sum(r['whole_gold_covered_by_target'] for r in reports),
        'gap_categories': dict(Counter(g for r in reports for g in r['gaps'])),
        'claims_with_gaps_or_not_applicable': sum(bool(r['gaps']) for r in reports),
        'scripted_responses': sum(r['scripted_responses'] for r in reports),
        'read_transitions': read_transition_summary(frames, reports, records),
        'supplemental_action_mass': action_mass(records, ids),
        'original48_control': {'compact_sha256': CONTROL_SHA, 'records_sha256': CONTROL_ROWS_SHA,
            'reexecuted': False, 'action_mass': action_mass(old_records, original_fit_ids),
            'read_transitions': control['read_transitions'], 'initial_complete_claims': control['initial_complete_witness_claims'],
            'post_read_complete_claims': control['post_read_complete_witness_claims']},
        'joint_tokens_max': max((r['packing']['input_tokens']+r['packing']['target_tokens'] for r in records), default=0),
        'sequence_tokens_total': sum(r['packing']['input_tokens']+r['packing']['target_tokens'] for r in records),
        'four_layer_exposure': exposure,
        'model_calls': 0, 'optimizer_steps': 0, 'training_authorized': False,
        'model_tool_success_claimed': False, 'source_family_unseen_claimed': False,
        'tune_validation_dev_test_claim_gold_deserialized': False,
        'unselected_train_bytes_streamed_for_hash_and_ID_only': True,
        'selection_before_content_read': True, 'selection_unchanged_after_outcomes': True,
        'scope': 'conditional_metadata_selected_TRAIN_program_teacher_preparation_not_independent_test_or_model_gain',
        'elapsed_seconds': time.perf_counter()-started, 'cpu_seconds': usage.ru_utime+usage.ru_stime, 'maxrss_kib': usage.ru_maxrss,
        'private_file_sha256': {p.name: sha(p.read_bytes()) for p in out.iterdir() if p.is_file()}}
    require(all(sha((out/name).read_bytes()) == digest for name, digest in frozen.items()), 'selection_changed_after_outcomes')
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
    resource.setrlimit(resource.RLIMIT_CPU, (900, 900))
    resource.setrlimit(resource.RLIMIT_AS, (4*1024**3, 4*1024**3))
    signal.alarm(900)
    prepare(args.source_git, args.source_archive, args.source_sha, args.wrapper_sha)


if __name__ == '__main__':
    main()
