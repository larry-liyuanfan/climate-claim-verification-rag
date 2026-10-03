"""Cost-first, post-exit conditional scoring; failures retain the three-slot denominator."""
from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
from typing import Any

from climate_rag.scifact_adapter_regression import ADAPTER_SHA, SCORING_MANIFEST, load_frozen, require
from climate_rag.scifact_grounding import parse_abstract
from climate_rag.scifact_read_continuation import finalise
from climate_rag.scifact_semantic_contract import MODEL_SHA, SCORING_NAMES, checked, sha, write_once
from audit_scifact_bounded_closeout import physical, tariff
from run_scifact_read_continuation_operator import OUTPUT, validate_release


def score(args: Any) -> dict[str, Any]:
    release = json.loads(checked(args.release, args.release_sha))
    validate_release(release, Path(__file__).resolve().parents[1])
    proof_path = OUTPUT/'allocation/worker-exit.json'
    proof = json.loads(proof_path.read_bytes())
    require(proof['child_reaped'] is True and proof['release_sha256'] == args.release_sha, 'exit_before_score')
    directory = OUTPUT/'inference'
    reservations = sorted(directory.glob('slot-*-reserved.json'))
    records = []
    unresolved = 0
    for reserved in reservations:
        path = reserved.with_name(reserved.name.replace('-reserved',''))
        try:
            record = json.loads(path.read_bytes())
            require(record['identity']['release_sha256'] == args.release_sha, 'record_release_changed')
            require(record['identity'] == json.loads(reserved.read_bytes())['identity'], 'reservation_identity_changed')
            require(isinstance(record['attempt'],dict), 'invalid_attempt')
        except (OSError,ValueError,KeyError,TypeError):
            unresolved += 1
            continue
        records.append(record)
    require(len(reservations) <= 3 and len(records) <= 3, 'three_calls_maximum')
    costs = tariff([r['attempt'] for r in records])
    costs.update(planned_slots=3, reserved_slots=len(reservations), known_slot_records=len(records),
        unresolved_reserved_slots=unresolved, not_reserved_slots=3-len(reservations),
        model_calls=None if unresolved else len(records), model_calls_known_lower_bound=len(records),
        repair_calls=0, warmup_calls=0, reranker_calls=0, model_tool_executions=0,
        new_call_seconds_not_original_episode_remainder=120,
        token_totals_are_lower_bounds=bool(unresolved or costs['token_totals_are_lower_bounds']))
    physical_index, hash_errors = {}, {}
    for path in directory.rglob('*'):
        if path.is_file():
            name = path.relative_to(directory).as_posix()
            try:
                physical_index[name] = sha(path.read_bytes())
            except OSError as exc:
                hash_errors[name] = type(exc).__name__
    costs['physical_hash_errors'] = hash_errors
    write_once(OUTPUT/'cost-before-quality.json', costs | {'worker_exit_sha256':sha(proof_path.read_bytes()),
        'physical_sha256':physical_index})
    pending = {'status':'partial_or_unknown_no_quality', 'costs':costs, 'gold_loaded':False, 'planned_slots':3}
    if proof['returncode'] or proof['parent_wait_interrupted'] or len(records) != 3 or costs['token_totals_are_lower_bounds'] or hash_errors:
        return pending
    run = json.loads(checked(directory/'run.json',proof['run_sha256']))
    require(run['rows'] == records, 'run_record_mismatch')
    integrity = json.loads(checked(directory/'adapter-integrity.json',run['adapter_integrity_sha256']))
    expected_active = {'active_adapters':['default'],'lora_layers':72,'enabled':True,'merged':False,'trainable_parameters':0}
    require(integrity['tensor_count'] == 144 and integrity['all_checkpoint_values_equal']
        and integrity['active_state'] == expected_active, 'active_adapter_integrity')
    bundle = json.loads(checked(Path(release['conditional_inputs']),release['conditional_inputs_sha256']))['rows']
    require([r['claim_id'] for r in records] == [r['claim_id'] for r in bundle], 'original_three_order')
    _, corpus_bytes = load_frozen(args.inference_dir)
    corpus = {d.doc_id:d for d in (parse_abstract(json.loads(line)) for line in corpus_bytes.splitlines())}
    wires = []
    for n,(row,saved) in enumerate(zip(bundle,records,strict=True),1):
        require(saved['identity'] == records[0]['identity'], 'common_identity')
        attempt = saved['attempt']
        d = attempt['diagnostics']
        require(d['base_model_sha256'] == MODEL_SHA and d['adapter_sha256'] == ADAPTER_SHA
            and d['adapter_state'] == expected_active
            and d['regression_policy_sha256'] == run['provider_binding_sha256']
            and d['actual_prompt_sha256'] == row['expected_state']['prompt_sha256']
            and d['actual_schema_sha256'] == row['expected_state']['schema_sha256']
            and attempt['usage']['input_tokens'] == row['expected_state']['prompt_tokens'], 'physical_active_input_binding')
        wire, raw = physical(directory/f'private-responses/slot-{n:02d}',[attempt],False)
        wires.append(wire)
        text = raw[d['output_sha256']]
        try:
            recalculated = finalise(text,row,corpus)
        except ValueError:
            require(saved['terminal']['status'] == 'invalid_no_repair'
                and saved['terminal']['prediction'] is None and attempt['status'] == 'validation_failed',
                'failed_parse_status_or_prediction_changed')
        else:
            require(recalculated == saved['terminal'], 'original_parse_render_prediction_changed')
    # No gold is read before cost persistence, exit proof and all physical/input checks.
    from climate_rag.scifact_grounding import parse_gold
    from climate_rag.scifact_scoring import parse_prediction, score_original
    require({p.name for p in args.scoring_dir.iterdir()} == SCORING_NAMES, 'scoring_allowlist')
    manifest = json.loads(checked(args.scoring_dir/'manifest.json',SCORING_MANIFEST))
    payload = {name:checked(args.scoring_dir/name,digest) for name,digest in manifest['scoring_file_sha256'].items()}
    ids = {r['claim_id'] for r in bundle}
    gold = [parse_gold(r,corpus) for line in payload['gold.jsonl'].splitlines() if (r:=json.loads(line))['id'] in ids]
    require(len(gold) == 3 and {g.claim_id for g in gold} == ids, 'three_gold_shape')
    # Nonterminal/invalid decisions are not abstentions; empty only for scorer coverage accounting.
    predictions = [parse_prediction(r['terminal'].get('prediction') or {'id':r['claim_id'],'evidence':{}},corpus) for r in records]
    membership = json.loads(checked(Path(release['conditional_inputs']).parent/'private-membership.json',release['private_membership_sha256']))['fit_overlap']
    require(len(membership) == 3 and sum(membership) == 1, 'one_plus_two_strata')
    subgroups = {}
    for flag in (True,False):
        selected = {row['claim_id'] for row,m in zip(bundle,membership,strict=True) if m is flag}
        subgroups['fit_overlap' if flag else 'no_direct_fit_overlap'] = score_original(
            [g for g in gold if g.claim_id in selected], [p for p in predictions if p.claim_id in selected])
    return {'status':'conditional_probe_scored_not_autonomous_success', 'costs':costs,
        'quality':score_original(gold,predictions), 'fit_subgroups':subgroups,
        'terminal_statuses':dict(Counter(r['terminal']['status'] for r in records)),
        'raw_action_proposals':dict(Counter(r['attempt']['diagnostics'].get('raw_wire_action') for r in records)),
        'wire_audits':wires, 'gold_loaded_after_inference_exit':True,
        'nonterminal_empty_predictions_are_failures_not_abstentions':True,
        'scripted_read_not_autonomous':True, 'independent_test':False, 'validation_gate':False,
        'further_execution_authorized':False}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('release','inference-dir','scoring-dir'):
        parser.add_argument('--'+name,type=Path,required=True)
    parser.add_argument('--release-sha',required=True)
    args = parser.parse_args()
    try:
        result = score(args)
    except Exception as exc:
        write_once(OUTPUT/'score-failed.json', {'exception_type':type(exc).__name__,
            'costs_retained':(OUTPUT/'cost-before-quality.json').is_file()})
        raise
    write_once(OUTPUT/'score.json',result)
    if result['status'] == 'partial_or_unknown_no_quality':
        raise SystemExit(2)


if __name__ == '__main__':
    main()
