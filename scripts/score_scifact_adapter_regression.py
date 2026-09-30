"""Exit-bound costs, physical wire/event audit, then existing four-route scorer."""
from __future__ import annotations

import argparse
from collections import Counter
import json
import math
from pathlib import Path
from typing import Any

from climate_rag.scifact_adapter_regression import (
    ADAPTER_SHA, FIT_SHA, PREVIOUS_SOURCE, PROTOCOL, SCORING_MANIFEST,
    load_frozen, policy_identity, require,
)
from climate_rag.scifact_diagnostic_scoring import score_diagnostic
from climate_rag.scifact_grounding import parse_abstract, parse_gold
from climate_rag.scifact_semantic_contract import MODEL_SHA, ROUTES, SCORING_NAMES, checked, encoded, sha, write_once
from audit_scifact_bounded_closeout import audit_slot, physical, tariff
from run_scifact_adapter_regression_operator import OUTPUT, validate_release
from run_scifact_grounding_train_operator import ROOT
from score_scifact_semantic_pair import terminal_counts, validate_prepared_scoring


def collect_costs(output: Path, identity: dict[str, Any]) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    directory = output / 'inference'
    records: list[dict[str, Any]] = []
    rows: list[dict[str, Any]] = []
    unresolved = 0
    rerank_calls = rerank_pairs = 0
    rerank_by_route: dict[str, Any] = {route: {'status_counts': {}, 'completed_operations': 0,
        'failed_or_uncompleted_operations': 0, 'elapsed_ms_known_sum': 0.0, 'elapsed_ms_missing': 0,
        'requested_pairs_known_lower_bound': 0} for route in ROUTES}
    reservations = sorted(directory.glob('slot-*-reserved.json'))
    for reservation in reservations:
        number = reservation.name.split('-')[1]
        require(json.loads(reservation.read_bytes())['identity'] == identity, 'slot_reservation_identity')
        raw_path = directory / f'slot-{number}-raw.json'
        try:
            raw = json.loads(raw_path.read_bytes())
            require(raw['regression_identity'] == identity, 'raw_identity')
            records.extend(raw['generation_attempts'])
            rerank_calls += sum(e.get('tool') == 'rerank' and e.get('status') != 'skipped' for e in raw['events'])
            rerank_pairs += raw['rerank_pairs']
            route_cost = rerank_by_route[raw['route']]
            route_cost['requested_pairs_known_lower_bound'] += raw['rerank_pairs']
            for event in raw['events']:
                if event.get('tool') != 'rerank' or event.get('status') == 'skipped':
                    continue
                status = event.get('status', 'unrecorded')
                route_cost['status_counts'][status] = route_cost['status_counts'].get(status, 0) + 1
                route_cost['completed_operations' if status == 'completed' else 'failed_or_uncompleted_operations'] += 1
                elapsed = event.get('elapsed_ms')
                if elapsed is None:
                    route_cost['elapsed_ms_missing'] += 1
                else:
                    require(isinstance(elapsed, (float, int)) and math.isfinite(elapsed) and elapsed >= 0,
                            'rerank_time_invalid')
                    route_cost['elapsed_ms_known_sum'] += elapsed
        except (OSError, ValueError, KeyError):
            unresolved += 1
            continue
        row_path = directory / f'slot-{number}.json'
        if row_path.exists():
            row = json.loads(row_path.read_bytes())
            require(row['regression_identity'] == identity
                    and all(row['result'][k] == v for k, v in raw.items()), 'durable_raw_slot_mismatch')
            rows.append(row)
    require(len(reservations) <= 48 and len(records) <= 168 and rerank_calls <= 36 and rerank_pairs <= 720,
            'regression_reservation_call_bounds')
    costs = tariff(records)
    costs.update(planned_slots=48, reserved_slots=len(reservations), completed_slot_records=len(rows),
        not_reserved_slots=48-len(reservations), unresolved_reserved_slots=unresolved,
        actual_model_calls=None if unresolved else len(records),
        model_calls_known_lower_bound=len(records), reranker_tokens=None,
        rerank_operations_known_lower_bound=rerank_calls, rerank_requested_pairs_known_lower_bound=rerank_pairs,
        rerank_by_route=rerank_by_route,
        rerank_pair_boundary='requested_pairs_include_failed_operations_not_completed_forward_count',
        reranker_tokens_status='not_measured_not_zero',
        token_totals_are_lower_bounds=bool(unresolved or costs['token_totals_are_lower_bounds']),
        all_slots_retained=True, warmup_or_preflight_calls=0)
    return costs, rows


def score(args: Any) -> dict[str, Any]:
    source = Path(__file__).resolve().parents[1]
    release = json.loads(checked(args.release, args.release_sha))
    validate_release(release, source)
    require(args.output == OUTPUT, 'fixed_scoring_output')
    proof = json.loads((args.output / 'allocation/worker-exit.json').read_bytes())
    require(proof['child_reaped'] is True and proof['release_sha256'] == args.release_sha, 'inference_must_exit_first')
    directory = args.output / 'inference'
    policy = policy_identity(source)
    identity = {'source_git': release['source_git'], 'source_archive_sha256': release['source_archive_sha256'],
        'release_sha256': args.release_sha, 'policy': policy, 'policy_sha256': sha(encoded(policy)),
        'inference_archive_sha256': release['inference_archive_sha256'], 'gold_loaded': False}
    costs, rows = collect_costs(args.output, identity)
    physical_index = {p.relative_to(directory).as_posix(): sha(p.read_bytes())
                      for p in directory.rglob('*') if p.is_file()}
    write_once(args.output / 'cost-before-quality.json', costs | {'physical_sha256': physical_index,
        'worker_exit_sha256': sha((args.output / 'allocation/worker-exit.json').read_bytes())})
    pending = {'status': 'incomplete_or_unknown_cost_no_quality', 'costs': costs,
               'gold_loaded': False, 'planned_slots': 48, 'validation_gate': False}
    if proof['returncode'] != 0 or proof.get('parent_wait_interrupted') or len(rows) != 48 or costs['token_totals_are_lower_bounds']:
        return pending
    run_path = directory / 'run.json'
    run = json.loads(checked(run_path, proof['run_sha256']))
    require(run['runs'] == rows and all(run[k] == v for k, v in identity.items()), 'run_identity')
    integrity = json.loads(checked(directory / 'adapter-integrity.json', run['adapter_integrity_sha256']))
    expected_state = {'active_adapters': ['default'], 'lora_layers': 72,
                      'enabled': True, 'merged': False, 'trainable_parameters': 0}
    require(integrity['active_state'] == expected_state and integrity['tensor_count'] == 144
            and integrity['all_checkpoint_values_equal'] is True
            and all(integrity[k] == v for k, v in identity.items()), 'checkpoint_active_proof')
    claims, corpus_bytes = load_frozen(args.inference_dir)
    require([(r['claim_id'], r['route']) for r in rows] == [(c['id'], route) for c in claims for route in ROUTES],
            'full_original_four_route_matrix')
    raw_per_slot = []
    for index, row in enumerate(rows, 1):
        require(row['result']['regression_protocol'] == PROTOCOL, 'regression_protocol_binding')
        for attempt in row['result']['generation_attempts']:
            binding = attempt['diagnostics']
            require(binding['base_model_sha256'] == MODEL_SHA and binding['adapter_sha256'] == ADAPTER_SHA
                    and binding['regression_policy_sha256'] == identity['policy_sha256']
                    and binding['adapter_state'] == expected_state
                    and all(len(binding[key]) == 64 for key in ('actual_prompt_sha256', 'actual_schema_sha256')),
                    'physical_call_active_binding')
        wire, raw = physical(directory / f'private-responses/slot-{index:02d}', row['result']['generation_attempts'], False)
        raw_per_slot.append((wire, raw))
    # Physical identity/cost checked before loading any scoring gold.
    require({p.name for p in args.scoring_dir.iterdir()} == SCORING_NAMES, 'scoring_allowlist')
    manifest = json.loads(checked(args.scoring_dir / 'manifest.json', SCORING_MANIFEST))
    require(manifest['execution_source_git'] == PREVIOUS_SOURCE, 'historical_bundle_source_not_current_execution')
    payload = {n: checked(args.scoring_dir / n, h) for n, h in manifest['scoring_file_sha256'].items()}
    validate_prepared_scoring(payload, manifest)
    corpus = {d.doc_id: d for d in (parse_abstract(json.loads(line)) for line in corpus_bytes.splitlines())}
    gold = [parse_gold(json.loads(line), corpus) for line in payload['gold.jsonl'].splitlines()]
    selected = json.loads(payload['selected-strata.json'])
    require([g.claim_id for g in gold] == [s['id'] for s in selected] == [c['id'] for c in claims], 'gold_order')
    by_id = {g.claim_id: g for g in gold}
    audits = [audit_slot(row, by_id[row['claim_id']], corpus, raw, gap=False)
              for row, (_, raw) in zip(rows, raw_per_slot, strict=True)]
    legacy = [{'id': s['id'], 'component': s['component'], 'stratum': s['legacy_stratum']} for s in selected]
    # Includes exact result.answer -> to_original_prediction -> row.prediction equality.
    quality = score_diagnostic(gold, corpus, rows, legacy)
    for route in ROUTES:
        routed = [(r, a) for r, a in zip(rows, audits, strict=True) if r['route'] == route]
        quality['routes'][route]['terminal_decisions'] = terminal_counts(gold, [r for r, _ in routed])
        for key in ('chain', 'raw_action_proposals', 'attempt_statuses'):
            totals: Counter[str] = Counter()
            for _, audit in routed:
                totals.update(audit[key])
            quality['routes'][route][key] = dict(totals)
    fit = json.loads(checked(ROOT / 'posthoc/scifact-grounding-candidate-40d84a377bd1/fit/records.json', FIT_SHA))
    fit_ids, fit_components = {r['claim_id'] for r in fit}, {r['component'] for r in fit}
    overlap = {s['id'] for s in selected if s['component'] in fit_components}
    require(overlap == {s['id'] for s in selected if s['id'] in fit_ids} and len(overlap) == 1, 'fixed_overlap')
    read_ids = {s['id'] for s in selected if s['legacy_stratum'] == 'top20_doc_replenishable'}
    groups = {'read_opportunity_fit_overlap': read_ids & overlap, 'read_opportunity_no_direct_fit_overlap': read_ids-overlap}
    subgroup = {name: {'claims': len(ids), 'exposed_regression_not_heldout': True,
        'quality': score_diagnostic([g for g in gold if g.claim_id in ids], corpus,
            [r for r in rows if r['claim_id'] in ids], [s for s in legacy if s['id'] in ids])}
        for name, ids in groups.items()}
    return {'status': 'old_TRAIN_four_route_regression_scored_not_gate', 'identity': identity,
        'costs': costs, 'quality': quality, 'read_opportunity_fit_subgroups': subgroup,
        'planned_slots': 48, 'gold_loaded_after_inference_exit': True,
        'wire_audits': [wire for wire, _ in raw_per_slot], 'independent_test': False,
        'validation_gate': False, 'further_execution_authorized': False,
        'interpretation': 'Same active adapter across new four routes; no fresh base/G baseline, no isolated LoRA causal attribution. All twelve exposed queries retained.'}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('output', 'inference-dir', 'scoring-dir', 'release'):
        parser.add_argument('--' + name, type=Path, required=True)
    parser.add_argument('--release-sha', required=True)
    args = parser.parse_args()
    try:
        report = score(args)
    except Exception as exc:
        write_once(args.output / 'score-failed.json', {'status': 'quality_blocked_existing_costs_retained',
            'exception_type': type(exc).__name__, 'validation_gate': False, 'retry_authorized': False})
        raise
    write_once(args.output / 'score.json', report)
    if report['status'] != 'old_TRAIN_four_route_regression_scored_not_gate':
        raise SystemExit(2)


if __name__ == '__main__':
    main()
