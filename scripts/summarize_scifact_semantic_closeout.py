"""Stdlib-only, content-free closeout of the frozen completed semantic pair.

Runs in place on Spartan. No inference, prompt reconstruction, new selection,
retrieval, model training or dev/test reads. Frozen GPU scoring already ran
physical()/audit_slot(gap=True); this collector binds those results to hashes.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
import subprocess
import tarfile
from collections import Counter
from pathlib import Path
from typing import Any

ROOT = Path('/data/gpfs/projects/punim2936/portfolio_20260903/climate-public-retrieval-v2')
SOURCE = '99cd9ff707697ea395a9bc067f3ce91395071cdb'
RELEASE = 'climate-scifact-semantic-pair-20260930-v1'
POLICIES = ('scifact-gap-original-v1', 'scifact-gap-semantic-policy-v1')
ROUTES = ('fixed_retrieval', 'fixed_rerank', 'deterministic_extra', 'adaptive')
STRATA = ('initial_doc_opportunity', 'top20_doc_replenishable', 'gold_absent_top20', 'nei')
METRICS = ('abstract_label_only', 'abstract_rationalized', 'sentence_selection', 'sentence_label')
RUN_SHA = ('557b7b8ddc7f4e40c1ac1abbe2abd68af37536427f3b106e9e0063e3f6ae746a',
           '486f1c5866011e926b47fe2848189ce4c121213ab7fc86fc22a2c18ea68403f8')
SCORE_SHA = '5cee393c8d5c671def659885d12d3f663465cc633e4f6d0eab83c06fde07fa73'
PROTOCOL_SHA = '5595143cfb6d276184e91f40856efa388ab440dc78527fe7fcffd52c12bb296d'
INPUT_SHA = 'c4907db623b644c9c883a807f07597f87746595da7fc6435714452304e50a44d'
PREP = 'data/scifact-original-20260930/prepared-r1/'
LEGACY = 'data/scifact-train-diagnostic-20260930-r1/'


def require(ok: bool) -> None:
    if not ok:
        raise ValueError('compact contract failed; private values suppressed')


def sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def checked(path: Path, expected: str) -> bytes:
    require(path.is_file() and not path.is_symlink() and path.stat().st_size < 50_000_000)
    raw = path.read_bytes()
    require(sha(raw) == expected)
    return raw


def member_bytes(archive: tarfile.TarFile, name: str) -> bytes:
    matches = [m for m in archive.getmembers() if m.name == name]
    require(len(matches) == 1 and matches[0].isfile() and matches[0].size < 40_000_000)
    stream = archive.extractfile(matches[0])
    if stream is None:
        raise ValueError('missing frozen regular member')
    return stream.read()


def numbers(data: dict[str, Any], keys: tuple[str, ...]) -> dict[str, Any]:
    result = {k: data[k] for k in keys}
    require(all(v is None or type(v) in (bool, int, float) for v in result.values()))
    require(all(not isinstance(v, float) or math.isfinite(v) for v in result.values()))
    return result


def metric_summary(data: dict[str, Any]) -> dict[str, Any]:
    return {k: numbers(data[k], ('correct', 'predicted', 'relevant', 'precision', 'recall', 'f1'))
            for k in METRICS}


def route_summary(route: dict[str, Any]) -> dict[str, Any]:
    # Never copy claim_diagnostics, current_opportunity_by_claim or arbitrary maps.
    return {
        'official_metrics': metric_summary(route['official_point_score']['metrics']),
        'by_legacy_stratum': {s: metric_summary(route['by_legacy_stratum'][s]['metrics']) for s in STRATA},
        'terminal': numbers(route['terminal_decisions'], ('selected_claims', 'validated_nonempty_answer',
            'valid_model_abstention', 'failure_or_budget_termination', 'nei_claims', 'nei_false_evidence',
            'nei_valid_model_abstention', 'nei_failure_empty_prediction', 'nonempty_answer_coverage',
            'claim_verdict_accuracy')),
        'tokens': numbers(route['known_tokens_including_failures'], ('input_tokens', 'output_tokens')),
        'latency_and_unknowns': numbers(route, ('whole_question_p50_ms', 'whole_question_p95_ms',
            'unknown_usage_attempts', 'token_totals_are_lower_bounds')),
        'raw_action_proposals': {a: route['raw_action_proposals'].get(a, 0)
                                 for a in ('answer', 'abstain', 'read', 'rewrite', 'rerank')},
        'chain': numbers(route['chain'], ('model_events', 'execution_success', 'newly_seen_evidence',
            'newly_seen_then_gold_label_complete_rationale', 'strict_whole_final_answer_correct_slots')),
        'tools': numbers(route['actual_tools'], ('slots_with_tool_opportunity', 'slots_with_model_selected_event')),
    }


def visibility_summary(rows: list[Any], gold: dict[int, Any]) -> dict[str, int]:
    counts: Counter[str] = Counter()
    for row in rows:
        result = row['result']
        # This frozen pair has exactly one generation call, not an inferred
        # universal last-context rule for future multi-step experiments.
        require(len(result['generation_attempts']) == len(result['visible_attempts']) == 1)
        visible: dict[str, set[int]] = {}
        for v in result['visible_attempts'][0]['visible']:
            visible.setdefault(str(v['doc_id']), set()).add(v['sentence_index'])
        predictions = row['prediction']['evidence']
        evidence = gold[row['claim_id']]['evidence']
        for doc, prediction in predictions.items():
            counts['predicted_documents'] += 1
            counts['predicted_documents_not_matching_official_gold'] += doc not in evidence
            counts['predicted_documents_with_more_than_three_sentences'] += len(prediction['sentences']) > 3
        for doc, rationales in evidence.items():
            if not any(0 < len(r['sentences']) <= 3 and set(r['sentences']) <= visible.get(doc, set()) for r in rationales):
                continue
            counts['gold_documents_with_complete_legal_rationale_visible'] += 1
            prediction = predictions.get(doc)
            if prediction is None:
                counts['visible_gold_document_omitted'] += 1
            elif prediction['label'] != rationales[0]['label']:
                counts['visible_gold_document_wrong_label'] += 1
            else:
                counts['visible_gold_document_correct_label'] += 1
                counts['correct_label_complete_rationale_any_predicted_position'] += any(
                    set(r['sentences']) <= set(prediction['sentences']) for r in rationales)
                counts['correct_label_complete_rationale_in_first_three'] += any(
                    set(r['sentences']) <= set(prediction['sentences'][:3]) for r in rationales)
    return {k: counts[k] for k in ('predicted_documents', 'predicted_documents_not_matching_official_gold',
        'predicted_documents_with_more_than_three_sentences', 'gold_documents_with_complete_legal_rationale_visible',
        'visible_gold_document_omitted', 'visible_gold_document_wrong_label', 'visible_gold_document_correct_label',
        'correct_label_complete_rationale_any_predicted_position', 'correct_label_complete_rationale_in_first_three')}


def population_summary(assignment: dict[str, Any], audit: list[Any], ledger: dict[str, Any],
                       runs: list[Any], gold_lines: list[bytes]) -> dict[str, Any]:
    eligible = set(assignment['eligible_train_ids'])
    require(len(eligible) == 531 and {r['id'] for r in audit} == eligible)
    component = {i: str(assignment['claim_component'][str(i)]) for i in eligible}
    consumed = set(ledger['model_consumed_ids'])
    uncertain = set(ledger['uncertain_ids'])
    for run in runs:
        rows = run['runs']
        ids = {r['claim_id'] for r in rows}
        require(len(ids) == 12 and len(rows) == 48 and ids <= eligible)
        require({(r['claim_id'], r['route']) for r in rows} == {(i, rt) for i in ids for rt in ROUTES})
        for row in rows:
            result = row['result']
            if result.get('generation_attempts') or result.get('model_calls', 0) > 0:
                consumed.add(row['claim_id'])
            else:
                uncertain.add(row['claim_id'])
    uncertain -= consumed
    require(consumed | uncertain <= eligible)
    excluded_components = {component[i] for i in consumed | uncertain}
    remaining = {i for i in eligible if component[i] not in excluded_components}
    labels: Counter[str] = Counter()
    documents = rationale_sets = usable_sets = 0
    seen = set()
    for line in gold_lines:
        row = json.loads(line)
        if row['id'] not in remaining:
            continue
        seen.add(row['id'])
        evidence = row['evidence']
        present_labels = {r['label'] for rats in evidence.values() for r in rats}
        require(present_labels <= {'SUPPORT', 'CONTRADICT'})
        label = 'NEI' if not evidence else '+'.join(sorted(present_labels))
        labels[label] += 1
        documents += len(evidence)
        rationale_sets += sum(len(rats) for rats in evidence.values())
        usable_sets += sum(0 < len(r['sentences']) <= 3 for rats in evidence.values() for r in rats)
    require(seen == remaining)
    return {'eligible_train_claims': len(eligible), 'eligible_train_components': len(set(component.values())),
        'cumulative_model_consumed_claims': len(consumed), 'uncertain_claims': len(uncertain),
        'excluded_components': len(excluded_components), 'excluded_eligible_claims': len(eligible - remaining),
        'remaining_claims': len(remaining), 'remaining_components': len({component[i] for i in remaining}),
        'remaining_gold_label_scope': dict(labels), 'remaining_annotated_claim_document_pairs': documents,
        'remaining_rationale_sets': rationale_sets, 'remaining_rationale_sets_at_most_three_sentences': usable_sets,
        'remaining_legacy_stratum_claims': dict(Counter(r['stratum'] for r in audit if r['id'] in remaining)),
        'new_selection_performed': False, 'all_eligible_train_had_prior_gold_preparation_exposure': True}


def collect() -> dict[str, Any]:
    source = ROOT / 'envs/scifact-semantic-source-99cd9ff'
    source_tar = ROOT / 'envs/scifact-semantic-execution-99cd9ff.tar'
    checked(source_tar, 'b3fa6e96b56e466b2861f0eea4465c0bb94a48f82fb2b3692eb235e4ff5d2a2a')
    frozen_sources = {}
    with tarfile.open(source_tar) as archive:
        for name in ('scripts/audit_scifact_train_closeout.py', 'scripts/audit_scifact_bounded_closeout.py',
                     'scripts/score_scifact_semantic_pair.py'):
            raw = member_bytes(archive, name)
            require(raw == (source / name).read_bytes())
            frozen_sources[name] = sha(raw)
    spec = importlib.util.spec_from_file_location('frozen_independent', source / 'scripts/audit_scifact_train_closeout.py')
    if spec is None or spec.loader is None:
        raise ValueError('missing frozen stdlib helper')
    independent = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(independent)
    scoring_tar = ROOT / 'envs/scifact-semantic-bundles-99cd9ff/scoring.tar'
    checked(scoring_tar, '7e2303364b69be920de017179485bbb6f9d0e5d48367717576b3411c92c8ca8e')
    with tarfile.open(scoring_tar) as archive:
        gold = {r['id']: r for r in (json.loads(line) for line in member_bytes(archive, 'gold.jsonl').splitlines())}
    score = json.loads(checked(ROOT / 'runs' / (RELEASE + '-' + POLICIES[1]) / 'score.json', SCORE_SHA))
    require(score['source_git'] == SOURCE and score['protocol_sha256'] == PROTOCOL_SHA)
    require(score['initial_contexts_comparable'] is True and score['agent_benefit_established'] is False)
    require(score['status'] == 'paired_train_diagnostic_only')
    runs, policies = [], {}
    for policy, expected in zip(POLICIES, RUN_SHA):
        base = ROOT / 'runs' / (RELEASE + '-' + policy)
        run = json.loads(checked(base / 'inference/run.json', expected))
        op = json.loads((base / 'operator-status.json').read_bytes())
        require(op['status'] == op['stage'] == 'complete' and op['source_git'] == SOURCE)
        require(op['official_dev_read'] is False and op['inference_sha256'] == expected and run['gold_loaded'] is False)
        flight_raw = checked(base / 'inference/runtime-preflight.json', run['preflight_sha256'])
        flight = json.loads(flight_raw)
        require(flight['status'] == 'passed' and flight['attempted_calls'] == 4 and flight['gap'] is True)
        require(flight['official_data_read'] is False and flight['policy'] == policy)
        if policy == POLICIES[1]:
            require(op['score_sha256'] == SCORE_SHA and op['accepted_predecessor_run_sha256'] == RUN_SHA[0])
        p = score['policies'][policy]
        require(p['run_sha256'] == expected and len(p['slot_wire_audits']) == 48)
        independent.reconcile_private_files(base / 'inference/private-responses/preflight', flight['records'])
        for index, row in enumerate(run['runs'], 1):
            independent.verify_cost(row['result'])
            independent.reconcile_private_files(base / f'inference/private-responses/slot-{index:02d}', row['result']['generation_attempts'])
            require(json.loads((base / f'inference/slot-{index:02d}.json').read_bytes()) == row)
        for route in ROUTES:
            predictions = {r['claim_id']: r['prediction'] for r in run['runs'] if r['route'] == route}
            independent_metrics = independent.rescore(gold, predictions)
            original_metrics = p['quality']['routes'][route]['official_point_score']['metrics']
            require(all(math.isclose(v, original_metrics[m][k], rel_tol=1e-12, abs_tol=1e-12)
                        for m, values in independent_metrics.items() for k, v in values.items()))
        wire_keys = ('bytes', 'complete_responses', 'empty_grammar_receipts', 'grammar_logs')
        costs = {}
        for phase in ('train', 'preflight'):
            c = score['costs'][policy][phase]
            costs[phase] = numbers(c, ('calls', 'unknown_usage_attempts', 'token_totals_are_lower_bounds',
                                       'generation_ms_known_sum', 'generation_ms_missing'))
            costs[phase]['tokens'] = numbers(c['known_tokens_including_failures'], ('input_tokens', 'output_tokens'))
        policies[policy] = {'run_sha256': expected, 'operator_sha256': sha((base / 'operator-status.json').read_bytes()),
            'preflight_sha256': sha(flight_raw), 'costs': costs,
            'routes': {r: route_summary(p['quality']['routes'][r]) for r in ROUTES},
            'posthoc_visible_evidence_error_layers': {route: visibility_summary(
                [r for r in run['runs'] if r['route'] == route], gold) for route in ROUTES},
            'train_wire_audit': {k: sum(numbers(w, wire_keys)[k] for w in p['slot_wire_audits']) for k in wire_keys},
            'preflight_wire_audit': numbers(p['preflight_wire_audit'], wire_keys),
            'validation_repairs': sum(r['result']['validation_repairs'] for r in run['runs']),
            'overlapping_timing_ms_do_not_add': {'operator': op['elapsed_seconds'] * 1000,
                **numbers(run, ('model_load_ms', 'reranker_load_ms', 'bm25_build_ms'))}}
        runs.append(run)
    input_path = ROOT / 'envs/scifact-semantic-inputs-c4907db6.tar'
    checked(input_path, INPUT_SHA)
    names = (PREP + 'private-group-assignment.json', LEGACY + 'private/sampling-audit.json',
             PREP + 'gold/claims_train.jsonl', PREP + 'preparation-manifest.json')
    with tarfile.open(input_path) as archive:
        payload = {n: member_bytes(archive, n) for n in names}
    require(sha(payload[names[0]]) == '6d79861daef5d385878913fe10681dff95df444e6f7b1b76ab7c07f4d4d70a12')
    require(sha(payload[names[1]]) == 'ce38a8d705296a89f3079c672b535dd05c3d64bd22cf2ffdae60013e138768e9')
    manifest = json.loads(payload[names[3]])
    require(sha(payload[names[2]]) == manifest['output_file_sha256']['gold/claims_train.jsonl'])
    ledger = json.loads(checked(ROOT / 'posthoc/scifact-semantic-preparation-779e49883570/private/consumption-ledger.json',
        '65eac7e26e5603ee8672b44eb6b0dce0d6d8544c8759cfa36184daaf0a518a73'))
    population = population_summary(json.loads(payload[names[0]]), json.loads(payload[names[1]]), ledger,
                                    runs, payload[names[2]].splitlines())
    slurm = subprocess.check_output(['sacct', '-n', '-P', '-j', '31706518,31706520', '--format=JobIDRaw,State,ExitCode,Start,End,ElapsedRaw,TotalCPU,CPUTimeRAW,MaxRSS,ReqTRES,AllocTRES'], text=True)
    return {'schema_version': 'scifact-semantic-content-free-closeout-v1', 'execution_source_git': SOURCE,
        'collector_sha256': sha(Path(__file__).read_bytes()), 'score_sha256': SCORE_SHA,
        'frozen_audit_source_sha256': frozen_sources, 'independent_official_metric_groups_equal': 32,
        'independent_physical_wire_multiplicity_recheck': True,
        'protocol_sha256': PROTOCOL_SHA, 'policies': policies, 'remaining_train': population,
        'slurm_fields': ['JobIDRaw', 'State', 'ExitCode', 'Start', 'End', 'ElapsedRaw', 'TotalCPU', 'CPUTimeRAW', 'MaxRSS', 'ReqTRES', 'AllocTRES'],
        'slurm_rows': [line.split('|') for line in slurm.splitlines() if line.strip()],
        'frozen_scoring_ran_physical_and_audit_slot_gap_true': True,
        'collector_reexecuted_physical_or_models': False, 'new_selection_or_training': False,
        'dev_or_test_read_in_this_collection': False, 'independent_test': False,
        'claim_verdict_accuracy': None, 'agent_benefit_established': False}


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    require(args.output.resolve().is_relative_to(ROOT / 'posthoc') and not args.output.exists())
    result = collect()
    text = json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + '\n'
    # Defense in depth against accidentally exporting a per-item source subtree.
    require(not any('"' + k + '"' in text for k in ('claim_id', 'claim', 'evidence', 'answer', 'prediction',
        'claim_diagnostics', 'current_opportunity_by_claim', 'sentences', 'documents', 'records')))
    with args.output.open('x', encoding='utf-8', newline='\n') as handle:
        handle.write(text)
    print(json.dumps({'status': 'compact_written', 'sha256': sha(text.encode()), 'bytes': len(text.encode())}))


if __name__ == '__main__':
    main()
