"""Redacted paired-score closeout; never load gold, model weights or raw text for export."""
from __future__ import annotations

import argparse
from collections import Counter
import copy
import json
from pathlib import Path
import subprocess

from export_scifact_prospective_commit import row_summary
from run_scifact_grounding_train_operator import ROOT, require, sha
from climate_rag.scifact_read_continuation import ordered_write
from climate_rag.scifact_relation_verifier import project_assessment
from scifact_paired_comparison import ARMS, VERSIONS

ORIGINAL = ROOT/'runs/scifact-evidence-commit-paired-v2-v3-20261002'


def summarize_row(row):
    projected = copy.deepcopy(row)
    for step in projected['steps']:
        if step['stage'] == 'verify' and 'relation' in (step.get('decision') or {}):
            step['decision'] = project_assessment(step['decision'])
    result = row_summary(projected)
    result['issued_positive_refs'] = len(row['verdict_refs'])
    result['max_legal_commit_options'] = max(
        (len(s['observation'].get('commit_selections', {})) for s in row['steps']
         if s['stage'] == 'plan'), default=0)
    for event, step in zip(result['trace'], row['steps'], strict=True):
        decision = step.get('decision') or {}
        if 'relation' in decision:
            event['direct_sentence_count'] = len(decision['direct_sentence_ids'])
            event['background_sentence_count'] = len(decision['background_sentence_ids'])
            event['qualifier_counts'] = dict(Counter(decision['qualifiers'].values()))
            event['uncertainty'] = decision['uncertainty']
    return result


def select_cases(cases):
    selected = []
    def add(label, predicate):
        for i, case in enumerate(cases):
            if i not in [n for n, _ in selected] and predicate(case):
                selected.append((i, label))
                return
    def ok(case, version, arm):
        return case[version][arm]['strict_whole_answer']
    add('fixed_top1_v3_positive_win', lambda c: c['positive'] and not ok(c,'v2','fixed_top1') and ok(c,'v3','fixed_top1'))
    add('fixed_policy_v3_positive_loss', lambda c: c['positive'] and any(
        ok(c,'v2',a) and not ok(c,'v3',a) for a in ARMS[:2]))
    add('fixed_all_difference_or_common_failure', lambda c: c['positive'] and (
        ok(c,'v2','fixed_all') != ok(c,'v3','fixed_all') or not ok(c,'v3','fixed_all')))
    add('v3_adaptive_loses_to_fixed', lambda c: c['positive'] and not ok(c,'v3','adaptive')
        and any(ok(c,'v3',a) for a in ARMS[:2]))
    if len(selected) < 4:
        add('v3_adaptive_only_positive', lambda c: c['positive'] and ok(c,'v3','adaptive')
            and not any(ok(c,'v3',a) for a in ARMS[:2]))
    add('incorrect_positive_abstention', lambda c: c['positive']
        and c['v3']['adaptive']['terminal_action'] == 'abstain')
    if len(selected) < 5:
        add('correct_NEI_abstention_interface_boundary', lambda c: not c['positive']
            and ok(c,'v3','adaptive') and c['v3']['adaptive']['terminal_action'] == 'abstain')
    return [{'case': f'case_{i+1:02d}', 'illustrative_selection': label, **cases[i]}
            for i, label in selected[:5]]


def export(replay: Path, destination: Path):
    def read(path):
        return json.loads(path.read_bytes())
    completed, reservation = read(replay/'replay-exit.json'), read(replay/'replay-reserved.json')
    require(completed['original_unchanged'] and completed['model_calls'] == 0, 'replay_incomplete')
    source_check = read(replay/'source-tree-after.json')
    require(source_check['status'] == 'PASS' and source_check['source_git'] == reservation['replay_source_git']
            and source_check['source_archive_sha256'] == reservation['replay_archive_sha256'], 'replay_source_identity')
    before, after = read(replay/'original-manifest-before.json'), read(replay/'original-manifest-after.json')
    require(before == after, 'original_changed')
    def original(name):
        path = ORIGINAL/name
        require(sha(path) == after[name], 'original_changed_after_replay')
        return read(path)
    comparison = read(replay/'comparison.json')
    require(comparison['status'] == 'paired_scored', 'no_paired_quality')
    reports = [read(replay/v/'quality.json') for v in ('v2','v3')]
    require(comparison['protocol_results'] == reports, 'quality_comparison_binding')
    require([r['protocol'] for r in reports] == list(VERSIONS), 'protocol_order')
    require(reports[0]['initial_retrieval_cost'] == reports[1]['initial_retrieval_cost'],
            'shared_historical_preparation_identity')
    cost = original('cost-before-gold.json')
    require(comparison['original_cost_sha256'] == sha(ORIGINAL/'cost-before-gold.json'), 'cost_binding')
    result = {'schema_version':'scifact-paired-score-closeout-v1', 'original_job':'31980221',
        'original_slurm_status':'FAILED', 'original_no_quality_preserved':True,
        'replay_status':comparison['status'], 'replay':reservation, 'replay_exit':completed,
        'source_tree_check':source_check,
        'comparison':{k:v for k,v in comparison.items() if k != 'protocol_results'},
        'cost':{k:cost[k] for k in ('unique_physical_calls','unknown_usage_attempts',
            'known_token_lower_bound','total_tokens','planned_slots','shared_preparation_seconds')},
        'historical_shared_preparation_cost_once':reports[0]['initial_retrieval_cost'],
        'versions':{}, 'paired_cases':[], 'artifact_sha256':{
            name:sha(replay/name) for name in ('comparison.json','replay-exit.json','replay-reserved.json',
                'v2/quality.json','v3/quality.json','original-manifest-before.json','original-manifest-after.json',
                'source-tree-after.json')},
        'limitations':['Consumed conditional TRAIN24, not independent test or online SLA.',
            'Original GPU FAILED/no_quality remains immutable; CPU replay repairs audit only.',
            'Across-version adaptive is verifier plus feedback joint change, not isolated feedback effect.',
            'Fixed unresolved and adaptive explicit NEI abstention have asymmetric interfaces.',
            'Shared preparation counted once; phase, backend and Slurm timings overlap.',
            'Raw claims, original IDs, gold and response text remain on Spartan.']}
    ids = [c['claim_id'] for c in reports[0]['arms']['fixed_top1']['cases']]
    require(len(ids) == len(set(ids)) == 24, 'all_cases_required')
    cases = [{'positive':reports[0]['arms']['fixed_top1']['cases'][i]['positive'], 'v2':{}, 'v3':{}}
             for i in range(24)]
    for v, report in zip(('v2','v3'), reports, strict=True):
        phase = original('phase-'+v+'.json')
        per_version = {'original_failure':original(v+'/no-quality.json'),
            'physical_cost':phase['physical_generation_cost'], 'phase_elapsed_seconds':phase['elapsed_seconds'],
            'model_load_seconds':original(v+'/model-load.json')['load_and_input_validation_seconds'], 'arms':{}}
        for arm in ARMS:
            values = report['arms'][arm]
            require([c['claim_id'] for c in values['cases']] == ids, 'case_roster')
            per_arm = {k:v for k,v in values.items() if k != 'cases'}
            per_arm['nei_denominator'] = values['planned']-values['positive_count']
            per_arm['physical_cost'] = report['arm_physical_generation_cost'][arm]
            for i, case in enumerate(values['cases']):
                row = original(f"{v}/inference/{case['claim_id']}-{arm}/result.json")
                cases[i][v][arm] = {**summarize_row(row), 'strict_whole_answer':case['strict_whole_answer']}
            per_arm['terminal_counts'] = dict(Counter(c[v][arm]['terminal_action'] or 'unresolved' for c in cases))
            per_arm['feedback_totals'] = {key:sum(c[v][arm][key] for c in cases) for key in (
                'proposed_verify','executed_verify','valid_verify','derived_continue_after_feedback',
                'derived_continue_after_insufficient')}
            per_version['arms'][arm] = per_arm
        result['versions'][v] = per_version
    result['paired_cases'] = select_cases(cases)
    result['slurm'] = subprocess.check_output(['sacct','-j','31980221,'+reservation['cpu_job'],'-nP',
        '-o','JobID,State,ExitCode,Elapsed,MaxRSS,TotalCPU,AllocTRES'],text=True).strip().splitlines()
    result['exporter_sha256'] = sha(Path(__file__))
    ordered_write(destination, result)
    print(json.dumps({'file':destination.name,'sha256':sha(destination),'bytes':destination.stat().st_size,
                      'paired_cases':len(result['paired_cases'])}))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--replay',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args = parser.parse_args()
    export(args.replay,args.output)
