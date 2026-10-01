"""Export aggregate and five pseudonymous cases only; never export raw text/gold."""
from collections import Counter
import hashlib
import json
from pathlib import Path
import subprocess

ROOT = Path('/data/gpfs/projects/punim2936/portfolio_20260903/climate-public-retrieval-v2')
RUN = ROOT / 'runs/scifact-evidence-commit-prospective24-v1-20261002-confirmation-v1'
STAGE = ROOT / 'envs/prospective24-source-da243036871f'
ARMS = ('fixed_top1', 'fixed_all', 'adaptive')
JOB = '31956320'


def read(name):
    return json.loads((RUN / name).read_bytes())


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def row_summary(row):
    trace = []
    proposed = executed = valid = continues = after_insufficient = 0
    had_feedback = False
    insufficient_feedback = False
    for step in row['steps']:
        decision = step.get('decision') or {}
        event = {'stage': step['stage'], 'status': step['status'],
                 'physical_call': step.get('physical_attempt_id') is not None}
        if step['stage'] == 'plan':
            action = decision.get('action')
            event['action'] = action
            if action == 'verify':
                event['source_alias'] = decision['source_id']
                proposed += 1
                continues += int(had_feedback)
                after_insufficient += int(insufficient_feedback)
            elif action == 'commit':
                selected = step['observation']['commit_selections'][decision['selection_id']]
                refs = {r['ref_id']: r for r in row['verdict_refs']}
                event['selected_source_aliases'] = [refs[k]['source_alias'] for k in selected]
            elif action == 'abstain':
                event['reason'] = decision['reason']
        else:
            event['source_alias'] = step['source_id']
            event['label'] = decision.get('label')
            event['sentence_count'] = len(decision.get('sentence_ids', []))
            executed += int(event['physical_call'])
            valid += int(event['physical_call'] and step['status'] == 'valid')
            had_feedback = True
            insufficient_feedback = decision.get('label') == 'INSUFFICIENT'
        trace.append(event)
    terminal = row.get('terminal') or {}
    assembled = row.get('assembled') or {}
    documents = assembled.get('documents', [])
    return {'state': row['state'], 'reason': row['reason'],
            'terminal_action': terminal.get('action'), 'physical_calls': row['physical_calls'],
            'elapsed_seconds': row['elapsed_seconds'],
            'proposed_verify': proposed, 'executed_verify': executed, 'valid_verify': valid,
            'derived_continue_after_feedback': continues,
            'derived_continue_after_insufficient': after_insufficient,
            'committed': [{'source_alias': d['source_id'], 'label': d['label'],
                           'sentence_count': len(d['sentence_ids'])} for d in documents],
            'feedback_status_counts': dict(Counter(f['status'] for f in row['verification_feedback'])),
            'trace': trace}


def main():
    accounting = subprocess.check_output(['sacct', '-j', JOB, '-n', '-P',
        '--format=JobID,State,ExitCode,Start,End,ElapsedRaw,MaxRSS,TotalCPU,AllocCPUS,ReqMem,AllocTRES'], text=True)
    accounting_rows = [line.split('|') for line in accounting.strip().splitlines()]
    root_job = next(r for r in accounting_rows if r[0] == JOB)
    if root_job[1] not in {'COMPLETED', 'FAILED', 'TIMEOUT', 'OUT_OF_MEMORY', 'CANCELLED'}:
        raise ValueError('job_not_terminal')
    reservation = read('reserved.json')
    if reservation['job_id'] != JOB or reservation['source_git'] != 'da243036871f61eef2e039a1618b8a3e1e1a00ac':
        raise ValueError('wrong_run')
    if reservation['release_sha256'] != 'db794da36ea061b5c0f537dd83829c7bd69e6763663410f407c6f5a5f72123ad':
        raise ValueError('wrong_release')
    completed = read('complete.json') if (RUN / 'complete.json').exists() else None
    quality = read('quality.json') if (RUN / 'quality.json').exists() else None
    cost = read('cost-before-gold.json') if (RUN / 'cost-before-gold.json').exists() else None
    if quality and (quality['status'] != 'scored' or not completed
                    or completed['quality_sha256'] != sha(RUN / 'quality.json')):
        raise ValueError('quality_binding')
    if cost and completed and completed['cost_sha256'] != sha(RUN / 'cost-before-gold.json'):
        raise ValueError('cost_binding')
    report = {'protocol': reservation['protocol'], 'job_id': JOB,
        'source_git': reservation['source_git'], 'release_sha256': reservation['release_sha256'],
        'status': completed['status'] if completed else 'incomplete_operator',
        'accounting_fields': ['JobID', 'State', 'ExitCode', 'Start', 'End', 'ElapsedRaw', 'MaxRSS', 'TotalCPU', 'AllocCPUS', 'ReqMem', 'AllocTRES'],
        'accounting_rows': accounting_rows,
        'artifact_sha256': {name: sha(RUN / name) for name in
            ('reserved.json', 'complete.json', 'worker-exit.json', 'runtime.json', 'quality.json',
             'cost-before-gold.json', 'no-quality.json', 'inference/initial-frames.json',
             'prepared/selection.json', 'prepared/component-reservations.json', 'prepared/preparation.json') if (RUN / name).exists()},
        'cost': cost, 'arms': {}, 'paired_cases': [],
        'limitations': ['Conditional component TRAIN24 diagnostic; all531 eligible TRAIN early gold-preparation-seen; not independent/source-family-unseen.',
            'Mandatory verify prerequisite is not spontaneous tool demand or proof of Agent gain.',
            'Continue counts are derived sequences, not a separate model action.',
            'New shared CPU preparation cost is counted once outside three-arm model cost; prepared frames are reused during inference. Allocation/backend time is not online SLA or monetary cost.',
            'Case correctness is existing scorer output; no raw gold, claim text, source text or physical responses exported.'],
        'exporter_sha256': sha(Path(__file__))}
    if (RUN / 'no-quality.json').exists():
        report['no_quality'] = read('no-quality.json')
    preparation_path = ROOT / 'posthoc/scifact-evidence-commit-prospective24-v1-20261002/preparation.json'
    preparation = None
    if preparation_path.is_file() and sha(preparation_path) == '23e1835a0e4f1330c2d11d0d70a2dbfa5c4655c2192169b05240433f5e5aae72':
        preparation = json.loads(preparation_path.read_bytes())
        report['input_scope'] = preparation['scope']
        report['shared_preparation_cost'] = preparation['shared_preparation_cost']
        report['preparation_sha256'] = sha(preparation_path)
    else:
        report['input_scope'] = 'unknown_preparation_identity'
        report['shared_preparation_cost'] = {'status': 'unknown', 'seconds': None}
    if quality:
        if (preparation is None or quality['selection_sha256'] != preparation['selection_sha256']
                or preparation['model_calls'] != 0):
            raise ValueError('prepared_selection_binding')
        report['selection_sha256'] = quality['selection_sha256']
        report['limitations'].append('Fixed empty-positive outputs remain unresolved; adaptive explicit abstention has a different NEI interface. Old and new24 are not claim-level pairs.')
        report['comparison'] = {k: quality[k] for k in (
            'hypothesis', 'positive_recovered_arms', 'fixed_baseline_arm',
            'adaptive_minus_fixed_positive_correct_by_arm', 'limits', 'comparison_limits')}
        ids = [c['claim_id'] for c in quality['arms']['fixed_top1']['cases']]
        summaries = {}
        for arm in ARMS:
            values = quality['arms'][arm]
            report['arms'][arm] = {k: v for k, v in values.items() if k != 'cases'}
            report['arms'][arm]['nei_denominator'] = values['planned'] - values['positive_count']
            arm_rows = []
            for position, case in enumerate(values['cases']):
                if case['claim_id'] != ids[position]:
                    raise ValueError('case_order_mismatch')
                row = read(f"inference/{case['claim_id']}-{arm}/result.json")
                summary = row_summary(row)
                summary['strict_whole_answer'] = case['strict_whole_answer']
                summaries[position, arm] = summary
                arm_rows.append(summary)
            totals = {key: sum(r[key] for r in arm_rows) for key in (
                'proposed_verify', 'executed_verify', 'valid_verify',
                'derived_continue_after_feedback', 'derived_continue_after_insufficient')}
            totals['terminal_counts'] = dict(Counter(r['terminal_action'] or 'unresolved' for r in arm_rows))
            totals['reason_counts'] = dict(Counter(r['reason'] for r in arm_rows))
            totals['model_selected_source_aliases'] = dict(Counter(e['source_alias'] for r in arm_rows
                for e in r['trace'] if e['stage'] == 'plan' and e.get('action') == 'verify'))
            totals['first_model_selected_source_aliases'] = dict(Counter(next((e['source_alias'] for e in r['trace']
                if e['stage'] == 'plan' and e.get('action') == 'verify'), 'none') for r in arm_rows))
            totals['trajectory_patterns'] = dict(Counter(' > '.join(e['stage'] + ':' + str(e.get('action', e.get('label', e['status'])))
                for e in r['trace']) for r in arm_rows))
            report['arms'][arm]['trajectory'] = totals
        report['paired_outcomes'] = {}
        for baseline in ('fixed_top1', 'fixed_all'):
            grouped = {}
            for positive, label in ((True, 'evidence_bearing'), (False, 'NEI')):
                counts = Counter()
                for i, scored_case in enumerate(quality['arms']['adaptive']['cases']):
                    if scored_case['positive'] != positive:
                        continue
                    a = summaries[i, 'adaptive']['strict_whole_answer']
                    b = summaries[i, baseline]['strict_whole_answer']
                    counts['both_correct' if a and b else 'adaptive_only' if a else 'baseline_only' if b else 'neither'] += 1
                grouped[label] = dict(counts)
            report['paired_outcomes'][baseline] = grouped
        adaptive_rows = [read(f'inference/{i}-adaptive/result.json') for i in ids]
        report['adaptive_choice_limits'] = {
            'max_issued_positive_refs_in_episode': max(len(r['verdict_refs']) for r in adaptive_rows),
            'episodes_with_multiple_positive_refs': sum(len(r['verdict_refs']) > 1 for r in adaptive_rows),
            'max_legal_commit_options_in_actual_plan': max(len(s['observation'].get('commit_selections', {}))
                for r in adaptive_rows for s in r['steps'] if s['stage'] == 'plan'),
            'note': 'Document selection and feedback use do not imply an observed choice among multiple positive subsets.'}
        candidates = []
        def choose(label, predicate):
            for i in range(len(ids)):
                pair = {a: summaries[i, a] for a in ARMS}
                if i not in [p[0] for p in candidates] and predicate(pair):
                    candidates.append((i, label))
                    break
        choose('adaptive_correct_commit_both_baselines_incorrect', lambda p: p['adaptive']['strict_whole_answer'] and p['adaptive']['terminal_action'] == 'commit' and not p['fixed_top1']['strict_whole_answer'] and not p['fixed_all']['strict_whole_answer'])
        choose('baseline_correct_adaptive_incorrect', lambda p: not p['adaptive']['strict_whole_answer'] and (p['fixed_top1']['strict_whole_answer'] or p['fixed_all']['strict_whole_answer']))
        choose('all_arms_correct_positive_commit', lambda p: all(r['strict_whole_answer'] and r['terminal_action'] == 'commit' for r in p.values()))
        choose('correct_model_abstention_vs_incorrect_fixed_all_commit', lambda p: p['adaptive']['strict_whole_answer'] and p['adaptive']['terminal_action'] == 'abstain' and p['fixed_all']['terminal_action'] == 'commit' and not p['fixed_all']['strict_whole_answer'])
        choose('incorrect_adaptive_commit_vs_fixed_unresolved', lambda p: p['adaptive']['terminal_action'] == 'commit' and not p['adaptive']['strict_whole_answer'] and p['fixed_all']['state'] == 'unresolved')
        choose('incorrect_model_abstention', lambda p: p['adaptive']['terminal_action'] == 'abstain' and not p['adaptive']['strict_whole_answer'])
        choose('continued_feedback_without_complete_answer_recovery', lambda p: p['adaptive']['derived_continue_after_feedback'] > 0 and not p['adaptive']['strict_whole_answer'])
        for i, label in candidates[:5]:
            report['paired_cases'].append({'case': f'case_{i + 1:02d}', 'illustrative_selection': label,
                                          'arms': {a: summaries[i, a] for a in ARMS}})
    if cost and cost['unassigned_physical_generation_cost']['unique_physical_calls']:
        report['limitations'].append('Unassigned cost has unknown arm attribution; empty per-arm zero aggregates are not evidence of zero actual cost.')
    report['reporting_revision'] = 'prospective24_same_three_arm_reporting_no_rescoring'
    destination = STAGE / 'prospective24-confirmation-31956320-compact.json'
    with destination.open('x') as stream:
        json.dump(report, stream, sort_keys=True, indent=2)
        stream.write('\n')
    print(json.dumps({'file': destination.name, 'sha256': sha(destination),
                      'bytes': destination.stat().st_size, 'status': report['status']}))


if __name__ == '__main__':
    main()
