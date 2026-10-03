"""Synthetic redaction/selection only; no model or dataset evidence."""
import copy
import json

from export_scifact_paired_score import summarize_row, select_cases


def test_relation_trace_uses_scoring_projection_and_no_raw_fields():
    row = {'claim_id':12345, 'state':'valid_terminal', 'reason':None,
        'terminal':{'action':'abstain'}, 'assembled':None, 'physical_calls':1,
        'elapsed_seconds':2, 'verdict_refs':[], 'verification_feedback':[{'status':'insufficient'}],
        'steps':[{'stage':'verify','status':'valid','physical_attempt_id':'g00','source_id':'c1',
            'observation':{'raw_claim':'DO_NOT_EXPORT'}, 'decision':{
                'source_id':'c1','relation':'INSUFFICIENT','direct_sentence_ids':[],
                'background_sentence_ids':['c1:0'],'minimal_sentence_ids':[],
                'qualifiers':{'population':'unknown'},'uncertainty':'missing_direct_evidence'}}]}
    before = copy.deepcopy(row)
    result = summarize_row(row)
    assert row == before
    event = result['trace'][0]
    assert event['label'] == 'INSUFFICIENT' and event['sentence_count'] == 0
    assert event['background_sentence_count'] == 1 and event['direct_sentence_count'] == 0
    assert result['issued_positive_refs'] == result['max_legal_commit_options'] == 0
    encoded = json.dumps(result)
    assert 'DO_NOT_EXPORT' not in encoded and '12345' not in encoded and 'c1:0' not in encoded


def case(positive=True):
    return {'positive':positive, **{v:{a:{'strict_whole_answer':False,'terminal_action':'unresolved'}
        for a in ('fixed_top1','fixed_all','adaptive')} for v in ('v2','v3')}}


def test_case_selection_retains_win_loss_baseline_loss_and_nei_without_duplicates():
    cases = [case() for _ in range(4)] + [case(False)]
    cases[0]['v3']['fixed_top1']['strict_whole_answer'] = True
    cases[1]['v2']['fixed_top1']['strict_whole_answer'] = True
    cases[3]['v3']['fixed_top1']['strict_whole_answer'] = True
    cases[4]['v3']['adaptive'].update(strict_whole_answer=True,terminal_action='abstain')
    result = select_cases(cases)
    assert len(result) == len({c['case'] for c in result}) == 5
    assert result[0]['illustrative_selection'] == 'fixed_top1_v3_positive_win'
    assert result[1]['illustrative_selection'] == 'fixed_policy_v3_positive_loss'
    assert result[3]['illustrative_selection'] == 'v3_adaptive_loses_to_fixed'
    assert result[4]['illustrative_selection'] == 'correct_NEI_abstention_interface_boundary'
    assert all(set(c) == {'case','illustrative_selection','positive','v2','v3'} for c in result)


def test_no_missing_category_is_fabricated():
    assert select_cases([]) == []
    same = case()
    for v in ('v2','v3'):
        for arm in same[v].values():
            arm.update(strict_whole_answer=True,terminal_action='commit')
    assert select_cases([same]) == []
