from __future__ import annotations

import copy
import json

import pytest

from climate_rag.scifact_grounding import Abstract, GoldClaim, Rationale
from climate_rag.scifact_semantic_contract import encoded, sha
from climate_rag.scifact_semantic_policy import POLICIES
from climate_rag.scifact_terminal import source_from_abstract
from replay_scifact_selected_read import fixed_read, replay, summarise
from test_scifact_train_diagnostic import TinyTokenizer


def fixture():
    corpus = {i:Abstract(i,f'Doc {i}',('First.','Second.','Third.','Fourth.'),False) for i in range(1,8)}
    sources = [source_from_abstract(corpus[i]) for i in range(1,7)]
    opportunity = {'initial':{'candidate_doc_ids':list(range(1,7)),
                              'initial_identity':{'historical':'different'}},
        'witness':{'fixture_history':{'read_ids':['c5']}},'witness_read_doc_ids':[6]}
    review = {'id':1,'legacy_stratum':'top20_doc_replenishable',
              'current_opportunity':{p:copy.deepcopy(opportunity) for p in POLICIES}}
    return corpus,sources,review


def test_replay_uses_original_read_and_existing_bounded_bare_entrypoint():
    corpus,sources,review = fixture()
    queries = []
    def retrieve(query,width):
        queries.append((query,width))
        return sources
    row = replay({'id':1,'claim':'Claim'},review,retrieve,corpus,TinyTokenizer())
    assert queries == [('Claim',20)]
    assert row['fixture_responses'] == 2 and row['scripted_completed_reads'] == 1
    assert row['model_calls'] == row['reranker_calls'] == 0
    assert 6 not in row['states'][0]['visible'] and row['states'][1]['visible'] == {6:[0,1,2,3]}
    assert all(len(state[k]) == 64 for state in row['states']
               for k in ('state_sha256','schema_sha256','prompt_sha256','observation_sha256'))
    assert row['candidate_order_sha256'] == sha(encoded(list(range(1,7))))
    assert row['candidate_order_unchanged'] and not row['initial_identity_unchanged']


@pytest.mark.parametrize('mutation',['candidate_order','policy_disagreement','extra_read'])
def test_saved_witness_cannot_be_replaced_or_retargeted(mutation):
    _,_,review = fixture()
    candidates = list(range(1,7))
    if mutation == 'candidate_order':
        candidates.reverse()
    elif mutation == 'policy_disagreement':
        review['current_opportunity'][POLICIES[1]]['witness_read_doc_ids'] = [5]
    else:
        review['current_opportunity'][POLICIES[0]]['witness']['fixture_history']['read_ids'].append('c4')
    with pytest.raises(ValueError):
        fixed_read(review,candidates)


@pytest.mark.parametrize('rationale',[(0,2),(0,1,2,3)])
def test_first3_credit_opportunity_is_not_any_visible_rationale(rationale):
    corpus,sources,review = fixture()
    row = replay({'id':1,'claim':'Claim'},review,lambda *_:sources,corpus,TinyTokenizer())
    result = summarise([row],{1:GoldClaim(1,'Claim',{6:(Rationale('SUPPORT',rationale),)},())})
    assert result['initial_complete_first3_eligible_claims'] == 0
    assert result['original_read_witnesses_with_complete_first3_rationale'] == (len(rationale) <= 3)
    assert result['original_read_witnesses_newly_reachable'] == (len(rationale) <= 3)
    result.pop('private_diagnosis')
    public = json.dumps(result)
    assert 'claim_id' not in public and 'doc_ids' not in public and 'Claim' not in public


def test_nonwitness_query_only_abstains_and_does_not_search():
    corpus,sources,review = fixture()
    review['legacy_stratum'] = 'nei'
    for old in review['current_opportunity'].values():
        old.pop('witness')
        old.pop('witness_read_doc_ids')
    row = replay({'id':1,'claim':'Claim'},review,lambda *_:sources,corpus,TinyTokenizer())
    assert row['fixture_responses'] == 1 and row['scripted_completed_reads'] == 0
    result = summarise([row],{1:GoldClaim(1,'Claim',{},())})
    assert result['opportunity_changes'] == {'nei->nei':1}
