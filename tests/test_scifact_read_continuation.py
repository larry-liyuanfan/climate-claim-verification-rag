"""Only synthetic CPU state/call fixtures; no model or gold dataset consumption."""
from __future__ import annotations

import copy
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from climate_rag.agent_protocol import ModelResponseValidationError
from climate_rag.scifact_read_continuation import finalise, ordered_write, policy, validate_state
from climate_rag.scifact_semantic_contract import encoded, sha
from prepare_scifact_read_continuation import reconstruct
from replay_scifact_selected_read import replay
from run_scifact_read_continuation import call_once
from run_scifact_read_continuation_operator import OUTPUT, RESOURCE, validate_release
from test_scifact_adapter_regression import instrument, release as old_release
from test_scifact_selected_read import fixture
from test_scifact_train_diagnostic import TinyTokenizer

SOURCE = Path(__file__).resolve().parents[1]


def prepared():
    corpus,sources,review = fixture()
    tokenizer = TinyTokenizer()
    claim = {'id':1,'claim':'Claim'}
    def retrieval(*_):
        return sources
    expected = replay(claim,review,retrieval,corpus,tokenizer)
    row = reconstruct(claim,review,expected,retrieval,corpus,tokenizer)
    return row,corpus,tokenizer


def test_ordered_roundtrip_preserves_exact_prompt(tmp_path):
    row,_,tokenizer = prepared()
    ordered_write(tmp_path/'inputs.json',row)
    loaded = json.loads((tmp_path/'inputs.json').read_bytes())
    validate_state(loaded,tokenizer)
    assert loaded['expected_state']['prompt_sha256'] == row['expected_state']['prompt_sha256']
    with pytest.raises(FileExistsError):
        ordered_write(tmp_path/'inputs.json',row)


def test_canonical_state_hash_does_not_prove_prompt_order():
    row,_,tokenizer = prepared()
    reordered = json.loads(encoded(row))
    assert sha(encoded(reordered['observation'])) == row['expected_state']['observation_sha256']
    with pytest.raises(ValueError,match='physical_prompt_order_changed'):
        validate_state(reordered,tokenizer)


@pytest.mark.parametrize('field',['observation_sha256','schema_sha256','state_sha256','prompt_sha256','prompt_tokens'])
def test_existing_witness_must_match_every_hash_and_count(field):
    corpus,sources,review = fixture()
    tokenizer = TinyTokenizer()
    claim = {'id':1,'claim':'Claim'}
    expected = replay(claim,review,lambda *_:sources,corpus,tokenizer)
    expected['states'][1][field] = 123 if field == 'prompt_tokens' else '0'*64
    with pytest.raises(ValueError,match='original_replay_metadata_changed'):
        reconstruct(claim,review,expected,lambda *_:sources,corpus,tokenizer)


class FixtureProvider:
    def __init__(self,tokenizer,response):
        self.base = SimpleNamespace(model=instrument(SimpleNamespace()),tokenizer=tokenizer)
        self.response,self.calls = response,0
    def start_slot(self,path):
        path.mkdir()
    def generate(self,obs,schema,max_output_tokens,remaining_seconds):
        self.calls += 1
        assert remaining_seconds == 120 and max_output_tokens == 512
        assert {'read','rewrite','rerank'} <= set(obs['allowed_actions'])
        if isinstance(self.response,Exception):
            raise self.response
        return {'raw':json.dumps(self.response),'usage':{'input_tokens':self.tokens,'output_tokens':5},'diagnostics':{}}


@pytest.mark.parametrize('response',[
    {'action':'read','source_ids':['c4']},
    {'action':'rewrite','query':'Claim alternate'},
    {'action':'rerank'},
    {'action':'answer','documents':[{'source_id':'c5','label':'SUPPORTS','sentence_ids':['c5:0','c5:2']}]},
    {'action':'abstain','reason':'insufficient_evidence'},
    {'action':'answer','documents':[{'source_id':'c5','label':'SUPPORTS','sentence_ids':['c4:0']}]},
])
def test_one_call_no_repair_and_no_proposed_tool_execution(tmp_path,response):
    row,corpus,tokenizer = prepared()
    provider = FixtureProvider(tokenizer,response)
    provider.tokens = row['expected_state']['prompt_tokens']
    (tmp_path/'private-responses').mkdir()
    before = copy.deepcopy(row)
    result = call_once(provider,row,corpus,tmp_path,1,{'fixture':True})
    assert provider.calls == 1 and result['model_tool_executions'] == 0 and row == before
    if response['action'] in {'read','rewrite','rerank'}:
        assert result['terminal']['status'] == 'proposed_not_executed' and result['terminal']['prediction'] is None
    if response['action'] == 'answer' and response['documents'][0]['sentence_ids'][0] == 'c4:0':
        assert result['terminal']['status'] == 'invalid_no_repair'
    with pytest.raises(FileExistsError):
        call_once(provider,row,corpus,tmp_path,1,{'fixture':True})
    assert provider.calls == 1


def test_unknown_failure_retains_reservation_and_stops(tmp_path):
    row,corpus,tokenizer = prepared()
    provider = FixtureProvider(tokenizer,ModelResponseValidationError({}, {'output_usage_unknown':True}))
    (tmp_path/'private-responses').mkdir()
    with pytest.raises(ValueError,match='partial_call_stop'):
        call_once(provider,row,corpus,tmp_path,1,{'fixture':True})
    record = json.loads((tmp_path/'slot-01.json').read_bytes())
    assert not record['attempt']['usage_known'] and provider.calls == 1
    assert record['terminal']['prediction'] is None and (tmp_path/'slot-01-reserved.json').is_file()


def test_answer_uses_original_ordered_citations_and_scorer_export():
    row,corpus,_ = prepared()
    decision = {'action':'answer','documents':[{'source_id':'c5','label':'REFUTES','sentence_ids':['c5:2','c5:0']}]}
    result = finalise(json.dumps(decision),row,corpus)
    assert result['prediction'] == {'id':1,'evidence':{'6':{'label':'CONTRADICT','sentences':[2,0]}}}
    assert result['autonomous_success'] is False


@pytest.mark.parametrize('decision,error', [
    ({'action':'read','source_ids':['c5']}, 'read_loop'),
    ({'action':'rewrite','query':'Claim'}, 'rewrite_constraint_or_loop'),
    ({'action':'rerank'}, None),
])
def test_schema_valid_tool_proposal_is_not_automatically_controller_legal(decision,error):
    row,corpus,_ = prepared()
    result = finalise(json.dumps(decision),row,corpus)
    assert result['proposal_validation'] == 'schema_valid'
    assert result['controller_error'] == error and result['controller_legal'] == (error is None)
    assert result['model_tool_executions'] == 0


def test_release_rejects_draft_and_extra_calls():
    r = old_release() | {'purpose':'three_frozen_read_conditional_calls','output':str(OUTPUT),
        'policy':policy(SOURCE),'resource_cap':RESOURCE,'max_worker_seconds':480,'max_operator_seconds':870,
        'scoring_timeout_seconds':90,'conditional_inputs':str(OUTPUT.parent.parent/'posthoc/scifact-read-continuation-fixture/conditional-inputs.json'),
        'conditional_inputs_sha256':'c'*64,'preparation_compact_sha256':'d'*64,'private_membership_sha256':'e'*64}
    validate_release(r,SOURCE)
    draft = r | {'authorization':'DRAFT_NOT_AUTHORIZED'}
    validate_release(draft,SOURCE,draft=True)
    with pytest.raises(ValueError,match='exact_release_required'):
        validate_release(draft,SOURCE)
    r['policy']['max_generator_calls'] = 4
    with pytest.raises(ValueError):
        validate_release(r,SOURCE)


def scored_fixture(tmp_path,monkeypatch):
    from climate_rag.scifact_adapter_regression import ActiveAdapterProvider, active_state
    from climate_rag.scifact_grounding import GoldClaim, Rationale
    from climate_rag.scifact_semantic_contract import write_once
    from test_bounded_scifact_runtime import provider_fixture
    import run_scifact_read_continuation_operator as operator
    import score_scifact_read_continuation as scorer
    out = tmp_path/'runs/probe'
    directory = out/'inference'
    (directory/'private-responses').mkdir(parents=True)
    (out/'allocation').mkdir()
    prep = tmp_path/'posthoc/scifact-read-continuation-fixture'
    prep.mkdir(parents=True)
    action = {'action':'answer','documents':[{'source_id':'c5','label':'SUPPORTS','sentence_ids':['c5:0','c5:2']}]}
    old,calls,_ = provider_fixture(tmp_path,monkeypatch,[action,{'action':'read','source_ids':['c5']},{'invalid':True}])
    provider = ActiveAdapterProvider.__new__(ActiveAdapterProvider)
    provider.__dict__.update(old.__dict__)
    instrument(provider.base.model)
    corpus,sources,review = fixture()
    rows=[]
    for i in (1,2,3):
        claim={'id':i,'claim':'Claim'}
        expected=replay(claim,review,lambda *_:sources,corpus,provider.base.tokenizer)
        rows.append(reconstruct(claim,review,expected,lambda *_:sources,corpus,provider.base.tokenizer))
    ordered_write(prep/'conditional-inputs.json',{'rows':rows})
    write_once(prep/'private-membership.json',{'fit_overlap':[True,False,False]})
    r=old_release() | {'purpose':'three_frozen_read_conditional_calls','output':str(out),
        'policy':policy(SOURCE),'resource_cap':RESOURCE,'max_worker_seconds':480,'max_operator_seconds':870,
        'scoring_timeout_seconds':90,'conditional_inputs':str(prep/'conditional-inputs.json'),
        'conditional_inputs_sha256':sha((prep/'conditional-inputs.json').read_bytes()),'preparation_compact_sha256':'d'*64,
        'private_membership_sha256':sha((prep/'private-membership.json').read_bytes())}
    write_once(tmp_path/'release.json',r)
    digest=sha((tmp_path/'release.json').read_bytes())
    binding=r['policy'] | {'adapter_active':True}
    provider.bind(binding)
    identity={'release_sha256':digest,'provider_binding_sha256':sha(encoded(binding))}
    records=[call_once(provider,row,corpus,directory,i,identity) for i,row in enumerate(rows,1)]
    assert len(calls) == 3
    integrity={'active_state':active_state(provider.base.model),'tensor_count':144,'all_checkpoint_values_equal':True}
    write_once(directory/'adapter-integrity.json',integrity)
    run=identity | {'rows':records,'adapter_integrity_sha256':sha((directory/'adapter-integrity.json').read_bytes())}
    write_once(directory/'run.json',run)
    write_once(out/'allocation/worker-exit.json',{'child_reaped':True,'returncode':0,'parent_wait_interrupted':None,
        'release_sha256':digest,'run_sha256':sha((directory/'run.json').read_bytes())})
    scoring=tmp_path/'scoring'
    scoring.mkdir()
    gold=[GoldClaim(i,'Claim',{6:(Rationale('SUPPORT',(0,2)),)},()).gold_row() for i in (1,2,3)]
    payload={name:b'{}\n' for name in scorer.SCORING_NAMES-{'manifest.json'}}
    payload['gold.jsonl']=b'\n'.join(json.dumps(g).encode() for g in gold)
    manifest={'scoring_file_sha256':{name:sha(raw) for name,raw in payload.items()}}
    for name,raw in payload.items():
        (scoring/name).write_bytes(raw)
    write_once(scoring/'manifest.json',manifest)
    monkeypatch.setattr(scorer,'SCORING_MANIFEST',sha((scoring/'manifest.json').read_bytes()))
    monkeypatch.setattr(scorer,'OUTPUT',out)
    monkeypatch.setattr(operator,'OUTPUT',out)
    monkeypatch.setattr(operator,'ROOT',tmp_path)
    monkeypatch.setattr(scorer,'load_frozen',lambda *_:([],b'\n'.join(json.dumps({
        'doc_id':d.doc_id,'title':d.title,'abstract':list(d.sentences),'structured':False}).encode() for d in corpus.values())))
    return scorer,SimpleNamespace(release=tmp_path/'release.json',release_sha=digest,
        inference_dir=tmp_path/'unused',scoring_dir=scoring),out


def test_three_slot_physical_scoring_keeps_nonterminal_and_invalid_denominator(tmp_path,monkeypatch):
    scorer,args,out=scored_fixture(tmp_path,monkeypatch)
    report=scorer.score(args)
    assert report['costs']['model_calls'] == 3
    assert report['quality']['claim_count'] == 3 and report['quality']['metrics']['abstract_rationalized']['correct'] == 1
    assert report['terminal_statuses'] == {'terminal':1,'proposed_not_executed':1,'invalid_no_repair':1}
    assert report['fit_subgroups']['fit_overlap']['claim_count'] == 1
    assert report['fit_subgroups']['no_direct_fit_overlap']['claim_count'] == 2
    assert (out/'cost-before-quality.json').is_file() and not report['validation_gate']


def test_changed_prediction_rejected_after_cost_persistence(tmp_path,monkeypatch):
    scorer,args,out=scored_fixture(tmp_path,monkeypatch)
    directory=out/'inference'
    record=json.loads((directory/'slot-01.json').read_bytes())
    record['terminal']['prediction']['evidence']['6']['sentences']=[1]
    (directory/'slot-01.json').write_bytes(encoded(record))
    run=json.loads((directory/'run.json').read_bytes())
    run['rows'][0]=record
    (directory/'run.json').write_bytes(encoded(run))
    proof=json.loads((out/'allocation/worker-exit.json').read_bytes())
    proof['run_sha256']=sha((directory/'run.json').read_bytes())
    (out/'allocation/worker-exit.json').write_bytes(encoded(proof))
    with pytest.raises(ValueError,match='original_parse_render_prediction_changed'):
        scorer.score(args)
    assert (out/'cost-before-quality.json').is_file()


def test_partial_reservation_is_unknown_and_does_not_load_gold(tmp_path,monkeypatch):
    scorer,args,out=scored_fixture(tmp_path,monkeypatch)
    (out/'inference/slot-03.json').unlink()
    args.scoring_dir=tmp_path/'must-not-read-gold'
    result=scorer.score(args)
    assert not result['gold_loaded'] and result['costs']['planned_slots'] == 3
    assert result['costs']['model_calls'] is None and result['costs']['unresolved_reserved_slots'] == 1


def test_torn_record_keeps_unknown_cost_before_quality(tmp_path,monkeypatch):
    scorer,args,out=scored_fixture(tmp_path,monkeypatch)
    (out/'inference/slot-03.json').write_bytes(b'{"attempt":')
    args.scoring_dir=tmp_path/'must-not-read-gold'
    result=scorer.score(args)
    assert result['costs']['model_calls'] is None and result['costs']['model_calls_known_lower_bound'] == 2
    assert (out/'cost-before-quality.json').is_file() and not result['gold_loaded']


def test_unreadable_record_cannot_fail_again_before_cost_write(tmp_path,monkeypatch):
    scorer,args,out=scored_fixture(tmp_path,monkeypatch)
    original=Path.read_bytes
    blocked=out/'inference/slot-03.json'
    def read(path):
        if path == blocked:
            raise PermissionError('fixture')
        return original(path)
    monkeypatch.setattr(Path,'read_bytes',read)
    args.scoring_dir=tmp_path/'must-not-read-gold'
    result=scorer.score(args)
    assert result['costs']['physical_hash_errors'] == {'slot-03.json':'PermissionError'}
    assert result['costs']['model_calls'] is None and not result['gold_loaded']
    assert (out/'cost-before-quality.json').is_file()


def test_invalid_wire_cannot_smuggle_saved_prediction(tmp_path,monkeypatch):
    scorer,args,out=scored_fixture(tmp_path,monkeypatch)
    directory=out/'inference'
    record=json.loads((directory/'slot-03.json').read_bytes())
    record['terminal']['prediction']={'id':3,'evidence':{'6':{'label':'SUPPORT','sentences':[0,2]}}}
    (directory/'slot-03.json').write_bytes(encoded(record))
    run=json.loads((directory/'run.json').read_bytes())
    run['rows'][2]=record
    (directory/'run.json').write_bytes(encoded(run))
    proof=json.loads((out/'allocation/worker-exit.json').read_bytes())
    proof['run_sha256']=sha((directory/'run.json').read_bytes())
    (out/'allocation/worker-exit.json').write_bytes(encoded(proof))
    with pytest.raises(ValueError,match='failed_parse_status_or_prediction_changed'):
        scorer.score(args)


def test_old_json_metadata_integer_keys_compare_semantically():
    corpus,sources,review=fixture()
    from climate_rag.scifact_grounding import Abstract
    from climate_rag.scifact_terminal import source_from_abstract
    from climate_rag.scifact_semantic_policy import POLICIES
    corpus[10]=Abstract(10,'Ten',('Zero.','One.'),False)
    sources[0]=source_from_abstract(corpus[10])
    for p in POLICIES:
        review['current_opportunity'][p]['initial']['candidate_doc_ids'][0]=10
    tokenizer=TinyTokenizer()
    claim={'id':1,'claim':'Claim'}
    expected=replay(claim,review,lambda *_:sources,corpus,tokenizer)
    loaded=json.loads(encoded(expected))
    row=reconstruct(claim,review,loaded,lambda *_:sources,corpus,tokenizer)
    validate_state(row,tokenizer)
