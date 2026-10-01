"""CPU fixtures only; no downloaded weights, official claims, GPU or new inference."""
from __future__ import annotations

import copy
import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
import time
from types import SimpleNamespace

import pytest

from climate_rag.scifact_adapter_regression import (
    ADAPTER_SHA, ActiveAdapterProvider, INFERENCE_SHA, SCORING_SHA, TRAINING_SHA,
    active_state, policy_identity,
)
from climate_rag.scifact_grounding import Abstract, GoldClaim, Rationale
from climate_rag.scifact_semantic_contract import MODEL_SHA, ROUTES, encoded, sha, write_once
from climate_rag.scifact_terminal import source_from_abstract
import run_scifact_adapter_regression as runner
import run_scifact_adapter_regression_operator as operator
import score_scifact_adapter_regression as scorer
from test_bounded_scifact_runtime import ABSTAIN, provider_fixture

SOURCE = Path(__file__).resolve().parents[1]


def release():
    return {'authorization': 'coordinator_exact_hash_release', 'purpose': 'old12_four_route_regression',
        'output': str(operator.OUTPUT), 'policy': policy_identity(SOURCE),
        'adapter_training_sha256': TRAINING_SHA, 'adapter_model_sha256': ADAPTER_SHA,
        'inference_archive_sha256': INFERENCE_SHA, 'scoring_archive_sha256': SCORING_SHA,
        'model_archive_sha256': operator.ARCHIVES['input'][1],
        'max_worker_seconds': 6300, 'max_operator_seconds': 6900, 'scoring_timeout_seconds': 180,
        'resource_cap': operator.RESOURCE, 'training_authorized': False, 'validation_authorized': False,
        'official_dev_test_read': False, 'no_automatic_retry': True,
        'source_git': 'a'*40, 'source_archive_sha256': 'b'*64}


def instrument(model, layers=72):
    model.active_adapters, model.training = ['default'], False
    model.layers = [SimpleNamespace(lora_A={'default': object()}, lora_B={'default': object()},
        active_adapters=['default'], disable_adapters=False, merged=False) for _ in range(layers)]
    model.named_modules = lambda: [(str(i), m) for i, m in enumerate(model.layers)]
    model.parameters = lambda: [SimpleNamespace(requires_grad=False)]
    return model


@pytest.mark.parametrize('fault', ['disabled','merged','wrong_active','wrong_count','trainable','training'])
def test_active_adapter_is_not_inferred_from_checkpoint_hash(fault):
    model = instrument(SimpleNamespace())
    assert active_state(model)['lora_layers'] == 72
    if fault == 'disabled':
        model.layers[0].disable_adapters = True
    elif fault == 'merged':
        model.layers[0].merged = True
    elif fault == 'wrong_active':
        model.active_adapters = ['other']
    elif fault == 'wrong_count':
        model.layers.pop()
    elif fault == 'trainable':
        model.parameters = lambda: [SimpleNamespace(requires_grad=True)]
    else:
        model.training = True
    with pytest.raises(ValueError):
        active_state(model)


@pytest.mark.parametrize('field,value', [('authorization','DRAFT_NOT_AUTHORIZED'), ('purpose','evaluate_validation'),
    ('adapter_model_sha256','0'*64), ('max_worker_seconds',7000), ('training_authorized',True)])
def test_draft_or_wrong_release_cannot_execute(field,value):
    r = release()
    operator.validate_release(r,SOURCE)
    r[field] = value
    with pytest.raises(ValueError):
        operator.validate_release(r,SOURCE)
    draft = release() | {'authorization':'DRAFT_NOT_AUTHORIZED'}
    operator.validate_release(draft,SOURCE,draft=True)


def test_policy_has_unchanged_budgets_and_zero_extra_smoke():
    p = policy_identity(SOURCE)
    assert p['planned_slots'] == 48 and p['max_generator_calls'] == 36*3+12*5 == 168
    assert p['max_rerank_calls'] == 36 and p['max_rerank_pairs'] == 36*20
    assert p['warmup_or_synthetic_preflight_calls'] == 0 and p['gap'] is False
    assert p['budget']['timeout_seconds'] == 120
    assert 'existing_CommonPacking' in p['packing'] and 'c0' in p['alias_policy']


@pytest.mark.parametrize('failure', ['timeout','interrupt','wait_error'])
def test_child_group_is_killed_and_reaped_before_any_scoring(tmp_path,monkeypatch,failure):
    order = []
    class Child:
        pid = 12345
        def wait(self,timeout=None):
            order.append(('wait',timeout))
            if timeout is not None:
                if failure == 'timeout':
                    raise subprocess.TimeoutExpired('fixture',timeout)
                if failure == 'interrupt':
                    raise KeyboardInterrupt()
                raise OSError('fixture')
            return -9
        def poll(self):
            return None
    def popen(*args,**kwargs):
        assert kwargs['start_new_session'] is True
        return Child()
    monkeypatch.setattr(operator.subprocess,'Popen',popen)
    monkeypatch.setattr(operator.os,'killpg',lambda pid,sig: order.append(('kill',pid)),raising=False)
    monkeypatch.setattr(signal,'SIGKILL',9,raising=False)
    proof = operator.bounded_child(['fixture'],tmp_path/'log',7)
    assert order == [('wait',7),('kill',12345),('wait',None)]
    assert proof['child_reaped'] and proof['returncode'] == -9
    assert proof['timed_out'] is (failure == 'timeout')
    assert bool(proof['parent_wait_interrupted']) is (failure != 'timeout')


def fixture_run(tmp_path,monkeypatch,checkpoint_binding=None):
    out = tmp_path/'output'
    directory = out/'inference'
    (directory/'private-responses').mkdir(parents=True)
    (out/'allocation').mkdir()
    monkeypatch.setattr(operator,'OUTPUT',out)
    r = release()
    if checkpoint_binding is not None:
        from climate_rag.scifact_mixed_checkpoint import binding_identity
        from climate_rag.scifact_mixed_inputs import seal
        r.update(checkpoint_binding=checkpoint_binding, **{k:v for k,v in binding_identity(checkpoint_binding).items()
            if k in {'adapter_training_sha256','adapter_model_sha256'}})
        r['policy'] = policy_identity(SOURCE,checkpoint_binding)
        monkeypatch.setattr(operator,'release_paths',lambda _: (out,Path(checkpoint_binding['directory'])))
        monkeypatch.setattr(scorer,'release_paths',operator.release_paths)
        # Metadata fixture only, deliberately direct != component overlap.
        roster = json.loads(Path(checkpoint_binding['training_release_file']).read_bytes())
        roster_path = Path(roster['prepared_directory'])/'roster.json'
        value = json.loads(roster_path.read_bytes())['payload']
        value['components'][str(value['claim_ids'][0])] = 'fixture-0'
        value['components'][str(value['claim_ids'][1])] = 'fixture-1'
        value['components'][str(value['claim_ids'][2])] = 'fixture-2'
        roster_path.write_text(json.dumps(seal(value),ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
        checkpoint_binding['roster_file_sha256'] = sha(roster_path.read_bytes())
        r['policy'] = policy_identity(SOURCE,checkpoint_binding)
    release_path = tmp_path/'release.json'
    write_once(release_path,r)
    digest = sha(release_path.read_bytes())
    identity = {'source_git':r['source_git'],'source_archive_sha256':r['source_archive_sha256'],
        'release_sha256':digest,'policy':r['policy'],'policy_sha256':sha(encoded(r['policy'])),
        'inference_archive_sha256':INFERENCE_SHA,'gold_loaded':False}
    old, calls, _ = provider_fixture(tmp_path,monkeypatch,[ABSTAIN]*48)
    provider = ActiveAdapterProvider.__new__(ActiveAdapterProvider)
    provider.__dict__.update(old.__dict__)
    instrument(provider.base.model)
    provider.bind(r['policy'])
    corpus = {i:Abstract(i,'Fixture',('Zero.','One.','Two.'),False) for i in range(100,107)}
    sources = [source_from_abstract(d) for d in corpus.values()]
    claims = [{'id':i,'claim':f'Fixture claim {i}'} for i in range(12)]
    rows = runner.execute_slots(claims,provider,lambda q,k:sources,lambda q,c:c,corpus,directory,identity)
    assert len(calls) == len(rows) == 48
    assert [row['route'] for row in rows] == list(ROUTES)*12
    assert all(row['result']['generation_attempts'][0]['diagnostics']['adapter_sha256'] == r['adapter_model_sha256'] for row in rows)
    assert all(row['result']['generation_attempts'][0]['diagnostics']['base_model_sha256'] == MODEL_SHA for row in rows)
    integrity = identity | {'active_state':active_state(provider.base.model), 'tensor_count':144,'all_checkpoint_values_equal':True}
    write_once(directory/'adapter-integrity.json',integrity)
    run = identity | {'runs':rows,'adapter_integrity_sha256':sha(encoded(integrity))}
    write_once(directory/'run.json',run)
    write_once(out/'allocation/worker-exit.json',{'child_reaped':True,'returncode':0,
        'parent_wait_interrupted':None,'release_sha256':digest,'run_sha256':sha(encoded(run))})
    monkeypatch.setattr(scorer,'load_frozen',lambda *_:(claims,b'\n'.join(json.dumps({
        'doc_id':d.doc_id,'title':d.title,'abstract':list(d.sentences),'structured':False}).encode() for d in corpus.values())))
    scoring = tmp_path/'scoring'
    scoring.mkdir()
    strata = ['top20_doc_replenishable']*3+['initial_doc_opportunity']*3+['gold_absent_top20']*3+['nei']*3
    selection = [{'id':i,'component':f'fixture-{i}','legacy_stratum':strata[i]} for i in range(12)]
    gold = [GoldClaim(i,claims[i]['claim'],{105:(Rationale('SUPPORT',(1,2)),)} if i<9 else {},()).gold_row() for i in range(12)]
    payload = {name:b'{}\n' for name in scorer.SCORING_NAMES-{'manifest.json'}}
    payload['gold.jsonl'] = b'\n'.join(json.dumps(g).encode() for g in gold)
    payload['selected-strata.json'] = encoded(selection)
    manifest = {'execution_source_git':scorer.PREVIOUS_SOURCE,'scoring_file_sha256':{n:sha(b) for n,b in payload.items()}}
    for name,raw in payload.items():
        (scoring/name).write_bytes(raw)
    write_once(scoring/'manifest.json',manifest)
    monkeypatch.setattr(scorer,'SCORING_MANIFEST',sha(encoded(manifest)))
    monkeypatch.setattr(scorer,'validate_prepared_scoring',lambda *_:None)  # exact real bundle tested separately
    fit = tmp_path/'posthoc/scifact-grounding-candidate-40d84a377bd1/fit'
    fit.mkdir(parents=True)
    write_once(fit/'records.json',[{'claim_id':0,'component':'fixture-0'}])
    monkeypatch.setattr(scorer,'ROOT',tmp_path)
    monkeypatch.setattr(scorer,'FIT_SHA',sha((fit/'records.json').read_bytes()))
    args = SimpleNamespace(output=out,inference_dir=tmp_path/'unused',scoring_dir=scoring,
        release=release_path,release_sha=digest)
    return args, rows, directory, identity


def test_full_four_route_fixture_uses_physical_event_and_prediction_scorers(tmp_path,monkeypatch):
    args,_,_,_ = fixture_run(tmp_path,monkeypatch)
    report = scorer.score(args)
    assert report['planned_slots'] == report['costs']['actual_model_calls'] == 48
    assert report['validation_gate'] is False and report['independent_test'] is False
    assert report['read_opportunity_fit_subgroups']['read_opportunity_component_overlap']['claims'] == 1
    assert report['read_opportunity_fit_subgroups']['read_opportunity_no_component_overlap']['claims'] == 2
    assert report['costs']['reranker_tokens'] is None


def test_row_prediction_cannot_differ_from_rendered_answer(tmp_path,monkeypatch):
    args, rows, directory,_ = fixture_run(tmp_path,monkeypatch)
    row = copy.deepcopy(rows[0])
    row['prediction']['evidence'] = {'105':{'label':'SUPPORT','sentences':[1,2]}}
    (directory/'slot-01.json').write_bytes(encoded(row))
    run = json.loads((directory/'run.json').read_bytes())
    run['runs'][0] = row
    (directory/'run.json').write_bytes(encoded(run))
    proof = json.loads((args.output/'allocation/worker-exit.json').read_bytes())
    proof['run_sha256'] = sha(encoded(run))
    (args.output/'allocation/worker-exit.json').write_bytes(encoded(proof))
    with pytest.raises(ValueError):
        scorer.score(args)


def test_missing_durable_slot_cost_is_unknown_not_dropped(tmp_path):
    directory = tmp_path/'inference'
    directory.mkdir()
    identity = {'fixture':True}
    write_once(directory/'slot-01-reserved.json',{'identity':identity})
    costs, rows = scorer.collect_costs(tmp_path,identity)
    assert costs['planned_slots'] == 48 and costs['not_reserved_slots'] == 47
    assert costs['unresolved_reserved_slots'] == 1 and costs['actual_model_calls'] is None
    assert costs['token_totals_are_lower_bounds'] and not rows


def test_rerank_failures_keep_requested_pairs_and_unknown_elapsed(tmp_path):
    directory = tmp_path/'inference'
    directory.mkdir()
    identity = {'fixture':True}
    for i,event in enumerate([{'tool':'rerank','status':'completed','elapsed_ms':12.5},
                              {'tool':'rerank','status':'running'}],1):
        write_once(directory/f'slot-{i:02d}-reserved.json',{'identity':identity})
        write_once(directory/f'slot-{i:02d}-raw.json',{'regression_identity':identity,
            'route':'fixed_rerank','generation_attempts':[],'events':[event],'rerank_pairs':20})
    costs,_ = scorer.collect_costs(tmp_path,identity)
    route = costs['rerank_by_route']['fixed_rerank']
    assert route['status_counts'] == {'completed':1,'running':1}
    assert route['completed_operations'] == route['failed_or_uncompleted_operations'] == 1
    assert route['elapsed_ms_known_sum'] == 12.5 and route['elapsed_ms_missing'] == 1
    assert costs['rerank_requested_pairs_known_lower_bound'] == 40
    assert costs['reranker_tokens'] is None


def test_actual_tiny_causal_adapter_active_vs_disabled_without_download(tmp_path):
    from peft import LoraConfig, get_peft_model
    from transformers.models.qwen3.configuration_qwen3 import Qwen3Config
    from transformers.models.qwen3.modeling_qwen3 import Qwen3ForCausalLM
    from run_scifact_grounding_candidate import restore_causal_adapter
    config = Qwen3Config(vocab_size=32, hidden_size=16, intermediate_size=24,
        num_hidden_layers=1,num_attention_heads=2,num_key_value_heads=2,head_dim=8)
    model = get_peft_model(Qwen3ForCausalLM(config),LoraConfig(r=8,lora_alpha=16,
        lora_dropout=0.0,target_modules=['q_proj','v_proj'],task_type='CAUSAL_LM'))
    model.save_pretrained(tmp_path/'adapter',safe_serialization=True)
    restored,_ = restore_causal_adapter(Qwen3ForCausalLM(config),tmp_path/'adapter')
    assert active_state(restored,2)['enabled'] is True
    with restored.disable_adapter():
        with pytest.raises(ValueError,match='disabled_or_merged'):
            active_state(restored,2)


def test_real_shell_outer_timer_covers_precheck_body(tmp_path):
    bash = shutil.which('bash') if os.name == 'posix' else 'E:/SoftWare/Git/bin/bash.exe'
    if not bash or not Path(bash).is_file():
        pytest.skip('real bash unavailable; required exact-export Linux check remains')
    wrapper = (SOURCE/'hpc/scifact_adapter_regression.sbatch').read_text()
    # Same executable prologue; shorten only time bounds for this CPU fixture.
    prefix = wrapper.split(': "${SLURM_JOB_ID')[0]
    assert 'exec timeout --signal=TERM --kill-after=30s 6900s' in prefix
    script = tmp_path/'outer-timer.sh'
    script.write_text(prefix.replace('--kill-after=30s 6900s','--kill-after=1s 1s')+'sleep 20\n',newline='\n')
    start = time.monotonic()
    result = subprocess.run([bash,script.as_posix()],capture_output=True,timeout=5)
    assert result.returncode == 124 and 0.7 <= time.monotonic()-start < 5
