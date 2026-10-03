"""Synthetic metadata/wire fixtures only; no official gold or model execution."""
from __future__ import annotations

import copy
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

import climate_rag.scifact_mixed_checkpoint as checkpoint
import climate_rag.scifact_mixed_launch as launch
from climate_rag.scifact_mixed_inputs import SCOPE, input_config_sha, seal
from climate_rag.scifact_mixed_training import CONFIG, VERSION, config_sha
from climate_rag.scifact_semantic_contract import checked, encoded, sha
import run_scifact_grounding_candidate as candidate
import run_scifact_grounding_tune_operator as tune


@pytest.fixture
def mixed(tmp_path, monkeypatch):
    root = tmp_path/'project'
    monkeypatch.setattr(checkpoint, 'ROOT', root)
    monkeypatch.setattr(launch, 'ROOT', root)
    train_git = 'c'*40
    adapter = root/'runs'/('scifact-mixed-training-'+train_git[:12])
    prepared = root/'posthoc'/('scifact-mixed-program-'+launch.PREPARATION_GIT[:12])
    release_path = root/'envs'/'training.json'
    def save(path, value):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes((json.dumps(value, ensure_ascii=False, indent=2)+'\n').encode())
        return sha(path.read_bytes())
    artifacts = {n:save(adapter/'final'/n, {'fixture':n})
        for n in ('README.md','adapter_config.json','adapter_model.safetensors')}
    ids = list(range(1000,1144))
    roster = seal({'claim_ids':ids, 'cohorts':{'old48':ids[:48], 'supp49':ids[48:97], 'NEI47':ids[97:]},
                   'components':{str(i):f'component-{i}' for i in ids}})
    roster_sha = save(prepared/'roster.json',roster)
    cpu = {'status':'complete','scope':SCOPE,'source_git':launch.PREPARATION_GIT,
        'source_bridge_validated':True,'input_config_sha256':input_config_sha(),
        'denominator':144,'rows':223,'planned_updates':36,
        'status_counts':{'ready':144,'failed':0,'unknown':0},'model_calls':0,'optimizer_steps':0,
        'files':{'prepared.json':'3'*64,'plan.json':'4'*64,'roster.json':roster_sha}}
    cpu_sha = save(prepared/'complete.json',cpu)
    monkeypatch.setattr(launch,'PREPARATION_SHA',cpu_sha)
    release = {'authorization':'coordinator_exact_hash_release','purpose':VERSION,
        'config_sha256':config_sha(),'output':adapter.as_posix(),'source_git':train_git,
        'source_archive_sha256':'5'*64,**launch.preparation_fields((prepared/'complete.json').read_bytes())}
    release_sha = save(release_path,release)
    training = {'version':VERSION,'status':'complete','actual_claims':144,'decision_records':223,
        'optimizer_steps':36,'automatic_retry':False,'artifacts':artifacts,
        'prepared_sha256':'1'*64,'plan_sha256':'2'*64}
    binding = {'kind':checkpoint.KIND,'directory':adapter.as_posix(),'training_source_git':train_git,
        'training_release_file':release_path.as_posix(),'training_release_sha256':release_sha,
        'training_source_archive_sha256':'5'*64,'training_complete_sha256':save(adapter/'complete.json',training),
        'adapter_files':artifacts,'prepared_envelope_sha256':'1'*64,'plan_envelope_sha256':'2'*64,
        'prepared_file_sha256':'3'*64,'plan_file_sha256':'4'*64,'roster_file_sha256':roster_sha,
        'started_sha256':save(adapter/'started.json',{'config_sha256':config_sha(),
            'prepared_sha256':'1'*64,'plan':{'sha256':'2'*64}}),
        'training_runtime_sha256':save(adapter/'training-runtime.json',{'config':CONFIG,'fresh_base_and_adapter':True})}
    execution = adapter.with_name(adapter.name+'-execution')
    binding['worker_exit_sha256'] = save(execution/'worker-exit.json',{
        'returncode':0,'child_started':True,'child_reaped':True,'interrupted':None})
    save(execution/'reserved.json',{'release_sha256':release_sha})
    return binding


def test_checkpoint_metadata_never_opens_training_payload(mixed, monkeypatch):
    seen = []
    original = Path.read_bytes
    def read(path):
        assert path.name not in {'prepared.json','plan.json','records.json'}
        seen.append(path.name)
        return original(path)
    monkeypatch.setattr(Path,'read_bytes',read)
    value = checkpoint.checkpoint_metadata(Path(mixed['directory']),mixed)
    assert value['training_exit_verified'] and not value['training_payload_opened']
    assert 'complete.json' in seen
    assert mixed['prepared_file_sha256'] != mixed['prepared_envelope_sha256']


@pytest.mark.parametrize('fault',['physical_as_logical','logical_as_physical','adapter','exit','source'])
def test_mixed_metadata_rejects_crossed_identity_or_failed_child(mixed,fault):
    binding = copy.deepcopy(mixed)
    if fault == 'physical_as_logical':
        binding['prepared_envelope_sha256'] = binding['prepared_file_sha256']
    elif fault == 'logical_as_physical':
        binding['prepared_file_sha256'] = binding['prepared_envelope_sha256']
    elif fault == 'adapter':
        binding['adapter_files']['adapter_model.safetensors'] = '0'*64
    elif fault == 'source':
        binding['training_source_archive_sha256'] = '0'*64
    else:
        path = Path(binding['directory']+'-execution')/'worker-exit.json'
        path.write_bytes(encoded({'returncode':2,'child_started':True,'child_reaped':True,'interrupted':None}))
        binding['worker_exit_sha256'] = sha(path.read_bytes())
    with pytest.raises(ValueError):
        checkpoint.checkpoint_metadata(Path(binding['directory']),binding)


def test_metadata_overlap_keeps_direct_and_component_distinct(mixed):
    result = checkpoint.fit_overlap(mixed,[{'id':1000,'component':'component-1000'},
        {'id':2000,'component':'component-1001'},{'id':3000,'component':'other'}])
    assert result['fit_claims'] == 144
    assert result['direct_claim_overlap'] == [1000]
    assert result['component_overlap'] == [1000,2000]
    assert result['no_overlap_is_not_independent_holdout']


def mixed_release(binding):
    from test_scifact_grounding_tune import release
    identity = checkpoint.binding_identity(binding)
    return release() | {'checkpoint_binding':binding,'next_validation_calls':0,
        'adapter_training_sha256':identity['adapter_training_sha256'],
        'adapter_model_sha256':identity['adapter_model_sha256'],
        'source_git':'a'*40,'source_archive_sha256':'b'*64}


def test_mixed_training_config_is_not_old_evaluation_config(mixed):
    release = mixed_release(mixed)
    assert config_sha() != release['config_sha256']
    tune.fixed_tune_release(release)
    for key,value in [('next_validation_calls',24),('partition','validation'),
        ('max_total_calls',25),('data_manifest_sha256','0'*64),('adapter_model_sha256',tune.ADAPTER_SHA)]:
        with pytest.raises(ValueError):
            tune.fixed_tune_release(release | {key:value})


def test_package_candidate_uses_only_accepted_redacted_receipt(mixed):
    from package_scifact_mixed_tune import release_candidate
    closeout = {'status':'training_accepted_evaluation_not_released','job_id':'31858295',
        'slurm_state':'COMPLETED','exit_code':'0:0','independent_coordinator_acceptance':True,
        'denominators':{'claims':144,'decision_records':223,'optimizer_updates':36},'checkpoint_binding':mixed}
    source = {'source_archive_sha256':'d'*64,'source_archive_bytes':123}
    release = release_candidate('a'*40,source,encoded(closeout),'e'*64)
    assert release['checkpoint_binding'] == mixed and release['max_total_calls'] == 24
    assert release['source_git'] != mixed['training_source_git']
    assert release['next_validation_calls'] == 0 and release['actual_PEFT_reload_required']
    assert release['requires_separate_coordinator_submission_authorization'] and not release['job_submitted']
    for key,value in [('exit_code','2:0'),('independent_coordinator_acceptance',False),
        ('denominators',{'claims':143,'decision_records':223,'optimizer_updates':36})]:
        with pytest.raises(ValueError,match='accepted_training_required'):
            release_candidate('a'*40,source,encoded(closeout | {key:value}),'e'*64)


def tune_run(tmp_path,monkeypatch,binding):
    from climate_rag.scifact_grounding import Abstract, GoldClaim, Rationale
    from climate_rag.scifact_grounding_eval import evaluate_arm
    from climate_rag.scifact_grounding_sft import context, token_record
    from test_scifact_grounding_candidate import Provider, Tokenizer
    corpus = {8:Abstract(8,'Fixture',('One source sentence.',),False)}
    ctx = context('Claim',[corpus[8]])
    rows = [{'claim_id':i,'context':ctx,'packing':token_record(Tokenizer(),ctx)} for i in range(12)]
    gold = [GoldClaim(i,'Claim',{8:(Rationale('SUPPORT',(0,)),)},()).gold_row() for i in range(12)]
    bundle,output = tmp_path/'bundle',tmp_path/'output'
    files = {}
    for name,value in [('inference/tune.json',rows),('scoring/tune.json',gold)]:
        path = bundle/name
        path.parent.mkdir(parents=True,exist_ok=True)
        path.write_bytes(encoded(value))
        files[name] = sha(path.read_bytes())
    monkeypatch.setattr(candidate,'load_bundle',lambda *args: ({'files':files},corpus))
    release = mixed_release(binding) | {'output':output.as_posix()}
    release_path = tmp_path/'evaluation.json'
    release_path.write_bytes(encoded(release))
    release_sha = sha(release_path.read_bytes())
    identity = {'checkpoint':checkpoint.binding_identity(binding),'release_sha256':release_sha,
        'source_git':release['source_git'],'source_archive_sha256':release['source_archive_sha256']}
    output.mkdir()
    (output/'adapter-integrity.json').write_bytes(encoded({'tensor_count':144,'all_checkpoint_values_equal':True,
        'evaluation_identity':identity}))
    for arm in ('base','adapted'):
        evaluate_arm(rows,corpus,Provider(),output/arm,arm)
    candidate.terminate_inference(output,'tune',tune.DATA_SHA,binding['training_complete_sha256'],identity)
    execution = output.with_name('output-execution')
    execution.mkdir()
    (execution/'worker-exit.json').write_bytes(encoded({'child_reaped':True,'returncode':0,
        'terminated_sha256':sha((output/'inference-terminated.json').read_bytes()),'release_sha256':release_sha}))
    return SimpleNamespace(bundle=bundle,output=output,data_sha=tune.DATA_SHA,partition='tune',
        release=release_path,release_sha=release_sha)


@pytest.mark.parametrize('fault',['none','missing_release','wrong_release','unknown_cost','nonzero_exit','physical_failure'])
def test_costs_and_provenance_precede_gold_and_never_open_validation(tmp_path,monkeypatch,mixed,fault):
    args = tune_run(tmp_path,monkeypatch,mixed)
    seen = []
    def check(path,digest):
        if path.parent.name == 'scoring':
            assert (args.output/'cost-before-quality.json').is_file()
            seen.append(path.name)
        return checked(path,digest)
    monkeypatch.setattr(candidate,'checked',check)
    if fault == 'missing_release':
        args.release = None
    elif fault == 'wrong_release':
        args.release_sha = '0'*64
    elif fault in {'unknown_cost','physical_failure'}:
        original = candidate.audit_arm
        def audit(*a):
            if fault == 'physical_failure':
                raise ValueError('synthetic_corrupt_physical_wire')
            result = original(*a)
            result['records'][0]['usage_known'] = False
            return result
        monkeypatch.setattr(candidate,'audit_arm',audit)
    elif fault == 'nonzero_exit':
        proof = args.output.with_name('output-execution')/'worker-exit.json'
        value = json.loads(proof.read_bytes())
        proof.write_bytes(encoded(value | {'returncode':2}))
    if fault in {'missing_release','wrong_release','physical_failure'}:
        with pytest.raises(ValueError):
            candidate.score(args)
    else:
        candidate.score(args)
    costs = json.loads((args.output/'cost-before-quality.json').read_bytes())
    assert not costs['gold_loaded']
    if fault == 'none':
        assert seen == ['tune.json']
        gate = json.loads((args.output/'gate.json').read_bytes())
        assert not gate['passed'] and gate['next_validation_calls'] == 0
    else:
        assert not seen and not (args.output/'score.json').exists()


def test_four_routes_bind_mixed_sha_and_empty_component_subgroup(tmp_path,monkeypatch,mixed):
    from test_scifact_adapter_regression import fixture_run
    import score_scifact_adapter_regression as scorer
    args,rows,_,identity = fixture_run(tmp_path,monkeypatch,mixed)
    assert identity['policy']['adapter_model_sha256'] == mixed['adapter_files']['adapter_model.safetensors']
    assert identity['policy']['max_generator_calls'] == 168 and identity['policy']['max_rerank_pairs'] == 720
    assert all(r['result']['generation_attempts'][0]['diagnostics']['adapter_sha256'] ==
        mixed['adapter_files']['adapter_model.safetensors'] for r in rows)
    report = scorer.score(args)
    assert report['fit_overlap_metadata']['direct_claim_overlap'] == []
    assert report['fit_overlap_metadata']['component_overlap'] == [0,1,2]
    group = report['read_opportunity_fit_subgroups']['read_opportunity_no_component_overlap']
    assert group['claims'] == 0 and group['quality'] is None and not group['available']
    assert not report['validation_gate']


@pytest.mark.parametrize('invalid',[False,True])
def test_provider_success_and_invalid_response_keep_new_adapter_identity(monkeypatch,mixed,invalid):
    import climate_rag.scifact_adapter_regression as regression
    from climate_rag.agent_protocol import ModelResponseValidationError
    provider = regression.ActiveAdapterProvider.__new__(regression.ActiveAdapterProvider)
    provider.base = SimpleNamespace(model=object())
    provider.gap,provider.name = False,'fixture'
    provider.render = lambda *a: 'fixture prompt'
    monkeypatch.setattr(regression,'active_state',lambda *_: {'enabled':True})
    policy = regression.policy_identity(Path(__file__).resolve().parents[1],mixed)
    provider.bind(policy)
    def generate(*a,**kw):
        if invalid:
            raise ModelResponseValidationError({'input_tokens':1,'output_tokens':1})
        return {'diagnostics':{}}
    monkeypatch.setattr(regression.LocalQwenBoundedSciFactProvider,'generate',generate)
    if invalid:
        with pytest.raises(ModelResponseValidationError) as failure:
            provider.generate({}, {}, 10,1.0)
        diagnostic = failure.value.diagnostics
    else:
        diagnostic = provider.generate({}, {},10,1.0)['diagnostics']
    assert mixed['adapter_files']['adapter_model.safetensors'] in provider.name
    assert diagnostic['adapter_sha256'] == mixed['adapter_files']['adapter_model.safetensors']
    assert diagnostic['regression_policy_sha256'] == sha(encoded(policy))
