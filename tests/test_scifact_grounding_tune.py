from __future__ import annotations

from contextlib import contextmanager
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

import run_scifact_grounding_candidate as entry
import run_scifact_grounding_tune_operator as operator
from climate_rag.scifact_grounding import Abstract
from climate_rag.scifact_grounding_sft import context, token_record
from test_scifact_grounding_candidate import Provider, Tokenizer


def release() -> dict[str, object]:
    return {'authorization': 'coordinator_exact_hash_release', 'purpose': 'evaluate_tune',
        'partition': 'tune', 'max_runtime_seconds': 1500, 'max_total_calls': 24,
        'training_authorized': False, 'validation_authorized': False,
        'adapter_training_sha256': operator.TRAINING_SHA, 'adapter_model_sha256': operator.ADAPTER_SHA,
        'data_manifest_sha256': operator.DATA_SHA, 'config_sha256': operator.CONFIG_SHA}


@pytest.mark.parametrize('key,value', [('authorization', 'DRAFT_NOT_AUTHORIZED'),
    ('purpose','train'), ('purpose','evaluate_validation'), ('partition','validation'),
    ('max_total_calls',25), ('training_authorized',True), ('validation_authorized',True),
    ('adapter_model_sha256','changed')])
def test_release_is_tune_only_and_checkpoint_fixed(key: str, value: object) -> None:
    operator.fixed_tune_release(release())
    with pytest.raises(ValueError):
        operator.fixed_tune_release(release() | {key: value})


@pytest.mark.parametrize('reaped,worker_code,parent_code,allowed', [
    (False,0,0,False),(True,0,0,True),(True,1,1,True),
    (True,-9,247,True),(True,1,0,False)])
@pytest.mark.parametrize('partition', ['tune', 'validation'])
def test_cpu_scorer_runs_only_after_matching_reaped_exit(tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
        reaped: bool, worker_code: int, parent_code: int, allowed: bool, partition: str) -> None:
    output = tmp_path/'run'
    execution = tmp_path/'run-execution'
    allocation = tmp_path/'run-allocation'
    for p in (output,execution,allocation):
        p.mkdir()
    termination = output/'inference-terminated.json'
    termination.write_text('{}')
    proof = {'child_reaped':reaped,'returncode':worker_code,'release_sha256':'release',
        'data_manifest_sha256':operator.DATA_SHA,'terminated_sha256':operator.sha(termination)}
    (execution/'worker-exit.json').write_text(json.dumps(proof))
    calls = []
    def run(command, **kwargs):
        assert (allocation/'before-scoring.json').is_file()
        assert command[2] == 'score' and command[3:5] == ['--partition',partition]
        calls.append(command)
        return SimpleNamespace(returncode=0)
    monkeypatch.setattr(operator.subprocess,'run',run)
    if allowed:
        assert operator.score_after_exit(tmp_path,output,tmp_path,'release',parent_code,allocation,partition=partition) == 0
        assert len(calls) == 1
    else:
        with pytest.raises(ValueError,match='paired_inference_exit_required'):
            operator.score_after_exit(tmp_path,output,tmp_path,'release',parent_code,allocation,partition=partition)
        assert calls == []


@pytest.mark.parametrize('fail', [False, True])
def test_existing_evaluator_uses_same_model_and_inputs_no_extra_calls(tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch, fail: bool) -> None:
    import climate_rag.local_bounded_scifact_provider as provider_module
    corpus = {8: Abstract(8,'Fixture',('One source sentence.',),False)}
    ctx = context('Claim',[corpus[8]])
    rows = [{'claim_id':i,'context':ctx,'packing':token_record(Tokenizer(),ctx)} for i in range(12)]
    adapter = tmp_path/'adapter'
    adapter.mkdir()
    (adapter/'complete.json').write_text(json.dumps({'data_manifest_sha256':'data',
        'config_sha256':entry.config_sha(),'adapter_files':{}}))
    (tmp_path/'inference').mkdir()
    (tmp_path/'inference/tune.json').write_text(json.dumps(rows))
    class Model:
        disabled = False
        @contextmanager
        def disable_adapter(self):
            self.disabled = True
            try:
                yield
            finally:
                self.disabled = False
    model = Model()
    seen = []
    class FixtureProvider(Provider):
        def generate(self,*args):
            seen.append((model.disabled,args[0]))
            return super().generate(*args)
    provider = FixtureProvider(fail=fail)
    provider.base.model = model
    def build(*args, **kwargs):
        assert kwargs['gap'] is False
        return provider
    monkeypatch.setattr(provider_module,'LocalQwenBoundedSciFactProvider',build)
    monkeypatch.setattr(entry,'release_check',lambda *args: {'adapter_training_sha256':'training'})
    monkeypatch.setattr(entry,'load_bundle',lambda *args: ({'files':{'inference/tune.json':'rows'}},corpus))
    monkeypatch.setattr(entry,'checked',lambda path,digest: path.read_bytes())
    monkeypatch.setattr(entry,'model_manifest',lambda *args: {})
    restored = []
    def restore(base,path):
        assert base is model and path == adapter/'final'
        restored.append(path)
        return model, {'all_checkpoint_values_equal':True}
    monkeypatch.setattr(entry,'restore_causal_adapter',restore)
    output = tmp_path/'output'
    entry.evaluate(SimpleNamespace(partition='tune',bundle=tmp_path,data_sha='data',
        adapter=adapter,output=output,model=tmp_path,model_manifest=tmp_path))
    assert len(restored) == 1 and model.disabled is False
    assert provider.calls == (1 if fail else 24)
    assert all(disabled for disabled,_ in seen[:12])
    if fail:
        assert not (output/'adapted').exists()
    else:
        assert all(not disabled for disabled,_ in seen[12:])
        assert [obs for _,obs in seen[:12]] == [obs for _,obs in seen[12:]]
    assert json.loads((output/'inference-terminated.json').read_bytes())['attempted_calls'] == provider.calls


@pytest.mark.parametrize('termination_exists', [False, True])
@pytest.mark.parametrize('partition', ['tune', 'validation'])
def test_interrupted_summary_is_cost_pending_not_zero_or_scored(tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch, termination_exists: bool, partition: str) -> None:
    output = tmp_path/'run'
    execution = tmp_path/'run-execution'
    allocation = tmp_path/'run-allocation'
    slot = output/'base/slot-00'
    slot.mkdir(parents=True)
    execution.mkdir()
    allocation.mkdir()
    (slot/'started.json').write_text('{}')
    (slot/'response.json').write_bytes(b'{"raw":')
    termination = output/'inference-terminated.json'
    if termination_exists:
        termination.write_text('{}')
    (execution/'worker-exit.json').write_text(json.dumps({
        'child_reaped': True, 'returncode': 1, 'release_sha256': 'release',
        'data_manifest_sha256': operator.DATA_SHA,
        'terminated_sha256': operator.sha(termination) if termination_exists else None}))
    def forbidden(*args, **kwargs):
        pytest.fail('must not launch quality scorer for unreconciled raw costs')
    monkeypatch.setattr(operator.subprocess, 'run', forbidden)
    assert operator.score_after_exit(tmp_path,output,tmp_path,'release',1,allocation,partition=partition) == 2
    pending = json.loads((allocation/'cost-audit-pending.json').read_bytes())
    assert pending['planned_slots'] == 24 and pending['durable_reservations'] == 1
    assert pending['partition'] == partition
    assert pending['unreserved_slots'] == 23 and pending['known_cost_totals'] is None
    assert pending['cost_audit_pending'] and pending['unknown_cost']
    assert not pending['quality_scored'] and not pending['validation_gate_open']
    assert not pending['retry_authorized']
    assert pending['physical_files_sha256']['base/slot-00/response.json']


@pytest.mark.parametrize('last_failure', ['parse', 'incomplete'])
@pytest.mark.parametrize('partition', ['tune', 'validation'])
def test_last_slot_stop_blocks_gate_despite_semantic_gain(tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch, last_failure: str, partition: str) -> None:
    from climate_rag.scifact_grounding import GoldClaim, Rationale
    from climate_rag.scifact_grounding_eval import evaluate_arm
    from climate_rag.scifact_semantic_contract import encoded, sha, write_once
    corpus = {8: Abstract(8,'Fixture',('One source sentence.',),False)}
    ctx = context('Claim',[corpus[8]])
    rows = [{'claim_id':i,'context':ctx,'packing':token_record(Tokenizer(),ctx)} for i in range(12)]
    gold = [GoldClaim(i,'Claim',{8:(Rationale('SUPPORT',(0,)),)},()).gold_row() for i in range(12)]
    bundle, output = tmp_path/'bundle', tmp_path/'output'
    (bundle/'inference').mkdir(parents=True)
    (bundle/'scoring').mkdir()
    output.mkdir()
    write_once(bundle/f'inference/{partition}.json',rows)
    write_once(bundle/f'scoring/{partition}.json',gold)
    manifest = {'files': {f'inference/{partition}.json':sha(encoded(rows)),
                          f'scoring/{partition}.json':sha(encoded(gold))}}
    monkeypatch.setattr(entry,'load_bundle',lambda *args: (manifest,corpus))
    class Base(Provider):
        def generate(self,*args):
            response = super().generate(*args)
            if self.calls == 1:
                response['raw'] = '{'
            return response
    class Adapted(Provider):
        def generate(self,*args):
            response = super().generate(*args)
            response['raw'] = '{"action":"answer","documents":[{"source_id":"c1","label":"SUPPORTS","sentence_ids":["c1:0"]}]}'
            if self.calls == 12:
                if last_failure == 'parse':
                    response['raw'] = '{'
                else:
                    response['diagnostics']['eos_observed'] = False
            return response
    for arm, provider in [('base',Base()),('adapted',Adapted())]:
        result = evaluate_arm(rows,corpus,provider,output/arm,arm)
        assert result['attempts'] == 12
    entry.terminate_inference(output,partition,'data','adapter')
    execution = output.with_name('output-execution')
    execution.mkdir()
    write_once(execution/'worker-exit.json',{'child_reaped':True,'returncode':0,
        'terminated_sha256':sha((output/'inference-terminated.json').read_bytes())})
    entry.score(SimpleNamespace(bundle=bundle,output=output,data_sha='data',partition=partition))
    scored = json.loads((output/'score.json').read_bytes())
    assert scored['base']['correctly_rationalized_documents'] == 0
    assert scored['adapted']['correctly_rationalized_documents'] == 11
    assert scored['base']['planned_unsuccessful'] == scored['adapted']['planned_unsuccessful'] == 1
    assert scored['base']['stop_required'] == 0
    assert scored['adapted']['stop_required'] == (last_failure == 'incomplete')
    if partition == 'tune':
        assert json.loads((output/'gate.json').read_bytes())['passed'] is (last_failure == 'parse')
    else:
        assert not (output/'gate.json').exists()
