from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

import run_scifact_grounding_candidate as entry
import run_scifact_grounding_tune_operator as tune
import run_scifact_grounding_validation_operator as validation


@pytest.fixture
def frozen_gate(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    base = {'input_identity':'same','claims':12,'attempts':12,'correctly_rationalized_documents':1,
            'nei_false_evidence':4,'planned_unsuccessful':0,'unknown_usage':0,'stop_required':0}
    score = {'base':base,'adapted':base | {'correctly_rationalized_documents':4}}
    score_path = tmp_path/'score.json'
    score_path.write_text(json.dumps(score))
    score_sha = tune.sha(score_path)
    gate = {'passed':True,'score_sha256':score_sha,'next_validation_calls':24,
        'data_manifest_sha256':tune.DATA_SHA,'adapter_training_sha256':tune.TRAINING_SHA}
    gate_path = tmp_path/'gate.json'
    gate_path.write_text(json.dumps(gate))
    monkeypatch.setattr(validation,'TUNE',tmp_path)
    monkeypatch.setattr(validation,'GATE_SHA',tune.sha(gate_path))
    monkeypatch.setattr(validation,'SCORE_SHA',score_sha)
    return gate,score,gate_path,score_path


def release():
    return {'authorization':'coordinator_exact_hash_release','purpose':'evaluate_validation',
        'partition':'validation','max_runtime_seconds':1500,'max_total_calls':24,
        'warmup_calls':0,'calls_per_arm':12,'training_authorized':False,'validation_authorized':True,
        'adapter_training_sha256':tune.TRAINING_SHA,'adapter_model_sha256':tune.ADAPTER_SHA,
        'data_manifest_sha256':tune.DATA_SHA,'config_sha256':tune.CONFIG_SHA,
        'output':str(tune.ROOT/'runs/scifact-grounding-validation-20261001-v1'),
        'tune_gate_sha256':validation.GATE_SHA,'tune_score_sha256':validation.SCORE_SHA}


@pytest.mark.parametrize('key,value',[('authorization','DRAFT_NOT_AUTHORIZED'),
    ('purpose','evaluate_tune'),('partition','tune'),('max_total_calls',25),('warmup_calls',1),
    ('training_authorized',True),('validation_authorized',False),('adapter_model_sha256','changed')])
def test_validation_release_rejects_cross_partition_or_changed_candidate(frozen_gate,key,value):
    validation.fixed_validation_release(release())
    with pytest.raises(ValueError):
        validation.fixed_validation_release(release() | {key:value})
    with pytest.raises(ValueError):
        tune.fixed_tune_release(release())


@pytest.mark.parametrize('changed',['gate_bytes','score_bytes','score_link','next_calls','passed','stop_required'])
def test_physical_and_semantic_gate_bindings_fail_closed(frozen_gate,monkeypatch,changed):
    gate,score,gate_path,score_path = frozen_gate
    validation.verify_frozen_gate(gate_path,release())
    if changed in ('gate_bytes','score_bytes'):
        path = gate_path if changed == 'gate_bytes' else score_path
        path.write_bytes(path.read_bytes()+b' ')
    else:
        if changed == 'score_link':
            gate['score_sha256'] = 'changed'
        elif changed == 'next_calls':
            gate['next_validation_calls'] = 25
        elif changed == 'passed':
            gate['passed'] = False
        else:
            score['adapted']['stop_required'] = 1
            score_path.write_text(json.dumps(score))
            monkeypatch.setattr(validation,'SCORE_SHA',tune.sha(score_path))
            gate['score_sha256'] = validation.SCORE_SHA
        gate_path.write_text(json.dumps(gate))
        monkeypatch.setattr(validation,'GATE_SHA',tune.sha(gate_path))
    with pytest.raises(ValueError):
        validation.verify_frozen_gate(gate_path,release())


def test_direct_evaluator_rejects_release_partition_mismatch(tmp_path,monkeypatch):
    payload = release() | {'output':str(tmp_path.resolve())}
    monkeypatch.setattr(entry,'checked',lambda *args:json.dumps(payload).encode())
    args=SimpleNamespace(release=tmp_path/'release.json',release_sha='x',data_sha=tune.DATA_SHA,
                         output=tmp_path,partition='tune')
    with pytest.raises(ValueError,match='release_partition_mismatch'):
        entry.release_check(args,'evaluate_validation')


def test_entrypoint_partition_is_not_chosen_from_release(monkeypatch):
    calls=[]
    monkeypatch.setattr(tune,'run_pair',lambda part,check:calls.append((part,check)))
    monkeypatch.setattr(validation,'run_pair',lambda part,check:calls.append((part,check)))
    tune.main()
    validation.main()
    assert calls == [('tune',tune.fixed_tune_release),('validation',validation.fixed_validation_release)]
