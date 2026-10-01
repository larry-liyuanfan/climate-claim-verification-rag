"""CPU startup integration through the real runtime consumer, not just schema."""
from __future__ import annotations

import copy
import hashlib
import json

import pytest

import package_scifact_document_verifier as packager
import run_scifact_document_verifier_operator as operator
import run_scifact_utility8_operator as runtime


@pytest.fixture
def startup(tmp_path, monkeypatch):
    deps = tmp_path / "deps"
    deps.mkdir()
    inventory = deps / "runtime-files.json"
    inventory.write_text('{}')
    observed = dict(python='3.10.4', os_name='posix', torch='2.1.2',
        python_executable=operator.FROZEN_FIELDS['python_executable'],
        versions={'torch': '2.1.2'}, module_files={'torch': '/synthetic/torch.py'},
        model_loaded=False, generation_calls=0)
    receipt = {k: v for k, v in observed.items() if k != 'python_executable'}
    receipt.update(status='imports_verified', dependency_errors=[], stderr_empty=True,
                   runtime_files_sha256=runtime.sha(inventory))
    path = deps / 'final-receipt.json'
    path.write_text(json.dumps(receipt))
    observation = tmp_path / 'accepted-runtime.json'
    observation.write_text(json.dumps(observed))
    output = tmp_path / 'run-r2'
    frozen = dict(copy.deepcopy(operator.FROZEN_FIELDS), output=str(output),
        runtime_receipt_sha256=runtime.sha(path), runtime_observation_sha256=runtime.sha(observation),
        runtime_files_sha256=runtime.sha(inventory))
    monkeypatch.setattr(operator, 'FROZEN_FIELDS', frozen)
    monkeypatch.setattr(packager, 'FROZEN_FIELDS', frozen)
    monkeypatch.setattr(operator, 'OUTPUT', output)
    monkeypatch.setattr(runtime, 'DEPS', deps)
    calls = []
    def observe(names):
        calls.append(names)
        return copy.deepcopy(observed)
    # Only imports are stubbed: release producer, validator, disk receipts/hash
    # checks, runtime consumer and startup/reservation execute for real.
    monkeypatch.setattr(runtime, 'import_observation', observe)
    release = packager.build_release('a'*40, 'b'*64, 'c'*64, path, observation)
    release['authorization'] = 'coordinator_exact_hash_release'
    return release, output, calls, path, observation, observed


def test_producer_through_actual_startup_and_runtime_consumer(startup):
    release, output, calls, _, _, _ = startup
    actual = operator.start_attempt(release, 'd'*64, 'synthetic-cpu')
    assert calls == [['torch']]
    assert actual['python_executable'] == release['python_executable']
    assert actual['model_loaded'] is False and actual['generation_calls'] == 0
    reservation = json.loads((output / 'reserved.json').read_bytes())
    assert reservation['attempt_id'] == 'r2' and reservation['infrastructure_retry'] == 1
    assert reservation['previous_attempt_job_id'] == '31914601'
    assert reservation['automatic_retry'] is False
    assert {p.name for p in output.iterdir()} == {'reserved.json', 'runtime.json'}
    with pytest.raises(FileExistsError):
        operator.start_attempt(release, 'd'*64, 'synthetic-cpu')


@pytest.mark.parametrize('key', sorted(operator.REQUIRED_RELEASE_KEYS))
def test_every_missing_consumer_key_fails_before_import_or_reservation(startup, key):
    release, output, calls, *_ = startup
    del release[key]
    with pytest.raises(ValueError, match='release_missing_fields:' + key):
        operator.start_attempt(release, 'd'*64, 'synthetic-cpu')
    assert not output.exists() and not calls


@pytest.mark.parametrize('key,value', [
    ('python_executable', '/usr/bin/python'), ('python_executable', ''),
    ('python_executable', None), ('python_executable', 'python'), ('attempt_id', 'r1'),
    ('infrastructure_retry', 2), ('automatic_retry', True),
    ('output', str(operator.ROOT / 'runs' / operator.PROTOCOL)),
])
def test_contract_drift_or_old_attempt_refused(startup, key, value):
    release, output, calls, *_ = startup
    release[key] = value
    with pytest.raises(ValueError, match='frozen_contract'):
        operator.start_attempt(release, 'd'*64, 'synthetic-cpu')
    assert not calls and not output.exists()


def test_real_runtime_import_drift_refuses_before_reservation(startup):
    release, output, calls, _, _, observed = startup
    observed['python_executable'] = '/different/python'
    with pytest.raises(ValueError, match='actual_runtime_import_drift'):
        operator.start_attempt(release, 'd'*64, 'synthetic-cpu')
    assert calls == [['torch']] and not output.exists()


def test_runtime_receipt_tamper_refuses_before_reservation(startup):
    release, output, calls, path, *_ = startup
    path.write_text('{}')
    with pytest.raises(ValueError, match='runtime_receipt_hash'):
        operator.start_attempt(release, 'd'*64, 'synthetic-cpu')
    assert not calls and not output.exists()


def test_constructor_binds_accepted_observation_and_never_authorizes(startup):
    _, _, _, path, observation, _ = startup
    draft = packager.build_release('a'*40, 'b'*64, 'c'*64, path, observation)
    with pytest.raises(ValueError, match='draft_is_not_executable'):
        operator.start_attempt(draft, 'd'*64, 'synthetic-cpu')
    observation.write_text('{}')
    with pytest.raises(ValueError, match='accepted_runtime_receipts_required'):
        packager.build_release('a'*40, 'b'*64, 'c'*64, path, observation)


def test_constructor_requires_observed_interpreter_even_with_valid_hash(startup, monkeypatch):
    _, _, _, path, observation, observed = startup
    del observed['python_executable']
    observation.write_text(json.dumps(observed))
    monkeypatch.setitem(packager.FROZEN_FIELDS, 'runtime_observation_sha256',
                        hashlib.sha256(observation.read_bytes()).hexdigest())
    with pytest.raises(ValueError, match='runtime_observation_contract'):
        packager.build_release('a'*40, 'b'*64, 'c'*64, path, observation)


def test_frozen_remote_output_is_posix_on_every_packaging_host():
    assert operator.FROZEN_FIELDS['output'] == operator.OUTPUT.as_posix()
    assert operator.FROZEN_FIELDS['output'].startswith('/data/gpfs/projects/')
    assert '\\' not in operator.FROZEN_FIELDS['output']
