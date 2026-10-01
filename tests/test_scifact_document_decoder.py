"""Real provider/LMFE/controller seam; fake model follows prescribed toy tokens.

These are grammar/integration tests, never estimates of model answer quality.
"""
import contextlib
import copy
import json
import types

import pytest
import torch

from climate_rag.agent_protocol import ModelResponseValidationError
from climate_rag.private_diagnostics_v3 import PrivateDiagnosticStore
from climate_rag.scifact_document_verifier import (
    DECODER_IMPLEMENTATION, PROTOCOL, DocumentJournal, DocumentVerifierProvider,
    audit_episode, call_input, run_episode,
)
from climate_rag.scifact_natural_contract import base_state
from climate_rag.scifact_utility_runtime import ledger_cost
from test_bounded_scifact_grammar import toy_data
from test_scifact_document_verifier import answer, inputs, verdict


def real_provider(tmp_path, actions):
    data, vocab, eos = toy_data()
    calls, callbacks = [], []
    responses = iter(actions)

    class Inputs(dict):
        def __init__(self, ids):
            self.input_ids = torch.tensor([ids])
            super().__init__(input_ids=self.input_ids)

        def to(self, device):
            return self

    class Tokenizer:
        def apply_chat_template(self, messages, **kwargs):
            assert kwargs == dict(tokenize=False, add_generation_prompt=True, enable_thinking=False)
            return json.dumps(messages)

        def encode(self, text, *, add_special_tokens=False):
            assert not add_special_tokens
            return [vocab.index(char) for char in text]

        def __call__(self, text, *, return_tensors, add_special_tokens):
            assert return_tensors == 'pt' and not add_special_tokens
            return Inputs(self.encode(text))

        def decode(self, ids, *, skip_special_tokens=True):
            return ''.join(vocab[int(i)] for i in ids if int(i) != eos)

    tokenizer = Tokenizer()

    def generate(**kwargs):
        calls.append(kwargs)
        callback = kwargs['prefix_allowed_tokens_fn']
        callbacks.append(callback)
        raw = json.dumps(next(responses), separators=(',', ':'))
        sent = kwargs['input_ids'][0].tolist()
        tokens = tokenizer.encode(raw) + [eos]
        assert len(tokens) <= kwargs['max_new_tokens']
        for token in tokens:
            if token not in callback(0, torch.tensor(sent)):
                raise ValueError('synthetic_prescribed_token_rejected')
            sent.append(token)
        return [sent]

    provider = DocumentVerifierProvider.__new__(DocumentVerifierProvider)
    provider.name, provider.tokenizer_data = 'synthetic-no-model', data
    provider.private_store = PrivateDiagnosticStore(tmp_path)
    provider.base = types.SimpleNamespace(tokenizer=tokenizer, name='synthetic',
        _torch=types.SimpleNamespace(inference_mode=contextlib.nullcontext),
        model=types.SimpleNamespace(device='cpu', generate=generate, training=False,
            named_parameters=lambda: [], named_modules=lambda: [],
            generation_config=types.SimpleNamespace(eos_token_id=eos)))
    return provider, calls, callbacks


@pytest.mark.parametrize('stage,action', [
    ('plan', {'source_id': 'c8', 'action': 'verify'}),
    ('plan', answer()), ('terminal', answer()),
    ('terminal', {'action': 'abstain', 'reason': 'insufficient_evidence'}),
    ('verify', verdict()), ('verify', verdict(label='REFUTES')),
    ('verify', verdict(label='INSUFFICIENT')),
])
def test_actual_generate_stage_decoder_and_immutable_input(tmp_path, stage, action):
    provider, calls, callbacks = real_provider(tmp_path, [action])
    frame, _ = inputs()
    obs, schema = call_input(frame, stage, 'c7' if stage == 'verify' else None,
                             [], 0, 0, ['c7', 'c8'] if stage == 'plan' else [])
    before = copy.deepcopy((obs, schema))
    result = provider.generate(obs, schema, 512, 120)
    assert json.loads(result['raw']) == action
    assert before == (obs, schema) and 'preview_only' not in obs
    assert result['usage']['input_tokens'] == provider.count_prompt(obs, schema)
    d = result['diagnostics']
    assert d['decoder_implementation'] == DECODER_IMPLEMENTATION
    assert d['decoder_stage'] == stage and d['eos_observed']
    assert d['grammar'] == ('lm-format-enforcer0.11.3/fresh-inline-anyOf' if stage == 'verify'
                            else 'scifact-bounded-prefix-v1')
    assert d['effective_parser_config']['max_json_array_length'] == 20
    assert len(calls) == len(callbacks) == 1


@pytest.mark.parametrize('stage,action', [
    ('plan', {'action': 'verify', 'source_id': 'c9'}),
    ('terminal', {'action': 'verify', 'source_id': 'c7'}),
    ('terminal', {'action': 'answer', 'documents': [verdict(), verdict()]}),
])
def test_actual_callback_rejects_without_retry_and_preserves_unknown_cost(tmp_path, stage, action):
    provider, calls, _ = real_provider(tmp_path, [action])
    frame, _ = inputs()
    obs, schema = call_input(frame, stage, None, [], 0, 0, ['c7', 'c8'] if stage == 'plan' else [])
    with pytest.raises(ModelResponseValidationError) as err:
        provider.generate(obs, schema, 512, 120)
    assert err.value.diagnostics['output_usage_unknown'] is True
    assert err.value.usage['input_tokens'] == provider.count_prompt(obs, schema)
    assert len(calls) == 1


def test_real_provider_journal_episode_and_physical_audit(tmp_path):
    provider, calls, callbacks = real_provider(tmp_path, [
        {'action': 'verify', 'source_id': 'c8'}, verdict('c8'),
        {'action': 'verify', 'source_id': 'c7'}, verdict(), answer()])
    frame, corpus = inputs()
    journal = DocumentJournal(provider, tmp_path/'ledger', max_generations=240,
                              protocol=PROTOCOL, physical_guard=base_state)
    row = run_episode(1, 'adaptive', frame, journal, corpus, tmp_path/'episode')
    assert row['state'] == 'valid_terminal'
    assert [s['stage'] for s in row['steps']] == ['plan', 'verify', 'plan', 'verify', 'terminal']
    assert len(calls) == len({id(c) for c in callbacks}) == 5
    assert ledger_cost(journal.directory)['unique_physical_calls'] == 5
    assert audit_episode(row, frame, journal.directory, tmp_path/'episode/private-responses',
                         provider.base.tokenizer, corpus) == row


@pytest.mark.parametrize('last_count,accepted', [(4, True), (5, False)])
def test_actual_generate_twenty_sentence_boundary(tmp_path, last_count, accepted):
    frame, _ = inputs(3)
    visible = {}
    for alias, doc in frame['alias_to_source'].items():
        for i in range(8):
            visible[f'{alias}:{i}'] = {'source_id': doc, 'text': f'Fixture {i}.'}
    frame['visible'] = visible
    frame['observation']['current_citable'] = [
        {'sentence_id': sid, 'text': value['text']} for sid, value in visible.items()]
    action = {'action': 'answer', 'documents': [
        {'source_id': alias, 'label': 'SUPPORTS',
         'sentence_ids': [f'{alias}:{i}' for i in reversed(range(n))]}
        for alias, n in zip(frame['document_order'], [8, 8, last_count])]}
    provider, calls, _ = real_provider(tmp_path, [action])
    obs, schema = call_input(frame, 'terminal', None, [], 0, 0, [])
    if accepted:
        assert json.loads(provider.generate(obs, schema, 512, 120)['raw']) == action
    else:
        with pytest.raises(ModelResponseValidationError) as err:
            provider.generate(obs, schema, 512, 120)
        assert err.value.diagnostics['exception_type'] == 'ValueError'
    assert len(calls) == 1


def test_decoder_time_charged_before_model_and_no_generation_when_exhausted(tmp_path, monkeypatch):
    from climate_rag import local_scifact_provider
    provider, calls, _ = real_provider(tmp_path, [answer()])
    frame, _ = inputs()
    obs, schema = call_input(frame, 'terminal', None, [], 0, 0, [])
    tick = iter([0, 121])
    monkeypatch.setattr(local_scifact_provider.time, 'perf_counter', lambda: next(tick))
    with pytest.raises(ModelResponseValidationError) as err:
        provider.generate(obs, schema, 512, 120)
    assert err.value.diagnostics['category'] == 'deadline_before_generation' and not calls
