"""Synthetic contract tests, not measured scientific/model performance."""
import copy
import hashlib
import json

import pytest

from climate_rag.scifact_document_verifier import (
    DocumentJournal, DocumentVerifierProvider, PROTOCOL, audit_episode, call_input,
    model_feedback, parse_verdict, render_prompt, run_episode,
)
from climate_rag.scifact_grounding import Abstract
from climate_rag.scifact_natural_contract import base_state
from climate_rag.scifact_terminal import source_from_abstract
from climate_rag.scifact_utility_contract import identity
from climate_rag.scifact_utility_runtime import ledger_cost
from test_scifact_natural_runtime import Backend as OldBackend
from test_bounded_scifact_runtime import ABSTAIN


class Backend(OldBackend):
    def render(self, obs, schema):
        return render_prompt(self.base.tokenizer, obs, schema)

    def count_prompt(self, obs, schema):
        return len(self.base.tokenizer.encode(self.render(obs, schema), add_special_tokens=False))

    def generate(self, obs, schema, maximum, seconds):
        response = super().generate(obs, schema, maximum, seconds)
        response['usage']['input_tokens'] = self.count_prompt(obs, schema)
        return response


def inputs(n=2):
    corpus = {i: Abstract(i, 'Synthetic', tuple(f'Original sentence {j}.' for j in range(9)), False)
              for i in range(77, 77+n)}
    visible = {}
    aliases = {}
    for i, doc in enumerate(corpus.values()):
        alias, source = f'c{7+i}', source_from_abstract(doc)
        aliases[alias] = str(doc.doc_id)
        for index in (2, 7):
            text = source.sentences[index]
            visible[f'{alias}:{index}'] = {'source_id': source.source_id, 'sentence_index': index,
                'text': text, 'text_sha256': hashlib.sha256(text.encode()).hexdigest(),
                'source_text_sha256': source.text_sha256}
    frame = {'visible': visible, 'alias_to_source': aliases, 'document_order': list(aliases),
        'observation': {'immutable_claim': 'Complete immutable fixture claim.',
            'current_citable': [{'sentence_id': s, 'text': v['text']} for s, v in visible.items()]}}
    return frame, corpus


def verdict(doc='c7', label='SUPPORTS'):
    return {'source_id': doc, 'label': label,
            'sentence_ids': [] if label == 'INSUFFICIENT' else [doc+':7', doc+':2']}


def answer(doc='c7'):
    return {'action': 'answer', 'documents': [verdict(doc)]}


def run(tmp_path, actions, arm='adaptive', n=2, faults=None, clock=None):
    frame, corpus = inputs(n)
    backend = Backend(actions, faults)
    journal = DocumentJournal(backend, tmp_path/'ledger', max_generations=240, protocol=PROTOCOL, physical_guard=base_state)
    row = run_episode(1, arm, frame, journal, corpus, tmp_path/'episode', **({'clock': clock} if clock else {}))
    return row, backend, journal, frame


def test_fixed_four_verifiers_then_terminal_shared_caps(tmp_path):
    row, backend, journal, frame = run(tmp_path,
        [verdict(f'c{7+i}') for i in range(4)] + [answer()], 'fixed', 5)
    assert row['physical_calls'] == row['tool_calls'] == 5
    assert [s['stage'] for s in row['steps']] == ['verify']*4 + ['terminal']
    assert row['state'] == 'valid_terminal'
    assert [s['source_id'] for s in row['steps'][:-1]] == frame['document_order'][:4]
    assert row['prediction']['evidence']['77']['sentences'] == [7, 2]
    with pytest.raises(ValueError, match='budget'):
        journal.generate({}, {}, 512, 10)
    assert backend.calls == ledger_cost(journal.directory)['unique_physical_calls'] == 5


def test_adaptive_feedback_next_input_same_evidence_no_extra_synthesis(tmp_path):
    row, backend, journal, frame = run(tmp_path, [
        {'action':'verify','source_id':'c8'}, verdict('c8'),
        {'action':'verify','source_id':'c7'}, verdict(), answer()])
    assert row['physical_calls'] == 5 and row['tool_calls'] == 3
    assert [s['stage'] for s in row['steps']] == ['plan','verify','plan','verify','terminal']
    for i in (0, 2, 4):
        assert row['steps'][i]['observation']['current_citable'] == frame['observation']['current_citable']
    f = row['steps'][2]['observation']['verification_feedback'][0]
    assert not f['citable'] and f['judgment'] == verdict('c8')
    assert f['original_source_id'] == '78' and f['source_text_sha256']
    assert row['steps'][4]['observation']['verifiable_documents'] == []
    request = json.loads((journal.directory/'g01.reserved.json').read_bytes())
    assert request['prompt_sha256'] == identity(backend.render(request['observation'], request['schema']))


def test_adaptive_never_forced_to_verify(tmp_path):
    row, backend, _, _ = run(tmp_path, [ABSTAIN])
    assert backend.calls == row['physical_calls'] == row['tool_calls'] == 1
    assert row['verification_feedback'] == [] and row['prediction']['evidence'] == {}


@pytest.mark.parametrize('label', ['SUPPORTS','REFUTES'])
def test_projection_multisentence_noncontiguous_preserves_order(label):
    frame, _ = inputs()
    original = copy.deepcopy(frame)
    obs, _ = call_input(frame, 'verify', 'c7', [], 0, 1, [])
    assert [s['sentence_id'] for s in obs['current_citable']] == ['c7:2','c7:7']
    assert obs['immutable_claim'] == original['observation']['immutable_claim']
    assert parse_verdict(verdict(label=label), frame, 'c7')['sentence_ids'] == ['c7:7','c7:2']
    assert frame == original


@pytest.mark.parametrize('ids', [['c7:0'],['c8:2'],['c0:2'],['feedback:0'],['c7:2','c7:2']])
def test_hidden_cross_document_renumbered_feedback_duplicate_rejected(ids):
    frame, _ = inputs()
    with pytest.raises(ValueError):
        parse_verdict(dict(verdict(), sentence_ids=ids), frame, 'c7')


def test_failed_verifier_feedback_is_not_nei(tmp_path):
    row, _, _, _ = run(tmp_path, [{'action':'verify','source_id':'c7'}, {'bad':True}, answer()])
    feedback = row['verification_feedback'][0]
    assert feedback['status'] == 'failed' and feedback['judgment'] is None
    assert row['steps'][2]['observation']['verification_feedback'] == model_feedback([feedback])
    assert row['physical_calls'] == 3 and row['prediction']['evidence']


def test_terminal_truncated_not_correct_nei(tmp_path):
    row, _, journal, _ = run(tmp_path, [ABSTAIN], faults={0:'no_eos'})
    assert row['prediction'] is None and row['state'] == 'unresolved'
    assert ledger_cost(journal.directory)['unique_physical_calls'] == 1


def test_document_insufficient_requires_real_terminal(tmp_path):
    row, _, _, _ = run(tmp_path, [verdict(label='INSUFFICIENT'), ABSTAIN], 'fixed', 1)
    assert row['physical_calls'] == 2 and row['terminal'] == ABSTAIN


def test_shared_deadline_keeps_cost_stops_next_stage(tmp_path):
    ticks = iter([0, 0, 121, 121, 121])
    row, _, journal, _ = run(tmp_path, [verdict()], 'fixed', 1, clock=lambda: next(ticks))
    assert row['prediction'] is None and row['physical_calls'] == 1
    assert ledger_cost(journal.directory)['unique_physical_calls'] == 1


def test_overflow_not_silently_repacked(tmp_path, monkeypatch):
    monkeypatch.setattr(Backend, 'render', lambda *args: 'x'*100000)
    row, backend, journal, frame = run(tmp_path, [ABSTAIN])
    assert row['reason'] == 'prompt_overflow_no_repacking' and backend.calls == 0
    assert row['steps'][0]['observation']['current_citable'] == frame['observation']['current_citable']
    assert ledger_cost(journal.directory)['unique_physical_calls'] == 0


def test_verifier_provider_uses_generic_same_model_path():
    from climate_rag.local_scifact_provider import LocalQwenSciFactProvider
    assert DocumentVerifierProvider.generate is LocalQwenSciFactProvider.generate


@pytest.mark.parametrize('arm,actions', [
    ('adaptive', [ABSTAIN]), ('fixed', [verdict(), verdict('c8'), answer()]),
    ('adaptive', [{'action':'verify','source_id':'c7'}, {'bad':True}, answer()]),
    ('adaptive', [{'action':'read','source_ids':['c7']}]),
])
def test_physical_audit_reconstructs_episode_without_model(tmp_path, arm, actions):
    row, backend, journal, frame = run(tmp_path, actions, arm)
    _, corpus = inputs()
    audited = audit_episode(row, frame, journal.directory, tmp_path/'episode/private-responses', backend.base.tokenizer, corpus)
    assert audited == row


@pytest.mark.parametrize('fault', ['feedback','prompt','cost','prediction','claim'])
def test_audit_refuses_tampered_feedback_prompt_or_prediction(tmp_path, fault):
    row, backend, journal, frame = run(tmp_path, [{'action':'verify','source_id':'c7'}, verdict(), answer()])
    _, corpus = inputs()
    if fault == 'feedback':
        row['steps'][2]['observation']['verification_feedback'][0]['judgment']['label'] = 'REFUTES'
    elif fault == 'prediction':
        row['prediction']['evidence'] = {}
    elif fault == 'claim':
        row['steps'][1]['observation']['immutable_claim'] = 'Replacement claim'
    else:
        path = journal.directory/('g01.reserved.json' if fault == 'prompt' else 'g01.finished.json')
        record = json.loads(path.read_bytes())
        record['prompt_sha256' if fault == 'prompt' else 'usage_known'] = 'wrong' if fault == 'prompt' else False
        path.write_text(json.dumps(record))
    with pytest.raises(ValueError):
        audit_episode(row, frame, journal.directory, tmp_path/'episode/private-responses', backend.base.tokenizer, corpus)


def test_unknown_cost_stops_without_terminal(tmp_path, monkeypatch):
    def failure(*args, **kwargs):
        raise RuntimeError('synthetic failed physical model call')
    monkeypatch.setattr(Backend, 'generate', failure)
    row, _, journal, _ = run(tmp_path, [], 'fixed')
    assert row['state'] == 'unresolved' and row['prediction'] is None
    assert row['physical_calls'] == row['verification_feedback'][0]['physical_attempt_id'].count('g') == 1
    assert row['verification_feedback'][0]['judgment'] is None
    assert ledger_cost(journal.directory)['unknown_usage_attempts'] == 1


def test_release_draft_and_budget_mutation_rejected():
    from run_scifact_document_verifier_operator import RESOURCE, OUTPUT, validate_release
    from prepare_scifact_natural import SELECTION_SHA
    from run_budget_agent_full_operator import ARCHIVES
    release = dict(authorization='draft', protocol=PROTOCOL, output=str(OUTPUT), resource_cap=RESOURCE,
        selection_sha256=SELECTION_SHA, max_generator_calls=240, planned_episodes=48,
        max_episode_seconds=120, max_episode_tools=5, max_episode_generations=5, max_worker_seconds=6360,
        warmup_generation_calls=0, training_authorized=False, protected_split_read=False,
        automatic_retry=False, adapter_loaded=False, reranker_loaded=False,
        model_archive_sha256=ARCHIVES['input'][1], source_git='a'*40)
    for key in ('source_archive_sha256','wrapper_sha256','runtime_receipt_sha256','runtime_files_sha256','initial_inventory_sha256'):
        release[key] = 'a'*64
    with pytest.raises(ValueError, match='draft'):
        validate_release(release)
    release['authorization'] = 'coordinator_exact_hash_release'
    validate_release(release)
    release['max_generator_calls'] = 241
    with pytest.raises(ValueError, match='contract'):
        validate_release(release)


def test_audit_uses_physical_time_not_claimed_row_elapsed(tmp_path):
    row, backend, journal, frame = run(tmp_path, [verdict(), ABSTAIN], 'fixed', 1)
    _, corpus = inputs(1)
    for path in journal.directory.glob('g*.finished.json'):
        record = json.loads(path.read_bytes())
        record['elapsed_ms'] = 70000
        path.write_text(json.dumps(record))
    row['elapsed_seconds'] = 1
    with pytest.raises(ValueError, match='physical_time'):
        audit_episode(row, frame, journal.directory, tmp_path/'episode/private-responses', backend.base.tokenizer, corpus)


def test_fixed_zero_adaptive_positive_not_misreported_zero():
    from climate_rag.scifact_grounding import GoldClaim, Rationale
    from score_scifact_document_verifier import summarize
    _, corpus = inputs(1)
    gold = GoldClaim(1, 'claim', {77: (Rationale('SUPPORT', (7, 2)),)}, (77,))
    bad = {'state':'unresolved', 'prediction':None, 'verification_feedback':[], 'physical_calls':1, 'elapsed_seconds':1}
    good = dict(bad, state='valid_terminal', prediction={'id':1,'evidence':{'77':{'label':'SUPPORT','sentences':[7,2]}}})
    summary = summarize([1], [gold], {(1,'fixed'):bad, (1,'adaptive'):good}, corpus)
    assert summary['arms']['adaptive']['positive_correct'] == 1
    assert summary['hypothesis'].startswith('adaptive_only_positive_recovery')


def test_truncated_reservation_remains_unassigned_unknown_before_gold(tmp_path):
    from score_scifact_document_verifier import arm_costs_before_gold
    (tmp_path/'g00.reserved.json').write_text('{')
    arms, unassigned = arm_costs_before_gold(tmp_path, [1])
    assert sum(a['unique_physical_calls'] for a in arms.values()) == 0
    assert unassigned['unique_physical_calls'] == unassigned['unknown_usage_attempts'] == 1
    assert unassigned['total_tokens'] is None and unassigned['model_backend_elapsed_ms'] is None
