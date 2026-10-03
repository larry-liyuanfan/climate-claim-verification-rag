"""Six synthetic fixture groups; real JournalProvider/runtime, no model/gold IO."""
import copy
import hashlib
import json

import pytest

from climate_rag.scifact_grounding import Abstract
from climate_rag.model_diagnostics import response_diagnostics
from climate_rag.scifact_observation_receipts import FROZEN_CONTRACT, VERSION, audit_observations
from climate_rag.scifact_terminal import source_from_abstract
from climate_rag.scifact_utility_contract import identity
from climate_rag.scifact_utility_runtime import run_matrix
from test_bounded_scifact_runtime import ABSTAIN, SyntheticBackend


def matrix(tmp_path, actions=None, continuation=None):
    tmp_path.mkdir(exist_ok=True)
    class Backend(SyntheticBackend):
        def start_slot(self, path):
            path.mkdir()
            self.actions = iter(actions if path.parent.name.endswith('-A') else [continuation or ABSTAIN])
        def generate(self, *args):
            response = super().generate(*args)
            if json.loads(response['raw']) == 'INVALID_JSON_FIXTURE':
                response['raw'] = '{'
                response['diagnostics'].update(raw_wire_action=None, private_attachment={
                    'sha256': hashlib.sha256(b'{').hexdigest(), 'attempted_bytes': 1, 'stored_bytes': 1,
                    'truncated': False, 'io_failed': False})
            diagnostic = response_diagnostics(response['raw'], output_tokens=response['usage']['output_tokens'],
                max_new_tokens=512, eos_observed=True, generation_elapsed_ms=0)
            diagnostic.update(response['diagnostics'])
            response['diagnostics'] = diagnostic
            return response
    actions = actions or [
        {'action': 'read', 'source_ids': ['c5', 'c6']},
        {'action': 'read', 'source_ids': ['c7', 'c8']},
        {'action': 'rerank'}, {}, ABSTAIN]
    corpus = {i: Abstract(i, f'Fixture {i}', (f'Original sentence {i}.',), False) for i in range(100, 120)}
    sources = [source_from_abstract(a) for a in corpus.values()]
    backend = Backend([], False)
    out = tmp_path / 'inference'
    claims = [{'id': i, 'claim': f'Fixture claim {i}'} for i in range(8)]
    run_matrix(claims, backend, lambda q, k: sources[:k], lambda q, c: list(reversed(c)), corpus, out)
    files = {p.relative_to(out).as_posix(): json.loads(p.read_bytes()) for p in out.rglob('*.json')}
    payload = {'version': VERSION, 'scope': 'synthetic_fixture', 'contract_revision': FROZEN_CONTRACT,
               'slots': [{'claim_id': c['id'], 'arm': a, 'claim': c['claim']} for c in claims for a in 'ABC']}
    return files, {'payload': payload, 'sha256': identity(payload)}, backend.base.tokenizer, corpus


def audit(fixture, files=None):
    original, roster, tokenizer, corpus = fixture
    return audit_observations([(p, json.dumps(v, ensure_ascii=False, separators=(',', ':')).encode())
                               for p, v in (original if files is None else files).items()], roster, tokenizer, corpus)


def result(report, slot='0-A'):
    return next(r for r in report['slots'] if f"{r['claim_id']}-{r['arm']}" == slot)


def mutate_result(files, change):
    change(files['0-A/result.json']['result'])
    change(files['0-A/raw-result.json'])


def test_real_runtime_read_rerank_repair_shared_prefix_and_unexecuted_proposals(tmp_path):
    fixture = matrix(tmp_path, continuation={'action': 'read', 'source_ids': ['c7', 'c8']})
    before = copy.deepcopy(fixture[0])
    report = audit(fixture)
    assert report['accepted_slots'] == report['planned_slots'] == 24, report['slots'][:3]
    assert len(report['physical_observations']) == 56
    assert sum(len(p['logical_references']) for p in report['physical_observations']) == 72
    initial = report['physical_observations'][0]
    assert len(initial['logical_references']) == 3
    assert initial['journal_prompt_identity'] != initial['prompt_utf8_sha256']
    a = [p for p in report['physical_observations'] if p['physical_slot'] == '0-A']
    assert a[3]['parsed_model_proposal'] is None and a[3]['parse_error'] == 'invalid_schema'
    assert a[3]['logical_references'][0]['state_event_index'] == a[4]['logical_references'][0]['state_event_index'] == 3
    assert a[3]['logical_references'][0]['state_event']['candidate_ids'][0] == 'c19'
    assert a[3]['capture']['alias_to_source']['c19'] == '119'
    b = next(p for p in report['physical_observations'] if p['physical_slot'] == '0-B')
    ref = b['logical_references'][0]
    assert ref['attempt_index'] == 1 and ref['frame_path'] == '0-B/frame-0.json'
    assert ref['state_event']['origin'] == 'scripted_intervention' and ref['state_event']['model_selected'] is False
    assert ref['decision_execution_audit']['actual_event_index'] is None
    assert b['parsed_model_proposal']['action'] == 'read' and b['supervision_target'] is None
    assert fixture[0] == before and not report['training_authorized'] and report['semantic_targets_created'] == 0


def test_prefix_parse_status_survives_controller_read_loop_and_missing_branch(tmp_path):
    fixture = matrix(tmp_path, actions=[{'action': 'read', 'source_ids': ['c0', 'c1', 'c2', 'c3', 'c4']}, ABSTAIN])
    report = audit(fixture)
    assert report['planned_slots'] == 24 and report['accepted_slots'] == 16, report['slots'][:3]
    assert result(report, '0-B')['status'] == 'incomplete'
    initial = report['physical_observations'][0]
    statuses = {r['slot']: r['attempt']['status'] for r in initial['logical_references']}
    assert statuses == {'0-A': 'validation_failed', '0-C': 'valid_decision'}
    assert initial['parsed_model_proposal']['action'] == 'read'
    answer = {'action': 'answer', 'documents': [{'source_id': 'c0', 'label': 'SUPPORTS', 'sentence_ids': ['c0:0']}]}
    repair = matrix(tmp_path / 'invalid-json', actions=['INVALID_JSON_FIXTURE', answer])
    repaired = audit(repair)
    assert repaired['accepted_slots'] == 16, repaired['slots'][:3]
    a = [p for p in repaired['physical_observations'] if p['physical_slot'] == '0-A']
    assert a[0]['parse_error'] == 'invalid_json' and a[1]['parsed_model_proposal']['action'] == 'answer'
    assert a[1]['capture']['observation']['feedback'] == (
        'invalid_json; choose a new legal action; read known preview candidates before citation')
    assert a[0]['logical_references'][0]['state_event_index'] == a[1]['logical_references'][0]['state_event_index'] == 0


def test_receipt_incomplete_duplicate_cross_slot_and_orphan_frames_keep_denominator(tmp_path):
    fixture = matrix(tmp_path)
    original = fixture[0]
    mutations = [
        lambda f: f.pop('ledger/g00.finished.json'),
        lambda f: f['ledger/g00.finished.json'].update(status='failed'),
        lambda f: f['ledger/g00.reserved.json'].update(slot='1-A'),
        lambda f: f['ledger/g01.reserved.json'].update(physical_attempt_id='g00'),
        lambda f: f['0-A/result.json']['logical_generation_ids'].append('g00'),
        lambda f: f.update({'0-A/frame-5.json': copy.deepcopy(f['0-A/frame-4.json'])}),
        lambda f: f['ledger/g00.finished.json']['response'].update(raw='{}'),
        lambda f: f.pop('ledger/g55.reserved.json'),
        lambda f: f.update({'999-A/frame-0.json': copy.deepcopy(f['0-A/frame-0.json'])}),
    ]
    for mutate in mutations:
        files = copy.deepcopy(original)
        mutate(files)
        report = audit(fixture, files)
        assert report['planned_slots'] == len(report['slots']) == 24
        assert report['accepted_slots'] < 24
        assert report['semantic_targets_created'] == 0
    files = copy.deepcopy(original)
    files.pop('ledger/g55.reserved.json')
    assert result(audit(fixture, files))['errors'] == ['finished_without_reservation']
    answer = {'action': 'answer', 'documents': [{'source_id': 'c0', 'label': 'SUPPORTS', 'sentence_ids': ['c0:0']}]}
    short = matrix(tmp_path / 'short', actions=[answer])
    assert audit(short)['accepted_slots'] == 24
    files = copy.deepcopy(short[0])
    assert 'ledger/g54.reserved.json' not in files and 'ledger/g54.finished.json' not in files
    files['ledger/g54.finished.json'] = dict(files['ledger/g00.finished.json'], physical_attempt_id='g54')
    assert result(audit(short, files))['errors'] == ['finished_without_reservation']
    for update_attachment in (False, True):
        files = copy.deepcopy(short[0])
        response = files['ledger/g00.finished.json']['response']
        changed = json.loads(response['raw'])
        changed['documents'][0]['label'] = 'REFUTES'  # Still legal, still answer: not an action/schema rejection.
        response['raw'] = json.dumps(changed)
        if update_attachment:
            response['diagnostics']['private_attachment'].update(
                sha256=hashlib.sha256(response['raw'].encode()).hexdigest(),
                attempted_bytes=len(response['raw'].encode()), stored_bytes=len(response['raw'].encode()))
            mutate_result(files, lambda r: r['generation_attempts'][0].update(diagnostics=response['diagnostics']))
        expected = 'raw_diagnostic_identity' if update_attachment else 'full_raw_response_identity'
        assert result(audit(short, files))['errors'] == [expected]
    artifacts = [(p, json.dumps(v).encode()) for p, v in original.items()]
    report = audit_observations(artifacts + [artifacts[0]], *fixture[1:])
    assert report['accepted_slots'] == 0 and report['planned_slots'] == 24


def test_candidate_schema_order_prompt_domains_and_unknown_candidates_rejected(tmp_path):
    fixture = matrix(tmp_path)
    for change in ('order', 'duplicate', 'unknown', 'schema', 'prompt_domain', 'json_order'):
        files = copy.deepcopy(fixture[0])
        if change in {'order', 'duplicate', 'unknown'}:
            def mutate(r):
                order = r['events'][0]['candidate_ids']
                if change == 'order':
                    order[5], order[6] = order[6], order[5]
                else:
                    order[5] = 'c0' if change == 'duplicate' else 'c999'
            mutate_result(files, mutate)
        elif change == 'schema':
            files['0-A/frame-0.json']['schema']['anyOf'].reverse()
        elif change == 'json_order':
            obs = files['ledger/g00.reserved.json']['observation']
            files['ledger/g00.reserved.json']['observation'] = dict(reversed(list(obs.items())))
        else:
            accepted = audit(fixture)['physical_observations'][0]
            files['ledger/g00.reserved.json']['prompt_sha256'] = accepted['prompt_utf8_sha256']
        report = audit(fixture, files)
        assert result(report)['status'] == 'incomplete', change
        assert result(report, '0-B')['status'] == result(report, '0-C')['status'] == 'incomplete'


def test_same_tool_different_valid_parameters_and_scripted_event_mislink(tmp_path):
    fixture = matrix(tmp_path, continuation={'action': 'read', 'source_ids': ['c5']})
    clean = audit(fixture)
    assert clean['accepted_slots'] == 24, clean['slots'][:3]
    for fault in ('audit_index', 'read_parameters', 'strict_action', 'scripted_link', 'prefix_proposal',
                  'after_terminal', 'diagnostic_sha', 'diagnostic_bytes', 'diagnostic_characters', 'diagnostic_tokens'):
        files = copy.deepcopy(fixture[0])
        if fault == 'audit_index':
            audits = files['0-A/result.json']['result']['decision_execution_audit']
            audits[0]['actual_event_index'], audits[1]['actual_event_index'] = 2, 1
        elif fault == 'read_parameters':
            # Both legal read events remain individually well-formed; wrong ordered parameters.
            mutate_result(files, lambda r: r['events'][1].update(requested_context=['c7', 'c8']))
        elif fault == 'strict_action':
            mutate_result(files, lambda r: r['generation_attempts'][0].update(action='rerank'))
        elif fault == 'scripted_link':
            files['0-B/result.json']['result']['decision_execution_audit'][1].update(
                actual_event_index=1, actual_event_status='completed')
        elif fault == 'prefix_proposal':
            prefix = files['0-A/prefix.json']
            prefix['payload']['decision']['source_ids'] = ['c9']
            prefix['sha256'] = identity(prefix['payload'])
            for arm in 'BC':
                for filename in ('result.json', 'raw-result.json'):
                    r = files[f'0-{arm}/{filename}']
                    (r['result'] if filename == 'result.json' else r)['shared_prefix_sha256'] = prefix['sha256']
        elif fault == 'after_terminal':
            # Consistent raw/diagnostics/attempt/prefix, but impossible post-terminal continuation.
            response = files['ledger/g00.finished.json']['response']
            response['raw'] = json.dumps(ABSTAIN)
            d = response['diagnostics']
            d.update(response_diagnostics(response['raw'], output_tokens=50, max_new_tokens=512,
                                         eos_observed=True, generation_elapsed_ms=0))
            d.update(raw_wire_action='abstain', private_attachment={
                'sha256': hashlib.sha256(response['raw'].encode()).hexdigest(),
                'attempted_bytes': len(response['raw'].encode()), 'stored_bytes': len(response['raw'].encode()),
                'truncated': False, 'io_failed': False})
            mutate_result(files, lambda r: r['generation_attempts'][0].update(action='abstain', diagnostics=d))
            prefix = files['0-A/prefix.json']
            prefix['payload'].update(response=response, decision=ABSTAIN)
            prefix['payload']['attempt'].update(action='abstain', diagnostics=d)
            prefix['sha256'] = identity(prefix['payload'])
        else:
            key = {'diagnostic_sha': 'output_sha256', 'diagnostic_bytes': 'output_bytes',
                   'diagnostic_characters': 'output_characters', 'diagnostic_tokens': 'output_tokens'}[fault]
            d = files['ledger/g00.finished.json']['response']['diagnostics']
            d[key] = '0' * 64 if fault == 'diagnostic_sha' else d[key] + 1
            mutate_result(files, lambda r: r['generation_attempts'][0].update(diagnostics=copy.deepcopy(d)))
        report = audit(fixture, files)
        assert result(report, '0-B' if fault == 'scripted_link' else '0-A')['status'] == 'incomplete', fault
        if fault == 'after_terminal':
            assert result(report)['errors'] == ['attempt_after_terminal_decision']
        if fault == 'prefix_proposal':
            assert result(report)['errors'] == ['prefix_proposal_not_physical_response']


def test_rewrite_normalized_raw_sha_external_roster_and_duplicate_json_keys(tmp_path):
    fixture = matrix(tmp_path, actions=[{'action': 'rewrite', 'query': 'Fixture   claim 0 context'}, ABSTAIN])
    # Only claim 0 has a constraint-preserving query; other claims can reject it honestly.
    report = audit(fixture)
    assert result(report)['status'] == 'accepted', result(report)
    assert fixture[0]['0-A/result.json']['result']['events'][1]['query_sha256'] == hashlib.sha256(
        b'Fixture claim 0 context').hexdigest()
    files = copy.deepcopy(fixture[0])
    mutate_result(files, lambda r: r['events'][1].update(query_sha256=identity('Fixture claim 0 context')))
    assert result(audit(fixture, files))['status'] == 'incomplete'
    wrong = copy.deepcopy(fixture[1])
    wrong['payload']['scope'] = 'exposed_train_pending_release'
    wrong['sha256'] = identity(wrong['payload'])
    with pytest.raises(ValueError, match='synthetic_external_roster_required'):
        audit_observations([], wrong, *fixture[2:])
    artifacts = [(p, json.dumps(v).encode()) for p, v in fixture[0].items()]
    artifacts.append(('duplicate-keys.json', b'{"id":1,"id":2}'))
    invalid = audit_observations(artifacts, *fixture[1:])
    assert invalid['accepted_slots'] == 0 and invalid['slots'][0]['errors'] == ['duplicate_json_key']
