"""Five bounded CPU groups. No real artifacts, corpus, weights, Slurm or gold."""
import copy
from dataclasses import replace
from fractions import Fraction
import json
import os
import signal
import subprocess
import sys

import pytest
import torch

from climate_rag.scifact_claim_group_mean import make_claim_group_mean_adamw
from climate_rag.scifact_mixed_inputs import (
    ORIGINS, SCOPE, build_inputs, check_released_shape, derived_row, legacy_envelopes,
    make_plan, nei_envelope, nei_inventory, seal, validate_inputs,
)
from climate_rag.scifact_mixed_training import run_synthetic_epoch
from climate_rag.scifact_program_capture import program_candidates
from climate_rag.scifact_semantic_contract import encoded, sha
from climate_rag.scifact_state_supervision import build_claim
from climate_rag.scifact_terminal import source_from_abstract
from climate_rag.scifact_utility_contract import identity
from test_scifact_claim_group_mean import PositionModel, fixtures, independent_clip, independent_mean
from test_scifact_nei_preparation import arguments, fixture as nei_fixture
from test_scifact_state_supervision import setup


def packed(envelopes):
    cohorts = {k: list(dict.fromkeys((e.get('legacy_row') or e['artifact'])['claim_id']
               for e in envelopes if e['cohort'] == k)) for k in ('old48', 'supp49', 'NEI47')}
    ids = sum(cohorts.values(), [])
    components = {str((e.get('legacy_row') or e['artifact'])['claim_id']):
                  e.get('component', e.get('legacy_row', {}).get('component')) for e in envelopes}
    roster = seal({'claim_ids': ids, 'cohorts': cohorts, 'components': components,
        'excluded_claim_ids': [99999], 'excluded_components': ['heldout'],
        'source_envelope_sha256': [identity(e) for e in envelopes]})
    return build_inputs(envelopes, roster, scope='synthetic_fixture',
                        inventory=seal({'file_sha256': 'fixture', 'private_file_sha256': {}}))


def synthetic_rows(claims=5):
    rows, tokens, ids = fixtures(4)
    extra, more, extra_ids = fixtures(1, variant=1)
    rows, tokens, ids = (rows+extra)[:], (tokens+more)[:], ids+extra_ids
    if claims == 144:
        rows, tokens = [], []
        template, tok, _ = fixtures(1)
        for i in range(144):
            if i < 48:
                count = 2 if i <= 44 else 1
                actions = ['read', 'answer'] if i == 0 else ['answer']*count if i < 46 else ['abstain']
            elif i < 97:
                count = 2 if i < 82 else 1
                actions = ['answer']*count if i < 94 else ['abstain']
            else:
                actions = ['abstain']
            for j, action in enumerate(actions):
                row = copy.deepcopy(template[0])
                row.update(claim_id=i, target={'action': action}, state_index=j if i == 0 else 0,
                           state_count=len(actions) if i == 0 else 1,
                           alternative=j if i != 0 else 0, alternative_count=len(actions) if i != 0 else 1,
                           weight_denominator=len(actions))
                rows.append(row)
                tokens.append(copy.deepcopy(tok[0]))
    envelopes = []
    for index, (row, tok) in enumerate(zip(rows, tokens, strict=True)):
        claim = row['claim_id']
        cohort = ('old48' if claim < 48 else 'supp49' if claim < 97 else 'NEI47') if claims == 144 else 'old48'
        reason, origin = ORIGINS[row['target']['action']]
        row.update(component=f'component-{claim}', teacher_reason=reason,
                   semantic_target_provenance='official_complete_original_fit_annotation'
                       if row['target']['action'] == 'answer' else 'no_semantic_label')
        row['record_sha256'] = sha(encoded({k: v for k, v in row.items() if k != 'record_sha256'}))
        if cohort == 'NEI47':
            envelopes.append({'kind': 'program_capture', 'cohort': cohort, 'component': row['component'],
                'target_origin': 'official_annotation_NEI', 'tokenized': tok,
                'source_file_sha256': 'fixture', 'artifact_name': f'{claim}.json',
                'artifact': {'claim_id': claim, 'candidates': [{'basis': 'official_annotation_NEI',
                    'target': {'action': 'abstain', 'reason': 'insufficient_evidence'}, 'tokenized': tok}]}})
        else:
            envelopes.append({'kind': 'legacy_program_artifact', 'cohort': cohort, 'source_file_sha256': 'fixture',
                'source_row_index': index, 'legacy_row': row, 'target_origin': origin,
                'event_lineage_available': False, 'tokenized': tok})
    return envelopes


def test_legacy_original_hash_packing_and_order_unchanged():
    claim, corpus, sources, tokenizer = setup()
    corpus = {i: replace(a, sentences=tuple(f'{i}: {s}' for s in a.sentences)) for i, a in corpus.items()}
    sources = [source_from_abstract(corpus[int(s.source_id)]) for s in sources]
    original = build_claim(claim, 'fixture', sources, corpus, tokenizer)['records']
    raw = (json.dumps(original, ensure_ascii=False) + '\n').encode()
    envelopes = legacy_envelopes(raw, 'old48', [claim.claim_id], corpus, tokenizer, scope='synthetic_fixture')
    assert envelopes[0]['legacy_row'] == json.loads(raw)[0]
    assert envelopes[0]['legacy_row']['epoch_normalizer'] == 48
    assert envelopes[0]['legacy_row']['packing'] == derived_row(envelopes[0], 144)['packing']
    assert envelopes[0]['event_lineage_available'] is False
    for change in ('tool_event', 'candidate_order', 'labels'):
        bad = copy.deepcopy(original)
        if change == 'tool_event':
            bad[0]['tool_event'] = {'status': 'completed'}
        elif change == 'candidate_order':
            bad[0]['candidates'].reverse()
        else:
            bad[0]['packing']['loss_mask_sha256'] = '0'*64
        bad[0]['record_sha256'] = sha(encoded({k: v for k, v in bad[0].items() if k != 'record_sha256'}))
        reason = {'tool_event': 'invented_legacy_event', 'candidate_order': 'visible_original_sentence',
                  'labels': 'packing_identity'}[change]
        with pytest.raises(ValueError, match=reason):
            legacy_envelopes((json.dumps(bad, ensure_ascii=False)+'\n').encode(),
                             'old48', [claim.claim_id], corpus, tokenizer, scope='synthetic_fixture')


def test_NEI_vs_context_origins_are_not_interchangeable():
    row, corpus, tokenizer, receipt = nei_fixture()
    artifact = program_candidates(*arguments(row, corpus, tokenizer, receipt))
    raw = (json.dumps(artifact, ensure_ascii=False) + '\n').encode()
    inventory = nei_inventory(encoded({'private_file_sha256': {'7.json': sha(raw)}}), scope='synthetic_fixture')
    envelope = nei_envelope(raw, expected_sha=sha(raw), component='fixture', inventory=inventory,
        artifact_name='7.json', corpus=corpus, tokenizer=tokenizer, scope='synthetic_fixture')
    assert derived_row(envelope, 144)['target_origin'] == 'official_annotation_NEI'
    changed = copy.deepcopy(envelope)
    changed['target_origin'] = 'context_insufficient'
    with pytest.raises(ValueError, match='NEI_origin'):
        derived_row(changed, 144)
    context = [e for e in synthetic_rows(144) if e['target_origin'] == 'context_insufficient']
    assert len(context) == 5
    for old in context:
        old['target_origin'] = 'official_annotation_NEI'
        with pytest.raises(ValueError, match='legacy_origin'):
            derived_row(old, 144)


def test_144_claims_223_rows_mass_and_36_update_plan():
    prepared = packed(synthetic_rows(144))
    data = validate_inputs(prepared)
    check_released_shape(data['records'], data['roster']['payload']['cohorts'])
    assert data['actual_claim_count'] == 144 and len(data['records']) == 223
    assert sum(Fraction(r['weight_numerator'], r['weight_denominator']) for r in data['records']) == 144
    assert {r['epoch_normalizer'] for r in data['records']} == {144}
    assert {r['source_envelope']['legacy_row']['epoch_normalizer'] for r in data['records']
            if r['source_envelope']['kind'] == 'legacy_program_artifact'} == {48}
    plan = make_plan(prepared)['payload']
    assert plan['optimizer_steps'] == 36
    assert all(len(g['claim_ids']) == 4 for g in plan['groups'])
    assert sorted(i for g in plan['groups'] for i in g['record_indices']) == list(range(223))
    fake_real = copy.deepcopy(data)
    fake_real['scope'] = SCOPE
    with pytest.raises(ValueError):
        validate_inputs(seal(fake_real))  # Counts alone cannot authorize a real package.


def test_arbitrary_N_tail_group_matches_independent_claim_mean_not_rowmean(tmp_path):
    prepared = packed(synthetic_rows())
    plan = make_plan(prepared)
    model = PositionModel(0.2)
    reference = copy.deepcopy(model)
    optimizer = make_claim_group_mean_adamw(reference)
    data = prepared['payload']
    expected_epoch = 0.0
    for group in plan['payload']['groups']:
        rows = [data['records'][i] for i in group['record_indices']]
        tokens = [data['tokenized'][i] for i in group['record_indices']]
        optimizer.zero_grad(set_to_none=True)
        loss = independent_mean(reference, rows, tokens)
        expected_epoch += float(loss.detach()) * len(group['claim_ids']) / 5
        loss.backward()
        independent_clip(reference)
        optimizer.step()
    report = run_synthetic_epoch(prepared, plan, model, tmp_path/'complete')
    assert [r['group_normalizer'] for r in report['group_results']] == [4, 1]
    assert report['epoch_observed_loss'] == pytest.approx(expected_epoch, rel=1e-11)
    for got, expected in zip(model.parameters(), reference.parameters(), strict=True):
        torch.testing.assert_close(got, expected, rtol=1e-11, atol=1e-12)
    assert report['optimizer_steps'] == 2
    # Different numbers of decisions/claim must not silently become a row mean.
    probe = PositionModel(0.2)
    per_row = [independent_mean(probe, [{**r, 'weight_numerator': 1, 'weight_denominator': 1}], [t])
               for r, t in zip(data['records'], data['tokenized'], strict=True)]
    assert not torch.isclose(torch.stack(per_row).mean(), independent_mean(probe, data['records'], data['tokenized']),
                             rtol=1e-8, atol=1e-9)


def test_missing_duplicate_contamination_and_failure_never_publish(tmp_path, monkeypatch):
    prepared = packed(synthetic_rows())
    plan = make_plan(prepared)
    for mutation in ('missing', 'duplicate', 'contamination', 'component', 'plan'):
        data = copy.deepcopy(prepared['payload'])
        schedule = copy.deepcopy(plan)
        if mutation == 'missing':
            data['records'].pop()
        elif mutation == 'duplicate':
            data['records'][-1] = data['records'][0]
        elif mutation in ('contamination', 'component'):
            roster = data['roster']['payload']
            if mutation == 'contamination':
                roster['excluded_claim_ids'].append(data['claim_ids'][0])
            else:
                roster['excluded_components'].append(next(iter(roster['components'].values())))
            data['roster'] = seal(roster)
        else:
            schedule['payload']['groups'][0]['claim_ids'].reverse()
            schedule = seal(schedule['payload'])
        model = PositionModel(0.2)
        with pytest.raises(ValueError):
            run_synthetic_epoch(seal(data), schedule, model, tmp_path/mutation)
        assert model.forward_calls == 0 and not (tmp_path/mutation/'complete.json').exists()
    def fail_save(path):
        raise OSError('synthetic final checkpoint failure')
    with pytest.raises(OSError):
        run_synthetic_epoch(prepared, plan, PositionModel(0.2), tmp_path/'save-failure', publish=fail_save)
    assert not (tmp_path/'save-failure/complete.json').exists()
    failed = json.loads((tmp_path/'save-failure/failed.json').read_text())
    assert failed['counts']['optimizer_step_completed'] == 2 and not failed['optimizer_step_outcome_unknown']
    import run_scifact_mixed_training as entry
    monkeypatch.delenv('SLURM_JOB_ID', raising=False)
    with pytest.raises(ValueError, match='allocated_GPU_required'):
        entry.release_inputs(tmp_path/'does-not-exist', '0'*64)
    execution = tmp_path/'worker-once'
    execution.mkdir()
    (execution/'reserved.json').write_text(json.dumps({'release_sha256': 'a'*64, 'parent_pid': os.getppid()}))
    monkeypatch.setenv('CLIMATE_MIXED_PARENT_PID', str(os.getppid()))
    with pytest.raises(ValueError, match='bounded_parent_required'):
        entry.claim_worker(execution, 'b'*64)
    entry.claim_worker(execution, 'a'*64)
    with pytest.raises(FileExistsError):
        entry.claim_worker(execution, 'a'*64)
    orphan = tmp_path/'orphan'
    orphan.mkdir()
    (orphan/'reserved.json').write_text(json.dumps({'release_sha256': 'a'*64, 'parent_pid': os.getppid()}))
    monkeypatch.delenv('CLIMATE_MIXED_PARENT_PID')
    with pytest.raises(ValueError, match='bounded_parent_required'):
        entry.claim_worker(orphan, 'a'*64)
    assert not (orphan/'worker-started.json').exists()
    actual_child = tmp_path/'actual-cpu-child-timeout'
    actual_child.mkdir()
    with pytest.raises(subprocess.TimeoutExpired):
        entry.bounded_worker([sys.executable, '-c', 'import time; time.sleep(5)'], actual_child, 0)
    actual_receipt = json.loads((actual_child/'worker-exit.json').read_text())
    assert actual_receipt['child_started'] and actual_receipt['child_reaped']
    assert type(actual_receipt['returncode']) is int
    for exception in (subprocess.TimeoutExpired('fixture', 1), InterruptedError('fixture signal')):
        calls = []
        class Child:
            def wait(self, timeout=None):
                calls.append(('wait', timeout))
                if timeout is not None:
                    raise exception
                return -9
            def poll(self):
                return None
            def kill(self):
                calls.append(('kill', None))
        monkeypatch.setattr(entry.subprocess, 'Popen', lambda *args, **kwargs: Child())
        directory = tmp_path/type(exception).__name__
        directory.mkdir()
        with pytest.raises(type(exception)):
            entry.bounded_worker(['synthetic-no-launch'], directory, 1)
        receipt = json.loads((directory/'worker-exit.json').read_text())
        assert receipt['child_reaped'] and receipt['returncode'] == -9
        assert calls == [('wait', 1), ('kill', None), ('wait', None)]
    calls.clear()
    def signal_during_creation(*args, **kwargs):
        signal.getsignal(signal.SIGTERM)(signal.SIGTERM, None)
        return Child()
    monkeypatch.setattr(entry.subprocess, 'Popen', signal_during_creation)
    directory = tmp_path/'signal-during-creation'
    directory.mkdir()
    with pytest.raises(InterruptedError):
        entry.bounded_worker(['synthetic-no-launch'], directory, 1)
    receipt = json.loads((directory/'worker-exit.json').read_text())
    assert receipt['child_started'] and receipt['child_reaped']
    assert calls == [('kill', None), ('wait', None)]
