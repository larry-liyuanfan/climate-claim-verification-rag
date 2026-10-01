"""Six synthetic CPU groups; no dataset, trained model, remote call or release."""
import copy
import json
import math
from types import SimpleNamespace

import pytest
import torch

from climate_rag import scifact_state_training_driver as driver
from climate_rag.scifact_bounded_runtime import ARMS, run_bounded_slot
from climate_rag.scifact_grounding import Abstract
from climate_rag.scifact_semantic_contract import encoded, sha
from climate_rag.scifact_state_supervision import packing_metadata, tokenize_target, validate_weights
from climate_rag.scifact_terminal import source_from_abstract, render_scifact_prompt
from climate_rag.scifact_utility_contract import UtilityDiagnostic, identity
from test_bounded_scifact_runtime import SyntheticBackend
from test_scifact_claim_group_mean import independent_mean, independent_clip
from test_scifact_state_supervision import Tokenizer


class TinyModel(torch.nn.Module):
    def __init__(self, scale=0.2, fail_at=None, failure=None):
        super().__init__()
        self.table = torch.nn.Parameter(torch.sin(torch.arange(130 * 130, dtype=torch.float64)).reshape(130, 130) * 0.08)
        self.forward_calls = 0
        self.scale, self.fail_at, self.failure = scale, fail_at, failure

    def forward(self, input_ids, attention_mask):
        self.forward_calls += 1
        logits = self.table[input_ids] * self.scale
        if self.forward_calls == self.fail_at:
            if self.failure == 'forward':
                raise RuntimeError('synthetic failure')
            if self.failure == 'loss':
                logits = logits * float('nan')
            if self.failure == 'gradient':
                self.table.register_hook(lambda grad: grad * float('nan'))
        return SimpleNamespace(logits=logits)


@pytest.fixture
def captured():
    before = torch.get_num_threads()
    torch.set_num_threads(1)
    corpus = {i: Abstract(i, f'Doc{i}', (f'Document {i} sentence.', 'Other sentence.'), False) for i in (10, 11)}
    tokenizer = Tokenizer()
    pairs = {}
    read, rerank = {'action': 'read', 'source_ids': ['c1']}, {'action': 'rerank'}
    abstain = {'action': 'abstain', 'reason': 'insufficient_evidence'}
    plans = [('read', [read, abstain]), ('rerank', [rerank, abstain]),
             ('terminal', [read, rerank, {'action': 'invalid'}, {'action': 'invalid'}, abstain])]
    for name, actions in plans:
        frames = []
        backend = SyntheticBackend(actions, False)
        backend.base.tokenizer = tokenizer
        result = run_bounded_slot(1, 'Synthetic claim', 'adaptive', ARMS[0], backend,
            lambda q, k: [source_from_abstract(corpus[i]) for i in corpus], lambda q, c: list(reversed(c)), corpus,
            diagnostic=UtilityDiagnostic(lambda kind, frame: frames.append(frame) if kind == 'frame' else None))['result']
        assert len(frames) == len(actions) and result['events'][-1]['status'] == 'completed'
        pairs[name] = (frames[-1], result['events'][-1])
    yield corpus, tokenizer, pairs
    torch.set_num_threads(before)


def seal_rows(rows):
    for row in rows:
        row['record_sha256'] = sha(encoded({k: v for k, v in row.items() if k != 'record_sha256'}))


def roster_for(rows, ids, provenance):
    payload = {'version': driver.ROSTER_VERSION, 'claim_ids': ids, 'claim_count': len(ids),
               'record_sha256': [r['record_sha256'] for r in rows], 'provenance_sha256': identity(provenance)}
    return {'payload': payload, 'sha256': identity(payload)}


def inputs(captured, n=5, counts=None):
    corpus, tokenizer, pairs = captured
    provenance = {'version': driver.VERSION, 'scope': 'synthetic_fixture',
        'corpus_sha256': driver.corpus_identity(corpus), 'tokenizer_sha256': identity('synthetic Tokenizer v1'),
        'capture_run_sha256': identity(pairs), 'supervision_selection_sha256': identity('synthetic selection')}
    ids, rows = list(range(100, 100 + n)), []
    for index, claim in enumerate(ids):
        count = (counts or [1 + i % 4 for i in range(n)])[index]
        capture, event = pairs['rerank']
        alternatives = count if index % 3 in (1, 2) else 2 if count == 4 else 1
        states = count // alternatives
        frame, sources, registry, order = driver.captured_state(capture, event, corpus)
        for record_index in range(count):
            alternative, state = divmod(record_index, states)
            alias = 'c1' if index % 2 == 0 else 'c0'
            label = 'REFUTES' if index % 3 == 1 else 'SUPPORTS'
            sentences = [f'{alias}:{alternative % 2}'] if index % 2 else [f'{alias}:0', f'{alias}:1']
            target = {'action': 'answer', 'documents': [{'source_id': alias, 'label': label, 'sentence_ids': sentences}]}
            tokens = tokenize_target(tokenizer, frame, target, sources, alias_to_source=registry, candidate_aliases=order)
            rows.append({'version': driver.VERSION, 'claim_id': claim, 'trajectory': 0, 'trajectory_count': 1,
                'alternative': alternative, 'alternative_count': alternatives, 'state_index': state, 'state_count': states,
                'weight_numerator': 1, 'weight_denominator': count, 'declared_claim_weight': 1,
                'epoch_normalizer': n, 'model_generated': False, 'decision_origin': 'externally_frozen_supervision',
                'action_target_provenance': 'frozen_captured_state_v1',
                'semantic_target_provenance': 'externally_verified_visible_rationale',
                'target': copy.deepcopy(target), 'packing': packing_metadata(tokens), 'status': 'ready', 'gap': None,
                'capture': copy.deepcopy(capture), 'tool_event': copy.deepcopy(event),
                'capture_sha256': identity(capture), 'tool_event_sha256': identity(event),
                'provenance_sha256': identity(provenance)})
    seal_rows(rows)
    return rows, roster_for(rows, ids, provenance), provenance


def prepare(captured, n=5, counts=None):
    rows, roster, provenance = inputs(captured, n, counts)
    return driver.prepare_training_inputs(rows, roster, captured[1], captured[0], provenance)


def test_real_controller_read_and_rerank_alias_prompt_mask_roundtrip(captured):
    corpus, tokenizer, pairs = captured
    for name, (capture, event) in pairs.items():
        original = copy.deepcopy(capture)
        frame, sources, registry, order = driver.captured_state(json.loads(json.dumps(capture)), event, corpus)
        assert registry == {'c0': '10', 'c1': '11'}
        if name != 'terminal':
            assert capture['observation']['feedback'].startswith('tool_completed')
        assert capture['visible']['c1:0']['source_id'] == '11'
        target = {'action': 'answer', 'documents': [{'source_id': 'c1', 'label': 'SUPPORTS', 'sentence_ids': ['c1:0']}]}
        tokens = tokenize_target(tokenizer, frame, target, sources, alias_to_source=registry, candidate_aliases=order)
        raw = json.dumps(target, ensure_ascii=False, separators=(',', ':'))
        prompt = render_scifact_prompt(tokenizer, capture['observation'], capture['schema'])
        assert tokens['input_ids'] == tokenizer.encode(prompt + raw + tokenizer.eos_token)
        assert tokens['labels'] == [-100] * tokens['input_tokens'] + tokens['input_ids'][tokens['input_tokens']:]
        assert capture == original and tokens['prompt_sha256'] == sha(prompt.encode())
    capture, event = pairs['terminal']
    assert 'read' not in capture['observation']['allowed_actions']
    assert event['candidate_ids'] == ['c1', 'c0']
    for fault in ('registry', 'order', 'wrong_event'):
        bad = copy.deepcopy(capture)
        bad_event = copy.deepcopy(event)
        if fault == 'registry':
            bad['alias_to_source'] = {'c0': '11', 'c1': '10'}
        elif fault == 'order':
            bad['candidate_aliases'] = ['c0', 'c1']
        else:
            bad_event['candidate_ids'] = bad_event['requested_context'] = ['c0', 'c1']
        with pytest.raises(ValueError, match='visible_selected_order' if fault == 'wrong_event' else None):
            driver.captured_state(bad, bad_event, corpus)


def test_claim_means_manual_gradients_adam_and_state_count_invariance(captured, tmp_path, monkeypatch):
    prepared = prepare(captured)
    plan = driver.make_epoch_plan(prepared)
    for scale in (0.2, 20.0):
        model, reference = TinyModel(scale), TinyModel(scale)
        optimizer = driver.make_claim_group_mean_adamw(reference)
        expected_grads, expected_losses = [], []
        for group in plan['payload']['groups']:
            rows = [prepared['payload']['records'][i] for i in group['record_indices']]
            tokens = [prepared['payload']['tokenized'][i] for i in group['record_indices']]
            optimizer.zero_grad(set_to_none=True)
            loss = independent_mean(reference, rows, tokens)
            loss.backward()
            expected_grads.append([p.grad.detach().clone() for p in reference.parameters()])
            expected_losses.append(float(loss.detach()))
            if len(group['claim_ids']) > 1:
                wrong_model = copy.deepcopy(reference)
                wrong_model.zero_grad(set_to_none=True)
                wrong_rows = [dict(r, claim_id=0, weight_numerator=1, weight_denominator=len(rows)) for r in rows]
                wrong_loss = independent_mean(wrong_model, wrong_rows, tokens)
                wrong_loss.backward()
                assert float(wrong_loss.detach()) != pytest.approx(float(loss.detach()), abs=1e-10)
                assert any(not torch.allclose(p.grad, q.grad, rtol=1e-8, atol=1e-10)
                           for p, q in zip(reference.parameters(), wrong_model.parameters(), strict=True))
            independent_clip(reference)
            optimizer.step()
        actual_grads, actual_optimizer = [], []
        original_clip, factory = torch.nn.utils.clip_grad_norm_, driver.make_claim_group_mean_adamw
        def clip(parameters, *args, **kwargs):
            parameters = list(parameters)
            actual_grads.append([p.grad.detach().clone() for p in parameters])
            return original_clip(parameters, *args, **kwargs)
        def make(model):
            result = factory(model)
            actual_optimizer.append(result)
            return result
        with monkeypatch.context() as patch:
            patch.setattr(torch.nn.utils, 'clip_grad_norm_', clip)
            patch.setattr(driver, 'make_claim_group_mean_adamw', make)
            report = driver.run_one_epoch(prepared, plan, model, tmp_path / str(scale))
        assert len(actual_grads) == report['counts']['clip_completed'] == report['counts']['optimizer_step_completed'] == 2
        for actual, expected in zip(actual_grads, expected_grads, strict=True):
            torch.testing.assert_close(actual, expected, rtol=1e-10, atol=1e-11)
        torch.testing.assert_close(list(model.parameters()), list(reference.parameters()), rtol=1e-11, atol=1e-12)
        for actual, expected in zip(model.parameters(), reference.parameters(), strict=True):
            for key in ('exp_avg', 'exp_avg_sq', 'step'):
                torch.testing.assert_close(actual_optimizer[0].state[actual][key], optimizer.state[expected][key])
        assert report['epoch_observed_loss'] == pytest.approx((expected_losses[0] * 4 + expected_losses[1]) / 5)
        assert report['epoch_observed_loss'] != pytest.approx(sum(expected_losses) / 48)
        assert report['epoch_observed_loss'] != pytest.approx(sum(expected_losses) / 2, abs=1e-10)
    one, two = prepare(captured, 1, [1]), prepare(captured, 1, [2])
    first, second = TinyModel(), TinyModel()
    driver.run_one_epoch(one, driver.make_epoch_plan(one), first, tmp_path / 'one')
    driver.run_one_epoch(two, driver.make_epoch_plan(two), second, tmp_path / 'two')
    torch.testing.assert_close(first.table, second.table, rtol=1e-11, atol=1e-12)


def test_non48_roster_seed_whole_claims_and_tail_group(captured):
    for n in (1, 3, 5, 7):
        prepared = prepare(captured, n)
        plan = driver.make_epoch_plan(prepared, seed=23)
        assert plan == driver.make_epoch_plan(prepared, seed=23)
        data = plan['payload']
        assert data['actual_claim_count'] == n and data['planned_optimizer_steps'] == math.ceil(n / 4)
        assert len(data['groups'][-1]['claim_ids']) == (n % 4 or 4)
        assert sorted(data['shuffled_claim_ids']) == prepared['payload']['claim_ids']
        indices = [i for g in data['groups'] for i in g['record_indices']]
        assert sorted(indices) == list(range(len(prepared['payload']['records'])))
        if n >= 5:
            assert data['shuffled_claim_ids'] != driver.make_epoch_plan(prepared, 24)['payload']['shuffled_claim_ids']


def test_all_missing_duplicate_gap_and_inconsistent_hierarchy_rejected_before_forward(captured, tmp_path, monkeypatch):
    model = TinyModel()
    zero_calls = []
    monkeypatch.setattr(torch.optim.AdamW, 'zero_grad', lambda *a, **k: zero_calls.append(1))
    faults = ('missing_state', 'duplicate_state', 'missing_claim', 'zero_rows', 'gap', 'duplicate_roster',
              'trajectory_counts', 'alternative_counts', 'state_counts', 'alias_rebound')
    for fault in faults:
        rows, roster, provenance = inputs(captured, 2, [3, 1])
        if fault == 'missing_state':
            rows.pop(1)
        elif fault == 'duplicate_state':
            rows[1] = copy.deepcopy(rows[0])
        elif fault == 'missing_claim':
            rows.pop()
        elif fault == 'zero_rows':
            rows = []
        elif fault == 'gap':
            rows[-1]['gap'] = 'unavailable state'
        elif fault == 'alias_rebound':
            corpus, tokenizer, _ = captured
            frames = []
            backend = SyntheticBackend([{'action': 'rerank'}, {'action': 'abstain', 'reason': 'insufficient_evidence'}], False)
            backend.base.tokenizer = tokenizer
            result = run_bounded_slot(100, 'Synthetic claim', 'adaptive', ARMS[0], backend,
                lambda q, k: [source_from_abstract(corpus[i]) for i in reversed(corpus)],
                lambda q, c: list(reversed(c)), corpus,
                diagnostic=UtilityDiagnostic(lambda kind, frame: frames.append(frame) if kind == 'frame' else None))['result']
            capture, event = frames[-1], result['events'][-1]
            frame, sources, registry, order = driver.captured_state(capture, event, corpus)  # Locally consistent.
            row = rows[1]
            row.update(capture=capture, tool_event=event, capture_sha256=identity(capture), tool_event_sha256=identity(event))
            row['packing'] = packing_metadata(tokenize_target(tokenizer, frame, row['target'], sources,
                                            alias_to_source=registry, candidate_aliases=order))
        elif fault.endswith('_counts'):
            for i, row in enumerate(rows[:3]):
                row.update(trajectory=0, trajectory_count=1, alternative=0, alternative_count=1,
                           state_index=0, state_count=1, weight_denominator=2 if i == 0 else 4)
                if fault == 'trajectory_counts':
                    row.update(trajectory=0 if i == 0 else 1, trajectory_count=1 if i == 0 else 2,
                               alternative=0 if i == 0 else i-1, alternative_count=2)
                elif fault == 'alternative_counts':
                    row.update(alternative=i, alternative_count=2 if i == 0 else 4)
                else:
                    row.update(state_index=i, state_count=2 if i == 0 else 4)
            validate_weights(rows, [100, 101], contract=driver.weights_for(2))  # Mass=1 alone used to pass.
        seal_rows(rows)
        roster = roster_for(rows, [100, 100] if fault == 'duplicate_roster' else [100, 101], provenance)
        with pytest.raises((ValueError, KeyError)):
            prepared = driver.prepare_training_inputs(rows, roster, captured[1], captured[0], provenance)
            driver.run_one_epoch(prepared, driver.make_epoch_plan(prepared), model, tmp_path / fault)
    assert model.forward_calls == 0 and zero_calls == []


def test_provenance_frame_tokens_mask_tamper_fail_before_any_forward(captured, tmp_path):
    for fault in ('provenance', 'frame', 'packing', 'token', 'mask', 'prepared_hash'):
        rows, roster, provenance = inputs(captured)
        model = TinyModel()
        with pytest.raises(ValueError):
            if fault == 'provenance':
                provenance['capture_run_sha256'] = 'f' * 64
            if fault == 'frame':
                rows[-1]['capture']['observation']['feedback'] = 'tampered'
            if fault == 'packing':
                rows[-1]['packing']['loss_mask_sha256'] = 'f' * 64
            prepared = driver.prepare_training_inputs(rows, roster, captured[1], captured[0], provenance)
            plan = driver.make_epoch_plan(prepared)
            if fault == 'token':
                prepared['payload']['tokenized'][-1]['input_ids'][-1] = 99
            if fault == 'mask':
                prepared['payload']['tokenized'][-1]['labels'][0] = 1
            if fault == 'prepared_hash':
                prepared['sha256'] = 'e' * 64
            driver.run_one_epoch(prepared, plan, model, tmp_path / fault)
        assert model.forward_calls == 0


def test_midrun_faults_cost_receipts_no_retry_and_only_final_success(captured, tmp_path, monkeypatch):
    prepared = prepare(captured)
    plan = driver.make_epoch_plan(prepared)
    fail_at = len(plan['payload']['groups'][0]['record_indices']) + 1
    for fault in ('forward', 'loss', 'gradient', 'step_after_mutation', 'post_step_nan', 'final_write'):
        model = TinyModel(fail_at=fail_at, failure=fault)
        output = tmp_path / fault
        factory, write = driver.make_claim_group_mean_adamw, driver.ordered_write
        def make(model):
            optimizer = factory(model)
            original = optimizer.step
            steps = []
            def step(*args, **kwargs):
                result = original(*args, **kwargs)
                steps.append(1)
                if len(steps) == 2:
                    if fault == 'step_after_mutation':
                        raise RuntimeError('after update')
                    if fault == 'post_step_nan':
                        model.table.data.fill_(float('nan'))
                return result
            optimizer.step = step
            return optimizer
        def store(path, value):
            if fault == 'final_write' and path.name == 'complete.pending.json':
                raise OSError('synthetic write failure')
            return write(path, value)
        with monkeypatch.context() as patch:
            patch.setattr(driver, 'make_claim_group_mean_adamw', make)
            patch.setattr(driver, 'ordered_write', store)
            with pytest.raises((RuntimeError, ValueError, OSError)):
                driver.run_one_epoch(prepared, plan, model, output)
        failure = json.loads((output / 'failed.json').read_bytes())
        assert not (output / 'complete.json').exists() and failure['automatic_retry'] is False
        assert failure['completed_groups'] == (2 if fault == 'final_write' else 1)
        assert failure['known_completed_optimizer_steps'] == (2 if fault in {'post_step_nan', 'final_write'} else 1)
        assert failure['optimizer_step_outcome_unknown'] == (fault == 'step_after_mutation')
        previous = model.forward_calls
        with pytest.raises(FileExistsError):
            driver.run_one_epoch(prepared, plan, model, output)
        assert model.forward_calls == previous
