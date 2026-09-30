"""Synthetic varying-position logits and real AdamW, no corpus or model weights."""
from __future__ import annotations

from collections import defaultdict
import copy
from types import SimpleNamespace
from typing import Any

import pytest
import torch

from climate_rag.scifact_claim_group_mean import (
    CONTRACT, make_claim_group_mean_adamw, optimizer_step_claim_group_mean,
)
from climate_rag.scifact_semantic_contract import encoded, sha
from climate_rag.scifact_state_supervision import CONFIG, VERSION, packing_metadata


class PositionModel(torch.nn.Module):  # type: ignore[misc]  # Torch is untyped in lint-only env.
    def __init__(self, scale: float) -> None:
        super().__init__()
        # Distinct per-input and per-position values; no shared constant logits.
        self.tokens = torch.nn.Parameter(torch.linspace(-0.15, 0.17, 49, dtype=torch.float64).reshape(7, 7))
        self.positions = torch.nn.Parameter(torch.cos(torch.arange(84, dtype=torch.float64)).reshape(12, 7)*0.07)
        self.scale = scale
        self.forward_calls = 0

    def forward(self, input_ids: Any, attention_mask: Any) -> Any:
        self.forward_calls += 1
        return SimpleNamespace(logits=(self.tokens[input_ids]+self.positions[:input_ids.shape[1]])*self.scale)


def fixtures(claim_count: int, variant: int = 0) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[int]]:
    rows, tokenized = [], []
    ids = [i+variant*1000 for i in [101, 205, 309, 413][:claim_count]]
    # 1, 2, 3, 4 records per claim, including two-state trajectories.
    for ordinal, claim in enumerate(ids):
        states = 2 if ordinal in (1, 3) else 1
        alternatives = (1, 1, 3, 2)[ordinal]
        for alt in range(alternatives):
            for state in range(states):
                prompt = [0, (ordinal+alt+2) % 7] + ([1] if state else [])
                # Assistant lengths differ within and across claims; token 1 is EOS and PAD.
                answer = [(j+ordinal+alt+variant+3) % 7 for j in range(1+(ordinal+alt+state+variant) % 4)] + [1]
                inputs, labels = prompt+answer, [-100]*len(prompt)+answer
                tokens = {'input_ids': inputs, 'labels': labels, 'attention_mask': [1]*len(inputs),
                          'input_tokens': len(prompt), 'target_tokens': len(answer),
                          'full_token_ids_sha256': sha(encoded(inputs)), 'loss_mask_sha256': sha(encoded(labels))}
                row = {'version': VERSION, 'claim_id': claim, 'trajectory': 0, 'trajectory_count': 1,
                       'alternative': alt, 'alternative_count': alternatives, 'state_index': state,
                       'state_count': states, 'weight_numerator': 1, 'weight_denominator': states*alternatives,
                       'declared_claim_weight': 1, 'epoch_normalizer': 48, 'model_generated': False,
                       'decision_origin': 'program_teacher', 'action_target_provenance': 'program_teacher_actual_state_v2',
                       'target': {'action': 'read'}, 'semantic_target_provenance': 'no_semantic_label',
                       'packing': packing_metadata(tokens)}
                row['record_sha256'] = sha(encoded(row))
                rows.append(row)
                tokenized.append(tokens)
    return rows, tokenized, ids


def independent_mean(model: Any, rows: list[dict[str, Any]], tokenized: list[dict[str, Any]]) -> Any:
    """Reference uses gather/log-softmax, not helper's CE or normalization code."""
    by_claim: dict[int, list[Any]] = defaultdict(list)
    for row, tokens in zip(rows, tokenized, strict=True):
        inputs = torch.tensor([tokens['input_ids']])
        logits = model(input_ids=inputs, attention_mask=torch.ones_like(inputs)).logits
        # Select every assistant prediction position explicitly (including EOS).
        positions = torch.arange(tokens['input_tokens']-1, len(tokens['input_ids'])-1)
        targets = inputs[0, tokens['input_tokens']:]
        losses = -torch.log_softmax(logits[0, positions, :], dim=-1).gather(1, targets[:, None]).squeeze(1)
        by_claim[row['claim_id']].append(losses.mean()*row['weight_numerator']/row['weight_denominator'])
    return torch.stack([torch.stack(losses).sum() for losses in by_claim.values()]).mean()


def independent_clip(model: Any) -> Any:
    """Global L2 reference, with Torch's clip epsilon; no clip helper call."""
    parameters = list(model.parameters())
    norm = torch.stack([p.grad.square().sum() for p in parameters]).sum().sqrt()
    coefficient = torch.clamp(1.0/(norm+1e-6), max=1.0)
    for parameter in parameters:
        parameter.grad.mul_(coefficient)
    return norm


@pytest.mark.parametrize('claim_count', [1, 3, 4])
@pytest.mark.parametrize('scale,clipped', [(0.2, False), (20.0, True)])
def test_preclip_gradients_and_two_adamw_updates_match_independent_mean(
    claim_count: int, scale: float, clipped: bool, monkeypatch: pytest.MonkeyPatch,
) -> None:
    model = PositionModel(scale)
    reference = copy.deepcopy(model)
    optimizer = make_claim_group_mean_adamw(model)
    other = torch.optim.AdamW(reference.parameters(), lr=1e-4, betas=(0.9, 0.999), eps=1e-8,
                              weight_decay=0.0, amsgrad=False, maximize=False, foreach=False,
                              fused=False, capturable=False, differentiable=False)
    original_clip = torch.nn.utils.clip_grad_norm_
    captured: list[list[Any]] = []
    after_clip: list[list[Any]] = []
    steps: list[int] = []
    zeros: list[int] = []
    original_step, original_zero = optimizer.step, optimizer.zero_grad

    def capture_clip(parameters: Any, *args: Any, **kwargs: Any) -> Any:
        params = list(parameters)
        captured.append([p.grad.detach().clone() for p in params])
        norm = original_clip(params, *args, **kwargs)
        after_clip.append([p.grad.detach().clone() for p in params])
        return norm

    def step(*args: Any, **kwargs: Any) -> Any:
        steps.append(1)
        return original_step(*args, **kwargs)

    def zero(*args: Any, **kwargs: Any) -> Any:
        zeros.append(1)
        return original_zero(*args, **kwargs)

    monkeypatch.setattr(torch.nn.utils, 'clip_grad_norm_', capture_clip)
    monkeypatch.setattr(optimizer, 'step', step)
    monkeypatch.setattr(optimizer, 'zero_grad', zero)
    for iteration in range(2):
        # Second synthetic group changes IDs, assistant lengths and targets;
        # AdamW moments persist. This is not an epoch over production claims.
        rows, tokens, ids = fixtures(claim_count, variant=iteration)
        other.zero_grad(set_to_none=True)
        expected_loss = independent_mean(reference, rows, tokens)
        expected_loss.backward()
        expected_gradients = [p.grad.detach().clone() for p in reference.parameters()]
        expected_norm = independent_clip(reference)
        other.step()
        result = optimizer_step_claim_group_mean(model, optimizer, rows, tokens, ids)
        assert len(captured) == len(steps) == len(zeros) == iteration+1
        assert (float(expected_norm) > 1) is clipped
        for actual, expected in zip(captured[-1], expected_gradients, strict=True):
            torch.testing.assert_close(actual, expected, rtol=1e-11, atol=1e-12)
        for actual, expected in zip(after_clip[-1], reference.parameters(), strict=True):
            torch.testing.assert_close(actual, expected.grad, rtol=1e-11, atol=1e-12)
        for actual, expected in zip(model.parameters(), reference.parameters(), strict=True):
            torch.testing.assert_close(actual, expected, rtol=1e-11, atol=1e-12)
            for name in ('exp_avg', 'exp_avg_sq', 'step'):
                torch.testing.assert_close(optimizer.state[actual][name], other.state[expected][name], rtol=1e-11, atol=1e-12)
        assert result['group_mean_loss'] == pytest.approx(float(expected_loss.detach()), rel=1e-12)
        assert result['epoch_metric_contribution'] == pytest.approx(float(expected_loss.detach())*claim_count/48)
        assert result['group_normalizer'] == claim_count
        assert result['gradient_norm_before_clip'] == pytest.approx(float(expected_norm), rel=1e-11)


def test_duplicate_identical_state_with_split_weight_preserves_gradient(monkeypatch: pytest.MonkeyPatch) -> None:
    model = PositionModel(0.2)
    expanded = copy.deepcopy(model)
    rows, tokens, ids = fixtures(2)
    alternative = copy.deepcopy(rows[0])
    alternative['alternative_count'], alternative['weight_denominator'] = 2, 2
    second = copy.deepcopy(alternative)
    second['alternative'] = 1
    for row in (alternative, second):
        row['record_sha256'] = sha(encoded({k: v for k, v in row.items() if k != 'record_sha256'}))
    captures: list[list[Any]] = []
    original_clip = torch.nn.utils.clip_grad_norm_
    def clip(parameters: Any, *args: Any, **kwargs: Any) -> Any:
        params = list(parameters)
        captures.append([p.grad.detach().clone() for p in params])
        return original_clip(params, *args, **kwargs)
    monkeypatch.setattr(torch.nn.utils, 'clip_grad_norm_', clip)
    first = optimizer_step_claim_group_mean(model, make_claim_group_mean_adamw(model), rows, tokens, ids)
    other = optimizer_step_claim_group_mean(expanded, make_claim_group_mean_adamw(expanded),
        [alternative, second]+rows[1:], [tokens[0], tokens[0]]+tokens[1:], ids)
    assert first['group_mean_loss'] == pytest.approx(other['group_mean_loss'], rel=1e-12)
    for original, duplicate in zip(captures[0], captures[1], strict=True):
        torch.testing.assert_close(original, duplicate, rtol=1e-11, atol=1e-12)
    for original, duplicate in zip(model.parameters(), expanded.parameters(), strict=True):
        torch.testing.assert_close(original, duplicate, rtol=1e-11, atol=1e-12)


def test_nonfinite_gradient_never_steps() -> None:
    model = PositionModel(1)
    optimizer = make_claim_group_mean_adamw(model)
    before = [p.detach().clone() for p in model.parameters()]
    model.tokens.register_hook(lambda gradient: torch.full_like(gradient, float('nan')))
    rows, tokens, ids = fixtures(1)
    with pytest.raises(RuntimeError, match='non-finite'):
        optimizer_step_claim_group_mean(model, optimizer, rows, tokens, ids)
    assert not optimizer.state
    for param, initial in zip(model.parameters(), before, strict=True):
        torch.testing.assert_close(param, initial)


@pytest.mark.parametrize('failure', ['missing_state', 'extra_id', 'duplicate_id', 'too_many', 'late_bad_tokens',
                                     'optimizer_lr', 'optimizer_decay', 'optimizer_epsilon'])
def test_reject_before_forward_or_optimizer_mutation(failure: str) -> None:
    model = PositionModel(1)
    optimizer = make_claim_group_mean_adamw(model)
    rows, tokens, ids = fixtures(4)
    if failure == 'missing_state':
        rows, tokens = rows[:-1], tokens[:-1]
    elif failure == 'extra_id':
        ids[-1] = 999
    elif failure == 'duplicate_id':
        ids[-1] = ids[0]
    elif failure == 'too_many':
        ids.append(999)
    elif failure == 'late_bad_tokens':
        tokens[-1]['labels'][0] = 1
    else:
        name = {'optimizer_lr': 'lr', 'optimizer_decay': 'weight_decay', 'optimizer_epsilon': 'eps'}[failure]
        optimizer.param_groups[0][name] = 0.3
    for param in model.parameters():
        param.grad = torch.full_like(param, 0.123)
    before = [p.detach().clone() for p in model.parameters()]
    with pytest.raises(ValueError):
        optimizer_step_claim_group_mean(model, optimizer, rows, tokens, ids)
    assert model.forward_calls == 0 and not optimizer.state
    for param, initial in zip(model.parameters(), before, strict=True):
        torch.testing.assert_close(param, initial)
        torch.testing.assert_close(param.grad, torch.full_like(param, 0.123))


def test_new_contract_explicit_and_historical_denominator_unchanged() -> None:
    assert CONFIG['epoch_normalizer'] == 48
    assert CONFIG['loss'] == 'causal_assistant_token_mean*record_weight/48_no_chunk_renormalization'
    assert CONTRACT['learning_rate'] == 1e-4 and CONTRACT['weight_decay'] == 0
    assert CONTRACT['betas'] == [0.9, 0.999] and CONTRACT['epsilon'] == 1e-8
    assert CONTRACT['fit_claims'] == 48 and CONTRACT['planned_optimizer_steps'] == 12 and CONTRACT['epochs'] == 1
    assert CONTRACT['training_authorized'] is False and CONTRACT['automatic_lr_or_epoch_rescaling'] is False
