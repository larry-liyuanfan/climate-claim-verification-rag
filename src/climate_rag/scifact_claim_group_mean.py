"""Prospective claim-group-mean optimizer contract, not a released trainer.

Historical state-supervision v2 (/48 at every step) stays unchanged. This new
interface reuses its validated record weights, not its optimizer denominator.
No model loading, dataset access, epoch runner, scheduler or job submission.
"""
from __future__ import annotations

from collections.abc import Mapping, Sequence
from fractions import Fraction
import math
from typing import Any

from .scifact_semantic_contract import encoded, sha
from .scifact_state_supervision import (
    VERSION as RECORD_VERSION,
    assistant_mean_loss, require, validate_tokenized, validate_weights,
)

VERSION = 'scifact-claim-group-mean-optimizer-v1-20261001'
CONTRACT: dict[str, Any] = {
    'version': VERSION, 'record_schema': RECORD_VERSION,
    'training_authorized': False, 'data_protocol_pending': True,
    'fit_claims': 48,
    'original_fit_ids_sha256': '22fab53c7ab0be1c1e3d53bb2c00a19300d72334696eb8cf2ecf87c2f3ff739c',
    'shuffle': 'original_48_claims_once_then_complete_claim_groups', 'seed': 20261001,
    'epochs': 1, 'claims_per_group': 4, 'planned_optimizer_steps': 12,
    'group_loss': 'sum(record_weight*causal_assistant_token_mean)/number_of_complete_group_claims',
    'epoch_metric_only_denominator': 48,
    'optimizer': 'AdamW', 'learning_rate': 1e-4, 'weight_decay': 0.0,
    'betas': [0.9, 0.999], 'epsilon': 1e-8, 'amsgrad': False, 'maximize': False,
    'foreach': False, 'fused': False, 'capturable': False, 'differentiable': False,
    'clip_max_norm': 1.0, 'clip_norm_type': 2.0,
    'clip_and_step_once_per_complete_group': True, 'automatic_lr_or_epoch_rescaling': False,
}


def contract_sha() -> str:
    return sha(encoded(CONTRACT))


def make_claim_group_mean_adamw(model: Any) -> Any:
    """Make the explicit optimizer only; does not execute or authorize training."""
    import torch
    parameters = [p for p in model.parameters() if p.requires_grad]
    require(bool(parameters), 'no_trainable_parameters')
    return torch.optim.AdamW(parameters, lr=1e-4, betas=(0.9, 0.999), eps=1e-8,
                             weight_decay=0.0, amsgrad=False, maximize=False,
                             foreach=False, fused=False, capturable=False, differentiable=False)


def validate_optimizer(model: Any, optimizer: Any) -> list[Any]:
    import torch
    require(isinstance(optimizer, torch.optim.AdamW), 'claim_group_requires_adamw')
    expected = {'lr': 1e-4, 'betas': (0.9, 0.999), 'eps': 1e-8, 'weight_decay': 0.0,
                'amsgrad': False, 'maximize': False, 'foreach': False, 'fused': False,
                'capturable': False, 'differentiable': False}
    require(all(all(group.get(k) == v for k, v in expected.items()) for group in optimizer.param_groups),
            'claim_group_optimizer_contract')
    parameters = [p for p in model.parameters() if p.requires_grad]
    owned = [p for group in optimizer.param_groups for p in group['params']]
    require(bool(parameters) and len(owned) == len(parameters)
            and {id(p) for p in owned} == {id(p) for p in parameters}, 'optimizer_parameter_ownership')
    return parameters


def optimizer_step_claim_group_mean(
    model: Any, optimizer: Any, records: Sequence[Mapping[str, Any]],
    tokenized: Sequence[Mapping[str, Any]], group_claim_ids: Sequence[int], device: Any = 'cpu',
) -> dict[str, Any]:
    """One complete group, mean over claims (not records, tokens or all 48).

    Validate every record before any forward/backward/optimizer mutation. Reuse
    v2's complete per-claim weight/provenance checks; its `epoch_normalizer=48`
    field remains historical record metadata and is NOT this step denominator.
    Caller must separately authorize/freeze data, shuffle and epoch boundaries.
    """
    import torch
    ids = list(group_claim_ids)
    require(1 <= len(ids) <= 4 and len(set(ids)) == len(ids)
            and all(type(i) is int for i in ids)
            and len(tokenized) == len(records), 'complete_claim_group_shape')
    validate_weights(records, ids)
    for row, tokens in zip(records, tokenized, strict=True):
        validate_tokenized(row, tokens)
    parameters = validate_optimizer(model, optimizer)
    optimizer.zero_grad(set_to_none=True)
    group_mean = 0.0
    for row, tokens in zip(records, tokenized, strict=True):
        inputs = torch.tensor([tokens['input_ids']], device=device)
        labels = torch.tensor([tokens['labels']], device=device)
        attention = torch.tensor([tokens['attention_mask']], device=device)
        loss = assistant_mean_loss(model(input_ids=inputs, attention_mask=attention).logits, labels)
        coefficient = float(Fraction(row['weight_numerator'], row['weight_denominator'])) / len(ids)
        weighted = loss * coefficient
        group_mean += float(weighted.detach().cpu())
        weighted.backward()
    require(math.isfinite(group_mean), 'nonfinite_group_mean')
    gradient_norm = torch.nn.utils.clip_grad_norm_(parameters, 1.0, norm_type=2.0, error_if_nonfinite=True)
    optimizer.step()
    return {'version': VERSION, 'contract_sha256': contract_sha(), 'group_mean_loss': group_mean,
            'group_normalizer': len(ids), 'epoch_metric_contribution': group_mean*len(ids)/48,
            'epoch_metric_only_denominator': 48, 'claim_count': len(ids), 'decision_records': len(records),
            'gradient_norm_before_clip': float(gradient_norm), 'clip_operations': 1, 'optimizer_steps': 1}
