"""A new, separately released epoch entry; the old synthetic/192-row guards stay.

No model/data loader here. The production CLI must verify a physical release
before calling the internal kernel. Public fixture runner remains CPU <=1M.
"""
from __future__ import annotations

from collections.abc import Callable, Mapping
import os
from pathlib import Path
import time
from typing import Any

from .scifact_claim_group_mean import claim_group_mean_update, make_claim_group_mean_adamw
from .scifact_mixed_inputs import VERSION as INPUT_VERSION, make_plan, unseal, validate_inputs, weights_for
from .scifact_read_continuation import ordered_write
from .scifact_semantic_contract import MODEL_SHA, TOKENIZER_SHA, encoded, sha
from .scifact_state_supervision import require

VERSION = 'scifact-mixed-claim-mean-train-v1-20261001'
CONFIG: dict[str, Any] = {
    'version': VERSION, 'input_version': INPUT_VERSION, 'model_sha256': MODEL_SHA,
    'tokenizer_sha256': TOKENIZER_SHA, 'actual_claims': 144, 'decision_records': 223,
    'epochs': 1, 'claims_per_update': 4, 'optimizer_updates': 36, 'seed': 20261001,
    'r': 8, 'alpha': 16, 'dropout': 0.0, 'target_modules': ['q_proj', 'v_proj'],
    'base_dtype': 'bfloat16', 'adapter_dtype': 'record_actual_not_assumed_bfloat16',
    'gradient_checkpointing_use_reentrant': False, 'learning_rate': 1e-4,
    'optimizer': 'explicit_claim_group_mean_adamw', 'clip_norm': 1.0,
    'checkpoint': 'final_only', 'resume': False, 'automatic_retry': False,
    'first_update_pilot_counted_in_36': True, 'max_input_tokens': 8192, 'max_target_tokens': 512,
}


def config_sha() -> str:
    return sha(encoded(CONFIG))


def finite_trainable(model: Any, optimizer: Any) -> None:
    import torch
    values = [p for p in model.parameters() if p.requires_grad]
    values += [v for state in optimizer.state.values() for v in state.values() if torch.is_tensor(v)]
    require(bool(values) and all(bool(torch.isfinite(v).all()) for v in values), 'nonfinite_trainable_or_optimizer')


def _run_epoch(prepared: Mapping[str, Any], plan: Mapping[str, Any], model: Any, output: Path, *,
               device: Any, publish: Callable[[Path], dict[str, Any]] | None = None,
               first_update: Callable[[float], dict[str, Any]] | None = None) -> dict[str, Any]:
    data = validate_inputs(prepared)  # Including tail records, before creating optimizer.
    require(dict(plan) == make_plan(prepared), 'mixed_plan_drift')
    schedule = unseal(plan)
    output.mkdir(parents=True, exist_ok=False, mode=0o700)
    counts: dict[str, int] = {}
    completed: list[dict[str, Any]] = []
    current: int | None = None
    def observe(event: str) -> None:
        counts[event] = counts.get(event, 0) + 1
        if event.startswith('optimizer_step'):
            ordered_write(output/f'{current:03d}-{event}.json', {'group': current, 'counts': dict(counts)})
    ordered_write(output/'started.json', {'version': VERSION, 'config_sha256': config_sha(),
        'prepared_sha256': prepared['sha256'], 'plan': dict(plan), 'automatic_retry': False})
    started = time.perf_counter()
    try:
        optimizer = make_claim_group_mean_adamw(model)
        finite_trainable(model, optimizer)
        model.train()
        seen: list[int] = []
        for current, group in enumerate(schedule['groups']):
            ordered_write(output/f'{current:03d}-reserved.json', group)
            indices = group['record_indices']
            result = claim_group_mean_update(model, optimizer, [data['records'][i] for i in indices],
                [data['tokenized'][i] for i in indices], group['claim_ids'], device=device,
                weight_contract=weights_for(data['actual_claim_count']), observe=observe)
            finite_trainable(model, optimizer)
            result['actual_epoch_N'] = data['actual_claim_count']
            result['epoch_metric_contribution'] = result['group_mean_loss'] * result['claim_count'] / data['actual_claim_count']
            ordered_write(output/f'{current:03d}-completed.json', result)
            completed.append(result)
            seen.extend(indices)
            if current == 0 and first_update is not None:
                ordered_write(output/'first-step-pilot.json', {**first_update(started),
                    'optimizer_steps': 1, 'counted_in_full_epoch': True, 'optimizer_reinitialized': False,
                    'remaining_optimizer_steps': schedule['optimizer_steps']-1})
        require(sorted(seen) == list(range(len(data['records'])))
                and counts['optimizer_step_completed'] == len(completed) == schedule['optimizer_steps']
                and sum(r['claim_count'] for r in completed) == data['actual_claim_count'], 'mixed_exact_once_epoch')
        artifacts = publish(output) if publish is not None else {}
        report = {'version': VERSION, 'status': 'complete', 'actual_claims': data['actual_claim_count'],
            'decision_records': len(seen), 'optimizer_steps': len(completed), 'counts': counts,
            'epoch_observed_loss': sum(r['epoch_metric_contribution'] for r in completed),
            'metric_scope': 'observed_training_path_not_final_model_quality', 'group_results': completed,
            'plan_sha256': plan['sha256'], 'prepared_sha256': prepared['sha256'], 'artifacts': artifacts,
            'elapsed_seconds': time.perf_counter()-started, 'automatic_retry': False}
        ordered_write(output/'complete.pending.json', report)
        os.link(output/'complete.pending.json', output/'complete.json')
        return report
    except BaseException as exc:
        ordered_write(output/'failed.json', {'version': VERSION, 'exception_type': type(exc).__name__,
            'group': current, 'counts': counts, 'completed_groups': len(completed),
            'optimizer_step_outcome_unknown': counts.get('optimizer_step_started', 0) > counts.get('optimizer_step_completed', 0),
            'automatic_retry': False, 'resume_authorized': False})
        raise


def run_synthetic_epoch(prepared: Mapping[str, Any], plan: Mapping[str, Any], model: Any, output: Path,
                        *, publish: Callable[[Path], dict[str, Any]] | None = None) -> dict[str, Any]:
    data = validate_inputs(prepared)
    params = list(model.parameters()) + list(model.buffers())
    require(data['scope'] == 'synthetic_fixture' and bool(params)
            and all(p.device.type == 'cpu' for p in params)
            and sum(p.numel() for p in model.parameters()) <= 1_000_000, 'cpu_tiny_synthetic_only')
    return _run_epoch(prepared, plan, model, output, device='cpu', publish=publish)
