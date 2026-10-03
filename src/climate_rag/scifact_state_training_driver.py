"""CPU synthetic training seam; no real-data loader, model loader, CLI or release.

Captured observations are never reconstructed. A future real-data/training
release is separate; this runner deliberately accepts only synthetic provenance.
"""
from __future__ import annotations

import copy
from collections.abc import Mapping, Sequence
from dataclasses import asdict
import math
import os
from pathlib import Path
import random
from typing import Any

from .agent_v3 import Source
from .scifact_claim_group_mean import claim_group_mean_update, make_claim_group_mean_adamw
from .scifact_grounding import Abstract
from .scifact_read_continuation import ordered_write
from .scifact_state_supervision import (
    WeightContract, frame_contract, require, tokenize_target, validate_tokenized, validate_weights,
)
from .scifact_terminal import source_from_abstract
from .scifact_utility_contract import identity

VERSION = 'scifact-captured-state-training-input-v1-20261001'
ROSTER_VERSION = 'scifact-external-state-roster-v1'
PLAN_VERSION = 'scifact-complete-claim-epoch-plan-v1'


def weights_for(n: int) -> WeightContract:
    return WeightContract(VERSION, 'externally_frozen_supervision', 'frozen_captured_state_v1',
                          'externally_verified_visible_rationale', n)


def corpus_identity(corpus: Mapping[int, Abstract]) -> str:
    return identity([asdict(corpus[i]) for i in sorted(corpus)])


def captured_state(capture: Mapping[str, Any], tool_event: Mapping[str, Any],
                   corpus: Mapping[int, Abstract]) -> tuple[dict[str, Any], list[Source], dict[str, str], list[str]]:
    """Old utility8 order comes ONLY from its corresponding completed tool event."""
    require(set(capture) <= {'observation', 'schema', 'visible', 'prompt_tokens',
                            'alias_to_source', 'candidate_aliases'}, 'capture_metadata_fields')
    require(tool_event['status'] == 'completed' and tool_event['tool'] in
            {'retrieve', 'read', 'rerank', 'rewrite'}, 'completed_state_event_required')
    registry = dict(capture['alias_to_source'])
    order = list(tool_event['candidate_ids'])
    require('candidate_aliases' not in capture or capture['candidate_aliases'] == order,
            'captured_candidate_order_drift')
    sources = [source_from_abstract(corpus[int(doc)]) for doc in registry.values()]
    by_id = {s.source_id: s for s in sources}
    require(all(type(doc) is str and doc in by_id for doc in registry.values()), 'source_registry_identity')
    frame = copy.deepcopy({k: capture[k] for k in ('observation', 'schema')})
    contract = frame_contract(frame, sources, alias_to_source=registry, candidate_aliases=order)
    selected = tool_event['requested_context']
    require(0 < len(selected) <= 5 and len(selected) == len(set(selected)) and set(selected) <= set(order),
            'selected_context_contract')
    require(tool_event['source_sha256'] == {a: by_id[registry[a]].text_sha256 for a in order},
            'event_registry_source_hash')
    require(contract['visible'] == capture['visible'] and all(s.split(':')[0] in selected for s in contract['visible']),
            'captured_visible_identity')
    visible_order = list(dict.fromkeys(s.split(':')[0] for s in contract['visible']))
    require(visible_order == [a for a in selected if a in visible_order], 'visible_selected_order')
    preview_order = [p['source_id'] for p in frame['observation']['preview_only']]
    expected_previews = [a for a in order if a not in selected]
    require(preview_order == expected_previews[:len(preview_order)], 'preview_candidate_order')
    for entry in frame['observation']['preview_only']:
        alias = entry['source_id']
        source = by_id[registry[alias]]
        require(alias not in selected and entry['preview'] == (source.title + ' ' + source.sentences[0])[:128],
                'original_preview_identity')
    return frame, sources, registry, order


def _hierarchy_complete(records: Sequence[Mapping[str, Any]], ids: Sequence[int]) -> None:
    for claim in ids:
        rows = [r for r in records if r['claim_id'] == claim]
        trajectories = {r['trajectory_count'] for r in rows}
        require(len(trajectories) == 1, 'inconsistent_trajectory_counts')
        require({r['trajectory'] for r in rows} == set(range(next(iter(trajectories)))), 'missing_trajectory')
        require(len({r['capture']['observation']['immutable_claim'] for r in rows}) == 1, 'claim_text_changed')
        for trajectory in {r['trajectory'] for r in rows}:
            current = [r for r in rows if r['trajectory'] == trajectory]
            alternatives = {r['alternative_count'] for r in current}
            require(len(alternatives) == 1, 'inconsistent_alternative_counts')
            require({r['alternative'] for r in current} == set(range(next(iter(alternatives)))), 'missing_alternative')
            for alternative in {r['alternative'] for r in current}:
                states = [r for r in current if r['alternative'] == alternative]
                counts = {r['state_count'] for r in states}
                require(len(counts) == 1 and {r['state_index'] for r in states} == set(range(next(iter(counts)))),
                        'missing_or_inconsistent_states')
                previous: dict[str, str] = {}
                for row in sorted(states, key=lambda r: r['state_index']):
                    registry = row['capture']['alias_to_source']
                    require(all(registry.get(alias) == source for alias, source in previous.items()),
                            'trajectory_alias_rebound_or_dropped')
                    require(len(set(registry.values())) == len(registry), 'trajectory_source_alias_reused')
                    previous = dict(registry)


def _unseal(value: Mapping[str, Any], reason: str) -> dict[str, Any]:
    require(set(value) == {'payload', 'sha256'} and value['sha256'] == identity(value['payload']), reason)
    return copy.deepcopy(dict(value['payload']))


def prepare_training_inputs(records: Sequence[Mapping[str, Any]], roster: Mapping[str, Any],
                            tokenizer: Any, corpus: Mapping[int, Abstract],
                            provenance: Mapping[str, Any]) -> dict[str, Any]:
    """Validate external selection in full; never select targets, drop rows or renormalize."""
    frozen = _unseal(roster, 'external_roster_identity')
    rows = copy.deepcopy(list(records))
    ids = frozen['claim_ids']
    require(frozen['version'] == ROSTER_VERSION and bool(ids) and all(type(i) is int for i in ids)
            and len(set(ids)) == len(ids) and frozen['claim_count'] == len(ids), 'external_claim_roster')
    require(frozen['record_sha256'] == [r['record_sha256'] for r in rows]
            and frozen['provenance_sha256'] == identity(provenance), 'roster_records_or_provenance')
    require(set(provenance) == {'version', 'scope', 'corpus_sha256', 'tokenizer_sha256',
                               'capture_run_sha256', 'supervision_selection_sha256'}
            and provenance['version'] == VERSION
            and provenance['scope'] in {'synthetic_fixture', 'exposed_train_pending_release'}, 'provenance_contract')
    require(all(isinstance(provenance[k], str) and len(provenance[k]) == 64
                and all(c in '0123456789abcdef' for c in provenance[k])
                for k in ('corpus_sha256', 'tokenizer_sha256', 'capture_run_sha256', 'supervision_selection_sha256')),
            'provenance_hash_format')
    require(provenance['corpus_sha256'] == corpus_identity(corpus), 'corpus_identity')
    validate_weights(rows, ids, contract=weights_for(len(ids)))
    _hierarchy_complete(rows, ids)
    tokenized = []
    for row in rows:
        require(row['gap'] is None and row['status'] == 'ready', 'unresolved_state_gap')
        require(row['provenance_sha256'] == identity(provenance)
                and row['capture_sha256'] == identity(row['capture'])
                and row['tool_event_sha256'] == identity(row['tool_event']), 'state_provenance_identity')
        frame, sources, registry, order = captured_state(row['capture'], row['tool_event'], corpus)
        tokens = tokenize_target(tokenizer, frame, row['target'], sources,
                                 alias_to_source=registry, candidate_aliases=order)
        require(row['capture']['prompt_tokens'] == tokens['input_tokens'], 'captured_prompt_count')
        validate_tokenized(row, tokens)
        tokenized.append(tokens)
    payload = {'version': VERSION, 'roster': copy.deepcopy(dict(roster)), 'claim_ids': ids,
               'actual_claim_count': len(ids), 'records': rows, 'tokenized': tokenized,
               'record_sha256': [r['record_sha256'] for r in rows], 'provenance': copy.deepcopy(dict(provenance)),
               'weight_contract': asdict(weights_for(len(ids))), 'real_training_authorized': False}
    return {'payload': payload, 'sha256': identity(payload)}


def _validated_inputs(prepared: Mapping[str, Any]) -> dict[str, Any]:
    data = _unseal(prepared, 'prepared_identity')
    ids, rows = data['claim_ids'], data['records']
    require(data['version'] == VERSION and data['actual_claim_count'] == len(ids)
            and data['weight_contract'] == asdict(weights_for(len(ids))), 'prepared_contract')
    frozen = _unseal(data['roster'], 'external_roster_identity')
    require(frozen['claim_ids'] == ids and frozen['record_sha256'] == data['record_sha256']
            == [r['record_sha256'] for r in rows]
            and frozen['provenance_sha256'] == identity(data['provenance']), 'prepared_roster_binding')
    validate_weights(rows, ids, contract=weights_for(len(ids)))
    _hierarchy_complete(rows, ids)
    require(len(rows) == len(data['tokenized']), 'tokenized_records_length')
    for row, tokens in zip(rows, data['tokenized'], strict=True):
        validate_tokenized(row, tokens)
    return data


def make_epoch_plan(prepared: Mapping[str, Any], seed: int = 20261001) -> dict[str, Any]:
    data = _validated_inputs(prepared)
    require(type(seed) is int, 'shuffle_seed')
    ids = list(data['claim_ids'])
    random.Random(seed).shuffle(ids)
    groups = []
    for offset in range(0, len(ids), 4):
        claims = ids[offset:offset + 4]
        indices = [i for claim in claims for i, row in enumerate(data['records']) if row['claim_id'] == claim]
        groups.append({'claim_ids': claims, 'record_indices': indices,
                       'record_sha256': [data['record_sha256'][i] for i in indices]})
    payload = {'version': PLAN_VERSION, 'prepared_sha256': prepared['sha256'],
               'roster_sha256': data['roster']['sha256'], 'actual_claim_count': len(ids),
               'seed': seed, 'epochs': 1, 'planned_optimizer_steps': math.ceil(len(ids) / 4),
               'shuffled_claim_ids': ids, 'groups': groups,
               'epoch_metric': 'sum(observed_group_mean * actual_group_claim_count) / actual_N'}
    return {'payload': payload, 'sha256': identity(payload)}


def _finite_state(model: Any, optimizer: Any) -> None:
    import torch
    values = list(model.parameters()) + list(model.buffers())
    values += [v for state in optimizer.state.values() for v in state.values() if torch.is_tensor(v)]
    require(all(bool(torch.isfinite(v).all()) for v in values), 'nonfinite_parameter_or_optimizer_state')


def run_one_epoch(prepared: Mapping[str, Any], plan: Mapping[str, Any], model: Any,
                  output: Path) -> dict[str, Any]:
    """Dependency-injected CPU tiny model only. No model loads, checkpoints or retries."""
    data = _validated_inputs(prepared)  # All records, including the last group, before ANY forward.
    scheduled = _unseal(plan, 'epoch_plan_identity')
    require(dict(plan) == make_epoch_plan(prepared, scheduled['seed']), 'epoch_plan_drift')
    parameters = list(model.parameters())
    require(data['provenance']['scope'] == 'synthetic_fixture' and bool(parameters)
            and all(p.device.type == 'cpu' for p in parameters + list(model.buffers()))
            and sum(p.numel() for p in parameters) <= 1_000_000, 'cpu_tiny_synthetic_only')
    output.mkdir(parents=True, exist_ok=False)
    counts: dict[str, int] = {}
    completed = []
    current_group: int | None = None
    def observe(event: str) -> None:
        counts[event] = counts.get(event, 0) + 1
        if event.startswith('optimizer_step'):
            ordered_write(output / f'{current_group:03d}-{event}.json',
                          {'event': event, 'counts': dict(counts), 'group': current_group})
    ordered_write(output / 'started.json', {'version': VERSION, 'plan': plan, 'automatic_retry': False})
    try:
        optimizer = make_claim_group_mean_adamw(model)
        _finite_state(model, optimizer)
        model.train()
        for current_group, group in enumerate(scheduled['groups']):
            ordered_write(output / f'{current_group:03d}-reserved.json', group)
            rows = [data['records'][i] for i in group['record_indices']]
            tokens = [data['tokenized'][i] for i in group['record_indices']]
            result = claim_group_mean_update(model, optimizer, rows, tokens, group['claim_ids'],
                                            weight_contract=weights_for(data['actual_claim_count']), observe=observe)
            _finite_state(model, optimizer)
            result['epoch_metric_contribution'] = result['group_mean_loss'] * len(group['claim_ids']) / data['actual_claim_count']
            result['actual_epoch_N'] = data['actual_claim_count']
            ordered_write(output / f'{current_group:03d}-completed.json', result)
            completed.append(result)
        report = {'version': VERSION, 'status': 'complete', 'plan_sha256': plan['sha256'],
                  'roster_sha256': data['roster']['sha256'], 'actual_claim_count': data['actual_claim_count'],
                  'decision_records': len(data['records']), 'completed_groups': len(completed),
                  'counts': counts, 'group_results': completed,
                  'epoch_observed_loss': sum(r['epoch_metric_contribution'] for r in completed),
                  'metric_scope': 'observed_along_training_path_not_final_model_evaluation',
                  'real_training': False, 'automatic_retry': False}
        ordered_write(output / 'complete.pending.json', report)
        # Atomic no-clobber publication: a partial write never becomes complete.json.
        os.link(output / 'complete.pending.json', output / 'complete.json')
        return report
    except BaseException as exc:
        ordered_write(output / 'failed.json', {'version': VERSION, 'status': 'failed',
            'exception_type': type(exc).__name__, 'group': current_group, 'completed_groups': len(completed),
            'counts': counts, 'known_completed_optimizer_steps': counts.get('optimizer_step_completed', 0),
            'optimizer_step_outcome_unknown': counts.get('optimizer_step_started', 0) > counts.get('optimizer_step_completed', 0),
            'automatic_retry': False, 'resume_authorized': False})
        raise
