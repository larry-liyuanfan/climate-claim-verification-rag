"""Synthetic-only utility8 receipt audit. Accepted observations are NOT targets.

No filesystem reader, gold loader, model call, trainer or job submission. The
caller supplies bytes and an independently sealed slot roster; every slot stays
in the denominator even when its physical lineage cannot be accepted.
"""
from __future__ import annotations

import copy
import hashlib
import json
from collections.abc import Mapping, Sequence
from typing import Any

from pydantic import ValidationError

from .agent_v3 import V3Budget, valid_search_query
from .scifact_grounding import Abstract
from .scifact_state_supervision import require
from .scifact_state_training_driver import captured_state
from .scifact_terminal import action_schema, parse_action, render_scifact_prompt
from .scifact_utility_contract import PROTOCOL, identity, read_intervention, validate_prefix
from .verification import normalise_claim

VERSION = 'scifact-accepted-observations-synthetic-v1-20261001'
FROZEN_CONTRACT = 'e22143cc771a3e5d10959a2cb3cb2f9238505520'
TOOLS = {'read', 'rewrite', 'rerank'}
TOOL_FEEDBACK = 'tool_completed: inspect current_citable; no semantic conclusion implied'


def _sha(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _pairs(items: list[tuple[str, Any]]) -> dict[str, Any]:
    require(len({k for k, _ in items}) == len(items), 'duplicate_json_key')
    return dict(items)


def _stable_attempt(attempt: Mapping[str, Any]) -> dict[str, Any]:
    return {k: v for k, v in attempt.items() if k not in {'status', 'error_code', 'shared_prefix_reference'}}


def _proposal(response: Mapping[str, Any], frame: Mapping[str, Any],
              order: list[str]) -> tuple[dict[str, Any] | None, str | None]:
    try:
        return parse_action(json.loads(response['raw']), frame['observation']['allowed_actions'],
                            frame['visible'], order, 5), None
    except json.JSONDecodeError:
        return None, 'invalid_json'
    except ValidationError:
        return None, 'invalid_schema'
    except ValueError as exc:
        return None, str(exc)


def _receipt(files: Mapping[str, Any], key: str, slot: str, frame: dict[str, Any],
             attempt: dict[str, Any], tokenizer: Any) -> dict[str, Any]:
    reserved, finished = (files[f'ledger/{key}.{phase}.json'] for phase in ('reserved', 'finished'))
    require(all(r['physical_attempt_id'] == key and r['slot'] == slot for r in (reserved, finished)),
            'physical_id_or_slot_mismatch')
    require(reserved['protocol'] == PROTOCOL and reserved['status'] == 'reserved_before_actual_generate'
            and reserved['usage'] is None and reserved['max_output_tokens'] == 512, 'reservation_contract')
    require(finished['status'] == 'returned' and finished['usage_known'] is True, 'generation_not_returned')
    response = finished['response']
    require(isinstance(response['raw'], str) and response['diagnostics']['physical_attempt_id'] == key
            and identity(response['diagnostics']) == identity(attempt['diagnostics'])
            and finished['usage'] == response['usage'] == attempt['usage']
            and attempt['usage_known'] is True, 'response_attempt_identity')
    attachment = response['diagnostics']['private_attachment']
    require(attachment['sha256'] == _sha(response['raw'].encode())
            and attachment['attempted_bytes'] == attachment['stored_bytes'] == len(response['raw'].encode())
            and attachment['truncated'] is False and attachment['io_failed'] is False, 'full_raw_response_identity')
    diagnostics = response['diagnostics']
    require(diagnostics['output_sha256'] == _sha(response['raw'].encode())
            and diagnostics['output_bytes'] == len(response['raw'].encode())
            and diagnostics['output_characters'] == len(response['raw'])
            and diagnostics['output_tokens'] == response['usage']['output_tokens'], 'raw_diagnostic_identity')
    require(all(identity(reserved[k]) == identity(frame[k]) for k in ('observation', 'schema')),
            'ordered_input_identity')
    prompt = render_scifact_prompt(tokenizer, frame['observation'], frame['schema'])
    token_ids = tokenizer.encode(prompt, add_special_tokens=False)
    require(reserved['prompt_sha256'] == identity(prompt)
            and reserved['token_ids_sha256'] == identity(token_ids), 'journal_prompt_or_tokens_identity')
    require(frame['prompt_tokens'] == attempt['input_prompt_tokens'] == len(token_ids)
            == response['usage']['input_tokens'] and type(response['usage']['output_tokens']) is int
            and 0 <= response['usage']['output_tokens'] <= 512, 'prompt_count_or_usage')
    require(attempt['allowed_actions'] == frame['observation']['allowed_actions']
            and attempt['visible_sentence_sha256'] == {s: v['text_sha256'] for s, v in frame['visible'].items()},
            'attempt_input_identity')
    return {'physical_attempt_id': key, 'physical_slot': slot, 'capture': copy.deepcopy(frame),
            'reserved': copy.deepcopy(reserved), 'finished': copy.deepcopy(finished),
            'response': copy.deepcopy(response), 'journal_prompt_identity': identity(prompt),
            'prompt_utf8_sha256': _sha(prompt.encode()), 'token_ids_identity': identity(token_ids),
            'supervision_target': None, 'semantic_correctness': 'unmeasured'}


def _transition(event: dict[str, Any], before: dict[str, Any], proposal: dict[str, Any],
                *, scripted: bool) -> None:
    kind = proposal['action']
    require(event['tool'] == kind and event['status'] == 'completed'
            and event['model_selected'] is (not scripted)
            and event.get('origin') == ('scripted_intervention' if scripted else None)
            and event['before_context'] == before['requested_context'], 'event_origin_or_before_state')
    order = event['candidate_ids']
    require(len(order) == len(set(order)) and bool(order), 'event_candidate_duplicates')
    if kind == 'read':
        require(event['requested_context'] == proposal['source_ids']
                and order == before['candidate_ids'], 'read_parameters_not_executed')
    elif kind == 'rerank':
        require(set(order) == set(before['candidate_ids']) and event['requested_context'] == order[:5],
                'rerank_candidate_set_changed')
    else:
        require(kind == 'rewrite' and event['query_sha256'] == _sha(normalise_claim(proposal['query']).encode())
                and event['requested_context'] == order[:5], 'rewrite_query_not_executed')
    if kind in {'read', 'rerank'}:
        require(event['source_sha256'] == before['source_sha256'], 'tool_source_hash_changed')


def _slot(files: Mapping[str, Any], declared: dict[str, Any], tokenizer: Any,
          corpus: Mapping[int, Abstract]) -> list[dict[str, Any]]:
    claim_id, arm, claim = declared['claim_id'], declared['arm'], normalise_claim(declared['claim'])
    slot = f'{claim_id}-{arm}'
    row = files[f'{slot}/result.json']
    require(files[f'{slot}/reserved.json'] == {'claim_id': claim_id, 'arm': arm}
            and row['claim_id'] == claim_id and row['arm'] == arm, 'slot_identity')
    require(row['status'] == 'finished' and row['result'] is not None, 'slot_not_finished')
    result = row['result']
    require(result['budget'] == V3Budget().model_dump() and result['diagnostic_protocol'] == PROTOCOL,
            'runtime_contract')
    raw = files[f'{slot}/raw-result.json']
    require(all(k in result and identity(v) == identity(result[k]) for k, v in raw.items())
            and all(k in raw for k in ('events', 'generation_attempts', 'outcome')), 'raw_result_changed')
    attempts, events, audits = result['generation_attempts'], result['events'], result['decision_execution_audit']
    require(bool(attempts) and len(audits) == len(attempts) and not result['trace_unlinked_events'],
            'attempt_audit_or_unlinked_events')
    prefix = None
    if arm != 'A':
        prefix = validate_prefix(files[f'{claim_id}-A/prefix.json'], claim, V3Budget())
        require(1 <= len(attempts) <= 2 and result['shared_prefix_attempts'] == 1
                and result['shared_prefix_sha256'] == files[f'{claim_id}-A/prefix.json']['sha256'],
                'shared_prefix_contract')
        inherited = {k: v for k, v in attempts[0].items() if k != 'shared_prefix_reference'}
        require(attempts[0]['shared_prefix_reference'] is True and identity(inherited) == identity(prefix['attempt']),
                'inherited_attempt_changed')
        require(identity(files[f'{slot}/initial-frame.json']) == identity(files[f'{claim_id}-A/initial-frame.json'])
                and identity(prefix['frame']) == identity(files[f'{slot}/initial-frame.json']['frame']),
                'shared_initial_changed')
    shared = int(arm != 'A')
    require(result['new_model_calls'] == len(attempts) - shared and result['model_calls'] == len(attempts),
            'logical_call_count')
    frames = {name for name in files if name.startswith(f'{slot}/frame-')}
    require(frames == {f'{slot}/frame-{i}.json' for i in range(len(attempts) - shared)},
            'orphan_or_missing_frame')
    ids = [a['diagnostics']['physical_attempt_id'] for a in attempts]
    require(len(ids) == len(set(ids)) and len(row['logical_generation_ids']) == len(ids)
            and set(row['logical_generation_ids']) == set(ids), 'duplicate_or_missing_logical_id')
    own = [r['physical_attempt_id'] for name, r in files.items()
           if name.startswith('ledger/') and name.endswith('.reserved.json') and r['slot'] == slot]
    require(len(row['physical_generation_ids']) == len(set(row['physical_generation_ids']))
            and set(own) == set(row['physical_generation_ids']) == set(ids[shared:]), 'physical_inventory_mismatch')
    initial = events[0]
    require(initial['tool'] == 'retrieve' and initial['status'] == 'completed'
            and initial['model_selected'] is False and initial['before_context'] == []
            and initial['query_sha256'] == _sha(claim.encode())
            and initial['requested_context'] == initial['candidate_ids'][:5], 'initial_retrieve_identity')
    state_index, cursor, feedback = 0, 1, None
    registry: dict[str, str] = {}
    reads: set[tuple[str, ...]] = set()
    observed = []
    for i, attempt in enumerate(attempts):
        if shared and i == 1:
            assert prefix is not None
            intervention, reason = read_intervention(files[f'{claim_id}-A/prefix.json']) if arm == 'B' else (
                {'action': 'rerank'}, None)
            require(reason is None and intervention is not None, 'scripted_intervention_unavailable')
            assert intervention is not None
            _transition(events[1], initial, intervention, scripted=True)
            state_index, cursor, feedback = 1, 2, TOOL_FEEDBACK
            if arm == 'B':
                reads.add(tuple(events[1]['requested_context']))
        frame_name = f'{slot}/frame-{i-shared}.json' if i >= shared else f'{slot}/initial-frame.json'
        frame = files[frame_name] if i >= shared else files[frame_name]['frame']
        _, _, current_registry, order = captured_state(frame, events[state_index], corpus)
        require(all(current_registry.get(k) == v for k, v in registry.items()), 'alias_rebound_or_dropped')
        registry = current_registry
        obs = frame['observation']
        require(obs['immutable_claim'] == claim and obs['feedback'] == feedback
                and obs['remaining_calls'] == 5 - i and obs['remaining_tools'] == 4 - state_index
                and attempt['requested_context'] == events[state_index]['requested_context'], 'state_or_feedback_mismatch')
        allowed = ['abstain'] + (['answer'] if frame['visible'] else [])
        executed = {e['tool'] for e in events[:state_index+1]}
        if i + 1 < 5 and state_index + 1 < 5:
            allowed += ['read'] + ([] if 'rewrite' in executed else ['rewrite'])
            allowed += [] if 'rerank' in executed else ['rerank']
        require(obs['allowed_actions'] == allowed, 'allowed_actions_runtime_contract')
        require(identity(frame['schema']) == identity(action_schema(obs['allowed_actions'], order, list(frame['visible']), 5)),
                'ordered_schema_mismatch')
        record = _receipt(files, ids[i], f'{claim_id}-A' if shared and i == 0 else slot, frame, attempt, tokenizer)
        if i == 0:
            initial_frame = files[f'{slot}/initial-frame.json']
            require(identity(initial_frame['frame']) == identity(frame)
                    and initial_frame['rendered_prompt_sha256'] == record['journal_prompt_identity']
                    and initial_frame['token_ids_sha256'] == record['token_ids_identity'], 'initial_prompt_identity')
            if prefix is not None:
                require(identity(prefix['response']) == identity(record['response'])
                        and identity({k: v for k, v in initial.items() if k != 'shared_prefix_reference'})
                        == identity(prefix['events'][0]), 'shared_response_or_event_changed')
        proposal, parse_error = _proposal(record['response'], frame, order)
        if i == 0 and f'{claim_id}-A/prefix.json' in files:
            capsule = validate_prefix(files[f'{claim_id}-A/prefix.json'], claim, V3Budget())
            require(identity(capsule['response']) == identity(record['response'])
                    and identity(capsule['frame']) == identity(frame)
                    and capsule['decision'] == proposal
                    and capsule['attempt']['status'] == ('validation_failed' if parse_error else 'valid_decision'),
                    'prefix_proposal_not_physical_response')
            require(identity(_stable_attempt(capsule['attempt'])) == identity(_stable_attempt(attempt)),
                    'prefix_stable_attempt_changed')
        if proposal and proposal['action'] in {'answer', 'abstain'} and not (shared and i == 0):
            require(i == len(attempts) - 1, 'attempt_after_terminal_decision')
        controller_error = None
        if proposal and not (shared and i == 0):
            if proposal['action'] == 'read' and (proposal['source_ids'] == attempt['requested_context']
                                                or tuple(proposal['source_ids']) in reads):
                controller_error = 'read_loop'
            if proposal['action'] == 'rewrite' and not valid_search_query(claim, normalise_claim(proposal['query'])):
                controller_error = 'rewrite_constraint_or_loop'
        # B/C reports a legal-schema proposal without executing it, even on a loop.
        expected_status = 'validation_failed' if parse_error or (controller_error and not shared) else 'valid_decision'
        require(attempt['status'] == expected_status and attempt.get('action') == (proposal['action'] if proposal else None),
                'strict_proposal_status_mismatch')
        if expected_status == 'validation_failed':
            require(attempt['error_code'] == (parse_error or controller_error), 'repair_reason_mismatch')
        if shared and i == 1 and proposal and proposal['action'] in TOOLS:
            require(result['outcome'] == 'proposed_not_executed' and attempt['proposed_decision'] == proposal
                    and attempt['proposal_validation'] == 'schema_valid'
                    and attempt['controller_legal'] is (controller_error is None)
                    and attempt['controller_error'] == controller_error, 'continuation_proposal_changed')
        event_index = None
        current_event_index = state_index
        if not shared and proposal and proposal['action'] in TOOLS and expected_status == 'valid_decision':
            event_index = cursor
            _transition(events[cursor], events[state_index], proposal, scripted=False)
            state_index, cursor, feedback = cursor, cursor + 1, TOOL_FEEDBACK
            if proposal['action'] == 'read':
                reads.add(tuple(proposal['source_ids']))
        elif expected_status == 'validation_failed':
            feedback = parse_error or controller_error
            if not shared and i + 1 < len(attempts):
                feedback = str(feedback) + '; choose a new legal action; read known preview candidates before citation'
        audit = audits[i]
        require(audit['attempt_index'] == i and audit['strict_action'] == attempt.get('action')
                and audit['raw_wire_action'] == attempt['diagnostics'].get('raw_wire_action')
                and audit['strict_status'] == expected_status and audit['actual_event_index'] == event_index
                and (event_index is None or type(audit['actual_event_index']) is int)
                and audit['actual_event_status'] == ('completed' if event_index is not None else None)
                and audit['next_attempt_index'] == (i+1 if i+1 < len(attempts) else None), 'decision_event_link_mismatch')
        if i + 1 < len(attempts):
            next_name = f'{slot}/frame-{i+1-shared}.json'
            require(audit['next_feedback'] == files[next_name]['observation']['feedback']
                    and audit['next_strict_action'] == attempts[i+1].get('action')
                    and audit['next_strict_status'] == attempts[i+1]['status'], 'next_attempt_audit_mismatch')
        else:
            require(all(audit[k] is None for k in ('next_feedback', 'next_strict_action', 'next_strict_status')),
                    'tail_audit_has_phantom_next_attempt')
        record.update(parsed_model_proposal=proposal, parse_error=parse_error)
        record['logical_reference'] = {'slot': slot, 'attempt_index': i, 'frame_path': frame_name,
            'attempt': copy.deepcopy(attempt), 'state_event_index': current_event_index,
            'state_event': copy.deepcopy(events[current_event_index]), 'decision_execution_audit': copy.deepcopy(audit),
            'shared_prefix_reference': bool(shared and i == 0)}
        observed.append(record)
    require(cursor == len(events), 'unbound_or_ambiguous_events')
    return observed


def audit_observations(artifacts: Sequence[tuple[str, bytes]], roster: Mapping[str, Any],
                       tokenizer: Any, corpus: Mapping[int, Abstract]) -> dict[str, Any]:
    """Fail closed per slot; no successful-trajectory selection or training output."""
    require(set(roster) == {'payload', 'sha256'} and roster['sha256'] == identity(roster['payload']), 'roster_identity')
    frozen = roster['payload']
    require(frozen['version'] == VERSION and frozen['scope'] == 'synthetic_fixture'
            and frozen['contract_revision'] == FROZEN_CONTRACT, 'synthetic_external_roster_required')
    slots = frozen['slots']
    require(bool(slots) and all(set(s) == {'claim_id', 'arm', 'claim'} and type(s['claim_id']) is int
                               and s['arm'] in {'A', 'B', 'C'} and isinstance(s['claim'], str) for s in slots)
            and len({(s['claim_id'], s['arm']) for s in slots}) == len(slots), 'slot_roster_invalid')
    reports = [dict(s, status='incomplete', errors=[], physical_ids=[]) for s in slots]
    observations: dict[str, Any] = {}
    try:
        require(len({p for p, _ in artifacts}) == len(artifacts), 'duplicate_artifact_path')
        files = {p: json.loads(b, object_pairs_hook=_pairs) for p, b in artifacts}
        planned = files['planned.json']
        require(planned['protocol'] == PROTOCOL and planned['slots'] == [
            {k: s[k] for k in ('claim_id', 'arm')} for s in slots], 'planned_roster_mismatch')
        slot_names = {f"{s['claim_id']}-{s['arm']}" for s in slots}
        ledger_ids = []
        for path, receipt in files.items():
            if path.startswith('ledger/'):
                key, phase, ext = path.removeprefix('ledger/').split('.')
                require(ext == 'json' and phase in {'reserved', 'finished'}
                        and len(key) == 3 and key[0] == 'g' and key[1:].isdigit()
                        and int(key[1:]) < 56 and receipt['physical_attempt_id'] == key
                        and receipt['slot'] in slot_names, 'ledger_path_id_or_slot')
                if phase == 'reserved':
                    ledger_ids.append(key)
                else:
                    require(f'ledger/{key}.reserved.json' in files, 'finished_without_reservation')
            elif path not in {'planned.json', 'run.json'}:
                owner, filename = path.split('/')
                require(owner in slot_names, 'artifact_outside_slot_roster')
                if filename.startswith('frame-'):
                    number = filename.removeprefix('frame-').removesuffix('.json')
                    require(number.isdigit() and filename == f'frame-{int(number)}.json', 'noncanonical_frame_path')
                else:
                    require(filename in {'reserved.json', 'result.json', 'raw-result.json',
                                         'prefix.json', 'initial-frame.json'}, 'unexpected_artifact_path')
        require(len(ledger_ids) == len(set(ledger_ids)), 'duplicate_physical_id')
    except (KeyError, IndexError, TypeError, ValueError) as exc:
        for report in reports:
            report['errors'] = [str(exc) if isinstance(exc, ValueError) else 'missing_or_malformed_artifact']
        files = {}
    staged: dict[str, list[dict[str, Any]]] = {}
    if files:
        for declared, report in zip(slots, reports, strict=True):
            name = f"{declared['claim_id']}-{declared['arm']}"
            source = files.get(f'{name}/result.json', {})
            try:
                require(isinstance(source, dict) and (source.get('result') is None or isinstance(source['result'], dict)),
                        'malformed_slot_result')
                report.update(input_status=source.get('status'),
                              reported_attempt_count=len((source.get('result') or {}).get('generation_attempts', [])),
                              reserved_physical_ids=[r['physical_attempt_id'] for p, r in files.items()
                                  if p.startswith('ledger/') and p.endswith('.reserved.json') and r['slot'] == name])
                rows = _slot(files, declared, tokenizer, corpus)
                staged[f"{declared['claim_id']}-{declared['arm']}"] = rows
                report.update(status='accepted', physical_ids=[r['physical_attempt_id'] for r in rows])
            except (KeyError, IndexError, TypeError, ValueError) as exc:
                report['errors'] = [str(exc) if isinstance(exc, ValueError) else 'missing_or_malformed_artifact']
        accepted_a = {r['claim_id'] for r in reports if r['arm'] == 'A' and r['status'] == 'accepted'}
        for report in reports:
            if report['arm'] != 'A' and report['claim_id'] not in accepted_a:
                report.update(status='incomplete', physical_ids=[])
                report['errors'].append('shared_A_lineage_not_accepted')
        for report in reports:
            if report['status'] != 'accepted':
                continue
            for row in staged[f"{report['claim_id']}-{report['arm']}"]:
                row = copy.deepcopy(row)
                reference = row.pop('logical_reference')
                key = row['physical_attempt_id']
                if key not in observations:
                    observations[key] = dict(row, logical_references=[])
                observations[key]['logical_references'].append(reference)
    return {'version': VERSION, 'roster': copy.deepcopy(dict(roster)), 'planned_slots': len(slots),
            'slots': reports, 'accepted_slots': sum(r['status'] == 'accepted' for r in reports),
            'physical_observations': list(observations.values()),
            'physical_receipt_inventory': [{'physical_attempt_id': r['physical_attempt_id'], 'slot': r['slot'],
                'finished_status': files.get(p.replace('.reserved.json', '.finished.json'), {}).get('status'),
                'accepted_observation': r['physical_attempt_id'] in observations}
                for p, r in files.items() if p.startswith('ledger/') and p.endswith('.reserved.json')],
            'artifact_sha256': [(p, _sha(b)) for p, b in artifacts],
            'training_authorized': False, 'semantic_targets_created': 0, 'agent_gain_measured': False}
