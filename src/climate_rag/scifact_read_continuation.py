"""Exact scripted-read conditional probe. No autonomous trajectory or gold access."""
from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

from .agent_v3 import valid_search_query
from .scifact_adapter_regression import policy_identity, require
from .scifact_semantic_contract import encoded, sha
from .scifact_terminal import (
    PROTOCOL as TERMINAL, parse_action, render_answer, render_scifact_prompt,
    source_from_abstract, to_original_prediction,
)
from .verification import normalise_claim

PROTOCOL = 'scifact-frozen-read-conditional-three-20261001-v1'
OLD_STATES_SHA = '5a49b53aa6ab7790231b224b5966f3e8e6f2fff1856219c1c177af16ded714b8'
CALL_SECONDS = 120


def policy(source: Path) -> dict[str, Any]:
    return {'protocol': PROTOCOL, 'inherited_regression_policy': policy_identity(source),
        'planned_slots': 3, 'max_generator_calls': 3, 'calls_per_slot': 1,
        'repair_calls': 0, 'warmup_calls': 0, 'reranker_calls': 0,
        'execute_model_proposed_tools': False, 'max_output_tokens': 512,
        'new_call_seconds': CALL_SECONDS, 'not_original_episode_remaining_time': True,
        'scripted_read_not_model_action': True, 'old_states_sha256': OLD_STATES_SHA,
        'scope': 'exposed_TRAIN_oracle_selected_read_conditional_not_autonomous_or_heldout'}


def ordered_write(path: Path, value: Any) -> None:
    """Physical order is significant to the prompt, unlike canonical state hashes."""
    raw = (json.dumps(value, ensure_ascii=False, separators=(',', ':'), sort_keys=False) + '\n').encode()
    with path.open('xb') as stream:
        stream.write(raw)
        stream.flush()
        os.fsync(stream.fileno())


def validate_state(row: dict[str, Any], tokenizer: Any) -> None:
    obs, schema, expected = row['observation'], row['schema'], row['expected_state']
    require(sha(encoded(obs)) == expected['observation_sha256'], 'observation_changed')
    require(sha(encoded(schema)) == expected['schema_sha256'], 'schema_changed')
    require(sha(encoded({'observation': obs, 'schema': schema})) == expected['state_sha256'], 'state_changed')
    prompt = render_scifact_prompt(tokenizer, obs, schema)
    require(sha(prompt.encode()) == expected['prompt_sha256'], 'physical_prompt_order_changed')
    tokens = tokenizer.encode(prompt, add_special_tokens=False)
    require(len(tokens) == expected['prompt_tokens'] and len(tokens) <= 8192, 'prompt_token_count_changed')
    require(sha(encoded(tokens)) == row['new_token_ids_sha256'], 'new_token_ids_hash_changed')
    require({'read', 'rewrite', 'rerank'} <= set(obs['allowed_actions']), 'original_tools_must_remain_offered')


def visible_contract(row: dict[str, Any], corpus: Any) -> dict[str, Any]:
    candidates = row['candidate_doc_ids']
    require(len(candidates) == len(set(candidates)), 'duplicate_candidates')
    for candidate in row['ordered_candidates']:
        require(source_from_abstract(corpus[candidate['doc_id']]).text_sha256 == candidate['source_sha256'],
                'candidate_source_changed')
    visible = {}
    for entry in row['observation']['current_citable']:
        sid = entry['sentence_id']
        alias, raw_index = sid.split(':')
        require(alias.startswith('c') and alias[1:].isdigit(), 'invalid_alias')
        doc = corpus[candidates[int(alias[1:])]]
        source = source_from_abstract(doc)
        index = int(raw_index)
        require(sid not in visible and 0 <= index < len(doc.sentences)
                and entry['text'] == doc.sentences[index], 'visible_source_changed')
        visible[sid] = {'source_id':str(doc.doc_id), 'sentence_index':index, 'text':entry['text'],
            'text_sha256':sha(entry['text'].encode()), 'source_text_sha256':source.text_sha256}
    return visible


def finalise(raw: str, row: dict[str, Any], corpus: Any) -> dict[str, Any]:
    visible = visible_contract(row, corpus)
    decision = parse_action(json.loads(raw), row['observation']['allowed_actions'], visible,
        [f'c{i}' for i in range(len(row['candidate_doc_ids']))], 5)
    if decision['action'] in {'read', 'rewrite', 'rerank'}:
        error = None
        if decision['action'] == 'read' and decision['source_ids'] == row['frozen_read_ids']:
            error = 'read_loop'
        if decision['action'] == 'rewrite' and not valid_search_query(
                row['observation']['immutable_claim'], normalise_claim(decision['query'])):
            error = 'rewrite_constraint_or_loop'
        return {'status':'proposed_not_executed', 'decision':decision, 'prediction':None,
            'proposal_validation':'schema_valid', 'controller_legal':error is None,
            'controller_error':error,
            'model_tool_executions':0, 'autonomous_success':False}
    answer = render_answer(decision, visible) if decision['action'] == 'answer' else None
    result = {'protocol':TERMINAL, 'answer':answer,
        'outcome':'ids_validated_semantics_unmeasured' if answer else 'model_abstention:'+decision['reason']}
    converted = to_original_prediction(row['claim_id'], result, corpus)
    return {'status':'terminal', 'decision':decision, 'result':result, **converted,
        'model_tool_executions':0, 'autonomous_success':False}
