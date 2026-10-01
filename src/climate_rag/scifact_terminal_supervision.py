"""Synthetic annotation-bound terminal candidates, not a training cohort.

Inputs are supplied accepted observations and complete original annotation bytes.
No file/model/gold loader, action policy, record weights or training release.
"""
from __future__ import annotations

import copy
import hashlib
import itertools
import json
import math
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

from .agent_v3 import Source
from .scifact_grounding import Abstract, GoldClaim, LABEL_TO_PROJECT, parse_gold
from .scifact_observation_receipts import VERSION as OBSERVATION_VERSION
from .scifact_state_supervision import require, tokenize_target
from .scifact_state_training_driver import captured_state, corpus_identity
from .scifact_terminal import parse_action, render_scifact_prompt
from .scifact_utility_contract import identity
from .verification import normalise_claim

VERSION = 'scifact-annotation-terminal-candidates-synthetic-v1-20261001'
MAX_CANDIDATES = 64


@dataclass
class _View:
    gold: GoldClaim
    frame: dict[str, Any]
    sources: list[Source]
    registry: dict[str, str]
    order: list[str]
    visible: dict[str, Any]
    source: dict[str, Any]
    provenance: dict[str, Any]
    state_event: dict[str, Any]


def _unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    require(len(pairs) == len({k for k, _ in pairs}), 'duplicate_annotation_key')
    return dict(pairs)


def _bind(report: Mapping[str, Any], reference: Mapping[str, Any], annotation: bytes,
          provenance: Mapping[str, Any], corpus: Mapping[int, Abstract], tokenizer: Any) -> _View:
    require(set(provenance) == {'payload', 'sha256'} and provenance['sha256'] == identity(provenance['payload']),
            'provenance_seal_changed')
    p = provenance['payload']
    require(set(p) == {'version', 'scope', 'annotation_sha256', 'accepted_report_sha256',
                      'corpus_sha256', 'reference_sha256', 'annotation_source', 'complete_original_row_declared'}
            and p['version'] == VERSION and p['scope'] == 'synthetic_fixture'
            and p['annotation_source'] == 'externally_supplied_synthetic_complete_original_row'
            and p['complete_original_row_declared'] is True, 'synthetic_annotation_contract_required')
    require(p['annotation_sha256'] == hashlib.sha256(annotation).hexdigest()
            and p['accepted_report_sha256'] == identity(report)
            and p['corpus_sha256'] == corpus_identity(corpus)
            and p['reference_sha256'] == identity(reference), 'annotation_report_corpus_reference_identity')
    require(report['version'] == OBSERVATION_VERSION and report['roster']['payload']['scope'] == 'synthetic_fixture'
            and report['training_authorized'] is False, 'accepted_observation_contract')
    require(set(reference) == {'physical_attempt_id', 'slot', 'attempt_index'}
            and type(reference['attempt_index']) is int, 'reference_contract')
    matches = [s for s in report['slots'] if f"{s['claim_id']}-{s['arm']}" == reference['slot']]
    require(len(matches) == 1 and matches[0]['status'] == 'accepted', 'slot_not_accepted')
    slot = matches[0]
    physical = [r for r in report['physical_observations'] if r['physical_attempt_id'] == reference['physical_attempt_id']]
    require(len(physical) == 1, 'physical_observation_not_unique')
    observation = physical[0]
    refs = [r for r in observation['logical_references'] if r['slot'] == reference['slot']
            and r['attempt_index'] == reference['attempt_index']]
    require(len(refs) == 1 and reference['physical_attempt_id'] in slot['physical_ids'], 'logical_reference_not_unique')
    logical = refs[0]
    row = json.loads(annotation, object_pairs_hook=_unique_object)
    require(set(row) == {'id', 'claim', 'evidence', 'cited_doc_ids'}, 'complete_original_annotation_required')
    gold = parse_gold(row, corpus)
    frame, sources, registry, order = captured_state(observation['capture'], logical['state_event'], corpus)
    require(gold.claim_id == slot['claim_id'] and normalise_claim(gold.claim) == normalise_claim(slot['claim'])
            == frame['observation']['immutable_claim'], 'annotation_claim_identity')
    prompt = render_scifact_prompt(tokenizer, frame['observation'], frame['schema'])
    tokens = tokenizer.encode(prompt, add_special_tokens=False)
    require(observation['journal_prompt_identity'] == identity(prompt)
            and observation['prompt_utf8_sha256'] == hashlib.sha256(prompt.encode()).hexdigest()
            and observation['token_ids_identity'] == identity(tokens)
            and observation['capture']['prompt_tokens'] == len(tokens), 'exact_captured_prompt_or_tokenizer_changed')
    response = observation['response']
    require(hashlib.sha256(response['raw'].encode()).hexdigest() == response['diagnostics']['output_sha256'],
            'observed_raw_identity')
    try:
        proposal = parse_action(json.loads(response['raw']), frame['observation']['allowed_actions'],
                                observation['capture']['visible'], order, 5)
    except ValueError:
        proposal = None
    require(proposal == observation['parsed_model_proposal'], 'observed_proposal_changed')
    origin = {'reference': dict(reference), 'response': copy.deepcopy(response),
              'parsed_model_proposal': copy.deepcopy(proposal), 'logical_reference': copy.deepcopy(logical),
              'capture': copy.deepcopy(observation['capture'])}
    return _View(gold, frame, sources, registry, order, observation['capture']['visible'],
                 origin, copy.deepcopy(dict(provenance)), copy.deepcopy(logical['state_event']))


def _match(target: Mapping[str, Any], view: _View) -> dict[str, Any]:
    canonical = parse_action(dict(target), view.frame['observation']['allowed_actions'], view.visible, view.order, 5)
    if canonical['action'] == 'abstain':
        require(not view.gold.evidence and canonical['reason'] == 'insufficient_evidence',
                'abstention_not_official_annotation_NEI')
        return {'basis': 'official_annotation_NEI', 'matched_rationales': [], 'whole_gold_document_coverage': True}
    require(canonical['action'] == 'answer' and bool(view.gold.evidence), 'only_annotation_terminal_candidates')
    matches = []
    for document in canonical['documents']:
        doc_id = int(view.registry[document['source_id']])
        indices = tuple(int(s.rsplit(':', 1)[1]) for s in document['sentence_ids'])
        alternatives = [i for i, alt in enumerate(view.gold.evidence.get(doc_id, ()))
                        if LABEL_TO_PROJECT[alt.label] == document['label'] and alt.sentences == indices]
        require(bool(alternatives), 'not_one_complete_original_OR_rationale')
        matches.append({'document_id': doc_id, 'alternative_indices': alternatives,
                        'first_three_cover_selected_rationale': len(indices) <= 3})
    alias_by_doc = {int(view.registry[a]): a for a in view.order}
    complete_visible_docs = {doc for doc, alts in view.gold.evidence.items() if doc in alias_by_doc and any(
        all(f'{alias_by_doc[doc]}:{i}' in view.visible for i in alt.sentences) for alt in alts)}
    require({r['document_id'] for r in matches} == complete_visible_docs,
            'supervision_omits_complete_current_visible_document')
    return {'basis': 'official_complete_visible_OR_rationale', 'matched_rationales': matches,
            'complete_current_visible_document_coverage': True,
            'whole_gold_document_coverage': {r['document_id'] for r in matches} == set(view.gold.evidence)}


def _record(target: dict[str, Any], view: _View, tokenizer: Any) -> dict[str, Any]:
    row: dict[str, Any] = {'target': copy.deepcopy(target), 'target_origin': 'program_derived_annotation_candidate',
                          'model_generated': False, 'status': 'unrepresentable', 'gap': None, 'tokenized': None}
    try:
        row.update(_match(target, view))
    except ValueError:
        row['gap'] = 'terminal_schema_unrepresentable'
        return row
    try:
        row['tokenized'] = tokenize_target(tokenizer, view.frame, target, view.sources,
            alias_to_source=view.registry, candidate_aliases=view.order)
        row['status'] = 'representable_candidate'
    except ValueError as exc:
        row['gap'] = str(exc)
    return row


def validate_terminal_candidate(target: Mapping[str, Any], report: Mapping[str, Any], reference: Mapping[str, Any],
                                annotation: bytes, provenance: Mapping[str, Any], corpus: Mapping[int, Abstract],
                                tokenizer: Any) -> dict[str, Any]:
    """Semantic/schema mismatch raises; token overflow is an explicit unrepresentable candidate."""
    view = _bind(report, reference, annotation, provenance, corpus, tokenizer)
    _match(target, view)
    return _record(dict(target), view, tokenizer)


def terminal_candidates(report: Mapping[str, Any], reference: Mapping[str, Any], annotation: bytes,
                        provenance: Mapping[str, Any], corpus: Mapping[int, Abstract], tokenizer: Any,
                        *, candidate_cap: int) -> dict[str, Any]:
    """All complete-visible documents × their exact OR alternatives, or explicit cap gap."""
    view = _bind(report, reference, annotation, provenance, corpus, tokenizer)
    result = _enumerate_bound(view, tokenizer, candidate_cap)
    result['observed_model'] = view.source
    return result


def _enumerate_bound(view: _View, tokenizer: Any, candidate_cap: int) -> dict[str, Any]:
    """Shared semantics only; each channel MUST first enforce its own strict binder."""
    require(type(candidate_cap) is int and 1 <= candidate_cap <= MAX_CANDIDATES, 'explicit_candidate_cap_1_to_64')
    choices = []
    incomplete = []
    for doc_id, alternatives in view.gold.evidence.items():
        alias = next((a for a in view.order if view.registry[a] == str(doc_id)), None)
        complete = []
        missing = []
        for index, alt in enumerate(alternatives):
            ids = [f'{alias}:{i}' for i in alt.sentences] if alias else []
            unseen = [sid for sid in ids if sid not in view.visible]
            if alias and not unseen:
                complete.append({'source_id': alias, 'label': LABEL_TO_PROJECT[alt.label], 'sentence_ids': ids})
            else:
                missing.append({'alternative_index': index, 'missing_original_sentence_indices': [
                    i for i in alt.sentences if alias is None or f'{alias}:{i}' not in view.visible]})
        if complete:
            assert alias is not None
            choices.append((view.order.index(alias), complete))
        if missing:
            incomplete.append({'document_id': doc_id, 'candidate_alias': alias,
                'selected_context': bool(alias and alias in view.state_event['requested_context']),
                'incomplete_alternatives': missing})
    choices.sort(key=lambda row: row[0])
    count = math.prod(len(alts) for _, alts in choices) if choices else (0 if view.gold.evidence else 1)
    result: dict[str, Any] = {'version': VERSION, 'claim_id': view.gold.claim_id,
        'provenance': view.provenance, 'frame': view.frame,
        'candidate_cap': candidate_cap, 'enumeration': 'all_complete_visible_documents_exact_OR_product',
        'candidate_count_before_cap': count, 'candidates': [], 'gaps': [], 'incomplete_annotation_view': incomplete,
        'budget_observation': {k: view.frame['observation'][k] for k in ('remaining_calls', 'remaining_tools', 'allowed_actions')},
        'cohort_created': False, 'weights_created': False, 'training_authorized': False}
    if count > candidate_cap:
        result.update(status='enumeration_overflow', gaps=['candidate_cap_exceeded_no_partial_enumeration'])
    elif not count:
        result.update(status='context_insufficient_for_annotation_view',
                      gaps=['positive_annotation_not_currently_complete_visible_no_action_target'])
    else:
        targets: list[dict[str, Any]] = ([{'action': 'abstain', 'reason': 'insufficient_evidence'}] if not view.gold.evidence else
            [{'action': 'answer', 'documents': list(combination)}
             for combination in itertools.product(*[alts for _, alts in choices])])
        result['candidates'] = [_record(target, view, tokenizer) for target in targets]
        require(len(result['candidates']) == count, 'candidate_denominator_changed')
        result['gaps'] = [r['gap'] for r in result['candidates'] if r['gap'] is not None]
        result['status'] = 'candidates_with_representation_gaps' if result['gaps'] else 'candidates_enumerated'
    return result
