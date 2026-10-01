"""FIT-only v2 supervision of production observations; no model execution CLI."""
from __future__ import annotations

import copy
from dataclasses import dataclass
from fractions import Fraction
import itertools
import json
import math
from collections.abc import Mapping, Sequence
from types import SimpleNamespace
from typing import Any

from .agent_v3 import Source
from .scifact_bounded_runtime import ARMS, run_bounded_slot
from .scifact_grounding import Abstract, GoldClaim, LABEL_TO_PROJECT
from .scifact_semantic_contract import encoded, sha
from .scifact_terminal import PROTOCOL, action_schema, parse_action, render_scifact_prompt, source_from_abstract
from .verification import normalise_claim

VERSION = 'scifact-production-state-supervision-fit-v2-20261001'
CONFIG: dict[str, Any] = {
    'version': VERSION, 'claims': 48, 'max_records': 192, 'max_records_per_claim': 4,
    'candidate_k': 20, 'max_input_tokens': 8192, 'max_output_tokens': 512,
    'capture': 'run_bounded_slot_adaptive_CommonPacking_unchanged',
    'annotation_view': 'complete_original_annotations_for_frozen_48_FIT_only',
    'candidate_pool': 'frozen_family_component_partition_fit_including_background',
    'teacher': 'direct_if_any_complete_document_else_one_existing_preview_read_else_context_abstain',
    'decision_origin': 'program_teacher', 'model_calls': 0,
    'claim_weight': 1, 'epoch_normalizer': 48, 'claims_per_optimizer_step': 4,
    'weight_hierarchy': 'claim/trajectory/alternative/state_frozen_before_tokenization',
    'loss': 'causal_assistant_token_mean*record_weight/48_no_chunk_renormalization',
    'training_authorized': False, 'new_nei_claims': False,
}


def require(value: bool, reason: str) -> None:
    if not value:
        raise ValueError(reason)


def config_sha() -> str:
    return sha(encoded(CONFIG))


class CaptureTeacher:
    """Scripted capture driver, never a learned policy or semantic evaluator."""
    gap = False
    name, kind, terminal_protocol = 'program-teacher-state-capture', 'fixture', PROTOCOL

    def __init__(self, tokenizer: Any, read_ids: Sequence[str] = ()) -> None:
        self.base = SimpleNamespace(tokenizer=tokenizer)
        self.read_ids = list(read_ids)
        self.frames: list[dict[str, Any]] = []

    def count_text(self, text: str) -> int:
        return len(self.base.tokenizer.encode(text, add_special_tokens=False))

    def count_prompt(self, observation: Mapping[str, Any], schema: Mapping[str, Any]) -> int:
        return self.count_text(render_scifact_prompt(self.base.tokenizer, observation, schema))

    def generate(self, observation: dict[str, Any], schema: dict[str, Any],
                 max_output_tokens: int, remaining_seconds: float) -> dict[str, Any]:
        self.frames.append(copy.deepcopy({'observation': observation, 'schema': schema}))
        decision = ({'action': 'read', 'source_ids': self.read_ids}
                    if self.read_ids and len(self.frames) == 1
                    else {'action': 'abstain', 'reason': 'insufficient_evidence'})
        raw = json.dumps(decision)
        return {'raw': raw, 'usage': {'input_tokens': self.count_prompt(observation, schema),
                                    'output_tokens': self.count_text(raw)}}


def capture(claim: GoldClaim, candidates: Sequence[Source], corpus: Mapping[int, Abstract],
            tokenizer: Any, read_ids: Sequence[str] = ()) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    require(0 < len(candidates) <= 20 and len({s.source_id for s in candidates}) == len(candidates),
            'candidate_contract')
    teacher = CaptureTeacher(tokenizer, read_ids)

    def retrieve(query: str, width: int) -> Sequence[Source]:
        require(query == normalise_claim(claim.claim) and width == 20, 'no_query_or_width_change')
        return candidates

    def no_rerank(*args: Any) -> Any:
        raise AssertionError('no_reranker')

    result = run_bounded_slot(claim.claim_id, claim.claim, 'adaptive', ARMS[0], teacher,
                              retrieve, no_rerank, corpus)['result']
    require(result['outcome'] == 'model_abstention:insufficient_evidence'
            and len(teacher.frames) == 1 + bool(read_ids)
            and all(a['status'] == 'valid_decision' for a in result['generation_attempts']),
            'capture_controller_failure_not_abstention')
    events = [e for e in result['events'] if e.get('tool') == 'read' and e.get('status') == 'completed']
    require(len(events) == bool(read_ids), 'scripted_read_execution_mismatch')
    return teacher.frames, {'scripted_responses': len(teacher.frames), 'program_teacher_reads': len(events),
                           'model_calls': 0, 'model_tool_executions': 0}


def frame_contract(frame: Mapping[str, Any], candidates: Sequence[Source], *,
                   alias_to_source: Mapping[str, str] | None = None,
                   candidate_aliases: Sequence[str] | None = None) -> dict[str, Any]:
    if alias_to_source is None and candidate_aliases is None:
        aliases = {f'c{i}': source for i, source in enumerate(candidates)}
    else:
        require(alias_to_source is not None and candidate_aliases is not None, 'explicit_alias_and_order_required')
        assert alias_to_source is not None and candidate_aliases is not None
        sources = {s.source_id: s for s in candidates}
        require(len(sources) == len(candidates) and len(set(alias_to_source.values())) == len(alias_to_source)
                and set(alias_to_source.values()) == set(sources)
                and set(alias_to_source) == {f'c{i}' for i in range(len(alias_to_source))}, 'stable_alias_registry')
        require(0 < len(candidate_aliases) <= 20 and len(set(candidate_aliases)) == len(candidate_aliases)
                and set(candidate_aliases) <= set(alias_to_source), 'current_candidate_order')
        aliases = {a: sources[alias_to_source[a]] for a in candidate_aliases}
    visible = {}
    observation = frame['observation']
    require(set(frame) == {'observation', 'schema'}, 'frame_extra_fields')
    require(set(observation) == {'immutable_claim', 'allowed_actions', 'current_citable', 'preview_only',
                               'feedback', 'remaining_calls', 'remaining_tools'}, 'teacher_fields_in_observation')
    for entry in observation['current_citable']:
        alias, raw = entry['sentence_id'].split(':')
        require(alias in aliases and raw.isdecimal(), 'visible_alias')
        source, index = aliases[alias], int(raw)
        require(0 <= index < len(source.sentences) and source.sentences[index] == entry['text']
                and entry['sentence_id'] not in visible, 'visible_original_sentence')
        visible[entry['sentence_id']] = {'source_id': source.source_id, 'sentence_index': index,
            'text': entry['text'], 'text_sha256': sha(entry['text'].encode()),
            'source_text_sha256': source.text_sha256}
    for preview in observation['preview_only']:
        require(preview['source_id'] in aliases and preview['citable'] is False, 'preview_contract')
    require(frame['schema'] == action_schema(observation['allowed_actions'], list(aliases), list(visible), 5),
            'production_schema_changed')
    return {'aliases': aliases, 'visible': visible}


def visible_choices(claim: GoldClaim, frame: Mapping[str, Any], candidates: Sequence[Source]) -> list[list[dict[str, Any]]]:
    """Each list contains OR alternatives for one currently visible document."""
    contract = frame_contract(frame, candidates)
    if not claim.evidence or 'answer' not in frame['observation']['allowed_actions']:
        return []
    choices = []
    for alias, source in contract['aliases'].items():
        doc_id = int(source.source_id)
        if doc_id not in claim.evidence:
            continue
        alternatives = []
        for alt in claim.evidence[doc_id]:
            ids = [f'{alias}:{i}' for i in alt.sentences]
            if all(i in contract['visible'] for i in ids):
                alternatives.append({'source_id': alias, 'label': LABEL_TO_PROJECT[alt.label], 'sentence_ids': ids})
        if alternatives:
            choices.append(alternatives)
    return choices


def answer_targets(claim: GoldClaim, frame: Mapping[str, Any], candidates: Sequence[Source],
                   limit: int) -> list[dict[str, Any]]:
    """Complete visible witnesses; unretrieved gold never erases a legal answer."""
    contract = frame_contract(frame, candidates)
    choices = visible_choices(claim, frame, candidates)
    if not choices:
        return []
    result = []
    for documents in itertools.islice(itertools.product(*choices), limit):
        target = {'action': 'answer', 'documents': list(documents)}
        # Semantic completeness does not excuse illegal total citation/output limits.
        parse_action(target, frame['observation']['allowed_actions'], contract['visible'], list(contract['aliases']), 5)
        result.append(target)
    return result


def read_plan(claim: GoldClaim, frame: Mapping[str, Any], candidates: Sequence[Source]) -> list[str]:
    obs = frame['observation']
    if 'read' not in obs['allowed_actions'] or obs['remaining_calls'] < 2 or obs['remaining_tools'] < 1:
        return []
    require(bool(claim.evidence), 'no_official_nei_in_authorized_fit')
    previews = {p['source_id'] for p in obs['preview_only']}
    citable = {s['sentence_id'].split(':')[0] for s in obs['current_citable']}
    # A teacher choice is one legal witness, never the unique correct action.
    # Keep retrieval order; missing other gold docs cannot invalidate this read.
    wanted = [f'c{i}' for i, source in enumerate(candidates)
              if int(source.source_id) in claim.evidence and f'c{i}' in previews | citable][:5]
    # Re-reading the identical initial selected set is a controller loop. A
    # narrower read may expose a previously truncated later sentence instead.
    if wanted == [f'c{i}' for i in range(min(5, len(candidates)))]:
        return wanted[:1] if len(wanted) > 1 else []
    return wanted


def tokenize_target(tokenizer: Any, frame: Mapping[str, Any], target: Mapping[str, Any],
                    candidates: Sequence[Source], *, alias_to_source: Mapping[str, str] | None = None,
                    candidate_aliases: Sequence[str] | None = None) -> dict[str, Any]:
    contract = frame_contract(frame, candidates, alias_to_source=alias_to_source,
                              candidate_aliases=candidate_aliases)
    canonical = parse_action(dict(target), frame['observation']['allowed_actions'],
                             contract['visible'], list(contract['aliases']), 5)
    prompt = render_scifact_prompt(tokenizer, frame['observation'], frame['schema'])
    prefix = list(tokenizer.encode(prompt, add_special_tokens=False))
    require(len(prefix) <= CONFIG['max_input_tokens'], 'actual_prompt_over_budget')
    raw = json.dumps(canonical, ensure_ascii=False, separators=(',', ':'))
    full = list(tokenizer.encode(prompt + raw + tokenizer.eos_token, add_special_tokens=False))
    require(full[:len(prefix)] == prefix, 'assistant_bpe_prefix_mismatch')
    target_ids = full[len(prefix):]
    require(type(tokenizer.eos_token_id) is int and bool(target_ids)
            and target_ids[-1] == tokenizer.eos_token_id, 'actual_eos_required')
    require(len(target_ids) <= CONFIG['max_output_tokens'], 'complete_target_over_budget')
    labels = [-100] * len(prefix) + target_ids
    return {'input_ids': full, 'attention_mask': [1] * len(full), 'labels': labels,
        'input_tokens': len(prefix), 'target_tokens': len(target_ids),
        'prompt_sha256': sha(prompt.encode()), 'schema_sha256': sha(encoded(frame['schema'])),
        'observation_sha256': sha(encoded(frame['observation'])), 'target_json_sha256': sha(raw.encode()),
        'full_token_ids_sha256': sha(encoded(full)), 'loss_mask_sha256': sha(encoded(labels))}


def build_claim(claim: GoldClaim, component: str, candidates: Sequence[Source],
                corpus: Mapping[int, Abstract], tokenizer: Any) -> dict[str, Any]:
    require(bool(claim.evidence), 'no_new_nei_fit_claim')
    initial, counters = capture(claim, candidates, corpus, tokenizer)
    frames = initial
    gaps: list[str] = []
    try:
        targets = answer_targets(claim, initial[0], candidates, 4)
    except ValueError:
        targets = []
        gaps.append('visible_positive_exceeds_terminal_contract')
    initial_choices = visible_choices(claim, initial[0], candidates)
    initial_complete = bool(initial_choices)
    plan = []
    gap = None
    if not targets and not gaps:
        plan = read_plan(claim, initial[0], candidates)
        if plan:
            frames, replay_counts = capture(claim, candidates, corpus, tokenizer, plan)
            require(frames[0] == initial[0], 'initial_replay_state_changed')
            for key in counters:
                counters[key] += replay_counts[key]
            try:
                targets = answer_targets(claim, frames[-1], candidates, 2)
            except ValueError:
                targets = []
                gaps.append('post_read_positive_exceeds_terminal_contract')
            if not targets:
                gap = 'post_read_no_complete_authorized_alternative'
        else:
            if any(int(s.source_id) in claim.evidence for s in candidates):
                # A retrieved but unexposed selected document may become usable
                # under a different legal read. Do not teach a false absence.
                gap = 'retrieved_positive_read_opportunity_unresolved'
            else:
                targets = [{'action': 'abstain', 'reason': 'insufficient_evidence'}]
    require(len(frames) in (1, 2), 'bounded_trajectory')
    candidates_rows = [{'doc_id': int(s.source_id), 'source_sha256': s.text_sha256} for s in candidates]
    report: dict[str, Any] = {'claim_id': claim.claim_id, 'component': component,
        'candidate_count': len(candidates), 'candidate_order_sha256': sha(encoded(candidates_rows)),
        'initial_complete': initial_complete, 'post_read_complete': bool(targets) if plan else None,
        'authorized_annotation_scope': CONFIG['annotation_view'], 'declared_claim_weight': 1,
        'planned_alternatives': len(targets), 'frames': len(frames), 'gaps': gaps,
        'official_document_count': len(claim.evidence),
        'official_alternative_count': sum(len(a) for a in claim.evidence.values()),
        'official_docs_in_candidates': sum(int(s.source_id) in claim.evidence for s in candidates),
        'whole_gold_covered_by_target': bool(targets) and targets[0]['action'] == 'answer'
            and len(targets[0]['documents']) == len(claim.evidence), **counters}
    final_choices = visible_choices(claim, frames[-1], candidates)
    report.update(initial_complete_document_count=len(initial_choices),
                  final_complete_document_count=len(final_choices),
                  available_complete_target_combinations=math.prod(len(c) for c in final_choices) if final_choices else 0,
                  target_alternatives_capped=bool(final_choices) and math.prod(len(c) for c in final_choices) > len(targets))
    records: list[dict[str, Any]] = []
    if gap:
        report['gaps'].append(gap)
    for alternative, answer in enumerate(targets):
        decisions = ([{'action': 'read', 'source_ids': plan}, answer] if plan else [answer])
        for number, (frame, decision) in enumerate(zip(frames, decisions, strict=True)):
            weight = Fraction(1, len(targets) * len(frames))
            row = {'version': VERSION, 'claim_id': claim.claim_id, 'component': component,
                'trajectory': 0, 'trajectory_count': 1, 'alternative': alternative, 'alternative_count': len(targets),
                'state_index': number, 'state_count': len(frames), 'frame': frame,
                'candidates': candidates_rows, 'target': decision, 'decision_origin': 'program_teacher',
                'semantic_target_provenance': ('official_complete_original_fit_annotation' if decision['action'] == 'answer' else 'no_semantic_label'),
                'action_target_provenance': 'program_teacher_actual_state_v2',
                'teacher_reason': ('complete_authorized_rationale_visible' if decision['action'] == 'answer'
                                   else 'existing_preview_read_to_authorized_evidence' if decision['action'] == 'read'
                                   else 'context_insufficient_for_authorized_annotation_view_not_official_nei'),
                'weight_numerator': weight.numerator, 'weight_denominator': weight.denominator,
                'declared_claim_weight': 1, 'epoch_normalizer': 48, 'model_generated': False}
            try:
                tokenized = tokenize_target(tokenizer, frame, decision, candidates)
            except ValueError as exc:
                report['gaps'].append(str(exc))
                continue
            row['packing'] = packing_metadata(tokenized)
            row['record_sha256'] = sha(encoded(row))
            records.append(row)
    require(len(records) <= 4, 'decision_record_ceiling')
    retained = sum((Fraction(r['weight_numerator'], r['weight_denominator']) for r in records), Fraction())
    report.update(retained_records=len(records), retained_weight_numerator=retained.numerator,
                  retained_weight_denominator=retained.denominator, training_ready=retained == 1 and not report['gaps'])
    return {'records': records, 'report': report, 'captured_frames': frames}


@dataclass(frozen=True)
class WeightContract:
    version: str = VERSION
    decision_origin: str = 'program_teacher'
    action_provenance: str = 'program_teacher_actual_state_v2'
    answer_provenance: str = 'official_complete_original_fit_annotation'
    epoch_normalizer: int = 48


def validate_weights(records: Sequence[Mapping[str, Any]], claim_ids: Sequence[int], *,
                     contract: WeightContract | None = None) -> None:
    spec = contract or WeightContract()
    require(type(spec.epoch_normalizer) is int and spec.epoch_normalizer > 0, 'positive_epoch_normalizer')
    require(len(set(claim_ids)) == len(claim_ids) and bool(claim_ids), 'claim_ids')
    require({r['claim_id'] for r in records} == set(claim_ids), 'missing_or_extra_claim')
    for claim_id in claim_ids:
        rows = [r for r in records if r['claim_id'] == claim_id]
        require(1 <= len(rows) <= 4, 'claim_record_bound')
        identities = {(r['trajectory'], r['alternative'], r['state_index']) for r in rows}
        require(len(identities) == len(rows), 'duplicate_supervision_state')
        mass = Fraction()
        for row in rows:
            require(row['version'] == spec.version and row['decision_origin'] == spec.decision_origin
                    and row['action_target_provenance'] == spec.action_provenance
                    and row['declared_claim_weight'] == 1 and row['epoch_normalizer'] == spec.epoch_normalizer,
                    'v2_provenance_or_normalizer')
            require(all(type(row[k]) is int and row[k] > 0 for k in
                        ('trajectory_count', 'alternative_count', 'state_count', 'weight_denominator', 'weight_numerator')),
                    'positive_hierarchy_counts')
            require(all(type(row[k]) is int and 0 <= row[k] < row[n] for k, n in
                        (('trajectory', 'trajectory_count'), ('alternative', 'alternative_count'), ('state_index', 'state_count'))),
                    'hierarchy_index')
            require(row['model_generated'] is False and row['semantic_target_provenance'] ==
                    (spec.answer_provenance if row['target']['action'] == 'answer' else 'no_semantic_label'),
                    'target_provenance')
            expected = Fraction(1, row['trajectory_count'] * row['alternative_count'] * row['state_count'])
            require(Fraction(row['weight_numerator'], row['weight_denominator']) == expected,
                    'weight_hierarchy')
            mass += expected
        require(mass == 1, 'missing_claim_weight_no_renormalization')


def packing_metadata(tokens: Mapping[str, Any]) -> dict[str, Any]:
    return {k: v for k, v in tokens.items() if k not in {'input_ids', 'attention_mask', 'labels'}}


def validate_tokenized(row: Mapping[str, Any], tokens: Mapping[str, Any]) -> None:
    require(row['record_sha256'] == sha(encoded({k: v for k, v in row.items() if k != 'record_sha256'})),
            'record_identity')
    require(row['packing'] == packing_metadata(tokens), 'packing_identity')
    n = tokens['input_tokens']
    inputs, labels = tokens['input_ids'], tokens['labels']
    require(type(n) is int and 0 < n < len(inputs) and len(inputs) == n + tokens['target_tokens']
            and tokens['attention_mask'] == [1] * len(inputs)
            and labels == [-100] * n + inputs[n:]
            and sha(encoded(inputs)) == tokens['full_token_ids_sha256']
            and sha(encoded(labels)) == tokens['loss_mask_sha256'], 'token_or_mask_identity')


def assistant_mean_loss(logits: Any, labels: Any) -> Any:
    """Causal token CE; EOS==PAD is irrelevant because masking is positional."""
    import torch
    import torch.nn.functional as functional
    shifted = labels[..., 1:]
    valid = shifted != -100
    require(bool(valid.any()), 'no_supervised_assistant_tokens')
    loss = functional.cross_entropy(logits[..., :-1, :].reshape(-1, logits.shape[-1]),
                                   shifted.reshape(-1), ignore_index=-100, reduction='sum')
    require(bool(torch.isfinite(loss)), 'nonfinite_loss')
    return loss / valid.sum()


def optimizer_step_for_claim_group(model: Any, optimizer: Any, records: Sequence[Mapping[str, Any]],
                                   tokenized: Sequence[Mapping[str, Any]], device: Any = 'cpu') -> dict[str, Any]:
    """Actual weighted backward/step, callable only by a separately released trainer.

    Fixed global denominator 48, including a last incomplete claim group. No
    per-chunk normalization; a complete epoch's unscaled numerators sum to the
    declared 48-claim objective. This function does not load a model or data.
    """
    import torch
    ids = list(dict.fromkeys(r['claim_id'] for r in records))
    require(1 <= len(ids) <= 4 and len(tokenized) == len(records), 'claim_group_shape')
    validate_weights(records, ids)
    optimizer.zero_grad(set_to_none=True)
    objective = 0.0
    for row, tokens in zip(records, tokenized, strict=True):
        validate_tokenized(row, tokens)
        inputs = torch.tensor([tokens['input_ids']], device=device)
        labels = torch.tensor([tokens['labels']], device=device)
        attention = torch.tensor([tokens['attention_mask']], device=device)
        loss = assistant_mean_loss(model(input_ids=inputs, attention_mask=attention).logits, labels)
        coefficient = float(Fraction(row['weight_numerator'], row['weight_denominator'])) / 48
        weighted = loss * coefficient
        objective += float(weighted.detach().cpu())
        weighted.backward()
    require(math.isfinite(objective), 'nonfinite_group_objective')
    gradient_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0, error_if_nonfinite=True)
    optimizer.step()
    return {'global_objective_contribution': objective, 'epoch_normalizer': 48,
            'claim_count': len(ids), 'decision_records': len(records),
            'gradient_norm_before_clip': float(gradient_norm), 'optimizer_steps': 1}


def candidates_from_rows(rows: Sequence[Mapping[str, Any]], corpus: Mapping[int, Abstract]) -> list[Source]:
    sources = [source_from_abstract(corpus[r['doc_id']]) for r in rows]
    require(len({s.source_id for s in sources}) == len(sources), 'duplicate_candidate')
    require(all(s.text_sha256 == row['source_sha256'] for s, row in zip(sources, rows, strict=True)),
            'candidate_source_changed')
    return sources
