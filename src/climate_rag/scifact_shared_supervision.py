"""Shared-corpus protocol/provenance around the unchanged v2 record builder."""
from __future__ import annotations

from collections import Counter
from collections.abc import Mapping, Sequence
from typing import Any

from .scifact_grounding import Abstract
from .scifact_semantic_contract import CORPUS_SHA, encoded, sha
from .scifact_state_supervision import (
    CONFIG as HISTORICAL_CONFIG, VERSION as RECORD_SCHEMA, build_claim,
    config_sha as historical_config_sha, require,
)
from .scifact_terminal import render_scifact_prompt

VERSION = 'scifact-shared-corpus-fit-state-v1-20261001'
CONFIG = dict(HISTORICAL_CONFIG, version=VERSION, record_schema=RECORD_SCHEMA,
              candidate_pool='all_5183_public_corpus_gold_free_retrieval',
              historical_structural_config_sha256=historical_config_sha(),
              only_experimental_change='candidate_pool_including_its_BM25_statistics',
              source_family_unseen_claim=False, optimizer_execution=False)


def config_sha() -> str:
    return sha(encoded(CONFIG))


def provenance() -> dict[str, Any]:
    return {'data_protocol': VERSION, 'data_protocol_sha256': config_sha(),
            'record_schema': RECORD_SCHEMA, 'candidate_pool': CONFIG['candidate_pool'],
            'candidate_pool_documents': 5183, 'corpus_sha256': CORPUS_SHA, 'source_family_unseen_claim': False}


def stamp_shared_protocol(built: dict[str, Any]) -> dict[str, Any]:
    """Keep v2 schema/validators; explicitly replace data origin, then rehash."""
    for row in built['records']:
        require(row['version'] == RECORD_SCHEMA, 'historical_record_schema')
        row['data_provenance'] = provenance()
        row['record_sha256'] = sha(encoded({k: v for k, v in row.items() if k != 'record_sha256'}))
    built['report']['data_provenance'] = provenance()
    return built


def build_shared_claim(*args: Any, **kwargs: Any) -> dict[str, Any]:
    # No controller, packing, teacher, target selection, weighting or cap fork.
    return stamp_shared_protocol(build_claim(*args, **kwargs))


def document_partitions(families: Mapping[str, Any]) -> dict[int, str]:
    owners, parts = families['family_component'], families['component_partition']
    return {int(d): parts[owners[str(f)]] if str(f) in owners else 'unowned'
            for d, f in families['document_family'].items()}


def describe_documents(ids: set[int], partitions: Mapping[int, str]) -> dict[str, Any]:
    return {'documents': len(ids), 'document_ids_sha256': sha(encoded(sorted(ids))),
            'known_owner_categories': dict(sorted(Counter(partitions[d] for d in ids).items()))}


def exposure_for_frame(frame: Mapping[str, Any], candidates: Sequence[Mapping[str, Any]],
                       corpus: Mapping[int, Abstract]) -> dict[str, Any]:
    """Public document text actually visible, not every schema/candidate alias."""
    observation = frame['observation']
    aliases = {f'c{i}': row for i, row in enumerate(candidates)}
    require(observation['feedback'] in (None, 'tool_completed: inspect current_citable; no semantic conclusion implied'),
            'unexpected_feedback_text_needs_exposure_audit')
    visible, previews = [], []
    for sentence in observation['current_citable']:
        alias, index = sentence['sentence_id'].split(':')
        source = aliases[alias]
        visible.append({'doc_id': source['doc_id'], 'source_sha256': source['source_sha256'],
                        'sentence_index': int(index), 'text_sha256': sha(sentence['text'].encode())})
    for preview in observation['preview_only']:
        source = aliases[preview['source_id']]
        require(preview['citable'] is False, 'preview_not_citable')
        document = corpus[source['doc_id']]
        require(preview['preview'] == (document.title+' '+document.sentences[0])[:128], 'actual_preview_text')
        previews.append({'doc_id': source['doc_id'], 'source_sha256': source['source_sha256'],
                         'text_sha256': sha(preview['preview'].encode())})
    return {'frame_sha256': sha(encoded(frame)), 'visible_sentences': visible, 'previews': previews,
            'feedback': observation['feedback'], 'remaining_calls': observation['remaining_calls'],
            'remaining_tools': observation['remaining_tools']}


def layer_inventory(records: Sequence[Mapping[str, Any]], captured: Sequence[Mapping[str, Any]],
                    indexed: Sequence[Mapping[str, Any]], partitions: Mapping[int, str],
                    corpus: Mapping[int, Abstract]) -> tuple[dict[str, Any], dict[str, Any]]:
    prompts, targets, runtime = [], [], []
    for record in records:
        require(record['data_provenance'] == provenance(), 'shared_record_protocol')
        entry = exposure_for_frame(record['frame'], record['candidates'], corpus)
        prompts.append(dict(entry, claim_id=record['claim_id'], record_sha256=record['record_sha256'],
                            rendered_prompt_sha256=record['packing']['prompt_sha256'],
                            trajectory=record['trajectory'], alternative=record['alternative'], state_index=record['state_index']))
        aliases = {f'c{i}': row for i, row in enumerate(record['candidates'])}
        target = record['target']
        selected = target.get('documents', [])
        ids = ([aliases[d['source_id']]['doc_id'] for d in selected] if target['action'] == 'answer'
               else [aliases[a]['doc_id'] for a in target.get('source_ids', [])])
        targets.append({'claim_id': record['claim_id'], 'record_sha256': record['record_sha256'],
                        'action': target['action'], 'doc_ids': ids, 'target_sha256': sha(encoded(target))})
    for claim in captured:
        runtime.append({'claim_id': claim['claim_id'], 'states': [exposure_for_frame(f, claim['candidates'], corpus)
                                                                  for f in claim['frames']]})
    def exposed(entries: Sequence[Mapping[str, Any]]) -> set[int]:
        return {s['doc_id'] for e in entries for s in e['visible_sentences']+e['previews']}
    private = {'indexed': list(indexed), 'retrieved': [{'claim_id': c['claim_id'], 'candidates': c['candidates']} for c in captured],
               'prepared_prompts': prompts, 'supervised_targets': targets,
               'captured_runtime_states_including_gaps': runtime}
    aggregate = {
        'indexed': describe_documents({r['doc_id'] for r in indexed}, partitions),
        'retrieved_candidates': describe_documents({r['doc_id'] for c in captured for r in c['candidates']}, partitions),
        'prepared_prompt_text': describe_documents(exposed(prompts), partitions),
        'prepared_citable_text': describe_documents({s['doc_id'] for p in prompts for s in p['visible_sentences']}, partitions),
        'prepared_preview_text': describe_documents({s['doc_id'] for p in prompts for s in p['previews']}, partitions),
        'supervised_target_documents': describe_documents({d for t in targets for d in t['doc_ids']}, partitions),
        'supervised_target_by_action': {a: describe_documents({d for t in targets if t['action'] == a for d in t['doc_ids']}, partitions)
                                        for a in ('answer', 'read', 'abstain')},
        'captured_runtime_text_including_gaps': describe_documents(exposed([s for r in runtime for s in r['states']]), partitions),
        'prepared_records': len(prompts), 'unique_prepared_prompt_hashes': len({p['rendered_prompt_sha256'] for p in prompts}),
        'unique_prepared_sentences': len({(s['doc_id'], s['sentence_index'], s['text_sha256']) for p in prompts for s in p['visible_sentences']}),
        'known_owner_mapping_is_descriptive_not_safe_negative_label': True,
        'actual_training_performed': False,
    }
    return private, aggregate


def paired_summary(old: Sequence[Mapping[str, Any]], new: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    require([r['claim_id'] for r in old] == [r['claim_id'] for r in new]
            and all(a['component'] == b['component'] for a, b in zip(old, new, strict=True)), 'same_ordered_fit_components')
    def state(r: Mapping[str, Any]) -> str:
        if r['gaps']:
            return 'gap'
        if r['initial_complete']:
            return 'initial_complete'
        return 'post_read_complete' if r['post_read_complete'] else 'context_abstain'
    transitions = Counter(state(a)+' -> '+state(b) for a, b in zip(old, new, strict=True))
    return {'aligned_claims': len(old), 'state_transitions': dict(sorted(transitions.items())),
            'old_records': sum(r['retained_records'] for r in old), 'new_records': sum(r['retained_records'] for r in new),
            'causal_model_improvement_claimed': False}


def read_transition_summary(captured: Sequence[Mapping[str, Any]], reports: Sequence[Mapping[str, Any]],
                            records: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Outcomes, not gates: distinguish executions from repeated teacher records."""
    by_id = {r['claim_id']: r for r in reports}
    reads = [c for c in captured if len(c['frames']) == 2]
    increments = []
    for claim in reads:
        identities = [{(s['sentence_id'], sha(s['text'].encode())) for s in f['observation']['current_citable']}
                      for f in claim['frames']]
        increments.append(len(identities[1]-identities[0]))
    return {'executed_read_claims': len(reads),
            'read_decision_records_including_alternative_duplicates': sum(r['target']['action'] == 'read' for r in records),
            'read_claims_with_new_citable_sentences': sum(n > 0 for n in increments),
            'new_citable_sentence_count_per_read_histogram': dict(sorted(Counter(str(n) for n in increments).items())),
            'read_claims_with_complete_post_read_rationale': sum(by_id[c['claim_id']]['post_read_complete'] is True for c in reads),
            'read_claims_with_gaps': sum(bool(by_id[c['claim_id']]['gaps']) for c in reads),
            'read_claims_with_no_new_citable_sentence': sum(n == 0 for n in increments)}


def matched_fullpool_state(old_states: Sequence[Mapping[str, Any]], captured: Sequence[Mapping[str, Any]],
                          records: Sequence[Mapping[str, Any]], tokenizer: Any) -> dict[str, Any]:
    by_id = {r['claim_id']: r for r in captured}
    matches = [r for r in old_states if r['frozen_read_ids'] and r['claim_id'] in by_id]
    require(len(matches) == 1, 'one_predeclared_matched_case')
    old = matches[0]
    new = by_id[old['claim_id']]
    candidates = [r['doc_id'] for r in new['candidates']]
    doc = old['candidate_doc_ids'][int(old['frozen_read_ids'][0][1:])]
    read_targets = [r['target']['source_ids'] for r in records if r['claim_id'] == old['claim_id'] and r['target']['action'] == 'read']
    same_read = bool(read_targets) and all(ids == old['frozen_read_ids'] for ids in read_targets)
    prompts = [render_scifact_prompt(tokenizer, f['observation'], f['schema']) for f in new['frames']]
    return {'case': 'predeclared_only_fit_overlap_in_old_three_reads',
            'old_witness_rank': old['candidate_doc_ids'].index(doc)+1,
            'new_witness_rank': candidates.index(doc)+1 if doc in candidates else None,
            'candidate_order_restored': candidates == old['candidate_doc_ids'],
            'initial_state_hash_restored': sha(encoded(new['frames'][0])) == old['states'][0]['state_sha256'],
            'initial_rendered_prompt_hash_restored': sha(prompts[0].encode()) == old['states'][0]['prompt_sha256'],
            'initial_prompt_token_count_restored': len(tokenizer.encode(prompts[0], add_special_tokens=False)) == old['states'][0]['prompt_tokens'],
            'same_read_ids_as_old': same_read,
            'post_read_state_hash_restored': same_read and len(new['frames']) == 2 and sha(encoded(new['frames'][1])) == old['states'][1]['state_sha256'],
            'post_read_rendered_prompt_hash_restored': same_read and len(prompts) == 2 and sha(prompts[1].encode()) == old['states'][1]['prompt_sha256'],
            'post_read_prompt_token_count_restored': same_read and len(prompts) == 2 and len(tokenizer.encode(prompts[1], add_special_tokens=False)) == old['states'][1]['prompt_tokens'],
            'natural_read_frames': len(new['frames'])-1}
