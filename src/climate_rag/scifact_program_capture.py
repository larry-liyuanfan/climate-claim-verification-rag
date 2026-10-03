"""Gold-free scripted initial-state capture, separate from physical model receipts."""
from __future__ import annotations

import copy
import json
from collections.abc import Mapping
from typing import Any

from .scifact_bounded_runtime import ARMS, run_bounded_slot
from .scifact_grounding import Abstract, parse_gold
from .scifact_retrieval import SciFactBM25
from .scifact_semantic_contract import sha
from .scifact_state_supervision import CaptureTeacher, require
from .scifact_state_training_driver import captured_state, corpus_identity
from .scifact_terminal import render_scifact_prompt, source_from_abstract
from .scifact_terminal_supervision import _View, _enumerate_bound, _unique_object
from .scifact_utility_contract import UtilityDiagnostic, identity
from .verification import normalise_claim

VERSION = 'scifact-program-initial-NEI-v1-20261001'
COUNTS = {'model_calls': 0, 'real_model_calls': 0, 'scripted_responses': 1,
          'model_tool_executions': 0, 'program_retrieve_events': 1, 'controller_provider_attempts': 1}


def capture_initial(inference: Mapping[str, Any], retrieve: SciFactBM25,
                    corpus: Mapping[int, Abstract], tokenizer: Any) -> dict[str, Any]:
    """Accept only {id,claim}; real BM25 top20 and production CommonPacking, no LLM."""
    require(set(inference) == {'id', 'claim'} and type(inference['id']) is int
            and isinstance(inference['claim'], str) and bool(normalise_claim(inference['claim'])),
            'gold_free_inference_contract')
    require(type(retrieve) is SciFactBM25, 'production_BM25_required')
    require({k: s.text_sha256 for k, s in retrieve.sources.items()} ==
            {str(k): source_from_abstract(v).text_sha256 for k, v in corpus.items()}, 'full_BM25_corpus_binding')
    require(retrieve.index.doc_ids == [str(k) for k in sorted(corpus)]
            and retrieve.index.texts == [retrieve.sources[k].title+' '+' '.join(retrieve.sources[k].sentences)
                                       for k in retrieve.index.doc_ids]
            and retrieve.index.k1 == 1.5 and retrieve.index.b == 0.75, 'actual_BM25_index_binding')
    teacher = CaptureTeacher(tokenizer)
    frames: list[dict[str, Any]] = []
    retrieved: list[dict[str, Any]] = []

    def observed(kind: str, frame: dict[str, Any]) -> None:
        if kind == 'initial_frame':
            frames.append(frame)

    def search(query: str, width: int) -> Any:
        require(not retrieved and query == normalise_claim(inference['claim']) and width == 20,
                'single_initial_BM25_top20')
        sources = list(retrieve(query, width))
        retrieved.append({'query_sha256': sha(query.encode()), 'width': width,
                          'document_ids': [s.source_id for s in sources],
                          'source_sha256': [s.text_sha256 for s in sources]})
        return sources

    def no_rerank(*args: Any) -> Any:
        raise AssertionError('scripted_abstain_must_not_rerank')

    result = run_bounded_slot(inference['id'], inference['claim'], 'adaptive', ARMS[0],
        teacher, search, no_rerank, corpus, diagnostic=UtilityDiagnostic(observed))['result']
    require(len(frames) == len(teacher.frames) == len(retrieved) == 1
            and result['outcome'] == 'model_abstention:insufficient_evidence'
            and result['model_calls'] == result['tool_calls'] == 1
            and len(result['events']) == len(result['generation_attempts']) == 1
            and result['generation_attempts'][0]['status'] == 'valid_decision',
            'program_capture_controller_contract')
    capture = frames[0]
    prompt = render_scifact_prompt(tokenizer, capture['observation'], capture['schema'])
    payload = {'version': VERSION, 'channel': 'program_capture', 'inference': dict(inference),
        'capture': capture, 'state_event': result['events'][0], 'retrieval': retrieved[0],
        'corpus_sha256': corpus_identity(corpus), 'prompt_identity': identity(prompt),
        'prompt_utf8_sha256': sha(prompt.encode()),
        'token_ids_identity': identity(tokenizer.encode(prompt, add_special_tokens=False)),
        'counts': dict(COUNTS), 'controller_counts': {'model_calls': result['model_calls'],
            'tool_calls': result['tool_calls'], 'meaning': 'fixture_provider_attempt_not_real_model_call'}}
    receipt = {'payload': payload, 'sha256': identity(payload)}
    validate_capture(receipt, corpus, tokenizer)
    return receipt


def validate_capture(receipt: Mapping[str, Any], corpus: Mapping[int, Abstract], tokenizer: Any
                     ) -> tuple[dict[str, Any], list[Any], dict[str, str], list[str]]:
    require(set(receipt) == {'payload', 'sha256'} and receipt['sha256'] == identity(receipt['payload']),
            'program_capture_seal')
    p = receipt['payload']
    require(set(p) == {'version', 'channel', 'inference', 'capture', 'state_event', 'retrieval',
                       'corpus_sha256', 'prompt_identity', 'prompt_utf8_sha256', 'token_ids_identity',
                       'counts', 'controller_counts'} and p['version'] == VERSION
            and p['channel'] == 'program_capture', 'strict_program_channel')
    require(set(p['inference']) == {'id', 'claim'} and type(p['inference']['id']) is int
            and isinstance(p['inference']['claim'], str), 'gold_free_inference_contract')
    capture, event = p['capture'], p['state_event']
    require(set(event) == {'tool', 'model_selected', 'before_context', 'status', 'query_sha256',
                           'requested_context', 'candidate_ids', 'elapsed_ms', 'source_sha256'},
            'strict_initial_retrieve_event_fields')
    require(set(capture) == {'observation', 'schema', 'visible', 'prompt_tokens', 'alias_to_source'},
            'full_five_field_program_capture')
    frame, sources, registry, order = captured_state(capture, event, corpus)
    obs = frame['observation']
    require(p['corpus_sha256'] == corpus_identity(corpus)
            and obs['immutable_claim'] == normalise_claim(p['inference']['claim'])
            and obs['remaining_calls'] == 5 and obs['remaining_tools'] == 4 and obs['feedback'] is None,
            'initial_claim_corpus_budgets')
    expected_actions = ['abstain'] + (['answer'] if capture['visible'] else []) + ['read', 'rewrite', 'rerank']
    require(obs['allowed_actions'] == expected_actions, 'production_initial_allowed_actions')
    require(event['tool'] == 'retrieve' and event['model_selected'] is False
            and event['before_context'] == [] and event.get('origin') is None
            and event['requested_context'] == order[:5]
            and order == list(registry) == [f'c{i}' for i in range(len(order))],
            'initial_retrieve_order')
    retrieval = p['retrieval']
    require(set(retrieval) == {'query_sha256', 'width', 'document_ids', 'source_sha256'}
            and retrieval['query_sha256'] == event['query_sha256'] == sha(obs['immutable_claim'].encode())
            and retrieval['width'] == 20
            and retrieval['document_ids'] == [registry[a] for a in order]
            and retrieval['source_sha256'] == [event['source_sha256'][a] for a in order],
            'BM25_initial_retrieval_binding')
    prompt = render_scifact_prompt(tokenizer, obs, frame['schema'])
    tokens = tokenizer.encode(prompt, add_special_tokens=False)
    require(p['prompt_identity'] == identity(prompt) and p['prompt_utf8_sha256'] == sha(prompt.encode())
            and p['token_ids_identity'] == identity(tokens) and capture['prompt_tokens'] == len(tokens),
            'exact_program_prompt_tokens')
    require(p['counts'] == COUNTS and p['controller_counts'] == {'model_calls': 1, 'tool_calls': 1,
            'meaning': 'fixture_provider_attempt_not_real_model_call'}, 'program_real_cost_separation')
    return frame, sources, registry, order


def program_candidates(receipt: Mapping[str, Any], annotation: bytes, provenance: Mapping[str, Any],
                       corpus: Mapping[int, Abstract], tokenizer: Any) -> dict[str, Any]:
    """Strict official-NEI binder; no model proposal or physical attempt identifiers."""
    require(set(provenance) == {'payload', 'sha256'} and provenance['sha256'] == identity(provenance['payload']),
            'program_provenance_seal')
    p = provenance['payload']
    require(set(p) == {'version', 'scope', 'annotation_sha256', 'capture_sha256', 'corpus_sha256',
                       'complete_original_row_declared', 'selection_sha256', 'parent_gold_sha256'}
            and p['version'] == VERSION and p['scope'] in {'synthetic_fixture', 'exposed_train_NEI47'}
            and p['complete_original_row_declared'] is True, 'program_annotation_contract')
    require(p['annotation_sha256'] == sha(annotation) and p['capture_sha256'] == receipt['sha256']
            and p['corpus_sha256'] == corpus_identity(corpus)
            and all(isinstance(p[k], str) and len(p[k]) == 64 and set(p[k]) <= set('0123456789abcdef')
                    for k in ('selection_sha256', 'parent_gold_sha256')), 'program_annotation_identity')
    row = json.loads(annotation, object_pairs_hook=_unique_object)
    require(set(row) == {'id', 'claim', 'evidence', 'cited_doc_ids'}, 'complete_original_annotation_required')
    gold = parse_gold(row, corpus)
    require(not gold.evidence, 'program_channel_official_NEI_only')
    frame, sources, registry, order = validate_capture(receipt, corpus, tokenizer)
    capture = receipt['payload']
    require(gold.claim_id == capture['inference']['id']
            and normalise_claim(gold.claim) == frame['observation']['immutable_claim'], 'annotation_claim_identity')
    view = _View(gold, frame, sources, registry, order, capture['capture']['visible'], {},
                 copy.deepcopy(dict(provenance)), copy.deepcopy(capture['state_event']))
    result = _enumerate_bound(view, tokenizer, 1)
    result.update(semantic_kernel_version=result['version'], version=VERSION,
                  program_capture=copy.deepcopy(dict(receipt)), channel='program_capture')
    return result
