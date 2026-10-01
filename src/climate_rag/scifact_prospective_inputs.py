"""Metadata-only prospective TRAIN selection and no-generation initial frames.

Shared corpus/early gold preparation exposure remains explicit. This does not
create an independent test set or change the evidence-commit decision policy.
"""
from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path
import time
from typing import Any

from .scifact_bounded_agent import SciFactBoundedAgent
from .scifact_bounded_runtime import CommonPacking
from .scifact_natural_selection import component_rows
from .scifact_semantic_contract import checked, encoded, sha, write_once
from .scifact_terminal import PROTOCOL as TERMINAL, render_scifact_prompt, source_from_abstract
from .scifact_utility_contract import UtilityDiagnostic, identity
from .scifact_natural_contract import require
from .verification import normalise_claim

PROTOCOL = 'scifact-evidence-commit-prospective24-v1-20261002'
SALT = PROTOCOL
POOL_COMPONENTS = (64, 'b421a1d70dd9e4f7c40b2b79731d392a4382e3d5c149a05f345a98270009a5eb')
POOL_IDS = (97, '56376abd5a8473a3738b37779221c6381c379864b5571e7057ee351db590068e')
SCOPE = 'conditional_metadata_selected_TRAIN_early_gold_preparation_seen_not_independent_test'


def select_metadata(pool: Any, metadata_sha: dict[str, str]) -> dict[str, Any]:
    rows = component_rows(pool)
    components, ids = {r['component'] for r in rows}, {r['id'] for r in rows}
    require((len(components), sha(encoded(sorted(components)))) == POOL_COMPONENTS
            and (len(ids), sha(encoded(sorted(ids)))) == POOL_IDS, 'remaining_pool_snapshot_drift_no_replacement')
    by_component = {c: [r['id'] for r in rows if r['component'] == c] for c in components}
    ordered = sorted(components, key=lambda c: (sha(encoded([SALT, 'component', c])), c))
    selected = [{'id': min(by_component[c], key=lambda i: (sha(encoded([SALT, 'claim', c, i])), i)),
                 'component': c} for c in ordered[:24]]
    require(len(selected) == 24, 'no_shortfall_or_replacement')
    return {'protocol': PROTOCOL, 'salt': SALT, 'scope': SCOPE, 'selected': selected,
            'ordering': 'sha256(semantic_encoded([salt,domain,component,(id)])),stable_ties',
            'pool_components': {'count': len(components), 'sha256': POOL_COMPONENTS[1]},
            'pool_ids': {'count': len(ids), 'sha256': POOL_IDS[1]},
            'input_metadata_sha256': metadata_sha, 'ordered_ids_sha256': identity([r['id'] for r in selected]),
            'pool_id_component_sha256': sha(encoded(sorted(rows, key=lambda r: r['id']))),
            'gold_preparation_seen_eligible_train': 531, 'source_family_unseen_claimed': False,
            'labels_text_outcomes_used_for_selection': False, 'replacement_allowed': False,
            'model_calls': 0, 'new_gold_deserialized': False}


def freeze_selection(out: Path, selection: Any) -> dict[str, str]:
    out.mkdir(mode=0o700)
    write_once(out/'selection.json', selection)
    write_once(out/'component-reservations.json', {'protocol': PROTOCOL,
        'selection_sha256': sha((out/'selection.json').read_bytes()),
        'state': 'reserved_before_query_read_keep_on_failure', 'selected': selection['selected']})
    return {name: sha((out/name).read_bytes()) for name in ('selection.json', 'component-reservations.json')}


def reserved_selection(out: Path, selection_sha: str, reservation_sha: str) -> Any:
    selection = json.loads(checked(out/'selection.json', selection_sha))
    reservation = json.loads(checked(out/'component-reservations.json', reservation_sha))
    rows = component_rows(selection['selected'])
    require(selection['protocol'] == PROTOCOL and selection['salt'] == SALT and selection['scope'] == SCOPE
            and len(rows) == len({r['component'] for r in rows}) == 24
            and selection['ordered_ids_sha256'] == identity([r['id'] for r in rows])
            and reservation == {'protocol': PROTOCOL, 'selection_sha256': selection_sha,
                'state': 'reserved_before_query_read_keep_on_failure', 'selected': rows},
            'selection_reservation_binding')
    return selection


class TokenCounter:
    name, kind, terminal_protocol = 'initial-frame-tokenizer-only', 'fixture', TERMINAL

    def __init__(self, tokenizer: Any) -> None:
        self.tokenizer = tokenizer

    def count_text(self, text: str) -> int:
        return len(self.tokenizer.encode(text, add_special_tokens=False))

    def count_prompt(self, observation: Any, schema: Any) -> int:
        return self.count_text(render_scifact_prompt(self.tokenizer, observation, schema))

    def generate(self, *args: Any, **kwargs: Any) -> Any:
        raise AssertionError('initial_frame_preparation_must_not_generate')


class _FrameReady(BaseException):
    """Stop at the existing observer before the first model attempt is created."""


def prepare_frame(claim: Any, retrieve: Any, tokenizer: Any) -> tuple[Any, Any]:
    require(set(claim) == {'id', 'claim'} and type(claim['id']) is int and isinstance(claim['claim'], str),
            'gold_free_claim_only')
    provider = TokenCounter(tokenizer)
    frames: list[Any] = []
    retrieval: list[Any] = []
    started = time.perf_counter()

    def tracked(query: str, width: int) -> Any:
        require(not retrieval and query == normalise_claim(claim['claim']) and width == 20,
                'single_original_initial_BM25_top20')
        began = time.perf_counter()
        sources = list(retrieve(query, width))[:width]
        retrieval.append({'query_sha256': sha(query.encode()), 'candidate_k': width,
            'source_ids': [s.source_id for s in sources],
            'source_sha256': [s.text_sha256 for s in sources],
            'elapsed_seconds': time.perf_counter()-began})
        require(len({s.source_id for s in sources}) == len(sources), 'unique_initial_sources')
        return sources

    def observer(kind: str, frame: Any) -> None:
        require(kind == 'initial_frame', 'unexpected_model_or_later_frame')
        frames.append(copy.deepcopy(frame))
        raise _FrameReady()

    def forbidden_rerank(*args: Any) -> Any:
        raise AssertionError('no_rerank_or_model_during_initial_preparation')

    try:
        SciFactBoundedAgent(provider, tracked, rerank=forbidden_rerank,
            packing_count=CommonPacking(provider, tokenizer)).run(claim['claim'], 'adaptive',
                diagnostic=UtilityDiagnostic(observer))
    except _FrameReady:
        pass
    require(len(frames) == len(retrieval) == 1, 'initial_frame_missing_no_replacement')
    frame = frames[0]
    aliases = {sid.rsplit(':', 1)[0] for sid in frame['visible']}
    frame['document_order'] = [a for a in frame['alias_to_source'] if a in aliases]
    # No fake generation prefix/receipt. Preparation is measured once, shared by all arms.
    frame['initial_retrieval'] = {'status': 'new_cpu_preparation_shared_three_arms', 'tool_debit': 1,
        'retrieval_elapsed_seconds': retrieval[0]['elapsed_seconds'],
        'cost': 'measured_shared_cpu_preparation_not_per_arm_free_online_retrieval'}
    receipt = {'claim_id': claim['id'], 'frame_sha256': identity(frame), **retrieval[0],
        'preparation_elapsed_seconds': time.perf_counter()-started,
        'model_calls': 0, 'physical_generation_prefix_created': False}
    return frame, receipt


def validate_frames(claims: Any, frames: Any, corpus: Any, tokenizer: Any) -> list[Any]:
    require(len(claims) == len(frames) == 24 and len({c['id'] for c in claims}) == 24, 'all24_frames')
    for claim, frame in zip(claims, frames, strict=True):
        require(set(claim) == {'id', 'claim'} and frame['observation']['immutable_claim'] == normalise_claim(claim['claim']),
                'frame_claim_mismatch')
        require(frame['observation']['current_citable'] == [
            {'sentence_id': sid, 'text': row['text']} for sid, row in frame['visible'].items()], 'visible_order_changed')
        for sid, row in frame['visible'].items():
            alias, index = sid.rsplit(':', 1)
            source = source_from_abstract(corpus[int(row['source_id'])])
            require(frame['alias_to_source'][alias] == row['source_id'] and str(row['sentence_index']) == index
                    and type(row['sentence_index']) is int and 0 <= row['sentence_index'] < len(source.sentences)
                    and row['text'] == source.sentences[row['sentence_index']]
                    and row['text_sha256'] == hashlib.sha256(row['text'].encode()).hexdigest()
                    and row['source_text_sha256'] == source.text_sha256, 'original_source_sentence_identity')
        visible_aliases = {sid.rsplit(':', 1)[0] for sid in frame['visible']}
        require(frame['document_order'] == [a for a in frame['alias_to_source'] if a in visible_aliases]
                and frame['prompt_tokens'] == len(tokenizer.encode(render_scifact_prompt(tokenizer,
                    frame['observation'], frame['schema']), add_special_tokens=False))
                and frame['initial_retrieval']['status'] == 'new_cpu_preparation_shared_three_arms', 'frame_packing_contract')
    return list(frames)
