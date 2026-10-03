"""Versioned program-artifact bridge; no gold, selection, physical replay or I/O.

Legacy rows remain nested byte-semantically unchanged, including historical /48.
The derived optimizer records use actual N, not the old metadata or row count.
Real preparation requires a separate exact-source release; these functions do
not load files, build a cohort, publish a bundle, or authorize training.
"""
from __future__ import annotations

import copy
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import asdict
from fractions import Fraction
import json
import math
import random
from typing import Any

from .scifact_grounding import Abstract
from .scifact_program_capture import VERSION as PROGRAM_VERSION, validate_capture
from .scifact_semantic_contract import encoded, sha
from .scifact_state_supervision import (
    WeightContract, candidates_from_rows, packing_metadata, require,
    tokenize_target, validate_tokenized, validate_weights,
)
from .scifact_utility_contract import identity

VERSION = 'scifact-mixed-program-input-v1-20261001'
SCOPE = 'exposed_train_old48_supp49_NEI47'
LEGACY_SHA = {
    'old48': '42e18a523026e617edeb0a1597e734e6d9431f324849caa487c8f798d51a58fa',
    'supp49': '49ebe0b020535f72507b6ed96fb01254327487f3d063f8160ff19f38d77f8630',
}
NEI_COMPACT_SHA = 'bbee6b6e0bf7d92ab6199911ddcc22896c933cdffcd57f035d0adeb75be62799'
NEI_SELECTION_SHA = 'a55441fa2a9bace07b307ccf2c0b622a033323a8923e2fb3e047a782e32c666a'
ROSTER_SHA = {
    'old48': '22fab53c7ab0be1c1e3d53bb2c00a19300d72334696eb8cf2ecf87c2f3ff739c',
    'supp49': '8ab8b61189a23350a7d55958a781f8da8c86fa7164e78cd0fdb5569ccf763b65',
    'NEI47': '066f4140a1cf1337ee8283fdc3b227aa5fa420cfcd5b18b2153edc7d2834651a',
}
ORIGINS = {
    'answer': ('complete_authorized_rationale_visible', 'visible_OR_rationale'),
    'read': ('existing_preview_read_to_authorized_evidence', 'program_read_witness'),
    'abstain': ('context_insufficient_for_authorized_annotation_view_not_official_nei', 'context_insufficient'),
}
HIERARCHY = ('trajectory', 'trajectory_count', 'alternative', 'alternative_count', 'state_index', 'state_count',
             'weight_numerator', 'weight_denominator', 'declared_claim_weight')


def input_config_sha() -> str:
    return sha(encoded({'version': VERSION, 'scope': SCOPE, 'legacy_files': LEGACY_SHA,
        'rosters': ROSTER_SHA, 'NEI_compact': NEI_COMPACT_SHA, 'NEI_selection': NEI_SELECTION_SHA,
        'target_origins': ORIGINS, 'actual_claims': 144, 'rows': 223, 'max_rows_per_claim': 4,
        'legacy_metadata_unchanged': True, 'reselection_or_renormalization': False}))


def seal(payload: Mapping[str, Any]) -> dict[str, Any]:
    value = copy.deepcopy(dict(payload))
    return {'payload': value, 'sha256': identity(value)}


def unseal(value: Mapping[str, Any]) -> dict[str, Any]:
    require(set(value) == {'payload', 'sha256'} and value['sha256'] == identity(value['payload']), 'mixed_seal')
    return copy.deepcopy(dict(value['payload']))


def weights_for(n: int) -> WeightContract:
    return WeightContract(VERSION, 'frozen_program_supervision', 'source_specific_program_artifact',
                          'visible_OR_rationale', n)


def legacy_envelopes(raw: bytes, cohort: str, claim_ids: Sequence[int],
                     corpus: Mapping[int, Abstract], tokenizer: Any, *, scope: str) -> list[dict[str, Any]]:
    require(cohort in LEGACY_SHA and scope in {SCOPE, 'synthetic_fixture'}, 'legacy_scope')
    digest = sha(raw)
    require(scope == 'synthetic_fixture' or digest == LEGACY_SHA[cohort], 'frozen_legacy_file')
    rows = json.loads(raw)
    require(isinstance(rows, list), 'legacy_rows_list')
    validate_weights(rows, claim_ids)
    result = []
    for index, row in enumerate(rows):
        require(not set(row) & {'tool_event', 'capture', 'physical_attempt', 'model_selected'}, 'invented_legacy_event')
        reason, origin = ORIGINS[row['target']['action']]
        require(row['teacher_reason'] == reason, 'legacy_origin')
        sources = candidates_from_rows(row['candidates'], corpus)
        tokens = tokenize_target(tokenizer, row['frame'], row['target'], sources)
        validate_tokenized(row, tokens)
        result.append({'kind': 'legacy_program_artifact', 'cohort': cohort,
            'source_file_sha256': digest, 'source_row_index': index, 'legacy_row': copy.deepcopy(row),
            'target_origin': origin, 'event_lineage_available': False, 'tokenized': tokens})
    return result


def nei_envelope(raw: bytes, *, expected_sha: str, component: str, inventory: Mapping[str, Any],
                 artifact_name: str, corpus: Mapping[int, Abstract], tokenizer: Any,
                 scope: str) -> dict[str, Any]:
    """Validate the already accepted NEI artifact; never reopen its annotation."""
    require(scope in {SCOPE, 'synthetic_fixture'} and sha(raw) == expected_sha, 'NEI_artifact_identity')
    compact = unseal(inventory)
    require(compact['private_file_sha256'].get(artifact_name) == expected_sha, 'NEI_inventory_binding')
    artifact = json.loads(raw)
    require(artifact['version'] == PROGRAM_VERSION and artifact['channel'] == 'program_capture'
            and artifact['status'] == 'candidates_enumerated' and artifact['gaps'] == []
            and len(artifact['candidates']) == artifact['candidate_count_before_cap'] == artifact['candidate_cap'] == 1,
            'NEI_program_candidate_required')
    p = unseal(artifact['provenance'])
    require(p['scope'] == ('exposed_train_NEI47' if scope == SCOPE else scope)
            and p['complete_original_row_declared'] is True
            and p['capture_sha256'] == artifact['program_capture']['sha256'], 'NEI_provenance')
    require(scope == 'synthetic_fixture' or p['selection_sha256'] == NEI_SELECTION_SHA, 'NEI_selection_binding')
    frame, sources, registry, order = validate_capture(artifact['program_capture'], corpus, tokenizer)
    require(frame == artifact['frame'] and artifact['claim_id'] == artifact['program_capture']['payload']['inference']['id'],
            'NEI_frame_identity')
    candidate = artifact['candidates'][0]
    require(candidate['basis'] == 'official_annotation_NEI'
            and candidate['target_origin'] == 'program_derived_annotation_candidate'
            and candidate['status'] == 'representable_candidate' and candidate['gap'] is None
            and candidate['model_generated'] is False
            and candidate['target'] == {'action': 'abstain', 'reason': 'insufficient_evidence'}, 'NEI_not_context_insufficient')
    tokens = tokenize_target(tokenizer, frame, candidate['target'], sources,
                             alias_to_source=registry, candidate_aliases=order)
    require(tokens == candidate['tokenized'], 'NEI_original_packing')
    return {'kind': 'program_capture', 'cohort': 'NEI47', 'source_file_sha256': expected_sha,
            'artifact_name': artifact_name, 'artifact': artifact, 'component': component,
            'target_origin': 'official_annotation_NEI', 'tokenized': tokens}


def nei_inventory(raw: bytes, *, scope: str) -> dict[str, Any]:
    require(scope in {SCOPE, 'synthetic_fixture'}
            and (scope == 'synthetic_fixture' or sha(raw) == NEI_COMPACT_SHA), 'frozen_NEI_inventory')
    return seal({'file_sha256': sha(raw), **json.loads(raw)})


def derived_row(envelope: Mapping[str, Any], n: int) -> dict[str, Any]:
    """Canonical optimizer envelope, not a manufactured observation/event."""
    require(envelope['kind'] in {'legacy_program_artifact', 'program_capture'}, 'source_kind')
    if envelope['kind'] == 'legacy_program_artifact':
        source = envelope['legacy_row']
        reason, origin = ORIGINS[source['target']['action']]
        require(envelope['cohort'] in LEGACY_SHA and envelope['event_lineage_available'] is False
                and source['teacher_reason'] == reason and envelope['target_origin'] == origin
                and not set(source) & {'tool_event', 'capture', 'physical_attempt', 'model_selected'}, 'legacy_origin_or_event')
        validate_tokenized(source, envelope['tokenized'])
        fields = {k: source[k] for k in HIERARCHY}
    else:
        artifact = envelope['artifact']
        candidate = artifact['candidates'][0]
        require(envelope['cohort'] == 'NEI47' and envelope['target_origin'] == 'official_annotation_NEI'
                and candidate['basis'] == 'official_annotation_NEI'
                and candidate['target'] == {'action': 'abstain', 'reason': 'insufficient_evidence'}
                and candidate['tokenized'] == envelope['tokenized'], 'NEI_origin')
        source = {'claim_id': artifact['claim_id'], 'component': envelope['component'], 'target': candidate['target']}
        fields = dict(trajectory=0, trajectory_count=1, alternative=0, alternative_count=1,
                      state_index=0, state_count=1, weight_numerator=1, weight_denominator=1, declared_claim_weight=1)
    contract = weights_for(n)
    row = {'version': VERSION, 'claim_id': source['claim_id'], 'component': source['component'],
        **fields, 'target': copy.deepcopy(source['target']), 'epoch_normalizer': n, 'model_generated': False,
        'decision_origin': contract.decision_origin, 'action_target_provenance': contract.action_provenance,
        'semantic_target_provenance': contract.answer_provenance if source['target']['action'] == 'answer' else 'no_semantic_label',
        'target_origin': envelope['target_origin'], 'source_envelope': copy.deepcopy(dict(envelope)),
        'packing': packing_metadata(envelope['tokenized'])}
    row['record_sha256'] = sha(encoded(row))
    return row


def hierarchy_complete(rows: Sequence[Mapping[str, Any]], ids: Sequence[int]) -> None:
    for claim in ids:
        current = [r for r in rows if r['claim_id'] == claim]
        require(len({r['component'] for r in current}) == 1, 'claim_component_drift')
        trajectories = {r['trajectory_count'] for r in current}
        require(len(trajectories) == 1 and {r['trajectory'] for r in current} == set(range(next(iter(trajectories)))),
                'missing_trajectory')
        for trajectory in {r['trajectory'] for r in current}:
            alternatives = [r for r in current if r['trajectory'] == trajectory]
            counts = {r['alternative_count'] for r in alternatives}
            require(len(counts) == 1 and {r['alternative'] for r in alternatives} == set(range(next(iter(counts)))),
                    'missing_alternative')
            for alternative in {r['alternative'] for r in alternatives}:
                states = [r for r in alternatives if r['alternative'] == alternative]
                counts = {r['state_count'] for r in states}
                require(len(counts) == 1 and {r['state_index'] for r in states} == set(range(next(iter(counts)))),
                        'missing_state')


def build_inputs(envelopes: Sequence[Mapping[str, Any]], roster: Mapping[str, Any], *, scope: str,
                 inventory: Mapping[str, Any]) -> dict[str, Any]:
    frozen = unseal(roster)
    ids = frozen['claim_ids']
    rows = [derived_row(e, len(ids)) for e in envelopes]
    value = seal({'version': VERSION, 'scope': scope, 'roster': dict(roster), 'inventory': dict(inventory),
        'claim_ids': ids, 'actual_claim_count': len(ids), 'records': rows,
        'tokenized': [copy.deepcopy(e['tokenized']) for e in envelopes],
        'weight_contract': asdict(weights_for(len(ids))), 'training_authorized': False})
    validate_inputs(value)
    return value


def validate_inputs(prepared: Mapping[str, Any]) -> dict[str, Any]:
    data = unseal(prepared)
    require(data['version'] == VERSION and data['scope'] in {SCOPE, 'synthetic_fixture'}, 'mixed_input_scope')
    ids, rows, tokens = data['claim_ids'], data['records'], data['tokenized']
    require(ids and all(type(i) is int for i in ids) and len(ids) == len(set(ids))
            and data['actual_claim_count'] == len(ids) and data['weight_contract'] == asdict(weights_for(len(ids))), 'mixed_N')
    roster, inventory = unseal(data['roster']), unseal(data['inventory'])
    require(roster['claim_ids'] == ids and set(roster['cohorts']) == {'old48', 'supp49', 'NEI47'}
            and sum((roster['cohorts'][k] for k in ('old48', 'supp49', 'NEI47')), []) == ids, 'explicit_cohort_roster')
    require(len(rows) == len(tokens) and roster['source_envelope_sha256'] ==
            [identity(r['source_envelope']) for r in rows], 'frozen_envelope_order')
    require(set(ids).isdisjoint(roster['excluded_claim_ids'])
            and set(roster['components'].values()).isdisjoint(roster['excluded_components']), 'excluded_partition_contamination')
    validate_weights(rows, ids, contract=weights_for(len(ids)))
    hierarchy_complete(rows, ids)
    components_by_cohort: dict[str, set[str]] = {k: set() for k in roster['cohorts']}
    for row, tok in zip(rows, tokens, strict=True):
        envelope = row['source_envelope']
        require(row == derived_row(envelope, len(ids)) and tok == envelope['tokenized'], 'derived_record_changed')
        validate_tokenized(row, tok)
        cohort = envelope['cohort']
        require(row['claim_id'] in roster['cohorts'][cohort]
                and row['component'] == roster['components'][str(row['claim_id'])], 'cohort_component_binding')
        components_by_cohort[cohort].add(row['component'])
        if data['scope'] == SCOPE:
            if cohort in LEGACY_SHA:
                require(envelope['source_file_sha256'] == LEGACY_SHA[cohort], 'legacy_fixed_source')
            else:
                require(inventory['file_sha256'] == NEI_COMPACT_SHA
                        and inventory['private_file_sha256'][envelope['artifact_name']] == envelope['source_file_sha256'],
                        'NEI_fixed_source')
    require(not (components_by_cohort['old48'] & (components_by_cohort['supp49'] | components_by_cohort['NEI47']))
            and not components_by_cohort['supp49'] & components_by_cohort['NEI47'], 'cross_cohort_component_overlap')
    if data['scope'] == SCOPE:
        require(all(sha(encoded(roster['cohorts'][c])) == h for c, h in ROSTER_SHA.items()), 'fixed_claim_rosters')
        original96 = roster['original96_ids']
        require(sha(encoded(original96)) == '7fcb6c6ff1c5421d2c9c8ac64d54db3bf4b2c7830da58a131d95b4600222d670'
                and [i for i in original96 if i in roster['cohorts']['supp49']] == roster['cohorts']['supp49']
                and [i for i in original96 if i in roster['cohorts']['NEI47']] == roster['cohorts']['NEI47'], 'original96_partition')
        require(sha(encoded({int(i): roster['components'][str(i)] for i in roster['cohorts']['old48']})) ==
                'e8c25bc8cd4069dbe056763987f5d89a016f89f28ad18a46b60244bc476ca086'
                and sha(encoded([roster['components'][str(i)] for i in original96])) ==
                '7e3402adc85b5338649f1b79de1cf80f56eeea9cc309e553ac45d96f04971835', 'fixed_component_rosters')
        for cohort, expected in (('old48', 93), ('supp49', 83)):
            sources = [r['source_envelope'] for r in rows if r['source_envelope']['cohort'] == cohort]
            require([e['source_row_index'] for e in sources] == list(range(expected)), 'legacy_no_drop_or_reorder')
        check_released_shape(rows, roster['cohorts'])
    return data


def check_released_shape(rows: Sequence[Mapping[str, Any]], cohorts: Mapping[str, Any]) -> None:
    """Count/mass gate is necessary, not sufficient: validate_inputs also checks IDs."""
    ids: list[int] = sum((cohorts[c] for c in ('old48', 'supp49', 'NEI47')), [])
    require([len(cohorts[c]) for c in ('old48', 'supp49', 'NEI47')] == [48, 49, 47]
                and len(ids) == 144 and len(rows) == 223, 'released_144_223_contract')
    counts = Counter(r['target']['action'] for r in rows)
    mass = {a: sum((Fraction(r['weight_numerator'], r['weight_denominator']) for r in rows
                       if r['target']['action'] == a), Fraction()) for a in ('answer', 'read', 'abstain')}
    require(counts == {'answer': 170, 'read': 1, 'abstain': 52}
                and mass == {'answer': Fraction(183, 2), 'read': Fraction(1, 2), 'abstain': Fraction(52)}, 'released_action_mass')


def make_plan(prepared: Mapping[str, Any]) -> dict[str, Any]:
    data = validate_inputs(prepared)
    ids = list(data['claim_ids'])
    random.Random(20261001).shuffle(ids)
    groups = []
    for start in range(0, len(ids), 4):
        claims = ids[start:start+4]
        indices = [i for claim in claims for i, row in enumerate(data['records']) if row['claim_id'] == claim]
        groups.append({'claim_ids': claims, 'record_indices': indices})
    return seal({'version': VERSION, 'prepared_sha256': prepared['sha256'], 'seed': 20261001,
        'epochs': 1, 'actual_N': len(ids), 'optimizer_steps': math.ceil(len(ids)/4), 'groups': groups,
        'epoch_metric': 'sum(group_mean*actual_group_claim_count)/actual_N'})
