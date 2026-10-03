"""Prospectively selected FIT-only supervision; never a training/evaluation CLI."""
from __future__ import annotations

from collections import Counter
from collections.abc import Callable, Mapping, Sequence
from fractions import Fraction
import json
from pathlib import Path
from typing import Any, BinaryIO

from .scifact_fit_selection import select_complete_fit
from .scifact_grounding import Abstract, GoldClaim
from .scifact_semantic_contract import CORPUS_SHA, checked, encoded, sha, write_once
from .scifact_shared_supervision import CONFIG as SHARED_CONFIG, build_shared_claim
from .scifact_state_supervision import VERSION as RECORD_SCHEMA, require

VERSION = 'scifact-supplemental-fit-v1-20261001'
SALT = VERSION
CONFIG: dict[str, Any] = {
    'version': VERSION, 'selection_salt': SALT, 'max_selected_claims': 96,
    'claims_per_component': 1, 'candidate_k': 20, 'max_records_per_claim': 4,
    'shared_structural_config_sha256': sha(encoded(SHARED_CONFIG)),
    'candidate_pool': 'all_5183_public_corpus_gold_free_retrieval',
    'annotation_view': 'only_frozen_selected_official_TRAIN_claims_complete_annotations',
    'record_schema': RECORD_SCHEMA, 'legacy_record_epoch_normalizer': 48,
    'legacy_epoch_normalizer_is_not_current_cohort_size_or_optimizer_denominator': True,
    'not_applicable_policy': 'retain_selection_and_zero_records_no_replacement',
    'selection_uses_labels_or_read_opportunity': False,
    'model_calls': 0, 'optimizer_steps': 0, 'training_authorized': False,
    'source_family_unseen_claim': False,
}


def config_sha() -> str:
    return sha(encoded(CONFIG))


def describe(values: set[Any]) -> dict[str, Any]:
    return {'count': len(values), 'sha256': sha(encoded(sorted(values)))}


def select_metadata(assignment: Mapping[str, Any], families: Mapping[str, Any],
                    ledger: Mapping[str, Any], split: Mapping[str, Any],
                    later_ids: Mapping[str, set[int]], input_sha: Mapping[str, str]) -> dict[str, Any]:
    """No claim text, labels, retrieval ranks, targets or read outcomes accepted."""
    eligible = set(assignment['eligible_train_ids'])
    require(len(eligible) == len(assignment['eligible_train_ids'])
            and all(type(i) is int for i in eligible), 'eligible_id_identity')
    components = {int(i): str(c) for i, c in assignment['claim_component'].items()}
    parts = families['component_partition']
    def groups(ids: set[int]) -> set[str]:
        require(ids <= set(components), 'unmapped_exclusion_id')
        return {components[i] for i in ids}
    require(groups(eligible) == set(parts) and not groups(eligible) & groups(set(assignment['dev_ids']))
            and set(parts.values()) <= {'fit', 'tune', 'validation'}, 'partition_and_dev_isolation')
    reserved = {p: groups(set(split[p])) for p in ('fit', 'tune', 'validation')}
    require(all(len(split[p]) == len(set(split[p])) == len(reserved[p])
                and set(split[p]) <= eligible and all(parts[c] == p for c in reserved[p]) for p in reserved),
            'original_reservation_partition')
    consumed, unknown = groups(set(ledger['model_consumed_ids'])), groups(set(ledger['uncertain_ids']))
    require(consumed | unknown == set(ledger['component_excluded']), 'ledger_component_binding')
    failed_ids = {a['claim_id'] for run in ledger['runs'] for a in run['attempts']
                  if a['claim_id'] is not None and a['status'] in ('failed', 'unknown')}
    exclusions = {'original_fit': reserved['fit'], 'reserved_tune': reserved['tune'],
                  'reserved_validation': reserved['validation'], 'model_consumed': consumed,
                  'uncertain': unknown, 'failed_unknown_attempt': groups(failed_ids)}
    exclusions.update({name: groups(ids) for name, ids in later_ids.items()})
    remaining = {c for c, p in parts.items() if p == 'fit'} - set().union(*exclusions.values())
    by_component = {c: [i for i in eligible if components[i] == c] for c in remaining}
    require(all(by_component.values()), 'empty_eligible_component')
    # Domain separation + canonical JSON tuple; stable string/int ties are explicit.
    ranked = sorted(remaining, key=lambda c: (sha(encoded([SALT, 'component', c])), c))
    ordered = {c: sorted(by_component[c], key=lambda i: (sha(encoded([SALT, 'claim', c, i])), i)) for c in ranked}
    chosen = ranked[:CONFIG['max_selected_claims']]
    return {'version': VERSION, 'config_sha256': config_sha(), 'salt': SALT,
        'ordering': 'sha256(canonical_json([salt,domain,component,(id)])),then_stable_id',
        'input_sha256': dict(input_sha),
        'candidate_id_to_component': {str(i): components[i] for c in ranked for i in ordered[c]},
        'ordered_candidate_components': ranked, 'ordered_ids_within_component': ordered,
        'selected_ordered_ids': [ordered[c][0] for c in chosen], 'selected_ordered_components': chosen,
        'remaining_components': describe(remaining),
        'eligible_ids_in_remaining_components': describe({i for c in remaining for i in by_component[c]}),
        'exclusions': {name: describe(ids) for name, ids in exclusions.items()},
        'conditional_eligibility_only': True, 'gold_preparation_seen': ledger['gold_preparation_seen_count'],
        'global_unexposed_or_source_family_unseen_attestation': False,
        'selected_before_new_content_read': True, 'replacement_or_outcome_selection_allowed': False}


def freeze_selection(out: Path, selection: Mapping[str, Any]) -> dict[str, str]:
    """Exclusive files. A failed/partial run keeps component reservations."""
    write_once(out/'selection-before-content.json', selection)
    selection_sha = sha((out/'selection-before-content.json').read_bytes())
    reservation = {'version': VERSION, 'selection_sha256': selection_sha,
        'config_sha256': config_sha(), 'state': 'reserved_before_content_do_not_reuse_on_failure',
        'ordered_claim_ids': selection['selected_ordered_ids'],
        'ordered_components': selection['selected_ordered_components']}
    write_once(out/'component-reservations.json', reservation)
    return {'selection-before-content.json': selection_sha,
            'component-reservations.json': sha((out/'component-reservations.json').read_bytes())}


def load_reserved_claims(open_stream: Callable[[], BinaryIO], train_sha: str, out: Path,
                         frozen: Mapping[str, str], corpus: Mapping[int, Abstract]) -> dict[int, GoldClaim]:
    selection = json.loads(checked(out/'selection-before-content.json', frozen['selection-before-content.json']))
    reservation = json.loads(checked(out/'component-reservations.json', frozen['component-reservations.json']))
    ids = selection['selected_ordered_ids']
    require(selection['version'] == VERSION and selection['config_sha256'] == config_sha()
            and reservation['selection_sha256'] == frozen['selection-before-content.json']
            and reservation['config_sha256'] == config_sha()
            and reservation['state'] == 'reserved_before_content_do_not_reuse_on_failure'
            and reservation['ordered_claim_ids'] == ids
            and reservation['ordered_components'] == selection['selected_ordered_components']
            and 0 < len(ids) <= 96 and len(set(ids)) == len(ids)
            and len(set(reservation['ordered_components'])) == len(ids), 'selection_reservation_binding')
    # The callback is invoked ONLY after both physical frozen records validate.
    with open_stream() as stream:
        return select_complete_fit(stream, train_sha, ids, corpus)


def data_provenance(frozen: Mapping[str, str]) -> dict[str, Any]:
    return {'data_protocol': VERSION, 'data_protocol_sha256': config_sha(),
        'record_schema': RECORD_SCHEMA, 'candidate_pool': CONFIG['candidate_pool'],
        'candidate_pool_documents': 5183, 'corpus_sha256': CORPUS_SHA,
        'selection_sha256': frozen['selection-before-content.json'],
        'reservation_sha256': frozen['component-reservations.json'],
        'source_family_unseen_claim': False,
        'legacy_epoch_normalizer_is_not_current_cohort_size_or_optimizer_denominator': True}


def prepare_selected_claim(claim: GoldClaim, component: str, candidates: Sequence[Any],
                           corpus: Mapping[int, Abstract], tokenizer: Any,
                           frozen: Mapping[str, str]) -> dict[str, Any]:
    scope = claim.label_scope
    if not claim.evidence:
        reason = 'official_nei_not_supported_by_frozen_teacher'
        return {'records': [], 'captured_frames': [], 'report': {
            'claim_id': claim.claim_id, 'component': component, 'status': 'not_applicable',
            'official_label_scope': scope, 'reason': reason, 'gaps': [reason],
            'initial_complete': False, 'post_read_complete': None, 'retained_records': 0,
            'retained_weight_numerator': 0, 'retained_weight_denominator': 1,
            'training_ready': False, 'program_teacher_reads': 0, 'scripted_responses': 0,
            'model_calls': 0, 'model_tool_executions': 0, 'whole_gold_covered_by_target': False,
            'authorized_annotation_scope': CONFIG['annotation_view'], 'data_provenance': data_provenance(frozen)}}
    result = build_shared_claim(claim, component, candidates, corpus, tokenizer)
    result['report'].update(status='prepared' if result['report']['training_ready'] else 'gap',
        official_label_scope=scope, authorized_annotation_scope=CONFIG['annotation_view'],
        data_provenance=data_provenance(frozen))
    for row in result['records']:
        row['data_provenance'] = data_provenance(frozen)
        row['record_sha256'] = sha(encoded({k: v for k, v in row.items() if k != 'record_sha256'}))
    return result


def action_mass(records: Sequence[Mapping[str, Any]], selected_ids: Sequence[int]) -> dict[str, Any]:
    """Claim-normalized effective weight, never reinterpret records as samples."""
    require(len(set(selected_ids)) == len(selected_ids) and bool(selected_ids), 'selected_ids')
    per_claim = {i: {a: Fraction() for a in ('answer', 'read', 'abstain')} for i in selected_ids}
    for row in records:
        require(row['claim_id'] in per_claim, 'record_outside_selected')
        per_claim[row['claim_id']][row['target']['action']] += Fraction(row['weight_numerator'], row['weight_denominator'])
    masses = {i: sum(actions.values(), Fraction()) for i, actions in per_claim.items()}
    require(all(0 <= mass <= 1 for mass in masses.values()), 'no_claim_upweighting')
    total = {a: sum((actions[a] for actions in per_claim.values()), Fraction()) for a in ('answer', 'read', 'abstain')}
    def rational(value: Fraction) -> dict[str, Any]:
        return {'numerator': value.numerator, 'denominator': value.denominator, 'value': float(value)}
    return {'selected_claims_denominator': len(selected_ids), 'decision_records': len(records),
        'records_by_action': dict(Counter(r['target']['action'] for r in records)),
        'claims_with_action': {a: sum(actions[a] > 0 for actions in per_claim.values()) for a in total},
        'effective_claim_mass_by_action': {a: rational(w) for a, w in total.items()},
        'mean_over_all_selected_claims_by_action': {a: rational(w/len(selected_ids)) for a, w in total.items()},
        'retained_mass_histogram': dict(Counter(str(m) for m in masses.values())),
        'missing_claim_mass': rational(sum((1-m for m in masses.values()), Fraction())),
        'missing_mass_renormalized': False, 'records_are_not_independent_claims': True}
