"""Synthetic export/consumption/error-layer checks, never real model scores."""
import importlib.util
import json
from pathlib import Path

import pytest

spec = importlib.util.spec_from_file_location('compact', Path(__file__).parents[1] / 'scripts/summarize_scifact_semantic_closeout.py')
compact = importlib.util.module_from_spec(spec)
spec.loader.exec_module(compact)


def test_numeric_allowlist_never_exports_private_subtree():
    assert compact.numbers({'calls': 1, 'claim_id': 'SECRET'}, ('calls',)) == {'calls': 1}
    assert compact.numbers({'cost': None}, ('cost',)) == {'cost': None}
    with pytest.raises(ValueError, match='private values suppressed'):
        compact.numbers({'calls': {'claim_id': 99}}, ('calls',))


@pytest.mark.parametrize('value', [float('nan'), float('inf'), 'private'])
def test_nonfinite_and_strings_rejected(value):
    with pytest.raises(ValueError):
        compact.numbers({'n': value}, ('n',))


def test_compatible_gold_after_first_three_is_not_full_rationale_credit():
    gold = {1: {'evidence': {'8': [{'label': 'SUPPORT', 'sentences': [4]}]}}}
    rows = [{'claim_id': 1, 'result': {'generation_attempts': [{}], 'visible_attempts': [{'visible': [
        {'doc_id': 8, 'sentence_index': 4}]}]}, 'prediction': {'evidence': {
            '8': {'label': 'SUPPORT', 'sentences': [0, 1, 2, 4]},
            '9': {'label': 'SUPPORT', 'sentences': [0]}}}}]
    report = compact.visibility_summary(rows, gold)
    assert report['correct_label_complete_rationale_any_predicted_position'] == 1
    assert report['correct_label_complete_rationale_in_first_three'] == 0
    assert report['predicted_documents_not_matching_official_gold'] == 1
    assert report['predicted_documents_with_more_than_three_sentences'] == 1
    assert 'claim_id' not in json.dumps(report)


def test_current_consumption_union_deduplicates_routes_arms_excludes_whole_component():
    assignment = {'eligible_train_ids': list(range(531)), 'claim_component': {str(i): str(i) for i in range(531)}}
    assignment['claim_component']['200'] = '12'
    audit = [{'id': i, 'stratum': compact.STRATA[i % 4]} for i in range(531)]
    audit[500]['stratum'] = None
    ledger = {'model_consumed_ids': list(range(12)), 'uncertain_ids': []}
    rows = [{'claim_id': i, 'route': r, 'result': {'generation_attempts': [{}]}}
            for i in range(12, 24) for r in compact.ROUTES]
    gold = [json.dumps({'id': i, 'evidence': {}}).encode() for i in range(531)]
    report = compact.population_summary(assignment, audit, ledger, [{'runs': rows}, {'runs': rows}], gold)
    assert report['cumulative_model_consumed_claims'] == 24
    assert report['excluded_components'] == 24
    assert report['excluded_eligible_claims'] == 25
    assert report['remaining_claims'] == 506
    assert report['remaining_components'] == 506
    assert report['new_selection_performed'] is False
    assert report['remaining_legacy_stratum_claims']['unclassified'] == 1
    compact.encode_compact(report)
    assert 'claim_id' not in json.dumps(report)


def test_realistic_aggregate_can_export_without_per_item_fields():
    report = {'policies': {'old': {'raw_action_proposals': {'answer_count': 11, 'abstain_count': 1}}},
              'slurm_fields': ['JobIDRaw', 'State'], 'slurm_rows': [['31706518', 'COMPLETED']]}
    assert json.loads(compact.encode_compact(report)) == report


@pytest.mark.parametrize('private', [{'claim_id': 123}, {'claim_diagnostics': []},
                                    {'answer': 'PRIVATE'}, {'unknown_rows': [{'id': 123}]}])
def test_embedded_raw_fields_and_per_item_arrays_fail_closed(private):
    with pytest.raises(ValueError, match='private values suppressed'):
        compact.encode_compact({'policies': {'old': private}})
