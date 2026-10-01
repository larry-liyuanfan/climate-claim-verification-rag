from copy import deepcopy

from audit_scifact_document_gold import diagnose_claim, match
from climate_rag.scifact_grounding import GoldClaim, Rationale


def fixture():
    frame = {'visible': {f'c{doc}:{i}': {'source_id': str(doc)} for doc in range(4) for i in range(5)}}
    feedback = [{'status': 'valid', 'model_selected': False,
        'provenance': {'original_source_id': str(doc)},
        'judgment': {'source_id': f'c{doc}', 'label': 'SUPPORTS' if doc == 0 else 'INSUFFICIENT',
                     'sentence_ids': ['c0:0','c0:1'] if doc == 0 else []}} for doc in range(4)]
    gold = GoldClaim(1, 'synthetic', {0: (Rationale('SUPPORT', (0, 1)),)}, (0,))
    row = {'claim_id': 1, 'arm': 'fixed', 'verification_feedback': feedback,
        'state': 'valid_terminal', 'prediction': {'id': 1, 'evidence': {'0': {'label': 'SUPPORT', 'sentences': [0, 1]}}}}
    return gold, frame, row


def test_paired_retention_and_unreachable_gold():
    gold, frame, row = fixture()
    result = diagnose_claim(gold, frame, row)
    assert result['verified_gold_doc_label_first3_pairs'] == {'verifier_1_final_1': 1}
    assert result['positive_verdict_projection_matches_strict_annotations'] is True
    no_candidate = GoldClaim(1, 'synthetic', {9: (Rationale('SUPPORT', (0,)),)}, (9,))
    result = diagnose_claim(no_candidate, frame, row)
    assert result['counts']['initial_first3_reachable_gold_docs'] == 0
    assert result['all_gold_docs_label_and_first3_verified'] is False


def test_label_evidence_and_mechanical_integration_loss_separate():
    gold, frame, row = fixture()
    row['state'] = 'unresolved'
    row['prediction'] = None
    result = diagnose_claim(gold, frame, row)
    assert result['counts']['verifier_first3_joint_lost_to_invalid_terminal'] == 1
    assert result['verified_gold_doc_label_first3_pairs'] == {'verifier_1_final_0': 1}
    row['verification_feedback'][0]['judgment']['label'] = 'REFUTES'
    result = diagnose_claim(gold, frame, row)
    assert result['counts']['verifier_label_correct_gold_docs'] == 0
    assert result['counts']['verifier_first3_evidence_correct_gold_docs'] == 1
    assert result['counts']['verifier_label_and_first3_correct_gold_docs'] == 0


def test_valid_terminal_drift_and_non_gold_positive_not_called_gold_accuracy():
    gold, frame, row = fixture()
    changed = deepcopy(row)
    changed['prediction']['evidence']['0']['label'] = 'CONTRADICT'
    changed['verification_feedback'][1]['judgment'].update(label='SUPPORTS', sentence_ids=['c1:0'])
    result = diagnose_claim(gold, frame, changed)
    assert result['counts']['verifier_first3_joint_lost_to_valid_terminal'] == 1
    assert result['non_gold_positive_verifier_judgments'] == 1
    assert result['all_gold_docs_label_and_first3_verified'] is True
    assert result['positive_verdict_projection_matches_strict_annotations'] is False


def test_original_alternative_or_and_first_three_order_not_any_hit():
    rats = (Rationale('SUPPORT', (0, 1)), Rationale('SUPPORT', (4,)))
    assert match('SUPPORTS', [1], rats) == (True, False)
    assert match('SUPPORTS', [2, 3, 1, 4], rats) == (True, False)
    assert match('SUPPORTS', [4], rats) == (True, True)
    assert match('REFUTES', [0, 1], rats) == (False, True)


def test_extra_unannotated_sentence_not_misattributed_to_terminal():
    gold, frame, row = fixture()
    row['verification_feedback'][0]['judgment']['sentence_ids'].append('c0:2')
    result = diagnose_claim(gold, frame, row)
    assert result['all_gold_docs_label_and_first3_verified'] is True
    assert result['counts']['verifier_full_selection_covers_rationale_gold_docs'] == 1
    assert result['counts']['verifier_selection_has_unannotated_sentences_gold_docs'] == 1
    assert result['positive_verdict_projection_matches_strict_annotations'] is False


def test_full_selection_reachability_separate_from_first_three_and_nei():
    _, frame, row = fixture()
    gold = GoldClaim(1, 'synthetic', {0: (Rationale('SUPPORT', (0, 1, 2, 3)),)}, (0,))
    row['verification_feedback'][0]['judgment']['sentence_ids'] = [f'c0:{i}' for i in range(4)]
    result = diagnose_claim(gold, frame, row)
    assert result['counts']['verifier_first3_evidence_correct_gold_docs'] == 0
    assert result['counts']['verifier_full_selection_covers_rationale_gold_docs'] == 1
    assert result['positive_verdict_projection_matches_strict_annotations'] is False
    nei = GoldClaim(1, 'synthetic', {}, ())
    assert diagnose_claim(nei, frame, row)['positive_verdict_projection_matches_strict_annotations'] is None
