"""Synthetic protocol/packing compatibility only, no real claim or model run."""
from __future__ import annotations

import copy
from typing import Any

import pytest
import test_scifact_state_supervision as fixtures

from climate_rag.scifact_grounding import Rationale
from climate_rag.scifact_semantic_contract import encoded, sha
from climate_rag.scifact_shared_supervision import (
    CONFIG, build_shared_claim, config_sha, layer_inventory,
    matched_fullpool_state, paired_summary, provenance, read_transition_summary,
)
from climate_rag.scifact_state_supervision import (
    CONFIG as OLD_CONFIG, VERSION as RECORD_SCHEMA, build_claim, tokenize_target, validate_tokenized, validate_weights,
)
from climate_rag.scifact_terminal import render_scifact_prompt


def prepared(evidence: Any = None) -> tuple[Any, ...]:
    existing_fixture: Any = fixtures.setup
    claim, corpus, candidates, tokenizer = existing_fixture(evidence)
    old = build_claim(claim, 'fixture', candidates, corpus, tokenizer)
    new = build_shared_claim(claim, 'fixture', candidates, corpus, tokenizer)
    captured = [{'claim_id': claim.claim_id, 'candidates': new['records'][0]['candidates'], 'frames': new['captured_frames']}]
    return claim, corpus, candidates, tokenizer, old, new, captured


@pytest.mark.parametrize('evidence', [None, {7: (Rationale('SUPPORT', (0, 2)),)}, {8: (Rationale('SUPPORT', (2,)),)}])
def test_only_external_protocol_metadata_changes_not_observation_tokens_weights(evidence: Any) -> None:
    _, _, candidates, tokenizer, old, new, _ = prepared(evidence)
    assert CONFIG['candidate_pool'] != OLD_CONFIG['candidate_pool']
    assert all(CONFIG[k] == v for k, v in OLD_CONFIG.items() if k not in ('version', 'candidate_pool'))
    assert len(old['records']) == len(new['records'])
    for a, b in zip(old['records'], new['records'], strict=True):
        assert {k: v for k, v in b.items() if k not in ('data_provenance', 'record_sha256')} == {
            k: v for k, v in a.items() if k != 'record_sha256'}
        assert b['version'] == RECORD_SCHEMA and b['data_provenance'] == provenance()
        assert b['data_provenance']['data_protocol_sha256'] == config_sha()
        assert b['record_sha256'] != a['record_sha256']
        validate_tokenized(b, tokenize_target(tokenizer, b['frame'], b['target'], candidates))
    validate_weights(new['records'], [1])
    if evidence and 8 in evidence:
        assert new['records'][0]['target']['action'] == 'abstain'
        assert new['records'][0]['semantic_target_provenance'] == 'no_semantic_label'


def test_four_layers_actual_preview_feedback_and_no_invented_training() -> None:
    _, corpus, _, _, _, new, captured = prepared({7: (Rationale('SUPPORT', (0, 2)),)})
    indexed = [{'doc_id': d} for d in corpus]
    parts = {d: 'fit' if d == 7 else 'unowned' for d in corpus}
    private, public = layer_inventory(new['records'], captured, indexed, parts, corpus)
    assert public['indexed']['documents'] == 8 and public['retrieved_candidates']['documents'] == 7
    assert public['prepared_prompt_text']['documents'] == 7 and public['supervised_target_documents']['documents'] == 1
    assert public['supervised_target_by_action']['read']['documents'] == 1
    assert public['actual_training_performed'] is False
    before, after = private['captured_runtime_states_including_gaps'][0]['states']
    assert before['feedback'] is None and after['feedback'].startswith('tool_completed:')
    assert after['remaining_calls'] == before['remaining_calls']-1
    assert after['remaining_tools'] == before['remaining_tools']-1
    bad = copy.deepcopy(new['records'])
    bad[0]['frame']['observation']['preview_only'][0]['preview'] = 'invented'
    with pytest.raises(ValueError, match='actual_preview_text'):
        layer_inventory(bad, captured, indexed, parts, corpus)
    bad = copy.deepcopy(new['records'])
    bad[0]['data_provenance']['candidate_pool'] = 'family-only'
    with pytest.raises(ValueError, match='shared_record_protocol'):
        layer_inventory(bad, captured, indexed, parts, corpus)


def test_paired_identity_and_predeclared_read_anchor_including_prompt() -> None:
    _, _, _, tokenizer, old, new, captured = prepared({7: (Rationale('SUPPORT', (0, 2)),)})
    assert paired_summary([old['report']], [new['report']])['state_transitions'] == {'post_read_complete -> post_read_complete': 1}
    states = []
    for frame in captured[0]['frames']:
        prompt = render_scifact_prompt(tokenizer, frame['observation'], frame['schema'])
        states.append({'state_sha256': sha(encoded(frame)), 'prompt_sha256': sha(prompt.encode()),
                       'prompt_tokens': len(tokenizer.encode(prompt, add_special_tokens=False))})
    prior: list[dict[str, Any]] = [{'claim_id': 1, 'frozen_read_ids': ['c6'], 'candidate_doc_ids': list(range(1, 8)), 'states': states}]
    result = matched_fullpool_state(prior, captured, new['records'], tokenizer)
    assert result['new_witness_rank'] == 7 and result['same_read_ids_as_old']
    assert all(result[k] for k in result if k.endswith('_restored'))
    prior[0]['states'][0]['prompt_sha256'] = '0'*64
    result = matched_fullpool_state(prior, captured, new['records'], tokenizer)
    assert result['initial_state_hash_restored'] and not result['initial_rendered_prompt_hash_restored']
    changed = copy.deepcopy(new['report'])
    changed['component'] = 'other'
    with pytest.raises(ValueError, match='same_ordered_fit_components'):
        paired_summary([old['report']], [changed])


def test_read_execution_not_duplicate_records_and_zero_increment_retained() -> None:
    _, _, _, _, _, built, captured = prepared({7: (Rationale('SUPPORT', (0,)), Rationale('SUPPORT', (2,)))})
    report = read_transition_summary(captured, [built['report']], built['records'])
    assert report['executed_read_claims'] == 1
    assert report['read_decision_records_including_alternative_duplicates'] == 2
    assert report['read_claims_with_new_citable_sentences'] == report['read_claims_with_complete_post_read_rationale'] == 1
    no_change = copy.deepcopy(captured)
    no_change[0]['frames'][1] = copy.deepcopy(no_change[0]['frames'][0])
    gap = dict(built['report'], gaps=['synthetic_no_gain'], post_read_complete=False)
    report = read_transition_summary(no_change, [gap], [])
    assert report['executed_read_claims'] == report['read_claims_with_no_new_citable_sentence'] == report['read_claims_with_gaps'] == 1
    assert report['read_claims_with_complete_post_read_rationale'] == 0
