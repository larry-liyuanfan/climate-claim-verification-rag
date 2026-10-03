"""Synthetic metadata/loader/state tests only; no real claims or model calls."""
from __future__ import annotations

import copy
import io
from pathlib import Path
from typing import Any

import pytest
import test_scifact_state_supervision as fixtures

import climate_rag.scifact_fit_selection as loader
from climate_rag.scifact_grounding import Abstract, GoldClaim, Rationale, parse_gold
from climate_rag.scifact_semantic_contract import encoded, sha
from climate_rag.scifact_shared_supervision import layer_inventory
from climate_rag.scifact_state_supervision import build_claim, validate_weights
from climate_rag.scifact_supplemental_fit import (
    CONFIG, SALT, action_mass, config_sha, data_provenance, freeze_selection,
    load_reserved_claims, prepare_selected_claim, select_metadata,
)


def metadata() -> tuple[Any, ...]:
    assignment: dict[str, Any] = {'eligible_train_ids': list(range(1, 213)), 'dev_ids': [500],
                  'claim_component': {str(i): f'c{(i-1)//2}' for i in range(1, 213)}}
    assignment['claim_component']['500'] = 'dev'
    parts = {f'c{i}': 'fit' for i in range(106)}
    parts.update(c1='tune', c2='validation')
    families = {'component_partition': parts}
    ledger = {'model_consumed_ids': [7], 'uncertain_ids': [9], 'component_excluded': ['c3', 'c4'],
              'gold_preparation_seen_count': 212, 'runs': [{'attempts': [{'claim_id': 11, 'status': 'failed'}]}]}
    split = {'fit': [1], 'tune': [3], 'validation': [5]}
    return assignment, families, ledger, split, {'later': {13}}, {'fixture': '0'*64}


def test_deterministic_component_first_hash_selection_with_all_exclusions() -> None:
    args = metadata()
    selected = select_metadata(*args)
    assert selected['remaining_components']['count'] == 99
    assert selected['eligible_ids_in_remaining_components']['count'] == 198
    assert len(selected['selected_ordered_ids']) == len(set(selected['selected_ordered_components'])) == 96
    assert not set(selected['selected_ordered_ids']) & set(range(1, 15))
    components = sorted([f'c{i}' for i in range(7, 106)], key=lambda c: (sha(encoded([SALT, 'component', c])), c))
    assert selected['selected_ordered_components'] == components[:96]
    for c, i in zip(selected['selected_ordered_components'], selected['selected_ordered_ids'], strict=True):
        ids = [j for j in range(1, 213) if args[0]['claim_component'][str(j)] == c]
        assert i == min(ids, key=lambda j: (sha(encoded([SALT, 'claim', c, j])), j))
    shuffled = copy.deepcopy(args)
    shuffled[0]['eligible_train_ids'].reverse()
    shuffled[0]['claim_component'] = dict(reversed(list(shuffled[0]['claim_component'].items())))
    assert select_metadata(*shuffled) == selected


def test_extra_reservation_or_unknown_component_changes_snapshot_not_silently_reused() -> None:
    args = metadata()
    original = select_metadata(*args)
    args[4]['additional_failed_attempt'] = {15}
    changed = select_metadata(*args)
    assert changed['remaining_components'] != original['remaining_components']
    assert changed['eligible_ids_in_remaining_components'] != original['eligible_ids_in_remaining_components']
    assert 15 not in changed['selected_ordered_ids'] and 16 not in changed['selected_ordered_ids']
    args[4]['unknown'] = {9999}
    with pytest.raises(ValueError, match='unmapped_exclusion_id'):
        select_metadata(*args)


def small_selection() -> dict[str, Any]:
    selected = select_metadata(*metadata())
    # Synthetic one-row setup, never the actual production selector.
    selected['selected_ordered_ids'] = [15]
    selected['selected_ordered_components'] = ['c7']
    return selected


def test_loader_frozen_first_then_selected_only_decode_and_full_member_hash(tmp_path: Path, monkeypatch: Any) -> None:
    selection = small_selection()
    corpus = {1: Abstract(1, 'title', ('evidence',), False)}
    raw = (b'{"id":999,"claim":"UNSELECTED","evidence":"NOT_A_GOLD_OBJECT"}\n'
           b'{"id":15,"claim":"selected","cited_doc_ids":[1],"evidence":{"1":[{"label":"SUPPORT","sentences":[0]}]}}\n')
    parsed: list[int] = []
    parse = parse_gold
    def spy(row: Any, docs: Any) -> Any:
        parsed.append(row['id'])
        assert (tmp_path/'component-reservations.json').is_file()
        return parse(row, docs)
    monkeypatch.setattr(loader, 'parse_gold', spy)
    opened: list[bool] = []
    def open_stream() -> io.BytesIO:
        assert (tmp_path/'selection-before-content.json').is_file()
        assert (tmp_path/'component-reservations.json').is_file()
        opened.append(True)
        return io.BytesIO(raw)
    with pytest.raises(FileNotFoundError):
        load_reserved_claims(open_stream, sha(raw), tmp_path,
            {'selection-before-content.json': '0'*64, 'component-reservations.json': '0'*64}, corpus)
    assert not opened
    frozen = freeze_selection(tmp_path, selection)
    with pytest.raises(ValueError, match='train_member_sha'):
        load_reserved_claims(open_stream, '0'*64, tmp_path, frozen, corpus)
    assert parsed == []
    claims = load_reserved_claims(open_stream, sha(raw), tmp_path, frozen, corpus)
    assert list(claims) == parsed == [15]
    assert len(opened) == 2
    with pytest.raises(FileExistsError):
        freeze_selection(tmp_path, selection)
    (tmp_path/'component-reservations.json').write_bytes(b'{}\n')
    with pytest.raises(ValueError):
        load_reserved_claims(open_stream, sha(raw), tmp_path, frozen, corpus)
    assert len(opened) == 2


@pytest.mark.parametrize('case', ['direct', 'read', 'context_abstain', 'mixed', 'official_nei'])
def test_selected_outcomes_preserve_teacher_frames_and_no_replacement(case: str, tmp_path: Path) -> None:
    setup: Any = fixtures.setup
    evidence = {'direct': {1: (Rationale('SUPPORT', (2,)),)},
        'read': {7: (Rationale('SUPPORT', (0,)), Rationale('SUPPORT', (2,)))},
        'context_abstain': {8: (Rationale('SUPPORT', (2,)),)},
        'mixed': {1: (Rationale('SUPPORT', (2,)),), 2: (Rationale('CONTRADICT', (0,)),)},
        'official_nei': {}}[case]
    claim, corpus, candidates, tokenizer = setup(evidence)
    if case == 'official_nei':
        claim = GoldClaim(claim.claim_id, claim.claim, {}, (1,))
    selection = small_selection()
    selection['selected_ordered_ids'] = [1]
    frozen = freeze_selection(tmp_path, selection)
    built = prepare_selected_claim(claim, 'c7', candidates, corpus, tokenizer, frozen)
    assert all(sha((tmp_path/n).read_bytes()) == h for n, h in frozen.items())
    assert built['report']['authorized_annotation_scope'] == CONFIG['annotation_view']
    if case == 'official_nei':
        assert not built['records'] and built['report']['status'] == 'not_applicable'
        assert not built['captured_frames'] and built['report']['official_label_scope'] == 'NOT_ENOUGH_INFO'
        assert action_mass([], [1])['missing_claim_mass']['value'] == 1
        return
    historical = build_claim(claim, 'c7', candidates, corpus, tokenizer)
    assert built['captured_frames'] == historical['captured_frames']
    assert built['report']['data_provenance'] == data_provenance(frozen)
    for old, new in zip(historical['records'], built['records'], strict=True):
        assert all(new[k] == v for k, v in old.items() if k != 'record_sha256')
        assert new['data_provenance']['data_protocol_sha256'] == config_sha()
    validate_weights(built['records'], [1])
    if case == 'mixed':
        assert built['report']['official_label_scope'] == 'MIXED' and built['report']['training_ready']
        assert {d['label'] for d in built['records'][0]['target']['documents']} == {'SUPPORTS', 'REFUTES'}
    if case == 'context_abstain':
        assert built['records'][0]['semantic_target_provenance'] == 'no_semantic_label'
    mass = action_mass(built['records'], [1, 2])  # missing second selected claim is not dropped
    assert mass['missing_claim_mass']['value'] == 1
    if case == 'read':
        assert mass['records_by_action']['read'] == 2 and mass['claims_with_action']['read'] == 1
        assert mass['effective_claim_mass_by_action']['read']['value'] == 0.5
        assert mass['mean_over_all_selected_claims_by_action']['read']['value'] == 0.25
    captured = [{'claim_id': 1, 'candidates': built['records'][0]['candidates'], 'frames': built['captured_frames']}]
    _, public = layer_inventory(built['records'], captured, [{'doc_id': d} for d in corpus],
        {d: 'fit' for d in corpus}, corpus, expected_provenance=data_provenance(frozen))
    assert public['prepared_records'] == len(built['records'])


def test_gap_not_reweighted_and_selection_survives(tmp_path: Path) -> None:
    setup: Any = fixtures.setup
    claim, corpus, _, tokenizer = setup({7: (Rationale('SUPPORT', (1,)),)})
    corpus[7] = Abstract(7, 'Long', ('x'*50000, 'Required.'), False)
    from climate_rag.scifact_terminal import source_from_abstract
    candidates = [source_from_abstract(corpus[i]) for i in range(1, 8)]
    frozen = freeze_selection(tmp_path, small_selection())
    result = prepare_selected_claim(claim, 'c7', candidates, corpus, tokenizer, frozen)
    assert result['report']['gaps'] and not result['records']
    assert action_mass(result['records'], [1])['missing_claim_mass']['value'] == 1
    assert all(sha((tmp_path/n).read_bytes()) == h for n, h in frozen.items())


def test_combined_action_mass_with_direct_duplicate_read_alternatives_and_zero() -> None:
    setup: Any = fixtures.setup
    claim, corpus, candidates, tokenizer = setup({7: (Rationale('SUPPORT', (0,)), Rationale('SUPPORT', (2,)))})
    read = build_claim(claim, 'read', candidates, corpus, tokenizer)['records']
    direct_claim = GoldClaim(2, 'Direct', {1: (Rationale('SUPPORT', (2,)),)}, ())
    direct = build_claim(direct_claim, 'direct', candidates, corpus, tokenizer)['records']
    mass = action_mass(direct+read, [1, 2, 3])
    assert mass['decision_records'] == 5
    assert mass['effective_claim_mass_by_action']['answer']['value'] == 1.5
    assert mass['effective_claim_mass_by_action']['read']['value'] == 0.5
    assert mass['mean_over_all_selected_claims_by_action']['read']['numerator'] == 1
    assert mass['mean_over_all_selected_claims_by_action']['read']['denominator'] == 6
    assert mass['missing_claim_mass']['value'] == 1
