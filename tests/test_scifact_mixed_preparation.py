"""Small real-factory fixtures; no real TRAIN source or model is opened."""
import copy
import json

import pytest

import climate_rag.scifact_mixed_preparation as prep
from climate_rag.scifact_grounding import GoldClaim, Rationale
from climate_rag.scifact_mixed_inputs import validate_inputs
from climate_rag.scifact_program_capture import program_candidates
from climate_rag.scifact_semantic_contract import encoded, sha
from climate_rag.scifact_state_supervision import build_claim
from climate_rag.scifact_terminal import source_from_abstract
from test_scifact_nei_preparation import arguments, fixture


def raw(value):
    return (json.dumps(value, ensure_ascii=False, separators=(',', ':'))+'\n').encode()


def inputs(monkeypatch):
    row, corpus, tokenizer, receipt = fixture()
    artifact = program_candidates(*arguments(row, corpus, tokenizer, receipt))
    candidates = [source_from_abstract(corpus[i]) for i in range(100, 120)]
    old = build_claim(GoldClaim(11, 'Water rises', {124: (Rationale('SUPPORT', (0,)),)}, ()),
                      'old-component', candidates, corpus, tokenizer)['records']
    supp = build_claim(GoldClaim(22, 'Water rises', {100: (Rationale('SUPPORT', (0,)),)}, ()),
                       'supp-component', candidates, corpus, tokenizer)['records']
    assert old[0]['target']['action'] == 'abstain' and supp[0]['target']['action'] == 'answer'
    values = {
        prep.OLD+'/private/selection-before-packing.json': {'fit': [11], 'tune': [88], 'validation': [99]},
        prep.OLD+'/private/families.json': {'component_partition': {'old-component': 'fit', 'supp-component': 'fit',
            'NEI-component': 'fit', 'heldout': 'validation'}},
        prep.SHARED+'/claim-reports.json': [{'claim_id': 11, 'component': 'old-component', 'training_ready': True}],
        prep.SHARED+'/fit-records.json': old,
        prep.SUPPLEMENT+'/fit-records.json': supp,
    }
    selected = {'selected_ordered_ids': [22, 7], 'selected_ordered_components': ['supp-component', 'NEI-component'],
                'config_sha256': 'c'*64}
    selection_sha = sha(raw(selected))
    values[prep.SUPPLEMENT+'/selection-before-content.json'] = selected
    values[prep.SUPPLEMENT+'/component-reservations.json'] = {'ordered_claim_ids': [22, 7],
        'ordered_components': selected['selected_ordered_components'], 'selection_sha256': selection_sha,
        'config_sha256': 'c'*64}
    reports = [{'claim_id': 22, 'component': 'supp-component', 'status': 'prepared', 'training_ready': True,
                'official_label_scope': 'SUPPORTS', 'retained_records': 1},
               {'claim_id': 7, 'component': 'NEI-component', 'status': 'not_applicable',
                'official_label_scope': 'NOT_ENOUGH_INFO', 'reason': 'official_nei_not_supported_by_frozen_teacher',
                'retained_records': 0}]
    values[prep.SUPPLEMENT+'/claim-reports.json'] = reports
    values[prep.SHARED+'/compact.json'] = {'private_file_sha256': {'fit-records.json': sha(raw(old))}}
    compact = {'actual_selected_claims': 2, 'selected_ids_sha256': sha(encoded([22, 7])),
        'selected_components_sha256': sha(encoded(selected['selected_ordered_components'])), 'config_sha256': 'c'*64,
        'private_file_sha256': {'selection-before-content.json': selection_sha, 'fit-records.json': sha(raw(supp))}}
    values[prep.SUPPLEMENT+'/compact.json'] = compact
    selection = prep.fixed_selection(compact, selected, values[prep.SUPPLEMENT+'/component-reservations.json'],
                                     reports, total=2, nei=1)
    values[prep.NEI+'/NEI-selection.json'] = selection
    values[prep.NEI+'/claim-status.json'] = [{'claim_id': 7, 'status': 'ready',
        'artifact': 'claim-00.json', 'artifact_sha256': sha(raw(artifact))}]
    values[prep.NEI+'/claim-00.json'] = artifact
    values[prep.NEI+'/compact.json'] = {'whole_cohort_data_ready': True,
        'status_counts': {'ready': 1, 'gap': 0, 'failed': 0, 'unknown': 0}, 'private_file_sha256': {
            name: sha(raw(values[prep.NEI+'/'+name])) for name in ('NEI-selection.json', 'claim-status.json', 'claim-00.json')}}
    files = {name: raw(value) for name, value in values.items()}
    monkeypatch.setattr(prep, 'PINS', {name: sha(payload) for name, payload in files.items() if not name.endswith('/claim-00.json')})
    monkeypatch.setitem(prep.LEGACY_SHA, 'old48', sha(raw(old)))
    monkeypatch.setitem(prep.LEGACY_SHA, 'supp49', sha(raw(supp)))
    reads = []
    def read(name):
        reads.append(name)
        return files[name]
    return files, reads, read, lambda: (corpus, tokenizer)


def run(out, read, assets):
    return prep.prepare_bundle(out, {'source_git': 'f'*40}, read, assets,
                               scope='synthetic_fixture', counts=(1, 1, 1))


def test_producer_uses_raw_factories_order_origins_and_physical_readback(tmp_path, monkeypatch):
    files, reads, read, assets = inputs(monkeypatch)
    calls = []
    old_factory, nei_factory = prep.legacy_envelopes, prep.nei_envelope
    def old(*args, **kwargs):
        calls.append(args[1])
        return old_factory(*args, **kwargs)
    def nei(*args, **kwargs):
        calls.append('NEI47')
        return nei_factory(*args, **kwargs)
    monkeypatch.setattr(prep, 'legacy_envelopes', old)
    monkeypatch.setattr(prep, 'nei_envelope', nei)
    out = tmp_path/'success'
    report = run(out, read, assets)
    assert report['status'] == 'complete' and report['status_counts'] == {'ready': 3, 'failed': 0, 'unknown': 0}
    assert report['denominator'] == report['rows'] == 3 and report['planned_updates'] == 1
    assert calls == ['old48', 'supp49', 'NEI47']
    restored = json.loads((out/'prepared.json').read_bytes())
    data = validate_inputs(restored)
    assert data['claim_ids'] == [11, 22, 7]
    assert [r['target_origin'] for r in data['records']] == ['context_insufficient', 'visible_OR_rationale', 'official_annotation_NEI']
    assert data['records'][0]['source_envelope']['legacy_row'] == json.loads(files[prep.SHARED+'/fit-records.json'])[0]
    assert data['records'][0]['source_envelope']['legacy_row']['epoch_normalizer'] == 48
    complete = json.loads((out/'complete.json').read_bytes())
    assert complete['source_bridge_validated'] and complete['input_config_sha256'] == prep.input_config_sha()
    assert complete['files'] == {name: sha((out/name).read_bytes()) for name in ('prepared.json', 'plan.json', 'roster.json')}
    assert not any('gold' in name or 'claims_train' in name for name in reads)
    with pytest.raises(FileExistsError):
        run(out, read, assets)


def test_bad_source_hash_or_roster_stops_before_assets_and_data(tmp_path, monkeypatch):
    files, reads, read, assets = inputs(monkeypatch)
    bad = prep.OLD+'/private/selection-before-packing.json'
    original = files[bad]
    files[bad] = original+b' '
    def no_assets():
        pytest.fail('assets should not open after failed metadata')
    report = run(tmp_path/'bad-hash', read, no_assets)
    assert report['status'] == 'failed' and report['status_counts']['unknown'] == report['denominator'] == 3
    assert not any(name.endswith('/fit-records.json') for name in reads)
    assert not (tmp_path/'bad-hash/complete.json').exists()
    diagnostic = json.loads((tmp_path/'bad-hash/private/failure-diagnostic.json').read_bytes())
    assert diagnostic['stage'] == 'metadata' and diagnostic['exception_type'] == report['fatal_type']
    assert diagnostic['message'] == 'physical_source_hash:'+bad and diagnostic['private_only']
    assert 'message' not in report and diagnostic['message'] not in (tmp_path/'bad-hash/compact.json').read_text()
    files[bad] = original
    selection = json.loads(files[prep.NEI+'/NEI-selection.json'])
    selection['selected_ordered_ids'] = [999]
    files[prep.NEI+'/NEI-selection.json'] = raw(selection)
    monkeypatch.setitem(prep.PINS, prep.NEI+'/NEI-selection.json', sha(raw(selection)))
    report = run(tmp_path/'bad-roster', read, no_assets)
    assert report['status'] == 'failed' and not (tmp_path/'bad-roster/complete.json').exists()
    diagnostic = json.loads((tmp_path/'bad-roster/private/failure-diagnostic.json').read_bytes())
    assert diagnostic['message'] == 'existing_NEI_selection_unchanged'


def test_missing_NEI_preserves_full_denominator_no_replacement(tmp_path, monkeypatch):
    files, reads, read, assets = inputs(monkeypatch)
    del files[prep.NEI+'/claim-00.json']
    report = run(tmp_path/'missing', read, assets)
    assert report['status'] == 'failed' and report['denominator'] == 3
    assert report['status_counts'] == {'ready': 2, 'failed': 1, 'unknown': 0}
    assert not (tmp_path/'missing/complete.json').exists()
    assert report['model_calls'] == report['optimizer_steps'] == 0


def test_sorted_nested_write_is_rejected_against_original_factory_tokens(tmp_path, monkeypatch):
    _, _, read, assets = inputs(monkeypatch)
    original_write = prep.ordered_write
    def changed_write(path, value):
        if path.name == 'prepared.json':
            value = json.loads(json.dumps(copy.deepcopy(value), sort_keys=True))
        original_write(path, value)
    monkeypatch.setattr(prep, 'ordered_write', changed_write)
    report = run(tmp_path/'bad-write', read, assets)
    assert report['status'] == 'failed' and report['last_stage'] == 'physical_readback'
    assert report['source_bridge_validated'] is False
    assert not (tmp_path/'bad-write/complete.json').exists()
