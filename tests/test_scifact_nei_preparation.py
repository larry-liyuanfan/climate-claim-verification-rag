"""Six synthetic groups; never load remote gold or invoke a model."""
import copy
import json

import pytest

import climate_rag.scifact_nei_preparation as prep
from climate_rag.scifact_grounding import Abstract, parse_gold
from climate_rag.scifact_program_capture import VERSION, capture_initial, program_candidates, validate_capture
from climate_rag.scifact_retrieval import SciFactBM25
from climate_rag.scifact_semantic_contract import checked, encoded, sha
from climate_rag.scifact_state_supervision import capture
from climate_rag.scifact_state_training_driver import corpus_identity
from climate_rag.scifact_terminal_supervision import terminal_candidates
from climate_rag.scifact_utility_contract import identity
from test_scifact_state_supervision import Tokenizer


def fixture():
    corpus = {i: Abstract(i, f'Doc{i}', (f'Water rises in {i}.', 'Climate varies.'), False) for i in range(100, 125)}
    row = {'id': 7, 'claim': '  Water   rises ', 'evidence': {}, 'cited_doc_ids': [124, 124]}
    tokenizer = Tokenizer()
    receipt = capture_initial({k: row[k] for k in ('id', 'claim')}, SciFactBM25(corpus), corpus, tokenizer)
    return row, corpus, tokenizer, receipt


def arguments(row, corpus, tokenizer, receipt):
    annotation = encoded(row)
    p = {'version': VERSION, 'scope': 'synthetic_fixture', 'annotation_sha256': sha(annotation),
         'capture_sha256': receipt['sha256'], 'corpus_sha256': corpus_identity(corpus),
         'complete_original_row_declared': True, 'selection_sha256': 'a'*64, 'parent_gold_sha256': 'b'*64}
    return receipt, annotation, {'payload': p, 'sha256': identity(p)}, corpus, tokenizer


def metadata_fixture():
    ids, components = [9, 2, 7], ['z', 'x', 'y']
    selection = {'selected_ordered_ids': ids, 'selected_ordered_components': components, 'config_sha256': 'a'*64}
    compact = {'actual_selected_claims': 3, 'selected_ids_sha256': sha(encoded(ids)),
        'selected_components_sha256': sha(encoded(components)), 'config_sha256': 'a'*64,
        'private_file_sha256': {'selection-before-content.json': 'b'*64}}
    reservation = {'ordered_claim_ids': ids, 'ordered_components': components,
                   'selection_sha256': 'b'*64, 'config_sha256': 'a'*64}
    reports = [{'claim_id': i, 'component': c, 'status': 'not_applicable', 'official_label_scope': 'NOT_ENOUGH_INFO',
                'reason': 'official_nei_not_supported_by_frozen_teacher', 'retained_records': 0}
               for i, c in zip(ids, components, strict=True)]
    reports[1].update(status='prepared', official_label_scope='SUPPORTS', reason=None, retained_records=1)
    rows = [{'id': i, 'claim': 'Water rises', 'evidence': {}, 'cited_doc_ids': [124]} for i in ids]
    rows[1]['evidence'] = {'100': [{'label': 'SUPPORT', 'sentences': [0]}]}
    return {'compact': compact, 'selection': selection, 'reservation': reservation, 'reports': reports}, rows


def test_zero_model_actual_initial_retrieval_and_full_capture(monkeypatch):
    calls = []
    original = SciFactBM25.__call__
    def recorded(self, query, width):
        calls.append((query, width))
        return original(self, query, width)
    monkeypatch.setattr(SciFactBM25, '__call__', recorded)
    row, corpus, tokenizer, receipt = fixture()
    p = receipt['payload']
    assert calls == [('Water rises', 20)]
    assert set(p['capture']) == {'observation', 'schema', 'visible', 'prompt_tokens', 'alias_to_source'}
    assert p['counts']['real_model_calls'] == p['counts']['model_calls'] == 0
    assert p['counts']['scripted_responses'] == p['controller_counts']['model_calls'] == 1
    assert p['state_event']['status'] == 'completed' and p['state_event']['before_context'] == []
    assert p['capture']['observation']['remaining_calls'] == 5
    assert p['capture']['observation']['remaining_tools'] == 4 and p['capture']['observation']['feedback'] is None
    expected = [s.source_id for s in original(SciFactBM25(corpus), 'Water rises', 20)]
    assert p['retrieval']['document_ids'] == expected
    with pytest.raises(ValueError, match='gold_free_inference'):
        capture_initial(row, SciFactBM25(corpus), corpus, tokenizer)
    with pytest.raises(ValueError, match='full_BM25_corpus_binding'):
        capture_initial({k: row[k] for k in ('id', 'claim')}, SciFactBM25({100: corpus[100]}), corpus, tokenizer)
    wrong_index = SciFactBM25(corpus)
    wrong_index.index = SciFactBM25({100: corpus[100]}).index
    with pytest.raises(ValueError, match='actual_BM25_index_binding'):
        capture_initial({k: row[k] for k in ('id', 'claim')}, wrong_index, corpus, tokenizer)


def test_only_NEI_complete_annotation_claim_source_binding():
    row, corpus, tokenizer, receipt = fixture()
    result = program_candidates(*arguments(row, corpus, tokenizer, receipt))
    candidate = result['candidates'][0]
    assert candidate['basis'] == 'official_annotation_NEI'
    assert candidate['target'] == {'action': 'abstain', 'reason': 'insufficient_evidence'}
    for field, value, reason in [('id', 8, 'annotation_claim_identity'), ('claim', 'Wrong', 'annotation_claim_identity'),
            ('evidence', {'100': [{'label': 'SUPPORT', 'sentences': [0]}]}, 'official_NEI_only')]:
        wrong = copy.deepcopy(row)
        wrong[field] = value
        with pytest.raises(ValueError, match=reason):
            program_candidates(*arguments(wrong, corpus, tokenizer, receipt))
    changed = dict(corpus)
    changed[100] = Abstract(100, 'Changed source', ('Different.',), False)
    with pytest.raises(ValueError):
        program_candidates(*arguments(row, changed, tokenizer, receipt))


def test_BM25_order_prompt_initial_contract_tamper_rejected_after_reseal():
    _, corpus, tokenizer, receipt = fixture()
    mutations = [lambda p: p['state_event'].update(before_context=['c0']),
        lambda p: p['state_event'].update(origin='scripted_intervention'),
        lambda p: p['state_event'].update(physical_attempt_id='g00'),
        lambda p: p['retrieval']['document_ids'].reverse(),
        lambda p: p.update(prompt_identity='0'*64),
        lambda p: p.update(token_ids_identity='0'*64),
        lambda p: p['capture']['observation'].update(remaining_calls=4),
        lambda p: p['counts'].update(model_calls=1),
        lambda p: p['capture'].update(physical_attempt_id='g00')]
    for mutate in mutations:
        wrong = copy.deepcopy(receipt)
        mutate(wrong['payload'])
        wrong['sha256'] = identity(wrong['payload'])
        with pytest.raises(ValueError):
            validate_capture(wrong, corpus, tokenizer)


def test_physical_program_separation_and_old_capture_compatibility():
    row, corpus, tokenizer, receipt = fixture()
    result = program_candidates(*arguments(row, corpus, tokenizer, receipt))
    assert 'observed_model' not in result and 'parsed_model_proposal' not in json.dumps(result)
    assert 'physical_attempt_id' not in json.dumps(result)
    assert not result['training_authorized'] and not result['cohort_created'] and not result['weights_created']
    candidates = list(SciFactBM25(corpus)(row['claim'], 20))
    frames, counts = capture(parse_gold(row, corpus), candidates, corpus, tokenizer)
    assert frames == [result['frame']] and counts['model_calls'] == 0 and counts['scripted_responses'] == 1
    # The physical binder has no new flag/bypass accepting program provenance.
    _, annotation, provenance, _, _ = arguments(row, corpus, tokenizer, receipt)
    with pytest.raises(ValueError, match='synthetic_annotation_contract_required'):
        terminal_candidates(receipt, {}, annotation, provenance, corpus, tokenizer, candidate_cap=1)


def test_frozen_subset_original_order_gold_after_freeze_and_tamper(tmp_path):
    meta, rows = metadata_fixture()
    _, corpus, _, _ = fixture()
    selection = prep.fixed_selection(**meta, total=3, nei=2)
    assert selection['selected_ordered_ids'] == [9, 7] and not selection['selection_before_first_label_exposure']
    frozen = prep.freeze(tmp_path, selection)
    reads = []
    raw = encoded(rows)
    def opened():
        assert all(checked(tmp_path/n, h) for n, h in frozen.items())
        reads.append(True)
        return raw
    restored = prep.reserved_gold(opened, sha(raw), tmp_path, frozen, corpus)
    assert list(restored) == [9, 7] and reads == [True]
    for tamper in ('order', 'missing', 'duplicate', 'positive', 'extra_field'):
        bad = copy.deepcopy(rows)
        if tamper == 'order':
            bad.reverse()
        elif tamper == 'missing':
            bad.pop()
        elif tamper == 'duplicate':
            bad[2] = bad[0]
        elif tamper == 'positive':
            bad[0]['evidence'] = bad[1]['evidence']
        else:
            bad[0]['injected'] = True
        content = encoded(bad)
        with pytest.raises(ValueError):
            prep.reserved_gold(lambda: content, sha(content), tmp_path, frozen, corpus)
    wrong = copy.deepcopy(meta)
    wrong['reports'].reverse()
    with pytest.raises(ValueError, match='original96_order_binding'):
        prep.fixed_selection(**wrong, total=3, nei=2)
    (tmp_path/'NEI-selection.json').write_bytes(b'{}')
    with pytest.raises(ValueError):
        prep.reserved_gold(opened, sha(raw), tmp_path, frozen, corpus)
    assert reads == [True]


def test_exclusivity_failure_full_denominator_token_masks_and_artifact_hash(tmp_path, monkeypatch):
    meta, rows = metadata_fixture()
    _, corpus, tokenizer, _ = fixture()
    raw = encoded(rows)
    release = {'scope': 'synthetic_fixture'}
    def execute(path, **kw):
        return prep.prepare_reserved(path, release, lambda: meta, lambda: raw, sha(raw),
                                     lambda: (corpus, tokenizer), total=3, nei=2, **kw)
    good = tmp_path/'good'
    result = execute(good)
    assert result['whole_cohort_data_ready'] and result['denominator'] == result['actual_records'] == 2
    assert (good/'complete.json').exists() and not (good/'failed.json').exists()
    marker = json.loads((good/'complete.json').read_bytes())
    assert marker['compact_sha256'] == sha((good/'compact.json').read_bytes())
    for name, digest in result['private_file_sha256'].items():
        checked(good/name, digest)
    first = json.loads((good/'claim-00.json').read_bytes())['candidates'][0]['tokenized']
    n = first['input_tokens']
    assert first['labels'] == [-100]*n + first['input_ids'][n:]
    assert first['input_ids'][-1] == tokenizer.eos_token_id
    with pytest.raises(FileExistsError):
        execute(good)
    original = prep.capture_initial
    def failing(inference, *args):
        if inference['id'] == 7:
            raise ValueError('synthetic_failure_no_retry')
        return original(inference, *args)
    monkeypatch.setattr(prep, 'capture_initial', failing)
    failed = tmp_path/'failed'
    result = execute(failed)
    assert result['denominator'] == 2 and result['status_counts'] == {'ready': 1, 'gap': 0, 'failed': 1, 'unknown': 0}
    assert result['scripted_responses_known'] == result['scripted_cost_unknown_claims'] == 1
    assert (failed/'failed.json').exists() and not (failed/'complete.json').exists()
    unknown = execute(tmp_path/'timeout', seconds=-1)
    assert unknown['denominator'] == unknown['status_counts']['unknown'] == 2
    assert unknown['actual_records'] == 0 and not unknown['whole_cohort_data_ready']
    with pytest.raises(ValueError):
        checked(good/'claim-00.json', '0'*64)
    for interrupt in (KeyboardInterrupt, TimeoutError):
        called = []
        def interrupted(inference, *args):
            called.append(inference['id'])
            raise interrupt('synthetic_SIGTERM_or_timeout')
        monkeypatch.setattr(prep, 'capture_initial', interrupted)
        interrupted_result = execute(tmp_path/interrupt.__name__)
        assert called == [9] and interrupted_result['status_counts']['unknown'] == 2
    monkeypatch.setattr(prep, 'capture_initial', original)
    def failed_check():
        raise ValueError('synthetic_final_invariant')
    invariant = tmp_path/'invariant'
    assert not execute(invariant, final_check=failed_check)['whole_cohort_data_ready']
    assert not (invariant/'complete.json').exists() and (invariant/'failed.json').exists()
    def failed_link(*args):
        raise OSError('synthetic_publish_failure')
    monkeypatch.setattr(prep.os, 'link', failed_link)
    publish = tmp_path/'publish'
    with pytest.raises(OSError, match='synthetic_publish_failure'):
        execute(publish)
    assert (publish/'failed.json').exists() and not (publish/'complete.json').exists()
