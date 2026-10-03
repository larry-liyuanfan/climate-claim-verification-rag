"""Synthetic CPU tests; production capture, no real claim or model weights."""
import copy
import io
import json
from types import SimpleNamespace
from pathlib import Path
import tarfile

import pytest
import torch

from climate_rag.scifact_fit_selection import IdScanner, select_complete_fit
from climate_rag.scifact_grounding import Abstract, GoldClaim, Rationale
from climate_rag.scifact_semantic_contract import encoded, sha
from climate_rag.scifact_state_supervision import (
    VERSION, answer_targets, build_claim, capture, frame_contract, optimizer_step_for_claim_group,
    packing_metadata, read_plan, tokenize_target, validate_tokenized, validate_weights,
)
from climate_rag.scifact_terminal import source_from_abstract


class Tokenizer:
    eos_token = '<EOS>'
    eos_token_id = pad_token_id = 1

    def apply_chat_template(self, messages, **kwargs):
        return json.dumps(messages, ensure_ascii=False) + '\nASSISTANT:'

    def encode(self, text, **kwargs):
        if text.endswith(self.eos_token):
            return [ord(c) + 2 for c in text[:-len(self.eos_token)]] + [1]
        return [ord(c) + 2 for c in text]


def setup(evidence=None):
    corpus = {i: Abstract(i, f'Doc {i}', ('Start.', 'Not evidence.', '完整依据.', 'End.'), False) for i in range(1, 9)}
    sources = [source_from_abstract(corpus[i]) for i in range(1, 8)]
    claim = GoldClaim(1, 'Claim', evidence or {1: (Rationale('SUPPORT', (2,)),)}, ())
    return claim, corpus, sources, Tokenizer()


def test_visible_positive_not_erased_by_unretrieved_gold_and_multiple_docs():
    claim, corpus, sources, tok = setup({1: (Rationale('SUPPORT', (2,)),), 8: (Rationale('SUPPORT', (2,)),)})
    result = build_claim(claim, 'fixture', sources, corpus, tok)
    assert result['records'][0]['target']['action'] == 'answer'
    assert result['records'][0]['target']['documents'][0]['sentence_ids'] == ['c0:2']
    assert result['report']['whole_gold_covered_by_target'] is False
    claim = GoldClaim(1, 'Claim', {1: claim.evidence[1], 2: (Rationale('CONTRADICT', (0, 2)),)}, ())
    result = build_claim(claim, 'fixture', sources, corpus, tok)
    assert len(result['records'][0]['target']['documents']) == 2
    assert result['report']['whole_gold_covered_by_target']
    assert result['records'][0]['target']['documents'][1]['sentence_ids'] == ['c1:0', 'c1:2']


def test_actual_read_feedback_budgets_full_rationale_and_weights():
    claim, corpus, sources, tok = setup({7: (Rationale('SUPPORT', (0, 2)),), 8: (Rationale('SUPPORT', (2,)),)})
    result = build_claim(claim, 'fixture', sources, corpus, tok)
    rows = result['records']
    assert [r['target']['action'] for r in rows] == ['read', 'answer']
    assert rows[0]['target']['source_ids'] == ['c6']
    before, after = [r['frame']['observation'] for r in rows]
    assert after['feedback'] and after['remaining_calls'] == before['remaining_calls'] - 1
    assert after['remaining_tools'] == before['remaining_tools'] - 1
    assert rows[1]['target']['documents'][0]['sentence_ids'] == ['c6:0', 'c6:2']
    assert result['report']['program_teacher_reads'] == 1
    assert result['report']['model_calls'] == 0 and not result['report']['whole_gold_covered_by_target']
    validate_weights(rows, [1])
    assert all(r['weight_denominator'] == 2 for r in rows)


def test_read_still_invisible_is_gap_not_partial_answer():
    claim, corpus, _, tok = setup({7: (Rationale('SUPPORT', (1,)),)})
    corpus[7] = Abstract(7, 'Long', ('x' * 50000, 'Required.'), False)
    sources = [source_from_abstract(corpus[i]) for i in range(1, 8)]
    result = build_claim(claim, 'fixture', sources, corpus, tok)
    assert not result['records'] and result['report']['gaps']
    assert not result['report']['training_ready']


@pytest.mark.parametrize('filler_count,expected', [(8, ['read', 'answer']), (12, [])])
def test_selected_truncated_document_not_in_previews_read_or_explicit_gap(filler_count, expected):
    claim, corpus, _, tok = setup({2: (Rationale('SUPPORT', (3,)),)})
    corpus[1] = Abstract(1, 'Long filler', tuple('x'*200 for _ in range(filler_count)), False)
    corpus[2] = Abstract(2, 'Relevant', ('small', 'z'*800, 'q'*800, 'Evidence.'), False)
    sources = [source_from_abstract(corpus[i]) for i in range(1, 8)]
    result = build_claim(claim, 'fixture', sources, corpus, tok)
    initial = result['captured_frames'][0]['observation']
    assert not any(p['source_id'] == 'c1' for p in initial['preview_only'])
    assert [r['target']['action'] for r in result['records']] == expected
    if expected:
        assert [s['sentence_id'] for s in initial['current_citable'] if s['sentence_id'].startswith('c1:')] == ['c1:0']
        assert result['records'][0]['target']['source_ids'] == ['c1']
        assert result['records'][1]['target']['documents'][0]['sentence_ids'] == ['c1:3']
    else:
        assert result['report']['gaps'] == ['retrieved_positive_read_opportunity_unresolved']


def test_unretrieved_official_positive_context_abstention_is_not_official_nei():
    claim, corpus, sources, tok = setup({8: (Rationale('SUPPORT', (2,)),)})
    result = build_claim(claim, 'fixture', sources, corpus, tok)
    row = result['records'][0]
    assert row['target']['action'] == 'abstain'
    assert row['semantic_target_provenance'] == 'no_semantic_label'
    assert 'not_official_nei' in row['teacher_reason']


def test_alternatives_or_not_union_and_eos_pad_mask_roundtrip():
    claim, corpus, sources, tok = setup({1: (Rationale('SUPPORT', (0, 2)), Rationale('SUPPORT', (3,)))})
    built = build_claim(claim, 'fixture', sources, corpus, tok)
    assert len(built['records']) == 2
    validate_weights(built['records'], [1])
    for row in json.loads(json.dumps(built['records'], ensure_ascii=False)):
        tokens = tokenize_target(tok, row['frame'], row['target'], sources)
        validate_tokenized(row, tokens)
        assert tokens['labels'][:tokens['input_tokens']] == [-100] * tokens['input_tokens']
        assert tokens['labels'][-1] == tok.pad_token_id == tok.eos_token_id
    with pytest.raises(ValueError, match='missing_claim_weight'):
        validate_weights(built['records'][:1], [1])


def test_hidden_teacher_fields_and_schema_drift_rejected():
    claim, corpus, sources, tok = setup()
    frames, _ = capture(claim, sources, corpus, tok)
    bad = copy.deepcopy(frames[0])
    bad['observation']['gold_label'] = 'SUPPORT'
    with pytest.raises(ValueError, match='teacher_fields'):
        frame_contract(bad, sources)
    bad = copy.deepcopy(frames[0])
    bad['schema']['extra'] = True
    with pytest.raises(ValueError, match='production_schema'):
        frame_contract(bad, sources)
    bad = copy.deepcopy(frames[0])
    bad['observation']['remaining_tools'] = 0
    assert read_plan(claim, bad, sources) == []


def test_no_truncation_or_bpe_prefix_assumption():
    claim, corpus, sources, tok = setup()
    frames, _ = capture(claim, sources, corpus, tok)
    target = answer_targets(claim, frames[0], sources, 1)[0]
    class Broken(Tokenizer):
        def encode(self, text, **kwargs):
            ids = super().encode(text, **kwargs)
            return [999] + ids[1:] if text.endswith(self.eos_token) else ids
    with pytest.raises(ValueError, match='assistant_bpe_prefix'):
        tokenize_target(Broken(), frames[0], target, sources)
    class Long(Tokenizer):
        def encode(self, text, **kwargs):
            ids = super().encode(text, **kwargs)
            return ids[:-1] + [4] * 600 + [1] if text.endswith(self.eos_token) else ids
    with pytest.raises(ValueError, match='complete_target_over_budget'):
        tokenize_target(Long(), frames[0], target, sources)


@pytest.mark.parametrize('raw,expected', [
    (b'{"claim":"fake \\"id\\": 7", "evidence":{"id":7},"id":2}', 2),
    (b'{"x":[{"id":7}],"\\u0069d":2}', 2),
    (b'{"id":2,"claim":"backslash \\\\ and [] {}"}', 2),
])
def test_id_first_scanner_ignores_nested_and_string_ids(raw, expected):
    assert IdScanner(raw).identifier() == expected


@pytest.mark.parametrize('raw', [
    b'{"id":2,"id":2}', b'{"id":2,"\\u0069d":2}', b'{"x":{"id":2}}',
    b'{"id":true}', b'{"id":"2"}', b'{"id":2.0}', b'{"id":2e0}', b'{"id":02}',
    b'{"id":2}{}', b'{"id":2,"s":"unfinished}', b'{"id":2,"x":NaN}',
    b'{"id":2,"x":' + b'[' * 65 + b'0' + b']' * 65 + b'}',
    b'{"id":2,"s":"' + b'x' * 1048576 + b'"}',
], ids=['duplicate-id', 'escaped-duplicate', 'nested-only', 'bool', 'string', 'float', 'exponent',
        'leading-zero', 'trailing', 'truncated', 'nan', 'deep', 'oversized'])
def test_malformed_or_ambiguous_id_selection_fails_closed(raw):
    with pytest.raises(ValueError):
        IdScanner(raw).identifier()


def test_complete_fit_only_deserialized_after_hash(monkeypatch):
    _, corpus, _, _ = setup()
    rows = [{'id':i, 'claim':'Claim', 'cited_doc_ids':[1], 'evidence':{'1':[
        {'label':'SUPPORT','sentences':[j % 4]} for j in range(5)]}} for i in (1, 2)]
    raw = b''.join(json.dumps(r).encode() + b'\n' for r in rows)
    original, seen = json.loads, []
    def spy(value, *args, **kwargs):
        if value.startswith(b'{'):
            seen.append(original(value)['id'])
        return original(value, *args, **kwargs)
    monkeypatch.setattr(json, 'loads', spy)
    selected = select_complete_fit(io.BytesIO(raw), sha(raw), [1], corpus)
    assert seen == [1] and len(selected[1].evidence[1]) == 5
    seen.clear()
    with pytest.raises(ValueError, match='train_member_sha'):
        select_complete_fit(io.BytesIO(raw), '0'*64, [1], corpus)
    assert not seen
    with pytest.raises(ValueError, match='coverage'):
        select_complete_fit(io.BytesIO(raw), sha(raw), [3], corpus)
    with pytest.raises(ValueError, match='duplicate_train_row'):
        select_complete_fit(io.BytesIO(raw + raw), sha(raw + raw), [1], corpus)


def toy_rows(claim_count, states=1, alternatives=1):
    tokens = {'input_ids':[0, 1, 1], 'labels':[-100, 1, 1], 'attention_mask':[1, 1, 1],
              'input_tokens':1, 'target_tokens':2, 'full_token_ids_sha256':sha(encoded([0,1,1])),
              'loss_mask_sha256':sha(encoded([-100,1,1]))}
    rows = []
    for i in range(claim_count):
        for alt in range(alternatives):
            for state in range(states):
                row = {'version':VERSION,'claim_id':i,'trajectory':0,'trajectory_count':1,
                       'alternative':alt,'alternative_count':alternatives,'state_index':state,'state_count':states,
                       'weight_numerator':1,'weight_denominator':states*alternatives,'declared_claim_weight':1,
                       'epoch_normalizer':48,'model_generated':False,'decision_origin':'program_teacher',
                       'action_target_provenance':'program_teacher_actual_state_v2','target':{'action':'read'},
                       'semantic_target_provenance':'no_semantic_label','packing':packing_metadata(tokens)}
                row['record_sha256'] = sha(encoded(row))
                rows.append(row)
    return rows, [tokens] * len(rows)


@pytest.mark.parametrize('claims,states,alternatives', [(1,1,1),(1,2,1),(1,1,4),(1,2,2),(4,1,1),(3,1,1)])
def test_actual_backward_fixed_48_denominator_short_chunk_and_two_turn(claims, states, alternatives):
    class Model(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.value = torch.nn.Parameter(torch.zeros(2))
        def forward(self, input_ids, attention_mask):
            return SimpleNamespace(logits=self.value.expand(1, 3, 2))
    model = Model()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    rows, tokens = toy_rows(claims, states, alternatives)
    result = optimizer_step_for_claim_group(model, optimizer, rows, tokens)
    assert torch.allclose(model.value, torch.tensor([-0.05,0.05]) * claims / 48)
    assert result['global_objective_contribution'] == pytest.approx(claims * 0.69314718 / 48)
    assert result['optimizer_steps'] == 1


def test_training_rejects_token_tamper_before_backward():
    rows, tokens = toy_rows(1)
    tokens[0]['labels'][0] = 1
    with pytest.raises(ValueError, match='token_or_mask_identity'):
        validate_tokenized(rows[0], tokens[0])


def test_narrow_input_loader_never_opens_manifest_other_files(tmp_path, monkeypatch):
    import prepare_scifact_state_supervision as prep
    from climate_rag.scifact_grounding_sft import fit_records
    claim, corpus, _, _ = setup()
    rows = []
    for i in range(48):
        rows.extend(fit_records(GoldClaim(i, claim.claim, claim.evidence, ()), corpus, str(i)))
    corpus_raw = b''.join(json.dumps({'doc_id':d.doc_id,'title':d.title,'abstract':d.sentences,
                                     'structured':False}).encode() + b'\n' for d in corpus.values())
    monkeypatch.setattr(prep, 'CORPUS_SHA', sha(corpus_raw))
    payloads = {'fit/records.json':encoded(rows),
                'private/selection-before-packing.json':encoded({'fit':list(range(48))}),
                'private/families.json':encoded({'corpus_sha256':sha(corpus_raw),'claim_grouping_recomputed':False}),
                'inference/corpus.jsonl':corpus_raw}
    hashes = {n:sha(v) for n,v in payloads.items()}
    manifest = encoded({'files':hashes | {'scoring/validation.json':'must-not-open'}})
    for name, raw in payloads.items():
        path = tmp_path/name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(raw)
    (tmp_path/'manifest.json').write_bytes(manifest)
    original, opened = Path.read_bytes, []
    def guarded(path):
        relative = path.relative_to(tmp_path).as_posix()
        assert relative in {*hashes, 'manifest.json'}
        opened.append(relative)
        return original(path)
    monkeypatch.setattr(Path, 'read_bytes', guarded)
    fit, _, _, _ = prep.narrow_inputs(tmp_path, hashes, sha(manifest))
    assert len(fit) == 48 and len(opened) == 5
    with pytest.raises(ValueError, match='sha_mismatch'):
        prep.narrow_inputs(tmp_path, hashes, '0'*64)


def test_original_rows_and_document_family_join():
    from prepare_scifact_state_supervision import validate_old_rows
    from climate_rag.scifact_grounding_sft import fit_records
    claim, corpus, _, _ = setup()
    rows = fit_records(claim, corpus, 'component')
    families = {'document_family':{'1':1},'family_component':{'1':'component'},'component_partition':{'component':'fit'}}
    assert validate_old_rows(rows, {1:claim}, families, corpus) == {1:'component'}
    with pytest.raises(ValueError, match='duplicate_v1'):
        validate_old_rows(rows + rows, {1:claim}, families, corpus)
    families['family_component']['1'] = 'other'
    with pytest.raises(ValueError, match='family_component'):
        validate_old_rows(rows, {1:claim}, families, corpus)


@pytest.mark.parametrize('kind', ['duplicate', 'link'])
def test_archive_member_rejects_duplicates_and_links(kind):
    from prepare_scifact_state_supervision import member
    stream = io.BytesIO()
    with tarfile.open(fileobj=stream, mode='w') as archive:
        info = tarfile.TarInfo('gold.jsonl')
        if kind == 'link':
            info.type, info.linkname = tarfile.SYMTYPE, 'other'
        archive.addfile(info)
        if kind == 'duplicate':
            archive.addfile(info)
    stream.seek(0)
    with tarfile.open(fileobj=stream) as archive, pytest.raises(ValueError, match='unique_regular'):
        member(archive, 'gold.jsonl')
