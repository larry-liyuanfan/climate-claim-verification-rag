"""Six synthetic annotation groups, actual controller → receipt adapter input."""
import copy
import hashlib
import json

import pytest

from climate_rag.model_diagnostics import response_diagnostics
from climate_rag.scifact_grounding import Abstract
from climate_rag.scifact_observation_receipts import FROZEN_CONTRACT, VERSION as RECEIPTS, audit_observations
from climate_rag.scifact_state_training_driver import corpus_identity
from climate_rag.scifact_terminal import source_from_abstract, render_scifact_prompt
from climate_rag.scifact_terminal_supervision import VERSION, terminal_candidates, validate_terminal_candidate
from climate_rag.scifact_utility_contract import identity
from climate_rag.scifact_utility_runtime import run_matrix
from test_bounded_scifact_runtime import ABSTAIN, SyntheticBackend
from test_scifact_state_supervision import Tokenizer


def observations(path, corpus=None):
    path.mkdir(exist_ok=True)
    corpus = corpus or {i: Abstract(i, f'Doc{i}', tuple(f'Sentence {i}.{j}.' for j in range(4)), False)
                        for i in range(100, 112)}
    tokenizer = Tokenizer()
    class Backend(SyntheticBackend):
        def start_slot(self, p):
            p.mkdir()
            claim_id = int(p.parent.name.split('-')[0])
            self.actions = iter([{'action': 'rewrite', 'query': f'Fixture claim {claim_id} context'},
                                 {'action': 'rerank'}, {'fixture_answer': True}]
                                if p.parent.name.endswith('-A') else [ABSTAIN])
        def generate(self, observation, *args):
            response = super().generate(observation, *args)
            if json.loads(response['raw']) == {'fixture_answer': True}:
                sid = observation['current_citable'][0]['sentence_id']
                response['raw'] = json.dumps({'action': 'answer', 'documents': [
                    {'source_id': sid.split(':')[0], 'label': 'REFUTES', 'sentence_ids': [sid]}]})
            raw = response['raw']
            response['diagnostics'] = response_diagnostics(raw, output_tokens=50, max_new_tokens=512,
                eos_observed=True, generation_elapsed_ms=0)
            response['diagnostics'].update(raw_wire_action=json.loads(raw)['action'], private_attachment={
                'sha256': hashlib.sha256(raw.encode()).hexdigest(), 'attempted_bytes': len(raw.encode()),
                'stored_bytes': len(raw.encode()), 'truncated': False, 'io_failed': False})
            return response
    backend = Backend([], False)
    backend.base.tokenizer = tokenizer
    def retrieve(query, width):
        ids = [107, 108, 109, *range(100, 107)] if query.endswith(' context') else range(100, 108)
        return [source_from_abstract(corpus[i]) for i in ids][:width]
    claims = [{'id': i, 'claim': f'Fixture claim {i}'} for i in range(8)]
    output = path / 'inference'
    run_matrix(claims, backend, retrieve, lambda q, rows: list(reversed(rows)), corpus, output)
    payload = {'version': RECEIPTS, 'scope': 'synthetic_fixture', 'contract_revision': FROZEN_CONTRACT,
        'slots': [{'claim_id': c['id'], 'arm': arm, 'claim': c['claim']} for c in claims for arm in 'ABC']}
    artifacts = [(p.relative_to(output).as_posix(), p.read_bytes()) for p in output.rglob('*.json')]
    report = audit_observations(artifacts, {'payload': payload, 'sha256': identity(payload)}, tokenizer, corpus)
    assert report['accepted_slots'] == 24, report['slots'][:3]
    return report, corpus, tokenizer


@pytest.fixture(scope='module')
def observed(tmp_path_factory):
    return observations(tmp_path_factory.mktemp('terminal-annotations'))


def inputs(observed, evidence, slot='0-A', attempt=2, cited=None):
    report, corpus, tokenizer = observed
    physical = next(p for p in report['physical_observations'] if any(
        r['slot'] == slot and r['attempt_index'] == attempt for r in p['logical_references']))
    ref = {'physical_attempt_id': physical['physical_attempt_id'], 'slot': slot, 'attempt_index': attempt}
    annotation = json.dumps({'id': 0, 'claim': '  Fixture   claim 0 ', 'evidence': evidence,
                             'cited_doc_ids': [111] if cited is None else cited}, ensure_ascii=False).encode()
    payload = {'version': VERSION, 'scope': 'synthetic_fixture',
        'annotation_source': 'externally_supplied_synthetic_complete_original_row', 'complete_original_row_declared': True,
        'annotation_sha256': hashlib.sha256(annotation).hexdigest(), 'accepted_report_sha256': identity(report),
        'corpus_sha256': corpus_identity(corpus), 'reference_sha256': identity(ref)}
    return report, ref, annotation, {'payload': payload, 'sha256': identity(payload)}, corpus, tokenizer


def evidence(doc, sentences=(0,), label='SUPPORT'):
    return {str(doc): [{'label': label, 'sentences': list(sentences)}]}


def test_official_nei_with_cited_metadata_vs_positive_missing_context(observed):
    args = inputs(observed, {}, cited=[111, 111])
    nei = terminal_candidates(*args, candidate_cap=8)
    assert nei['candidates'][0]['target'] == ABSTAIN
    assert nei['candidates'][0]['basis'] == 'official_annotation_NEI'
    assert nei['candidates'][0]['status'] == 'representable_candidate'
    assert not nei['training_authorized'] and not nei['cohort_created'] and not nei['weights_created']
    positive_args = inputs(observed, evidence(111))
    gap = terminal_candidates(*positive_args, candidate_cap=8)
    assert gap['status'] == 'context_insufficient_for_annotation_view' and gap['candidates'] == []
    with pytest.raises(ValueError, match='not_official_annotation_NEI'):
        validate_terminal_candidate(ABSTAIN, *positive_args)


def test_complete_or_labels_unions_partial_and_document_coverage(observed):
    gold = evidence(109, (0, 1))
    gold['109'].append({'label': 'SUPPORT', 'sentences': [2, 3]})
    gold.update(evidence(108, (0,), 'CONTRADICT'))
    gold.update(evidence(111))  # Unretrieved gold does not erase a valid local answer.
    args = inputs(observed, gold)
    result = terminal_candidates(*args, candidate_cap=8)
    assert result['candidate_count_before_cap'] == 2 and len(result['candidates']) == 2
    assert all(not r['whole_gold_document_coverage'] for r in result['candidates'])
    combined = next(r for r in result['candidates'] if len(r['target']['documents']) == 2)
    assert {d['label'] for d in combined['target']['documents']} == {'SUPPORTS', 'REFUTES'}
    target = result['candidates'][0]['target']
    alias = next(d['source_id'] for d in target['documents'] if d['label'] == 'SUPPORTS')
    for indices, label in (([0], 'SUPPORTS'), ([0, 1, 2, 3], 'SUPPORTS'), ([0, 1], 'REFUTES')):
        wrong = {'action': 'answer', 'documents': [{'source_id': alias, 'label': label,
                                                   'sentence_ids': [f'{alias}:{i}' for i in indices]}]}
        with pytest.raises(ValueError, match='not_one_complete_original_OR_rationale'):
            validate_terminal_candidate(wrong, *args)
    omitted = copy.deepcopy(target)
    omitted['documents'] = omitted['documents'][:1]
    with pytest.raises(ValueError, match='supervision_omits_complete_current_visible_document'):
        validate_terminal_candidate(omitted, *args)
    without_missing = copy.deepcopy(gold)
    without_missing.pop('111')
    whole = terminal_candidates(*inputs(observed, without_missing), candidate_cap=8)
    assert any(r['whole_gold_document_coverage'] for r in whole['candidates'])


def test_actual_rewrite_rerank_stable_alias_scripted_origin_and_wrong_model_proposal(observed):
    original = copy.deepcopy(observed[0])
    result = terminal_candidates(*inputs(observed, evidence(109)), candidate_cap=8)
    candidate = result['candidates'][0]
    source = result['observed_model']
    alias = candidate['target']['documents'][0]['source_id']
    assert alias == 'c9' and source['capture']['alias_to_source'][alias] == '109'
    assert source['logical_reference']['state_event']['candidate_ids'][0] == alias
    assert source['parsed_model_proposal']['documents'][0]['label'] == 'REFUTES'
    assert candidate['target']['documents'][0]['label'] == 'SUPPORTS'
    assert candidate['target_origin'] == 'program_derived_annotation_candidate' and not candidate['model_generated']
    rewrite = terminal_candidates(*inputs(observed, evidence(100), attempt=1), candidate_cap=8)
    assert rewrite['observed_model']['logical_reference']['state_event']['tool'] == 'rewrite'
    scripted = terminal_candidates(*inputs(observed, evidence(105), slot='0-B', attempt=1), candidate_cap=8)
    event = scripted['observed_model']['logical_reference']['state_event']
    assert event['origin'] == 'scripted_intervention' and event['model_selected'] is False
    assert observed[0] == original  # No gold prompt / capture or raw rewrite.


def test_candidate_and_selected_long_document_missing_evidence_remains_gap(observed, tmp_path):
    initial = terminal_candidates(*inputs(observed, evidence(105), attempt=0), candidate_cap=8)
    assert initial['candidates'] == [] and initial['incomplete_annotation_view'][0]['candidate_alias'] == 'c5'
    assert 'read' in initial['budget_observation']['allowed_actions']
    corpus = copy.deepcopy(observed[1])
    corpus[104] = Abstract(104, 'Long selected document', ('x' * 20000, 'Required unseen rationale.'), False)
    long = observations(tmp_path / 'long', corpus)
    gap = terminal_candidates(*inputs(long, evidence(104, (1,)), attempt=0), candidate_cap=8)
    assert gap['status'] == 'context_insufficient_for_annotation_view' and gap['candidates'] == []
    assert gap['incomplete_annotation_view'][0]['selected_context'] is True
    assert gap['incomplete_annotation_view'][0]['incomplete_alternatives'][0]['missing_original_sentence_indices'] == [1]
    assert gap['budget_observation']['remaining_calls'] == 5 and gap['budget_observation']['remaining_tools'] == 4


def test_annotation_claim_corpus_and_observed_model_provenance_tampering(observed):
    args = inputs(observed, evidence(109))
    for position in (0, 1, 2, 4):
        bad = list(copy.deepcopy(args))
        if position == 0:
            bad[0]['physical_observations'][0]['parsed_model_proposal']['action'] = 'abstain'
        elif position == 1:
            bad[1]['attempt_index'] = 0
        elif position == 2:
            bad[2] += b'\n'  # Full original bytes, not a normalized gold subset.
        else:
            bad[4][111] = Abstract(111, 'Changed', ('Changed.',), False)
        with pytest.raises(ValueError, match='annotation_report_corpus_reference_identity'):
            terminal_candidates(*bad, candidate_cap=8)
    for field, value in (('id', 7), ('claim', 'Different claim')):
        bad = list(copy.deepcopy(args))
        row = json.loads(bad[2])
        row[field] = value
        bad[2] = json.dumps(row).encode()
        bad[3]['payload']['annotation_sha256'] = hashlib.sha256(bad[2]).hexdigest()
        bad[3]['sha256'] = identity(bad[3]['payload'])
        with pytest.raises(ValueError, match='annotation_claim_identity'):
            terminal_candidates(*bad, candidate_cap=8)
    bad = list(copy.deepcopy(args))
    physical = next(p for p in bad[0]['physical_observations'] if p['physical_attempt_id'] == bad[1]['physical_attempt_id'])
    physical['parsed_model_proposal']['documents'][0]['label'] = 'SUPPORTS'
    bad[3]['payload']['accepted_report_sha256'] = identity(bad[0])
    bad[3]['sha256'] = identity(bad[3]['payload'])
    with pytest.raises(ValueError, match='observed_proposal_changed'):
        terminal_candidates(*bad, candidate_cap=8)


def test_exact_prompt_assistant_mask_enumeration_overflow_and_no_token_truncation(observed, tmp_path):
    gold = {str(doc): [{'label': 'SUPPORT', 'sentences': [i]} for i in (0, 1)] for doc in (109, 108)}
    args = inputs(observed, gold)
    overflow = terminal_candidates(*args, candidate_cap=3)
    assert overflow['candidate_count_before_cap'] == 4 and overflow['candidates'] == []
    assert overflow['status'] == 'enumeration_overflow'
    result = terminal_candidates(*args, candidate_cap=4)
    assert len(result['candidates']) == 4
    for row in result['candidates']:
        tokens = row['tokenized']
        n = tokens['input_tokens']
        assert tokens['labels'] == [-100] * n + tokens['input_ids'][n:]
        assert tokens['input_ids'][-1] == observed[2].eos_token_id
        prompt = render_scifact_prompt(observed[2], result['frame']['observation'], result['frame']['schema'])
        assert tokens['prompt_sha256'] == hashlib.sha256(prompt.encode()).hexdigest()
    four = terminal_candidates(*inputs(observed, evidence(109, (0, 1, 2, 3))), candidate_cap=1)
    assert four['candidates'][0]['matched_rationales'][0]['first_three_cover_selected_rationale'] is False
    class WideAssistant(Tokenizer):
        # A deterministic synthetic tokenizer with four tokens per assistant character;
        # prompt token IDs are unchanged, and EOS remains exactly one unmasked token.
        def encode(self, text, **kwargs):
            if text.endswith(self.eos_token):
                boundary = text.rindex('\nASSISTANT:') + len('\nASSISTANT:')
                prefix = super().encode(text[:boundary], **kwargs)
                suffix = text[boundary:-len(self.eos_token)]
                return prefix + [ord(c) + 2 for c in suffix for _ in range(4)] + [self.eos_token_id]
            return super().encode(text, **kwargs)
    large = terminal_candidates(*args[:-1], WideAssistant(), candidate_cap=4)
    assert large['candidate_count_before_cap'] == len(large['candidates']) == 4
    assert all(r['gap'] == 'complete_target_over_budget' for r in large['candidates'])
    assert [r['target'] for r in large['candidates']] == [r['target'] for r in result['candidates']]
    class BrokenPrefix(Tokenizer):
        def encode(self, text, **kwargs):
            tokens = super().encode(text, **kwargs)
            if text.endswith(self.eos_token):
                tokens[0] += 1
            return tokens
    broken = terminal_candidates(*args[:-1], BrokenPrefix(), candidate_cap=4)
    assert len(broken['candidates']) == 4 and all(r['gap'] == 'assistant_bpe_prefix_mismatch' for r in broken['candidates'])
    corpus = copy.deepcopy(observed[1])
    corpus[109] = Abstract(109, 'Nine sentences', tuple(f'S{i}.' for i in range(9)), False)
    nine = observations(tmp_path / 'nine', corpus)
    schema = terminal_candidates(*inputs(nine, evidence(109, tuple(range(9)))), candidate_cap=1)
    assert schema['candidates'][0]['gap'] == 'terminal_schema_unrepresentable'
    assert len(schema['candidates'][0]['target']['documents'][0]['sentence_ids']) == 9
