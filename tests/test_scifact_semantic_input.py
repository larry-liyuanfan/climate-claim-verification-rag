"""Input isolation/compatibility and failure accounting, not model-quality tests."""
import copy
import json
import time
from pathlib import Path

import pytest

from climate_rag.scifact_document_verifier import call_input
from climate_rag.scifact_evidence_commit import CommitState, ISOLATED_PROTOCOL, PROTOCOL, render_prompt
from climate_rag.scifact_evidence_commit_runtime import audit_episode
from climate_rag.scifact_grounding import GoldClaim, Rationale
from climate_rag.scifact_relation_verifier import RELATION_PROTOCOL
from climate_rag.scifact_utility_runtime import ledger_cost
from diagnose_scifact_semantic_input import diagnose_claim, invariant_probe
import run_scifact_evidence_commit_operator as operator
import scifact_evidence_input as inputs_adapter
from test_bounded_scifact_runtime import ABSTAIN
from test_scifact_evidence_commit import Backend, RUN, audit, choose, inputs, run, verdict


def test_old_input_contract_unchanged_and_new_projection_invariant():
    frame, _ = inputs(2)
    state = CommitState(1, 'adaptive', frame, 'old')
    for calls in range(5):
        state.calls = calls
        assert state.inputs('verify', 'c7', []) == call_input(frame, 'verify', 'c7', [], calls, 1, [])
    tokenizer = Backend([]).base.tokenizer
    probe = invariant_probe(frame, tokenizer)
    assert probe['documents'] == 2 and probe['cross_state_comparisons'] == 24
    assert probe['all_equal'] and probe['planner_real_budget_preserved']
    isolated = CommitState(1, 'adaptive', frame, 'new', protocol=ISOLATED_PROTOCOL)
    obs, schema = isolated.inputs('verify', 'c7', [])
    changed = copy.deepcopy(frame)
    changed['observation']['immutable_claim'] = 'A different synthetic claim'
    other = CommitState(1, 'adaptive', changed, 'new', protocol=ISOLATED_PROTOCOL)
    different, _ = other.inputs('verify', 'c7', [])
    assert render_prompt(tokenizer, obs, schema) != render_prompt(tokenizer, different, schema)
    with pytest.raises(ValueError, match='protocol'):
        CommitState(1, 'adaptive', frame, 'x', protocol='unknown')


@pytest.mark.parametrize('arm', ['fixed_top1', 'fixed_all', 'adaptive'])
def test_new_runtime_physical_audit_and_no_version_downgrade(tmp_path, arm):
    actions = [verdict()] if arm == 'fixed_top1' else (
        [verdict(), verdict('c8')] if arm == 'fixed_all' else
        [{'action':'verify', 'source_id':'c7'}, verdict(), choose])
    row, backend, journal, frame, corpus = run(tmp_path, actions, arm, protocol=ISOLATED_PROTOCOL)
    assert row['protocol'] == ISOLATED_PROTOCOL and row['state'] == 'valid_terminal'
    assert audit(row, backend, journal, frame, corpus, tmp_path) == row
    assert ledger_cost(journal.directory)['unique_physical_calls'] == len(actions)
    for step in row['steps']:
        if step['stage'] == 'verify':
            assert 'remaining' not in step['observation'] and 0 < step['remaining_seconds'] <= 120
        else:
            assert 'remaining' in step['observation']
    with pytest.raises(ValueError, match='episode_identity'):
        audit_episode(row, frame, journal.directory, tmp_path/'episode/private-responses',
                      backend.base.tokenizer, corpus, run_identity=RUN, protocol=PROTOCOL)
    forged = copy.deepcopy(row)
    forged['steps'][0]['observation']['remaining'] = {'physical_generations':999}
    if arm != 'adaptive':
        with pytest.raises(ValueError, match='policy'):
            audit(forged, backend, journal, frame, corpus, tmp_path)


@pytest.mark.parametrize('fault', ['unknown', 'schema', 'deadline'])
def test_new_failure_paths_remain_charged_and_unresolved(tmp_path, monkeypatch, fault):
    timer = {'offset':0.0}
    original = Backend.generate
    def generate(self, *args):
        if fault == 'unknown':
            raise RuntimeError('synthetic backend failure')
        result = original(self, *args)
        if fault == 'deadline':
            timer['offset'] = 120.0
        return result
    monkeypatch.setattr(Backend, 'generate', generate)
    actions = [{'invalid':True}] if fault == 'schema' else [verdict()]
    row, backend, journal, frame, corpus = run(tmp_path, actions, 'fixed_top1', protocol=ISOLATED_PROTOCOL,
        clock=lambda:time.monotonic()+timer['offset'])
    assert row['state'] == 'unresolved' and row['prediction'] is None
    assert ledger_cost(journal.directory)['unique_physical_calls'] == row['physical_calls'] == 1
    if fault != 'unknown':
        assert audit(row, backend, journal, frame, corpus, tmp_path) == row
    else:
        assert row['reason'] == 'unknown_physical_cost'


def test_new_remaining_caps_are_real_not_restored_by_projection(tmp_path):
    row, backend, journal, *_ = run(tmp_path, [ABSTAIN], protocol=ISOLATED_PROTOCOL)
    assert row['physical_calls'] == 1
    with pytest.raises(ValueError, match='shared_physical_budget'):
        journal.generate({}, {}, 512, 121)
    with pytest.raises(ValueError, match='shared_physical_budget'):
        journal.generate({}, {}, 513, 10)


def test_versioned_release_pins_existing_data_and_independent_execution_source():
    fields = operator.release_fields({'protocol':ISOLATED_PROTOCOL, 'input_protocol':inputs_adapter.PROTOCOL})
    release = dict(fields, source_git='b'*40, source_archive_sha256='c'*64, wrapper_sha256='d'*64,
                   authorization='coordinator_exact_hash_release')
    operator.validate_release(release)
    assert release['preparation_source_git'] != release['source_git']
    assert operator.output_path(release) != inputs_adapter.OUTPUT
    for changed in ({'authorization':'DRAFT_CPU_READY_NOT_AUTHORIZED'},
                    {'protocol':'unknown'}, {'frames_sha256':'f'*64}, {'output':str(inputs_adapter.OUTPUT)}):
        with pytest.raises(ValueError):
            operator.validate_release(dict(release, **changed))


@pytest.mark.parametrize('protocol', [ISOLATED_PROTOCOL, RELATION_PROTOCOL])
def test_packager_keeps_preparation_identity_separate_without_rewriting_compact(tmp_path, monkeypatch, protocol):
    import package_scifact_evidence_commit as package
    receipt = {'status':'imports_verified', 'dependency_errors':[], 'stderr_empty':True,
        'runtime_files_sha256':operator.FROZEN_FIELDS['runtime_files_sha256'],
        **{k:'synthetic' for k in ('python','os_name','torch','versions','module_files')}}
    observed = {k:receipt[k] for k in ('python','os_name','torch','versions','module_files')}
    observed.update(python_executable=operator.FROZEN_FIELDS['python_executable'],
                    model_loaded=False, generation_calls=0)
    r, o = tmp_path/'runtime.json', tmp_path/'observed.json'
    r.write_text(json.dumps(receipt))
    o.write_text(json.dumps(observed))
    monkeypatch.setattr(package, 'sha256', lambda p:operator.FROZEN_FIELDS[
        'runtime_receipt_sha256' if p == r else 'runtime_observation_sha256'])
    compact = Path(__file__).resolve().parents[1]/'docs/verified-runs/scifact-prospective24-cpu-31953981.json'
    before = compact.read_bytes()
    release = package.build_release('b'*40, 'c'*64, 'd'*64, r, o, compact, protocol=protocol)
    assert release['authorization'] == 'DRAFT_CPU_READY_NOT_AUTHORIZED'
    assert release['source_git'] == 'b'*40 and release['preparation_source_git'].startswith('da2430')
    with pytest.raises(ValueError, match='compact_contract'):
        package.build_release('b'*40, 'c'*64, 'd'*64, r, o, compact)
    bad = json.loads(before)
    bad['frames_sha256'] = 'f'*64
    altered = tmp_path/'altered.json'
    altered.write_text(json.dumps(bad))
    with pytest.raises(ValueError, match='frozen_contract'):
        package.build_release('b'*40, 'c'*64, 'd'*64, r, o, altered, protocol=protocol)
    assert compact.read_bytes() == before


def test_diagnosis_separates_label_selection_coverage_and_submission(tmp_path):
    frame, corpus = inputs(2)
    rows = {}
    for arm in ('fixed_top1', 'fixed_all', 'adaptive'):
        out = tmp_path/arm
        out.mkdir()
        actions = [verdict(label='REFUTES')] if arm == 'fixed_top1' else (
            [verdict(), verdict('c8')] if arm == 'fixed_all' else [ABSTAIN])
        row, *_ = run(out, actions, arm)
        rows[arm] = row
    gold = GoldClaim(1, 'synthetic', {77:(Rationale('SUPPORT',(7,2)),)}, (77,))
    result = diagnose_claim(gold, frame, rows, ['77','78'])
    assert result['coverage']['not_in_initial_top20'] == 0
    assert result['arms']['fixed_top1']['counts']['visited_label_disagreement'] == 1
    assert result['arms']['fixed_all']['counts']['strict_correct_annotated_verdicts'] == 1
    assert result['arms']['fixed_all']['counts']['final_unannotated_documents'] == 1
    assert result['arms']['adaptive']['counts']['visible_reachable_gold_not_attempted'] == 1
    packed = copy.deepcopy(frame)
    packed['visible'] = {k:v for k,v in frame['visible'].items() if k != 'c7:7'}
    diagnosis = diagnose_claim(gold, packed, rows, ['77','78'])
    assert diagnosis['coverage']['visible_document_missing_complete_rationale'] == 1
    missing = diagnose_claim(gold, frame, rows, ['78'])
    assert missing['coverage']['not_in_initial_top20'] == 1
    # Correct verifier evidence can exist but be omitted by a controller.
    dropped = copy.deepcopy(rows)
    dropped['fixed_all']['prediction']['evidence'].pop('77')
    diagnosis = diagnose_claim(gold, frame, dropped, ['77','78'])
    assert diagnosis['arms']['fixed_all']['counts']['correct_verdict_not_preserved_in_final'] == 1


def test_diagnosis_full_rationale_not_equal_first3_or_strict_annotation(tmp_path):
    rows = {}
    for arm in ('fixed_top1', 'fixed_all', 'adaptive'):
        out = tmp_path/arm
        out.mkdir()
        actions = [verdict()] if arm == 'fixed_top1' else (
            [verdict(), verdict('c8')] if arm == 'fixed_all' else
            [{'action':'verify','source_id':'c7'}, verdict(), choose])
        rows[arm], *_ = run(out, actions, arm)
    frame, _ = inputs(2)
    gold = GoldClaim(1, 'synthetic', {77:(Rationale('SUPPORT',(2,)),)}, (77,))
    result = diagnose_claim(gold, frame, rows, ['77','78'])
    count = result['arms']['fixed_top1']['counts']
    assert count['visited_first3_rationale_missing'] == 0
    assert count['visited_selected_unannotated_sentences'] == 1
    assert count['strict_correct_annotated_verdicts'] == 0
    assert json.dumps(result).find('synthetic') == -1  # no raw claim in report
