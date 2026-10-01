"""Synthetic contract tests, not scientific labels or measured model gains."""
import copy
import json
import time

import jsonschema
import pytest

from climate_rag.scifact_evidence_commit import CommitState, ISOLATED_PROTOCOL, PROTOCOL, render_prompt
from climate_rag.scifact_relation_verifier import (
    RELATION_PROTOCOL, QUALIFIERS, assessment_schema, parse_assessment, project_assessment,
)
from climate_rag.scifact_utility_runtime import ledger_cost
from diagnose_scifact_semantic_input import invariant_probe
from preflight_scifact_evidence_commit import probe
from preflight_scifact_relation_verifier import probe_relation
import run_scifact_evidence_commit_operator as operator
import scifact_evidence_input as inputs_adapter
from test_bounded_scifact_runtime import ABSTAIN
from test_scifact_evidence_commit import Backend, audit, choose, inputs, run, verdict


def assessment(doc="c7", relation="SUPPORTS"):
    ids = [f"{doc}:7", f"{doc}:2"]
    positive = relation != "INSUFFICIENT"
    return {"source_id":doc, "relation":relation,
        "qualifiers":{key:"aligned" if positive else "not_established" for key in QUALIFIERS},
        "direct_sentence_ids":ids, "background_sentence_ids":[],
        "minimal_sentence_ids":ids if positive else [],
        "uncertainty":"none" if positive else "missing_direct_evidence"}


@pytest.mark.parametrize("relation", ["SUPPORTS", "REFUTES", "INSUFFICIENT"])
def test_projection_is_original_order_not_label_repair(relation):
    frame, _ = inputs()
    raw = assessment(relation=relation)
    jsonschema.validate(raw, assessment_schema(frame, "c7"))
    parsed = parse_assessment(raw, frame, "c7")
    assert project_assessment(parsed) == verdict(label=relation)
    parsed["direct_sentence_ids"].clear()
    assert raw["direct_sentence_ids"]  # no alias mutation


@pytest.mark.parametrize("change", [
    {"relation":"UNKNOWN"}, {"source_id":"c8"}, {"extra":"not allowed"},
    {"direct_sentence_ids":["c8:7"]}, {"minimal_sentence_ids":["c7:7","c7:7"]},
    {"background_sentence_ids":["c7:7"]}, {"direct_sentence_ids":["c7:7"]},
    {"minimal_sentence_ids":[]}, {"uncertainty":"relation_unclear"},
    {"qualifiers":{k:"mismatched" for k in QUALIFIERS}},
])
def test_invalid_cross_field_assessments_rejected_not_downgraded(change):
    frame, _ = inputs()
    raw = dict(assessment(), **change)
    with pytest.raises(ValueError):
        parse_assessment(raw, frame, "c7")


def test_background_not_committed_and_no_first_three_truncation():
    import hashlib
    frame, _ = inputs()
    for index in range(8):
        text = f"Synthetic original sentence {index}."
        frame["visible"][f"c7:{index}"] = dict(frame["visible"]["c7:2"], sentence_index=index,
            text=text, text_sha256=hashlib.sha256(text.encode()).hexdigest())
    raw = assessment()
    raw.update(direct_sentence_ids=["c7:7","c7:3","c7:2","c7:4"],
               minimal_sentence_ids=["c7:7","c7:3","c7:2","c7:4"], background_sentence_ids=["c7:0"])
    projected = project_assessment(parse_assessment(raw, frame, "c7"))
    assert projected["sentence_ids"] == raw["minimal_sentence_ids"] and len(projected["sentence_ids"]) == 4
    assert "c7:0" not in projected["sentence_ids"]


def test_comparable_numeric_contradiction_and_joint_context_are_allowed():
    # Contract fixture: not a model semantic-accuracy result. The context sentence
    # is a necessary link even though it does not independently refute the claim.
    import hashlib
    frame, _ = inputs()
    frame["observation"]["immutable_claim"] = "In treated adult mice, marker X increased by 20%."
    for sid, text in (("c7:7", "The experiment studied treated adult mice."),
                      ("c7:2", "In this group, marker X decreased by 10%.")):
        frame["visible"][sid].update(text=text, text_sha256=hashlib.sha256(text.encode()).hexdigest())
    frame["observation"]["current_citable"] = [
        {"sentence_id":sid, "text":row["text"]} for sid,row in frame["visible"].items()]
    raw = assessment(relation="REFUTES")
    jsonschema.validate(raw, assessment_schema(frame, "c7"))
    parsed = parse_assessment(raw, frame, "c7")
    assert project_assessment(parsed)["sentence_ids"] == ["c7:7","c7:2"]
    state = CommitState(1,"fixed_top1",frame,"synthetic",protocol=RELATION_PROTOCOL)
    obs, schema = state.inputs("verify", "c7", [])
    prompt = render_prompt(Backend([]).base.tokenizer, obs, schema)
    assert "NOT that the observed result agrees" in prompt and "necessary antecedent" in prompt
    wrong_context = dict(raw, qualifiers=dict(raw["qualifiers"], entity_population="mismatched"))
    with pytest.raises(ValueError, match="positive_scope"):
        parse_assessment(wrong_context, frame, "c7")
    unresolved = dict(wrong_context, relation="INSUFFICIENT", minimal_sentence_ids=[], uncertainty="scope_mismatch")
    assert parse_assessment(unresolved, frame, "c7")["direct_sentence_ids"] == ["c7:7","c7:2"]


@pytest.mark.parametrize("arm", ["fixed_top1", "fixed_all", "adaptive"])
def test_runtime_projection_feedback_and_raw_audit(arm, tmp_path):
    actions = [assessment()] if arm == "fixed_top1" else (
        [assessment(), assessment("c8")] if arm == "fixed_all" else
        [{"action":"verify", "source_id":"c7"}, assessment(), choose])
    row, backend, journal, frame, corpus = run(tmp_path, actions, arm, protocol=RELATION_PROTOCOL)
    assert row["state"] == "valid_terminal"
    assert row["prediction"]["evidence"]["77"]["sentences"] == [7,2]
    assert row["verification_feedback"][0]["judgment"] == verdict()
    assert row["verification_feedback"][0]["assessment"] == dict(protocol=RELATION_PROTOCOL, **assessment())
    assert ledger_cost(journal.directory)["unique_physical_calls"] == len(actions)
    assert audit(row, backend, journal, frame, corpus, tmp_path) == row
    if arm == "adaptive":
        assert row["steps"][-1]["observation"]["verification_feedback"][0]["assessment"]
        assert row["steps"][-1]["observation"]["remaining"]["physical_generations"] == 3
    for field in ("judgment", "assessment"):
        changed = copy.deepcopy(row)
        changed["verification_feedback"][0][field]["source_id"] = "c8"
        with pytest.raises(ValueError, match="accounting_changed"):
            audit(changed, backend, journal, frame, corpus, tmp_path)


@pytest.mark.parametrize("continuation", ["abstain", "verify"])
def test_partial_uncertainty_is_feedback_not_global_nei(tmp_path, continuation):
    tail = [ABSTAIN] if continuation == "abstain" else [
        {"action":"verify", "source_id":"c8"}, assessment("c8"), choose]
    row, backend, journal, frame, corpus = run(tmp_path, [
        {"action":"verify", "source_id":"c7"}, assessment(relation="INSUFFICIENT"), *tail],
        protocol=RELATION_PROTOCOL)
    observation = row["steps"][2]["observation"]
    assert not observation["verdict_refs"]
    assert observation["verification_feedback"][0]["judgment"]["sentence_ids"] == []
    assert observation["verification_feedback"][0]["assessment"]["direct_sentence_ids"]
    assert row["physical_calls"] == (3 if continuation == "abstain" else 5)
    assert audit(row, backend, journal, frame, corpus, tmp_path) == row


@pytest.mark.parametrize("fault", ["unknown", "invalid", "deadline"])
def test_one_physical_failure_charged_without_label_repair(tmp_path, monkeypatch, fault):
    timer = {"offset":0.0}
    original = Backend.generate
    def generate(self, *args):
        if fault == "unknown":
            raise RuntimeError("synthetic backend failure")
        response = original(self, *args)
        if fault == "deadline":
            timer["offset"] = 120.0
        return response
    monkeypatch.setattr(Backend, "generate", generate)
    raw = dict(assessment(), uncertainty="scope_mismatch") if fault == "invalid" else assessment()
    row, backend, journal, frame, corpus = run(tmp_path, [raw], "fixed_top1", protocol=RELATION_PROTOCOL,
        clock=lambda:time.monotonic()+timer["offset"])
    assert row["state"] == "unresolved" and not row["verdict_refs"]
    assert row["physical_calls"] == ledger_cost(journal.directory)["unique_physical_calls"] == 1
    assert row["verification_feedback"][0]["status"] == "failed"
    assert row["verification_feedback"][0]["judgment"] is None
    if fault != "unknown":
        assert audit(row, backend, journal, frame, corpus, tmp_path) == row


def test_three_document_limit_is_not_repaired_or_hidden(tmp_path):
    row, backend, journal, frame, corpus = run(tmp_path, [
        {"action":"verify", "source_id":"c7"}, assessment(),
        {"action":"verify", "source_id":"c8"}, assessment("c8"), choose],
        n=3, protocol=RELATION_PROTOCOL)
    assert row["physical_calls"] == 5 and len(row["verification_feedback"]) == 2
    assert "c9" in frame["document_order"]
    assert not row["steps"][-1]["observation"]["verifiable_documents"]
    assert len(row["prediction"]["evidence"]) == 2
    assert audit(row, backend, journal, frame, corpus, tmp_path) == row


def test_isolation_old_versions_and_preflight_stress():
    frame, _ = inputs()
    tokenizer = Backend([]).base.tokenizer
    assert invariant_probe(frame, tokenizer, protocol=RELATION_PROTOCOL)["all_equal"]
    assert not probe(frame, tokenizer, protocol=RELATION_PROTOCOL)["overflow"]
    for old in (PROTOCOL, ISOLATED_PROTOCOL):
        state = CommitState(1,"fixed_top1",frame,"old",protocol=old)
        assert state.parse(verdict(), "c7", []) == verdict()
        with pytest.raises(ValueError):
            state.parse(assessment(), "c7", [])
    state = CommitState(1,"fixed_top1",frame,"new",protocol=RELATION_PROTOCOL)
    with pytest.raises(ValueError):
        state.parse(verdict(), "c7", [])
    obs, schema = state.inputs("verify", "c7", [])
    assert "remaining" not in obs
    assert "direct_sentence_ids" in render_prompt(tokenizer, obs, schema)


def test_pairwise_preflight_reports_no_quality_and_preserves_two_protocols():
    frame,_ = inputs()
    result = probe_relation(frame, Backend([]).base.tokenizer)
    assert result['invariance']['all_equal']
    assert result['prompt_probe_counts'][RELATION_PROTOCOL] == 39
    assert set(result['maximum_prompt_tokens']) == {ISOLATED_PROTOCOL,RELATION_PROTOCOL}
    assert result['maximum_synthetic_response_tokens_with_eos'] > 0


def test_versioned_release_keeps_v2_baseline_and_unique_path():
    versions = [operator.release_fields({"protocol":p, "input_protocol":inputs_adapter.PROTOCOL})
                for p in (ISOLATED_PROTOCOL, RELATION_PROTOCOL)]
    assert versions[0]["output"] != versions[1]["output"]
    assert versions[0]["attempt_id"] != versions[1]["attempt_id"]
    assert versions[1]["comparison_protocol"] == ISOLATED_PROTOCOL
    for fields in versions:
        release = dict(fields, source_git="b"*40, source_archive_sha256="c"*64,
                       wrapper_sha256="d"*64, authorization="coordinator_exact_hash_release")
        operator.validate_release(release)
        assert fields["max_episode_generations"] == 5 and fields["max_generator_calls"] == 240
        assert operator.output_path(release).as_posix() == fields["output"]
        assert fields["frames_sha256"] == versions[0]["frames_sha256"]


@pytest.mark.parametrize("relation", ["SUPPORTS", "REFUTES", "INSUFFICIENT"])
def test_actual_lmfe_and_inherited_generate_single_response(tmp_path, relation):
    from climate_rag.scifact_evidence_commit import EvidenceCommitProvider
    from test_scifact_document_decoder import real_provider
    provider, calls, callbacks = real_provider(tmp_path, [assessment(relation=relation)])
    provider.__class__ = EvidenceCommitProvider
    frame, _ = inputs()
    state = CommitState(1,"fixed_top1",frame,"synthetic",protocol=RELATION_PROTOCOL)
    obs, schema = state.inputs("verify", "c7", [])
    response = provider.generate(obs, schema, 512, 120)
    assert state.parse(json.loads(response["raw"]), "c7", []) == assessment(relation=relation)
    assert len(calls) == len(callbacks) == 1
    assert response["usage"]["input_tokens"] == provider.count_prompt(obs,schema)
    assert response["diagnostics"]["eos_observed"]
