"""Synthetic posthoc attribution only; no real TRAIN data or model calls."""
import hashlib
import json

import pytest

import diagnose_scifact_natural_outcomes as audit
from climate_rag.local_bounded_scifact_provider import observation_identity
from climate_rag.scifact_grounding import Abstract, GoldClaim, Rationale
from climate_rag.scifact_terminal import source_from_abstract
from climate_rag.scifact_utility_runtime import run_matrix
from test_scifact_natural_runtime import ABSTAIN, Backend


def corpus():
    return {i: Abstract(i, "Synthetic", tuple(f"Sentence {i}/{j}." for j in range(6)), False)
            for i in (10, 20)}


def gold(evidence=None):
    if evidence is None:
        evidence = {10: (Rationale("SUPPORT", (0, 1)), Rationale("SUPPORT", (4,))),
                    20: (Rationale("SUPPORT", (2,)),)}
    return GoldClaim(1, "Synthetic claim", evidence, tuple(evidence))


def prediction(evidence=None):
    return {"id": 1, "evidence": evidence or {}}


def initial(view, action="abstain", pred=None):
    return {"visible": view, "decision": {"action": action},
            "prediction": prediction() if pred is None else pred}


def test_or_alternatives_and_partial_multidocument_coverage():
    g = gold()
    partial = audit.coverage_detail(g, {10: [4]})
    assert partial["document_opportunities"] == [10]
    assert partial["gold_documents"] == 2 and not partial["coverage"]["complete"]
    alternate = audit.coverage_detail(g, {10: [4], 20: [2]})
    assert alternate["coverage"] == {"complete": True, "first3_reachable": True, "all_gold_sentences": False}
    first = audit.coverage_detail(g, {10: [0, 1], 20: [2]})
    assert first["coverage"]["complete"]


def test_candidate_pool_and_visible_context_are_distinct():
    g, docs = gold(), corpus()
    full = {10: [4], 20: [2]}
    pool_miss = audit.initial_diagnosis(g, {10: [4]}, initial({10: [4]}), docs)
    context_miss = audit.initial_diagnosis(g, full, initial({10: [4]}), docs)
    assert pool_miss["category"] == "candidate_pool_not_complete_under_frozen_contract"
    assert context_miss["category"] == "candidate_complete_but_initial_context_incomplete"
    assert context_miss["unexecuted_read_reachability"] == "unknown"
    assert context_miss["whole_corpus_opportunity"] == "not_measured"


def test_initial_tool_choice_is_not_classified_by_eventual_wrong_endpoint():
    view = {10: [4], 20: [2]}
    report = audit.initial_diagnosis(gold(), view, initial(view, "read", prediction()), corpus())
    assert report["category"] == "initial_sufficient_tool_selected"
    report = audit.initial_diagnosis(gold(), view, initial(view), corpus())
    assert report["category"] == "initial_sufficient_legal_abstention"


def test_complete_four_sentence_rationale_is_not_model_grounding_failure():
    g = gold({10: (Rationale("SUPPORT", (0, 1, 2, 3)),)})
    view = {10: [0, 1, 2, 3]}
    pred = prediction({"10": {"label": "SUPPORT", "sentences": [0, 1, 2, 3]}})
    report = audit.initial_diagnosis(g, view, initial(view, "answer", pred), corpus())
    assert report["initial_visible"]["coverage"]["complete"]
    assert not report["initial_visible"]["coverage"]["first3_reachable"]
    assert report["category"] == "initial_semantically_complete_but_strict_first3_unreachable"


def wire_frame():
    docs = corpus()
    aliases = {"c0": "20", "c1": "10"}  # Aliases differ from numeric order.
    visible = {}
    source_hashes = {}
    for alias, doc_id in aliases.items():
        source = source_from_abstract(docs[int(doc_id)])
        source_hashes[alias] = source.text_sha256
        for number in (4, 1):  # Non-monotonic visible order must not renumber sentences.
            text = source.sentences[number]
            visible[f"{alias}:{number}"] = {"source_id": doc_id, "sentence_index": number,
                "text": text, "text_sha256": hashlib.sha256(text.encode()).hexdigest(),
                "source_text_sha256": source.text_sha256}
    observation = {"current_citable": [{"sentence_id": k, "text": v["text"]} for k, v in visible.items()],
                   "preview_only": [{"source_id": "c9", "preview": "Not evidence", "citable": False}]}
    frame = {"alias_to_source": aliases, "visible": visible, "observation": observation, "prompt_tokens": 100}
    attempt = {"visible_sentence_sha256": {k: v["text_sha256"] for k, v in visible.items()},
               "diagnostics": {"actual_observation": observation_identity(observation)}, "input_prompt_tokens": 100}
    return frame, attempt, docs, source_hashes


def test_original_ids_survive_alias_and_sentence_reordering():
    frame, attempt, docs, hashes = wire_frame()
    assert audit.frame_visible(frame, attempt, docs, hashes) == {20: [4, 1], 10: [4, 1]}


@pytest.mark.parametrize("fault", ["alias", "text", "source", "order", "index", "prompt"])
def test_actual_frame_integrity_is_required(fault):
    frame, attempt, docs, hashes = wire_frame()
    if fault == "alias":
        frame["alias_to_source"] = {"c0": "10", "c1": "20"}
    elif fault == "text":
        frame["visible"]["c0:4"]["text"] = "Wrong sentence"
    elif fault == "source":
        hashes["c0"] = "0" * 64
    elif fault == "order":
        frame["observation"]["current_citable"].reverse()
    elif fault == "index":
        frame["visible"]["c0:4"]["sentence_index"] = 0
    else:
        attempt["input_prompt_tokens"] = 101
    with pytest.raises(ValueError):
        audit.frame_visible(frame, attempt, docs, hashes)


def linked_tool(before, after, *, scripted=False, action="abstain"):
    event = {"tool": "read", "status": "completed", "model_selected": not scripted,
             "before_context": ["c0"], "requested_context": ["c1"],
             "candidate_ids": ["c0", "c1"], "source_sha256": {"c0": "a", "c1": "b"}}
    if scripted:
        event["origin"] = "scripted_intervention"
    frames = [dict(initial(before, "read"), frame_sha256="before", observation_sha256="obs-before",
                   feedback=None),
              dict(initial(after, action), frame_sha256="after", observation_sha256="obs-after",
                   feedback="tool_completed:read")]
    frames[0]["decision"]["source_ids"] = ["c1"]
    frames[0]["source_state"] = dict(event, requested_context=["c0"])
    frames[1]["source_state"] = dict(event)
    frames[1]["executed_proposal"] = {"action": "read", "source_ids": ["c1"]}
    row = {"audit_status": "valid_terminal", "prediction": prediction(),
           "result": {"events": [event], "decision_execution_audit": [{"actual_event_index": 0,
               "attempt_index": 0, "next_attempt_index": 1, "strict_status": "valid_decision",
               "strict_action": "read", "actual_event_status": "completed", "next_feedback": "tool_completed:read"}]}}
    changes = [{"origin": "scripted" if scripted else "model_selected", "tool": "read", "status": "completed",
                "before_requested_context": ["c0"], "after_requested_context": ["c1"],
                "next_frame_change": {"post_tool_observation_sha256": "obs-after"}}]
    return row, frames, changes


def test_actual_tool_creates_complete_coverage_but_answer_stays_wrong():
    row, frames, changes = linked_tool({10: [4]}, {10: [4], 20: [2]})
    result, = audit.aligned_tools(gold(), "A", row, frames, changes, corpus())
    assert result["transition_insufficient_to_complete"]
    assert result["first3_transition"] and result["transition_outcome"] == "wrong"
    assert result["origin"] == "model_selected" and result["causal_effect"] == "not_identified"
    assert result["next_decision"]["action"] == "abstain"


def test_tool_complete_but_scoring_unreachable_does_not_imply_bad_discrimination():
    g = gold({10: (Rationale("SUPPORT", (0, 1, 2, 3)),)})
    row, frames, changes = linked_tool({}, {10: [0, 1, 2, 3]})
    result, = audit.aligned_tools(g, "A", row, frames, changes, corpus())
    assert result["transition_insufficient_to_complete"]
    assert not result["first3_transition"]
    assert result["transition_outcome"] == "strict_first3_unreachable"


def test_tool_loses_needed_doc_and_partial_doc_hit_does_not_complete_claim():
    row, frames, changes = linked_tool({10: [4], 20: [2]}, {10: [4]})
    result, = audit.aligned_tools(gold(), "A", row, frames, changes, corpus())
    assert result["before_coverage"]["coverage"]["complete"]
    assert not result["after_coverage"]["coverage"]["complete"]
    assert result["after_coverage"]["document_opportunities"] == [10]
    assert not result["transition_insufficient_to_complete"]
    assert result["transition_outcome"] == "not_applicable"


@pytest.mark.parametrize("fault", ["failed_tool", "missing_frame", "read_args", "link_action", "origin", "negative_index",
                                   "before_state", "candidate_change", "source_change"])
def test_unbound_or_unexecuted_change_never_gets_transition_credit(fault):
    row, frames, changes = linked_tool({10: [4]}, {10: [4], 20: [2]})
    if fault == "failed_tool":
        row["result"]["events"][0]["status"] = "failed"
    elif fault == "missing_frame":
        frames[1] = None
    elif fault == "read_args":
        frames[0]["decision"]["source_ids"] = ["c99"]
    elif fault == "link_action":
        row["result"]["decision_execution_audit"][0]["strict_action"] = "rerank"
    elif fault == "origin":
        row["result"]["events"][0]["origin"] = "scripted_intervention"
    elif fault == "negative_index":
        row["result"]["decision_execution_audit"][0]["attempt_index"] = -1
    elif fault == "before_state":
        frames[0]["source_state"]["requested_context"] = ["c99"]
    elif fault == "candidate_change":
        frames[0]["source_state"]["candidate_ids"] = ["c99"]
    else:
        frames[0]["source_state"]["source_sha256"] = {"c0": "modified"}
    result, = audit.aligned_tools(gold(), "A", row, frames, changes, corpus())
    assert result["attribution_status"] == "unknown"
    assert result["transition_insufficient_to_complete"] is None
    assert result["transition_outcome"] == "unknown"


def test_nei_failure_and_scripted_origin_stay_separate():
    g = gold({})
    row, frames, changes = linked_tool({}, {}, scripted=True)
    row.update(audit_status="unresolved", prediction=prediction())
    assert audit.endpoint(g, row, corpus()) == "unresolved"
    assert audit.initial_diagnosis(g, {}, initial({}), corpus())["category"] == "official_NEI_separate"
    result, = audit.aligned_tools(g, "B", row, frames, changes, corpus())
    assert result["origin"] == "scripted" and result["attribution_status"] == "NEI_not_applicable"
    assert result["transition_insufficient_to_complete"] is None


def synthetic_run(tmp_path, actions):
    class ObservedBackend(Backend):
        def generate(self, observation, schema, max_output_tokens, remaining_seconds):
            response = super().generate(observation, schema, max_output_tokens, remaining_seconds)
            response["diagnostics"]["actual_observation"] = observation_identity(observation)
            return response

    docs = {i: Abstract(i, "Synthetic", ("First fixture sentence.", "Second."), False) for i in range(100, 120)}
    sources = [source_from_abstract(a) for a in docs.values()]
    backend = ObservedBackend(actions)
    run_matrix([{"id": 1, "claim": "Fixture claim"}], backend, lambda q, k: sources[:k],
               lambda q, c: list(reversed(c)), docs, tmp_path / "inference", natural_fit=True)
    return docs, backend.base.tokenizer


def test_real_helper_integration_and_shared_branch_index(tmp_path):
    docs, tokenizer = synthetic_run(tmp_path, [{"action": "read", "source_ids": ["c8"]}] + [ABSTAIN] * 3)
    prefix = json.loads((tmp_path / "inference/1-A/prefix.json").read_bytes())
    assert len(audit.candidate_fulltext(prefix, docs)) == 20
    g = gold({108: (Rationale("SUPPORT", (0,)),)})
    a, frames, changes = audit.observed_slot(tmp_path, 1, "A", docs, tokenizer)
    assert a["audit_status"] == "valid_terminal" and len(frames) == 2
    item, = audit.aligned_tools(g, "A", a, frames, changes, docs)
    assert item["transition_insufficient_to_complete"] and item["transition_outcome"] == "wrong"
    b, frames, changes = audit.observed_slot(tmp_path, 1, "B", docs, tokenizer)
    assert b["audit_status"] == "valid_terminal"
    assert frames[0]["physical_attempt_id"] == "g00"
    assert 105 in frames[1]["visible"] and 108 not in frames[1]["visible"]
    item, = audit.aligned_tools(g, "B", b, frames, changes, docs)
    assert item["origin"] == "scripted" and not item["transition_insufficient_to_complete"]


def test_missing_initial_frame_keeps_denominator_and_does_not_shift_later_frames(tmp_path):
    docs, tokenizer = synthetic_run(tmp_path, [{"action": "read", "source_ids": ["c8"]}] + [ABSTAIN] * 3)
    (tmp_path / "inference/1-A/frame-0.json").unlink()
    row, frames, changes = audit.observed_slot(tmp_path, 1, "A", docs, tokenizer)
    assert row["audit_status"] == "unresolved" and row["prediction"] is None
    assert len(frames) == 2 and frames[0] is None and frames[1] is not None
    assert changes == []
    missing, _, _ = audit.observed_slot(tmp_path, 999, "C", docs, tokenizer)
    assert audit.endpoint(gold({}), missing, docs) == "unresolved"


def test_unknown_cost_cannot_be_nei_success(tmp_path):
    docs, tokenizer = synthetic_run(tmp_path, [ABSTAIN] * 3)
    path = tmp_path / "inference/ledger/g00.finished.json"
    value = json.loads(path.read_bytes())
    value["usage_known"] = False
    path.write_text(json.dumps(value))
    for arm in audit.ARMS:
        row, _, _ = audit.observed_slot(tmp_path, 1, arm, docs, tokenizer)
        assert audit.endpoint(gold({}), row, docs) == "unresolved"


def test_undeclared_physical_call_cannot_hide_unknown_cost(tmp_path):
    docs, tokenizer = synthetic_run(tmp_path, [ABSTAIN] * 3)
    # A real reserved call omitted from the row's declared IDs is not free.
    (tmp_path / "inference/ledger/g99.reserved.json").write_text(json.dumps({"slot": "1-A"}))
    row, _, _ = audit.observed_slot(tmp_path, 1, "A", docs, tokenizer)
    assert audit.endpoint(gold({}), row, docs) == "unresolved"


@pytest.mark.parametrize("arm", ["A", "B", "C"])
def test_duplicate_logical_reference_is_rejected(tmp_path, arm):
    docs, tokenizer = synthetic_run(tmp_path, [ABSTAIN] * 3)
    path = tmp_path / f"inference/1-{arm}/result.json"
    row = json.loads(path.read_bytes())
    row["logical_generation_ids"].append(row["logical_generation_ids"][0])
    path.write_text(json.dumps(row))
    row, _, _ = audit.observed_slot(tmp_path, 1, arm, docs, tokenizer)
    assert row["audit_status"] == "unresolved"


@pytest.mark.parametrize("state", ["", "PENDING", "RUNNING", "COMPLETING"])
def test_cli_readiness_rejects_nonterminal_before_reading_files(tmp_path, state):
    with pytest.raises(ValueError, match="job_not_terminal"):
        audit.require_ready(tmp_path, state)


def test_cli_readiness_requires_reap_and_persisted_cost(tmp_path):
    path = tmp_path / "worker-exit.json"
    proof = {"child_reaped": False, "release_sha256": audit.EXECUTION_RELEASE}
    path.write_text(json.dumps(proof))
    cost = tmp_path / "cost-before-gold.json"
    cost.write_text(json.dumps({"exit_proof_sha256": audit.sha(path)}))
    with pytest.raises(ValueError, match="child_not_reaped"):
        audit.require_ready(tmp_path, "COMPLETED")
    proof["child_reaped"] = True
    path.write_text(json.dumps(proof))
    with pytest.raises(ValueError, match="persisted_cost_exit_binding"):
        audit.require_ready(tmp_path, "FAILED")
    cost.write_text(json.dumps({"exit_proof_sha256": audit.sha(path)}))
    assert audit.require_ready(tmp_path, "COMPLETED")["persisted_cost_sha256"] == audit.sha(cost)
    cost.unlink()
    with pytest.raises(FileNotFoundError):
        audit.require_ready(tmp_path, "COMPLETED")
