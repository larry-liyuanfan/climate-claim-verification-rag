"""Synthetic source/selection/runtime tests, not measured Agent performance."""
import copy
from dataclasses import FrozenInstanceError, replace
import json

import pytest

from climate_rag.scifact_evidence_commit import (
    ARMS, CALL_CAPS, CommitState, EvidenceCommitProvider, PROTOCOL, VerdictRegistry, render_prompt,
)
from climate_rag.scifact_evidence_commit_runtime import CommitJournal, audit_episode, run_episode
from climate_rag.scifact_natural_contract import base_state
from climate_rag.scifact_utility_runtime import ledger_cost
from test_bounded_scifact_runtime import ABSTAIN
from test_scifact_document_decoder import real_provider
from test_scifact_document_verifier import Backend as OldBackend, inputs, verdict

RUN = "a" * 64


def choose(obs, index=-1):
    return {"action": "commit", "selection_id": list(obs["commit_selections"])[index]}


class Backend(OldBackend):
    def __init__(self, actions, faults=None):
        super().__init__([], faults)
        self.script = iter(actions)

    def render(self, obs, schema):
        return render_prompt(self.base.tokenizer, obs, schema)

    def generate(self, obs, schema, maximum, seconds):
        action = next(self.script)
        self.actions = iter([action(obs) if callable(action) else action])
        return super().generate(obs, schema, maximum, seconds)


def run(tmp_path, actions, arm="adaptive", n=2, faults=None, clock=None, protocol=PROTOCOL):
    frame, corpus = inputs(n)
    backend = Backend(actions, faults)
    journal = CommitJournal(backend, tmp_path/"ledger", max_generations=240,
                            protocol=protocol, physical_guard=base_state, run_identity=RUN)
    row = run_episode(1, arm, frame, journal, corpus, tmp_path/"episode", run_identity=RUN,
                      **({"clock": clock} if clock else {}))
    return row, backend, journal, frame, corpus


def audit(row, backend, journal, frame, corpus, tmp_path):
    # Real scoring consumes JSON from disk, not Python tuples/objects in memory.
    loaded = json.loads(json.dumps(row))
    return audit_episode(loaded, frame, journal.directory, tmp_path/"episode/private-responses",
                         backend.base.tokenizer, corpus, run_identity=RUN, protocol=journal.protocol)


@pytest.mark.parametrize("arm,calls", [("fixed_top1", 1), ("fixed_all", 4)])
def test_fixed_same_order_all_positives_no_final_model_call(tmp_path, arm, calls):
    row, backend, journal, frame, corpus = run(tmp_path,
        [verdict(f"c{7+i}") for i in range(calls)], arm, 5)
    assert row["physical_calls"] == backend.calls == calls
    assert row["tool_calls"] == calls+1 and row["state"] == "valid_terminal"
    assert [s["source_id"] for s in row["steps"]] == frame["document_order"][:calls]
    assert row["terminal"]["origin"] == arm+"_positive_refs"
    assert row["prediction"]["evidence"]["77"]["sentences"] == [7, 2]
    assert len(row["prediction"]["evidence"]) == calls
    assert audit(row, backend, journal, frame, corpus, tmp_path) == row
    with pytest.raises(ValueError, match="per_arm"):
        journal.generate({}, {}, 512, 10)


def test_adaptive_verify_commit_subsets_with_known_physical_provenance(tmp_path):
    row, backend, journal, frame, corpus = run(tmp_path, [
        {"action":"verify", "source_id":"c8"}, verdict("c8"),
        {"action":"verify", "source_id":"c7"}, verdict(), lambda obs: choose(obs, 0)])
    assert row["physical_calls"] == 5 and row["tool_calls"] == 3
    assert row["state"] == "valid_terminal"
    assert set(row["prediction"]["evidence"]) == {"78"}  # Model chooses first issued subset, not sorted by doc.
    assert len(row["steps"][-1]["observation"]["commit_selections"]) == 3
    assert row["steps"][-1]["observation"]["verifiable_documents"] == []
    assert row["steps"][2]["observation"]["commit_selections"].items() <= row["steps"][-1]["observation"]["commit_selections"].items()
    assert [r["physical_attempt_id"] for r in row["verdict_refs"]] == ["g01", "g03"]
    assert all(r["episode_id"].endswith(":1:adaptive") for r in row["verdict_refs"])
    assert audit(row, backend, journal, frame, corpus, tmp_path) == row


@pytest.mark.parametrize("negative", [verdict(label="INSUFFICIENT"), {"bad": True}])
@pytest.mark.parametrize("continuation", ["abstain", "verify"])
def test_negative_feedback_does_not_issue_ref_or_force_nei(tmp_path, negative, continuation):
    tail = [ABSTAIN] if continuation == "abstain" else [
        {"action":"verify", "source_id":"c8"}, verdict("c8"), choose]
    row, backend, journal, frame, corpus = run(tmp_path, [
        {"action":"verify", "source_id":"c7"}, negative, *tail])
    assert not row["steps"][2]["observation"]["verdict_refs"]
    assert row["steps"][2]["observation"]["verification_feedback"]
    assert row["physical_calls"] == (3 if continuation == "abstain" else 5)
    assert audit(row, backend, journal, frame, corpus, tmp_path) == row


def test_model_may_abstain_without_tool(tmp_path):
    row, backend, journal, frame, corpus = run(tmp_path, [ABSTAIN])
    assert row["physical_calls"] == 1 and row["tool_calls"] == 1 and row["verification_feedback"] == []
    assert row["prediction"]["evidence"] == {} and row["terminal"] == ABSTAIN
    assert audit(row, backend, journal, frame, corpus, tmp_path) == row


def test_fixed_empty_is_unresolved_not_correct_nei(tmp_path):
    row, backend, journal, frame, corpus = run(tmp_path, [verdict(label="INSUFFICIENT")], "fixed_top1")
    assert row["state"] == "unresolved" and row["prediction"] is None
    assert row["terminal"] is None and row["physical_calls"] == 1
    assert audit(row, backend, journal, frame, corpus, tmp_path) == row


def registry(frame=None, scope="episode-one"):
    frame = frame or inputs()[0]
    reg = VerdictRegistry(1, scope, frame)
    ref = reg.register("c7", verdict(), "g00", "b"*64)
    assert ref is not None
    return reg, ref, frame


def test_immutable_ref_cross_episode_frame_and_duplicate_rejected():
    reg, ref, frame = registry()
    with pytest.raises(FrozenInstanceError):
        ref.label = "REFUTES"
    other, _, _ = registry(scope="episode-two")
    with pytest.raises(ValueError, match="cross_episode"):
        other.assemble([ref.ref_id], frame)
    for ids in ([], [ref.ref_id, ref.ref_id], ["unknown"]):
        with pytest.raises(ValueError):
            reg.assemble(ids, frame)
    changed = copy.deepcopy(frame)
    changed["observation"]["immutable_claim"] = "Different claim"
    with pytest.raises(ValueError, match="stale"):
        reg.assemble([ref.ref_id], changed)
    reg._refs[ref.ref_id] = replace(ref, label="REFUTES")
    with pytest.raises(ValueError, match="tampered"):
        reg.assemble([ref.ref_id], frame)


def test_original_total_sentence_limit_never_truncates_fixed():
    frame, _ = inputs(3)
    # Expand only synthetic visible sentences; every identity/hash matches.
    import hashlib
    for alias, original in frame["alias_to_source"].items():
        for index in range(8):
            text = f"Original sentence {index}."
            frame["visible"][f"{alias}:{index}"] = dict(frame["visible"][f"{alias}:2"],
                sentence_index=index, text=text, text_sha256=hashlib.sha256(text.encode()).hexdigest())
    reg = VerdictRegistry(1, "scope", frame)
    for i, alias in enumerate(frame["document_order"]):
        reg.register(alias, {"source_id":alias, "label":"SUPPORTS",
            "sentence_ids":[f"{alias}:{j}" for j in range(8)]}, f"g{i}", "b"*64)
    refs = [r["ref_id"] for r in reg.records()]
    with pytest.raises(ValueError, match="total_sentence_budget"):
        reg.assemble(refs, frame)
    assert len(reg.records()) == 3 and len(reg.catalog()) == 6  # Three singletons + three pairs only.


@pytest.mark.parametrize("action", [
    {"action":"commit", "selection_id":"stale"},
    {"action":"commit", "selection_id":"stale", "label":"SUPPORTS"},
    {"action":"answer", "documents":[verdict()]},
    {"action":"verify", "source_id":"absent"},
])
def test_unissued_or_mutated_controller_fields_fail_without_retry(tmp_path, action):
    row, _, journal, _, _ = run(tmp_path, [action])
    assert row["prediction"] is None and row["physical_calls"] == 1
    assert ledger_cost(journal.directory)["unique_physical_calls"] == 1


@pytest.mark.parametrize("fault", ["refs", "catalog", "label", "wire", "run"])
def test_rebuild_from_wire_rejects_self_consistent_saved_claims(tmp_path, fault):
    row, backend, journal, frame, corpus = run(tmp_path, [
        {"action":"verify", "source_id":"c7"}, verdict(), choose])
    if fault == "refs":
        row["verdict_refs"][0]["label"] = "REFUTES"
    elif fault == "catalog":
        row["steps"][-1]["observation"]["commit_selections"] = {"invented": []}
    elif fault == "label":
        row["assembled"]["documents"][0]["label"] = "REFUTES"
    elif fault == "run":
        row["episode_id"] = row["episode_id"].replace(RUN, "c"*64)
    else:
        record = json.loads((journal.directory/"g01.finished.json").read_bytes())
        record["response"]["raw"] = json.dumps(verdict(label="REFUTES"))
        (journal.directory/"g01.finished.json").write_text(json.dumps(record))
    with pytest.raises(ValueError):
        audit(row, backend, journal, frame, corpus, tmp_path)


def test_unknown_cost_stops_no_assembly(tmp_path, monkeypatch):
    monkeypatch.setattr(Backend, "generate", lambda *a: (_ for _ in ()).throw(RuntimeError("synthetic")))
    row, _, journal, _, _ = run(tmp_path, [], "fixed_top1")
    assert row["reason"] == "unknown_physical_cost" and row["prediction"] is None
    assert row["physical_calls"] == ledger_cost(journal.directory)["unknown_usage_attempts"] == 1


@pytest.mark.parametrize("protocol", [PROTOCOL, "scifact-evidence-commit-semantic-v2-20261002"])
def test_actual_generate_lmfe_verify_ref_commit_raw_rendering_and_audit(tmp_path, protocol):
    # Prescribed synthetic tokens traverse actual inherited generate/LMFE callback.
    frame, corpus = inputs()
    state = CommitState(1, "adaptive", frame, f"{protocol}:{RUN}:1:adaptive", protocol=protocol)
    # The helper generator consumes this iterator at model invocation. The last
    # selection depends on the actual preceding physical request/ref identity.
    def dynamic_actions():
        yield {"action":"verify", "source_id":"c7"}
        yield verdict()
        obs, _ = state.inputs("plan", None, [])
        yield choose(obs)
    # Build a second real provider whose final action reads the issued state.
    provider, calls, callbacks = real_provider(tmp_path, dynamic_actions())
    provider.__class__ = EvidenceCommitProvider
    journal = CommitJournal(provider, tmp_path/"ledger", max_generations=240, protocol=protocol,
                            physical_guard=base_state, run_identity=RUN)
    import climate_rag.scifact_evidence_commit_runtime as runtime
    from unittest.mock import patch
    with patch.object(runtime, "CommitState", return_value=state):
        row = run_episode(1, "adaptive", frame, journal, corpus, tmp_path/"episode", run_identity=RUN)
    assert row["state"] == "valid_terminal" and row["physical_calls"] == 3
    assert len(calls) == len({id(c) for c in callbacks}) == 3
    assert row["assembled"]["documents"] == [verdict()]
    assert row["prediction"]["evidence"]["77"]["sentences"] == [7, 2]
    assert audit(row, provider, journal, frame, corpus, tmp_path) == row


def test_protocol_three_arms_still_240_total_ceiling():
    assert ARMS == ("fixed_top1", "fixed_all", "adaptive")
    assert 24 * sum(CALL_CAPS.values()) == 240


def test_fixed_physical_receipts_cannot_be_rescoped_even_with_rebuilt_refs(tmp_path):
    from climate_rag.scifact_evidence_commit import verifier_identity
    from climate_rag.scifact_evidence_commit_runtime import _result
    import hashlib
    row, backend, journal, frame, corpus = run(tmp_path, [verdict()], "fixed_top1")
    new_run = "c"*64
    state = CommitState(1, "fixed_top1", frame, f"{PROTOCOL}:{new_run}:1:fixed_top1")
    step = row["steps"][0]
    response = json.loads((journal.directory/"g00.finished.json").read_bytes())["response"]
    prompt = backend.render(step["observation"], step["schema"])
    state.accept(step, "c7", verifier_identity(prompt, step["schema"], hashlib.sha256(response["raw"].encode()).hexdigest()))
    assert state.next_call() is None
    forged = _result(state, frame, row["steps"], corpus, row["elapsed_seconds"])
    path = tmp_path/"episode/reserved.json"
    reservation = json.loads(path.read_bytes())
    reservation.update(run_identity=new_run, episode_id=state.episode_id)
    path.write_text(json.dumps(reservation))
    with pytest.raises(ValueError, match="actual_call_binding"):
        audit_episode(forged, frame, journal.directory, tmp_path/"episode/private-responses",
                      backend.base.tokenizer, corpus, run_identity=new_run)


@pytest.mark.parametrize("arm", ["fixed_top1", "adaptive"])
def test_final_assembly_deadline_has_same_unresolved_contract_as_auditor(tmp_path, monkeypatch, arm):
    import time
    timer = {"offset": 0.0}
    original = VerdictRegistry.assemble
    def assemble(self, ids, frame):
        result = original(self, ids, frame)
        if arm == "fixed_top1" or len(list((tmp_path/"ledger").glob("g*.finished.json"))) == 3:
            timer["offset"] = 120.0
        return result
    monkeypatch.setattr(VerdictRegistry, "assemble", assemble)
    actions = [verdict()] if arm == "fixed_top1" else [{"action":"verify", "source_id":"c7"}, verdict(), choose]
    row, backend, journal, frame, corpus = run(tmp_path, actions, arm, clock=lambda: time.monotonic()+timer["offset"])
    assert row["elapsed_seconds"] >= 120 and row["state"] == "unresolved" and row["prediction"] is None
    assert row["reason"] == "episode_deadline"
    assert audit(row, backend, journal, frame, corpus, tmp_path) == row
