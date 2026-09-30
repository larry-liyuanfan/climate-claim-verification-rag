"""Small synthetic audit counterexamples; no weights, scientific data or GPU."""
import copy
import hashlib
import importlib
import json
from pathlib import Path

import pytest

from climate_rag.local_bounded_scifact_provider import observation_identity
from climate_rag.scifact_bounded_runtime import ARMS, run_bounded_slot
from climate_rag.scifact_grounding import Abstract, GoldClaim, Rationale
from climate_rag.scifact_scoring import ClaimPrediction, PredictedAbstract
from climate_rag.scifact_terminal import source_from_abstract
from test_bounded_scifact_runtime import ABSTAIN, SyntheticBackend, wire


@pytest.fixture
def audit(monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1] / "scripts"))
    return importlib.import_module("audit_scifact_bounded_closeout")


def fixture(tmp_path, decisions, gap=False, fail_rerank=False):
    corpus = {i: Abstract(i, "Synthetic", ("Zero.", "One.", "Two."), False) for i in range(100, 107)}
    sources = [source_from_abstract(d) for d in corpus.values()]
    gold = GoldClaim(42, "Fixture claim", {105: (Rationale("SUPPORT", (1, 2)),)}, (105,))
    private = tmp_path / "private"
    private.mkdir()

    class Backend(SyntheticBackend):
        def generate(self, observation, schema, *args):
            response = super().generate(observation, schema, *args)
            payload = response["raw"].encode()
            d = response["diagnostics"]
            receipt = d["private_attachment"]
            receipt.update(dropped_bytes=0, stored_prefix_sha256=receipt["sha256"])
            d.update(grammar_log={"attempted_bytes": 0, "stored_bytes": 0, "dropped_bytes": 0,
                                  "truncated": False, "io_failed": False,
                                  "sha256": hashlib.sha256(b"").hexdigest(),
                                  "stored_prefix_sha256": hashlib.sha256(b"").hexdigest()},
                     output_sha256=receipt["sha256"], output_tokens=response["usage"]["output_tokens"],
                     generation_elapsed_ms=1.0, actual_observation=observation_identity(observation))
            (private / f"{len(self.observations)}-response.txt").write_bytes(payload)
            return response

    backend = Backend([wire(d) if gap else d for d in decisions], gap)

    def rerank(q, rows):
        if fail_rerank:
            raise ValueError("synthetic unavailable")
        return rows

    row = run_bounded_slot(42, gold.claim, "adaptive", ARMS[int(gap)], backend,
                           lambda q, k: sources[:k], rerank, corpus)
    raw = {hashlib.sha256(p.read_bytes()).hexdigest(): p.read_text() for p in private.iterdir()}
    return row, gold, corpus, raw, private


READ = {"action": "read", "source_ids": ["c5"]}
ANSWER = {"action": "answer", "documents": [{"source_id": "c5", "label": "SUPPORTS", "sentence_ids": ["c5:1", "c5:2"]}]}


@pytest.mark.parametrize("gap", [False, True])
def test_exact_wire_mapping_event_feedback_and_gold_chain(tmp_path, audit, gap):
    row, gold, corpus, raw, private = fixture(tmp_path, [READ, ANSWER], gap)
    physical, _ = audit.physical(private, row["result"]["generation_attempts"], gap)
    assert physical["complete_responses"] == physical["empty_grammar_receipts"] == 2
    report = audit.audit_slot(row, gold, corpus, raw)
    assert report["fixed_tools"] == {"retrieve:completed": 1}
    assert report["chain"]["execution_success"] == 1
    assert report["chain"]["newly_seen_then_gold_label_complete_rationale"] == 1
    assert report["chain"]["newly_seen_then_strict_whole_final_answer_correct"] == 1
    assert report["cost"]["calls"] == 2


def test_adaptive_abstains_not_credited_other_route(tmp_path, audit):
    row, gold, corpus, raw, _ = fixture(tmp_path, [READ, ABSTAIN])
    report = audit.audit_slot(row, gold, corpus, raw)
    assert report["chain"]["then_legal_decision"] == 1
    assert report["chain"]["then_final_cites_added"] == 0
    assert report["chain"]["strict_whole_final_answer_correct_slots"] == 0


def test_failed_event_keeps_cost_but_not_success(tmp_path, audit):
    row, gold, corpus, raw, _ = fixture(tmp_path, [{"action": "rerank"}], fail_rerank=True)
    report = audit.audit_slot(row, gold, corpus, raw)
    assert report["chain"]["model_events"] == 1
    assert report["chain"]["execution_success"] == 0
    assert report["cost"]["known_tokens_including_failures"]["output_tokens"] == 50


def test_read_then_removed_new_evidence_not_cited(tmp_path, audit):
    old = {"action": "answer", "documents": [{"source_id": "c0", "label": "SUPPORTS", "sentence_ids": ["c0:1"]}]}
    row, gold, corpus, raw, _ = fixture(tmp_path, [READ, {"action": "rerank"}, old])
    report = audit.audit_slot(row, gold, corpus, raw)
    assert report["chain"]["newly_seen_evidence"] == 1
    assert report["chain"]["newly_seen_then_final_cites"] == 0


@pytest.mark.parametrize("fault", ["feedback", "wire_action", "corpus_hash", "answer", "event_arguments", "candidate_universe", "links"])
def test_identity_and_trace_tampering_rejected(tmp_path, audit, fault):
    row, gold, corpus, raw, _ = fixture(tmp_path, [READ, ANSWER])
    r = row["result"]
    if fault == "feedback":
        r["visible_attempts"][1]["feedback"] = "self-reported success"
    elif fault == "wire_action":
        r["generation_attempts"][0]["action"] = "rerank"
    elif fault == "corpus_hash":
        r["visible_attempts"][1]["visible"][0]["text_sha256"] = "0" * 64
        r["visible_attempts"][1]["actual_observation"]["visible"][0]["sha256"] = "0" * 64
        r["generation_attempts"][1]["visible_sentence_sha256"]["c5:0"] = "0" * 64
        r["generation_attempts"][1]["diagnostics"]["actual_observation"] = copy.deepcopy(r["visible_attempts"][1]["actual_observation"])
    elif fault == "answer":
        r["answer"]["documents"][0]["citations"][0]["sentence_index"] = 0
    elif fault == "event_arguments":
        r["events"][1]["requested_context"] = ["c6"]
    elif fault == "candidate_universe":
        r["events"][0]["candidate_ids"].remove("c5")
    else:
        r["model_tool_links"][0]["event_index"] = 0
    with pytest.raises(ValueError):
        audit.audit_slot(row, gold, corpus, raw)


def test_gap_history_mapping_rejected(tmp_path, audit):
    row, gold, corpus, raw, _ = fixture(tmp_path, [READ, ANSWER], True)
    row["result"]["candidate_wire_audit"][1]["feedback"]["visibility_delta"]["tool_completion_reported"] = False
    with pytest.raises(ValueError, match="mapping_history"):
        audit.audit_slot(row, gold, corpus, raw)


def test_rejected_read_loop_followed_by_legal_abstention(tmp_path, audit):
    read_loop = {"action": "read", "source_ids": [f"c{i}" for i in range(5)]}
    row, gold, corpus, raw, _ = fixture(tmp_path, [read_loop, ABSTAIN])
    report = audit.audit_slot(row, gold, corpus, raw)
    assert report["attempt_statuses"]["validation_failed"] == 1
    assert report["chain"]["model_events"] == 0
    assert report["validation_repairs"] == 1


def test_no_physical_receipt_lost_or_hidden(tmp_path, audit):
    row, _, _, _, private = fixture(tmp_path, [ABSTAIN])
    records = row["result"]["generation_attempts"]
    (private / "extra-grammar.txt").write_text("hidden grammar failure")
    with pytest.raises(ValueError, match="grammar_receipt_multiset"):
        audit.physical(private, records, False)
    (private / "extra-response.txt").write_text(json.dumps(ABSTAIN))
    with pytest.raises(ValueError, match="wire/receipt"):
        audit.physical(private, records, False)


def test_reread_not_new_discovery(audit):
    gold = GoldClaim(1, "Synthetic", {1: (Rationale("SUPPORT", (0,)),)}, (1,))
    prediction = ClaimPrediction(1, {1: PredictedAbstract("SUPPORT", (0,))})
    report = audit.evidence_chain(prediction, gold, [{(1, 0)}, {(2, 0)}, {(1, 0)}],
                                 [{"status": "valid_decision"}] * 3,
                                 [{"status": "completed"}] * 2, [(0, 0), (1, 1)])
    assert report["added_citable_evidence"] == 2 and report["newly_seen_evidence"] == 1
    assert report["then_gold_label_complete_rationale"] == 1
    assert report["newly_seen_then_gold_label_complete_rationale"] == 0
    assert report["successful_event_with_next_legal_decision"] == 2


@pytest.mark.parametrize("sentences,extra,partial,whole", [((1,), False, 0, 0), ((1, 2), True, 1, 0), ((1, 2), False, 1, 1)])
def test_complete_rationale_and_strict_whole_are_separate(audit, sentences, extra, partial, whole):
    gold = GoldClaim(1, "Synthetic", {1: (Rationale("SUPPORT", (1, 2)),)}, (1,))
    docs = {1: PredictedAbstract("SUPPORT", sentences)}
    if extra:
        docs[2] = PredictedAbstract("CONTRADICT", (0,))
    prediction = ClaimPrediction(1, docs)
    report = audit.evidence_chain(prediction, gold, [{(2, 0)}, {(1, 1), (1, 2)}],
                                 [{"status": "valid_decision", "action": "read"}, {"status": "valid_decision", "action": "answer"}],
                                 [{"status": "completed"}], [(0, 0)], "ids_validated_semantics_unmeasured")
    assert report["then_gold_label_complete_rationale"] == partial
    assert report["then_strict_whole_final_answer_correct"] == whole


@pytest.mark.parametrize("status,action,outcome,success", [
    ("validation_failed", None, "validation_repair_exhausted", 0),
    ("terminal_failure", None, "controller_failure:RuntimeError", 0),
    ("valid_decision", "abstain", "deadline", 0),
    ("valid_decision", "abstain", "model_abstention:insufficient_evidence", 1),
])
def test_nei_structural_match_requires_valid_terminal(audit, status, action, outcome, success):
    gold = GoldClaim(1, "Synthetic NEI", {}, ())
    report = audit.evidence_chain(ClaimPrediction(1, {}), gold, [set()],
                                 [{"status": status, "action": action}], [], [], outcome)
    assert report["structural_whole_gold_match_slots"] == 1
    assert report["strict_whole_final_answer_correct_slots"] == success


def test_absent_cost_never_filled_as_zero(audit):
    result = audit.tariff([{"usage": {"input_tokens": 17}, "usage_known": False, "diagnostics": {}}])
    assert result["token_totals_are_lower_bounds"] and result["missing_fields"] == {"output_tokens": 1}
    assert result["generation_ms_known_sum"] is None and result["generation_ms_missing"] == 1


def test_pending_pair_gate_before_any_run_or_gold_read(tmp_path, audit):
    submission = {"jobs": [{"job_id": 1}, {"job_id": 2}]}
    with pytest.raises(ValueError, match="pair_not_completed"):
        audit.closeout(tmp_path / "does-not-exist", submission, {},
                       [{"JobIDRaw": "1", "State": "COMPLETED", "ExitCode": "0:0"},
                        {"JobIDRaw": "2", "State": "RUNNING", "ExitCode": "0:0"}])
