import json
import time

import pytest

from climate_rag.scifact_grounding import Abstract, GoldClaim, Rationale
from climate_rag.scifact_read_continuation import ordered_write
from climate_rag.scifact_semantic_contract import checked
from run_scifact_grounding_train_operator import sha
from run_scifact_natural_operator import slot_watchdog, validate_release
import score_scifact_natural as scoring
from test_bounded_scifact_runtime import ABSTAIN
from test_scifact_natural_runtime import matrix


def test_posthoc_physical_audit_roundtrip_and_tamper(tmp_path):
    _, backend, out = matrix(tmp_path, [ABSTAIN] * 3)
    corpus = {i: Abstract(i, "Synthetic", ("First fixture sentence.", "Second."), False) for i in range(100, 120)}
    rows = scoring.audit_rows(tmp_path, [1], corpus, backend.base.tokenizer)
    assert all(r["audit_status"] == "valid_terminal" for r in rows.values())
    path = out / "1-B/result.json"
    bad = json.loads(path.read_bytes())
    bad["prediction"]["id"] = 2
    path.write_text(json.dumps(bad))
    with pytest.raises(ValueError, match="prediction_identity"):
        scoring.audit_rows(tmp_path, [1], corpus, backend.base.tokenizer)


def test_strict_direct_recovery_vs_full_natural_and_failed_nei():
    corpus = {100: Abstract(100, "Synthetic", ("Evidence.",), False)}
    golds = [GoldClaim(i, "Fixture", {100: (Rationale("SUPPORT", (0,)),)}, ()) for i in (1, 2)]
    golds.append(GoldClaim(3, "NEI fixture", {}, ()))
    def pred(i, label):
        return {"id": i, "evidence": {"100": {"label": label, "sentences": [0]}}}
    rows = {}
    for i in (1, 2, 3):
        for arm in ("A", "B", "C"):
            rows[i, arm] = {"audit_status": "valid_terminal" if i != 3 else "unresolved",
                "initial_action": "answer" if i == 1 else "read",
                "prediction": pred(i, "CONTRADICT" if i == 1 and arm == "A" else "SUPPORT") if i != 3 else None,
                "audited_direct": pred(i, "CONTRADICT") if i == 1 else None}
    report = scoring.summarize([1, 2, 3], golds, rows, corpus)
    assert report["arms"]["A"]["strict_whole_answer_count"] == 1
    assert report["arms"]["B"]["strict_whole_answer_count"] == 2
    assert len(report["program_teacher_candidates"]) == 2
    assert all(r["claim_id"] == 1 and not r["training_authorized"] for r in report["program_teacher_candidates"])
    assert report["arms"]["B"]["cases"][1]["initial_tool_strategy_comparison_only"]
    assert not report["arms"]["B"]["cases"][1]["recoverable_direct_terminal_opportunity"]
    assert report["arms"]["B"]["cases"][2]["nei_failure_not_correct"]
    assert report["arms"]["B"]["planned"] == 3


def test_failure_saves_cost_before_tokenizer_or_gold(tmp_path, monkeypatch):
    prepared = tmp_path / "prepared"
    prepared.mkdir()
    ordered_write(prepared / "selection.json", {"selected": [{"id": i} for i in range(24)]})
    digest = sha(prepared / "selection.json")
    ordered_write(prepared / "preparation.json", {"selection_sha256": digest})
    monkeypatch.setattr(scoring, "SELECTION_SHA", digest)
    ordered_write(tmp_path / "worker-exit.json", {"child_reaped": True, "returncode": -9,
                                                 "parent_wait_interrupted": "TimeoutError"})
    monkeypatch.setattr(scoring, "select_complete_fit", lambda *a: pytest.fail("gold prohibited"))
    result = scoring.score_after_exit(tmp_path, lambda: pytest.fail("tokenizer must not load"))
    assert result["status"] == "no_quality" and result["gold_read"] is False
    assert (tmp_path / "cost-before-gold.json").is_file()
    assert json.loads((tmp_path / "cost-before-gold.json").read_bytes())["missing_slot_receipts"] == 72


def test_watchdog_inherits_prefix_time_and_ignores_completed_slot(tmp_path):
    slot = tmp_path / "1-B"
    slot.mkdir()
    ordered_write(slot / "reserved.json", {"started_unix": time.time() - 2, "deadline_seconds": 1})
    with pytest.raises(TimeoutError, match="inherited"):
        slot_watchdog(tmp_path)
    ordered_write(slot / "result.json", {"status": "failed"})
    slot_watchdog(tmp_path)


@pytest.mark.parametrize("fault", ["tokenizer", "partial_result"])
def test_cost_survives_tokenizer_or_partial_result_failure(tmp_path, monkeypatch, fault):
    prepared = tmp_path / "prepared"
    prepared.mkdir()
    ordered_write(prepared / "selection.json", {"selected": [{"id": i} for i in range(24)]})
    digest = sha(prepared / "selection.json")
    ordered_write(prepared / "preparation.json", {"selection_sha256": digest})
    monkeypatch.setattr(scoring, "SELECTION_SHA", digest)
    ordered_write(tmp_path / "worker-exit.json", {"child_reaped": True, "returncode": 0, "parent_wait_interrupted": None})
    for i in range(24):
        for arm in ("A", "B", "C"):
            directory = tmp_path / "inference" / f"{i}-{arm}"
            directory.mkdir(parents=True)
            ordered_write(directory / "result.json", {"result": {"events": []}})
    if fault == "partial_result":
        (tmp_path / "inference/0-A/result.json").write_text('{"partially_written":')
    monkeypatch.setattr(scoring, "select_complete_fit", lambda *a: pytest.fail("gold prohibited"))
    def broken_tokenizer():
        assert fault == "tokenizer"
        assert (tmp_path / "cost-before-gold.json").exists()
        raise ImportError("synthetic missing tokenizer dependency")
    result = scoring.score_after_exit(tmp_path, broken_tokenizer)
    assert result["status"] == "no_quality" and result["gold_read"] is False
    cost = json.loads((tmp_path / "cost-before-gold.json").read_bytes())
    assert cost["planned_slots"] == 72
    assert cost["unknown_tool_slots"] == int(fault == "partial_result")


def test_draft_cannot_execute_and_selection_hash_is_not_mutable(tmp_path):
    with pytest.raises(ValueError, match="draft"):
        validate_release({"authorization": "DRAFT_NOT_AUTHORIZED"})
    path = tmp_path / "selection.json"
    path.write_text("{}")
    with pytest.raises(ValueError):
        checked(path, "0" * 64)
