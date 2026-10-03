"""Synthetic utility8 invariants. No real corpus, labels, or models."""
import json
import hashlib
import io
from pathlib import Path
import sys

import pytest

from climate_rag.scifact_bounded_agent import SciFactBoundedAgent
from climate_rag.scifact_bounded_runtime import ARMS, run_bounded_slot
from climate_rag.scifact_grounding import Abstract, GoldClaim, Rationale
from climate_rag.scifact_read_continuation import ordered_write
from climate_rag.scifact_terminal import source_from_abstract
from climate_rag.scifact_utility_contract import UtilityDiagnostic, identity, read_intervention, seal_prefix
from climate_rag.scifact_utility_runtime import JournalProvider, direct_prediction, ledger_cost, reranker_cost, run_matrix
from test_bounded_scifact_runtime import ABSTAIN, SyntheticBackend

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from prepare_scifact_utility8 import select_eight, select_queries  # noqa: E402
from score_scifact_utility8 import summarize, audit_rows, tool_changes, score_after_exit  # noqa: E402


def setup(actions, tmp_path):
    corpus = {i: Abstract(i, "Fixture", ("One fixture sentence.", "Second fixture sentence."), False)
              for i in range(100, 120)}
    sources = [source_from_abstract(a) for a in corpus.values()]
    backend = SyntheticBackend(actions, False)
    journal = JournalProvider(backend, tmp_path / "ledger")
    journal.slot = "1-A"
    captured = {}
    def observe(kind, value):
        if kind in {"initial_frame", "prefix"}:
            captured[kind] = value
    row = run_bounded_slot(1, "Fixture claim", "adaptive", ARMS[0], journal,
                           lambda q, k: sources[:k], lambda q, c: c, corpus,
                           diagnostic=UtilityDiagnostic(observe))
    return captured, row, journal, corpus, sources


def resume(captured, journal, corpus, tool, rerank=lambda q, c: c):
    frames = []
    journal.slot = "1-C" if tool["action"] == "rerank" else "1-B"
    def no_retrieve(*args):
        pytest.fail("resume must not rerun initial retrieval")
    row = run_bounded_slot(1, "Fixture claim", "adaptive", ARMS[0], journal, no_retrieve, rerank, corpus,
        diagnostic=UtilityDiagnostic(lambda k, v: frames.append((k, v)), captured["prefix"], tool))
    return row, frames


def test_disk_roundtrip_shared_prefix_budget_and_physical_cost(tmp_path):
    captured, a, journal, corpus, _ = setup([ABSTAIN, ABSTAIN], tmp_path)
    path = tmp_path / "prefix.json"
    ordered_write(path, captured["prefix"])
    captured["prefix"] = json.loads(path.read_bytes())
    b, frames = resume(captured, journal, corpus, {"action": "read", "source_ids": ["c5"]})
    result = b["result"]
    assert result["new_model_calls"] == 1 and result["model_calls"] == 2
    assert result["tool_calls"] == 2 and result["validation_repairs"] == 0
    assert result["events"][0]["shared_prefix_reference"] is True
    assert result["events"][1]["origin"] == "scripted_intervention"
    assert not result["events"][1]["model_selected"]
    assert identity(frames[0][1]) == identity(captured["initial_frame"])
    obs = journal.backend.observations[-1]
    assert obs["remaining_calls"] == 4 and obs["remaining_tools"] == 3
    assert obs["feedback"] == "tool_completed: inspect current_citable; no semantic conclusion implied"
    assert len(result["visible_attempts"]) == 2
    assert result["visible_attempts"][1]["visible"][0]["doc_id"] == 105
    assert result["decision_execution_audit"][0]["next_feedback"] == obs["feedback"]
    assert ledger_cost(journal.directory, ["g00", "g00", "g01"])["unique_physical_calls"] == 2
    assert result["elapsed_ms"] >= captured["prefix"]["payload"]["elapsed_seconds"] * 1000
    assert a["result"]["generation_attempts"][0]["diagnostics"]["physical_attempt_id"] == "g00"


def test_rerank_reorders_but_alias_still_names_original_source(tmp_path):
    answer = {"action": "answer", "documents": [{"source_id": "c19", "label": "SUPPORTS", "sentence_ids": ["c19:0"]}]}
    captured, _, journal, corpus, _ = setup([ABSTAIN, answer], tmp_path)
    c, frames = resume(captured, journal, corpus, {"action": "rerank"}, lambda q, c: list(reversed(c)))
    assert list(c["prediction"]["evidence"]) == ["119"]
    assert frames[-1][1]["alias_to_source"]["c19"] == "119"
    assert c["result"]["rerank_pairs"] == 20 and c["result"]["new_model_calls"] == 1


@pytest.mark.parametrize("continuation", [{"action": "rerank"}, {"nonsense": True}])
def test_one_continuation_no_tool_or_repair(tmp_path, continuation):
    captured, _, journal, corpus, _ = setup([ABSTAIN, continuation], tmp_path)
    b, _ = resume(captured, journal, corpus, {"action": "read", "source_ids": ["c5"]})
    assert b["prediction"] is None
    assert b["result"]["outcome"] in {"proposed_not_executed", "continuation_invalid"}
    assert b["result"]["new_model_calls"] == 1
    assert b["result"]["validation_repairs"] == 0 and b["result"]["tool_calls"] == 2


def test_first_read_loop_not_repaired_into_one_source(tmp_path):
    loop = {"action": "read", "source_ids": [f"c{i}" for i in range(5)]}
    captured, _, _, _, _ = setup([loop, ABSTAIN], tmp_path)
    assert read_intervention(captured["prefix"]) == (None, "attempt0_read_loop")
    assert captured["prefix"]["payload"]["attempt"]["status"] == "valid_decision"  # before execution


def test_first_invalid_does_not_use_later_valid_action(tmp_path):
    captured, _, _, corpus, _ = setup([{"invalid": True}, ABSTAIN], tmp_path)
    assert read_intervention(captured["prefix"])[1] == "attempt0_invalid"
    assert direct_prediction(captured["prefix"], 1, corpus) is None


def test_first_tool_has_no_direct_baseline_and_capsule_immutable(tmp_path):
    captured, a, _, corpus, _ = setup([{"action": "read", "source_ids": ["c5"]}, ABSTAIN], tmp_path)
    assert direct_prediction(captured["prefix"], 1, corpus) is None
    assert len(captured["prefix"]["payload"]["events"]) == 1
    assert len(a["result"]["events"]) == 2
    assert read_intervention(captured["prefix"])[0] == {"action": "read", "source_ids": ["c5"]}


@pytest.mark.parametrize("fault", ["alias", "hash", "budget", "elapsed"])
def test_corrupt_prefix_no_new_generation(tmp_path, fault):
    captured, _, journal, corpus, _ = setup([ABSTAIN], tmp_path)
    p = captured["prefix"]["payload"]
    if fault == "alias":
        p["aliases"]["100"] = "c19"
    elif fault == "hash":
        p["sources"][0]["sha256"] = "f" * 64
    elif fault == "budget":
        p["budget"]["max_calls"] = 6
    else:
        p["elapsed_seconds"] = float("nan")
    captured["prefix"] = seal_prefix(p)
    b, _ = resume(captured, journal, corpus, {"action": "read", "source_ids": ["c5"]})
    assert b["result"]["outcome"].startswith("controller_failure:")
    assert ledger_cost(journal.directory)["unique_physical_calls"] == 1


def test_elapsed_prefix_consumes_original_deadline(tmp_path):
    captured, _, journal, corpus, _ = setup([ABSTAIN], tmp_path)
    p = captured["prefix"]["payload"]
    p["elapsed_seconds"] = 120
    captured["prefix"] = seal_prefix(p)
    b, _ = resume(captured, journal, corpus, {"action": "rerank"})
    assert b["result"]["new_model_calls"] == 0
    assert ledger_cost(journal.directory)["unique_physical_calls"] == 1


def test_observer_error_never_becomes_extra_model_repair(tmp_path):
    _, _, journal, _, sources = setup([ABSTAIN, ABSTAIN], tmp_path)
    def broken(kind, value):
        if kind == "prefix":
            raise ValueError("synthetic I/O failure")
    result = SciFactBoundedAgent(journal, lambda q, k: sources, rerank=lambda q, c: c).run(
        "Fixture claim", diagnostic=UtilityDiagnostic(broken))
    assert result["model_calls"] == 1 and result["validation_repairs"] == 0


def test_identical_text_is_two_physical_calls_and_unfinished_unknown(tmp_path):
    captured, _, journal, corpus, _ = setup([ABSTAIN, ABSTAIN], tmp_path)
    resume(captured, journal, corpus, {"action": "rerank"})
    assert ledger_cost(journal.directory)["unique_physical_calls"] == 2
    ordered_write(journal.directory / "g02.reserved.json", {"physical_attempt_id": "g02"})
    cost = ledger_cost(journal.directory)
    assert cost["unknown_usage_attempts"] == 1 and cost["total_tokens"] is None


def test_default_controller_same_decisions_and_counts_with_observer(tmp_path):
    captured, row, _, _, sources = setup([ABSTAIN], tmp_path)
    base = SyntheticBackend([ABSTAIN], False)
    result = SciFactBoundedAgent(base, lambda q, k: sources, rerank=lambda q, c: c).run("Fixture claim")
    for field in ("outcome", "answer", "model_calls", "tool_calls", "validation_repairs"):
        assert result[field] == row["result"][field]
    assert captured["prefix"]["payload"]["attempt"]["action"] == "abstain"


def test_continuation_invalid_loop_keeps_schema_vs_controller_legality(tmp_path):
    captured, _, journal, corpus, _ = setup([ABSTAIN, {"action": "read", "source_ids": ["c5"]}], tmp_path)
    row, _ = resume(captured, journal, corpus, {"action": "read", "source_ids": ["c5"]})
    a = row["result"]["generation_attempts"][-1]
    assert a["proposal_validation"] == "schema_valid" and not a["controller_legal"]
    assert a["controller_error"] == "read_loop" and row["prediction"] is None


def test_full_synthetic_matrix_24_unique_calls_cost_and_audit(tmp_path):
    class Backend(SyntheticBackend):
        def start_slot(self, path):
            path.mkdir()
    corpus = {i: Abstract(i, "Fixture", ("Fixture sentence.",), False) for i in range(100, 120)}
    sources = [source_from_abstract(a) for a in corpus.values()]
    backend = Backend([ABSTAIN] * 24, False)
    out = tmp_path / "inference"
    report = run_matrix([{"id": i, "claim": "Fixture claim"} for i in range(8)], backend,
                        lambda q, k: sources[:k], lambda q, c: list(reversed(c)), corpus, out)
    assert len(report["runs"]) == 24 and report["physical_generation_cost"]["unique_physical_calls"] == 24
    assert sum(r["logical_generation_cost"]["unique_physical_calls"] for r in report["runs"]) == 40
    assert all(r["result"]["new_model_calls"] == 1 for r in report["runs"])
    audited = audit_rows(tmp_path, list(range(8)), corpus)
    assert all(r["audit_status"] == "valid_terminal" for r in audited.values())
    changes = tool_changes(tmp_path, audited)
    assert len(changes) == 16 and all(r["origin"] == "scripted" for r in changes)
    assert all(r["next_frame_change"]["feedback"].startswith("tool_completed:") for r in changes)


def test_generation_57_rejected_before_backend(tmp_path):
    _, _, journal, _, _ = setup([ABSTAIN], tmp_path)
    for i in range(1, 56):
        ordered_write(journal.directory / f"g{i:02d}.reserved.json", {"physical_attempt_id": f"g{i:02d}"})
    with pytest.raises(RuntimeError, match="physical_generation_budget"):
        journal.generate({}, {}, 512, 120)
    assert len(list(journal.directory.glob("*.reserved.json"))) == 56


def test_reranker_unknown_tokens_remain_null(tmp_path):
    ordered_write(tmp_path / "r00.reserved.json", {"requested_pairs": 20})
    assert reranker_cost(tmp_path)["nonpadding_tokens"] is None
    ordered_write(tmp_path / "r00.finished.json", {"completed_pairs": 0, "elapsed_ms_including_swaps": 1,
        "observed_nonpadding_tokens": None, "batches": []})
    assert reranker_cost(tmp_path)["nonpadding_tokens"] is None
    assert reranker_cost(tmp_path)["generator_calls"] == 0


def test_selection_roles_order_and_no_replacement():
    def row(i):
        return {"claim_id": i, "component": str(i), "program_teacher_reads": 0,
                "initial_complete": False, "post_read_complete": None, "training_ready": False,
                "official_label_scope": "SUPPORTS"}
    old, new = [row(i) for i in range(48)], [row(i) for i in range(100, 196)]
    old[0]["program_teacher_reads"] = 1
    for r in (old[1], old[2], new[0], new[1]):
        r["initial_complete"] = True
    new[2]["training_ready"] = True
    new[3]["official_label_scope"] = new[4]["official_label_scope"] = "NOT_ENOUGH_INFO"
    selected = select_eight(old, new)
    assert [r["id"] for r in selected] == [0, 1, 2, 100, 101, 102, 103, 104]
    old[0]["program_teacher_reads"] = 0
    with pytest.raises(ValueError, match="unique_original_natural_read"):
        select_eight(old, new)


def test_only_selected_gold_free_queries_are_decoded(monkeypatch):
    import prepare_scifact_utility8 as preparation
    payload = b'{"id":1,"claim":"selected fixture"}\n{"id":2,"claim":"unselected private marker"}\n'
    original = preparation.json.loads
    def guarded(data, *args, **kwargs):
        assert b"unselected private marker" not in (data if isinstance(data, bytes) else data.encode())
        return original(data, *args, **kwargs)
    monkeypatch.setattr(preparation.json, "loads", guarded)
    assert select_queries(io.BytesIO(payload), hashlib.sha256(payload).hexdigest(), [1]) == [
        {"id": 1, "claim": "selected fixture"}]


def test_wrong_sentence_nei_failed_empty_and_complete_document_tradeoff():
    corpus = {i: Abstract(i, "Fixture", ("Relevant", "Wrong"), False) for i in (100, 101)}
    golds = [GoldClaim(i, "Fixture", {100: (Rationale("SUPPORT", (0,)),)}, ()) for i in range(6)]
    golds += [GoldClaim(i, "NEI fixture", {}, ()) for i in (6, 7)]
    rows = {(i, a): {"audit_status": "invalid_or_nonterminal", "prediction": None, "audited_direct": None}
            for i in range(8) for a in ("A", "B", "C")}
    rows[(0, "A")] = {"audit_status": "valid_terminal", "audited_direct": None,
        "prediction": {"id": 0, "evidence": {"100": {"label": "SUPPORT", "sentences": [1]}}}}
    report = summarize(list(range(8)), golds, rows, corpus)
    assert report["A"]["cases"][0]["strict_whole_answer"] is False
    assert report["A"]["cases"][6]["nei_failure_empty_prediction"] is True
    assert report["A"]["cases"][6]["nei_valid_abstention"] is False
    assert report["A"]["planned"] == 8 and report["A"]["official_micro"]["claim_count"] == 8
    golds[0] = GoldClaim(0, "Fixture", {i: (Rationale("SUPPORT", (0,)),) for i in (100, 101)}, ())
    rows[(0, "B")] = {"audit_status": "valid_terminal",
        "prediction": {"id": 0, "evidence": {"101": {"label": "SUPPORT", "sentences": [0]}}},
        "audited_direct": {"id": 0, "evidence": {"100": {"label": "SUPPORT", "sentences": [0]}}}}
    comparison = summarize(list(range(8)), golds, rows, corpus)["B"]["cases"][0]["direct_comparison"]
    assert comparison["gained_complete_documents"] == comparison["lost_complete_documents"] == 1
    assert comparison["candidate_for_manual_trajectory_review"] is False


def test_cost_saved_before_audit_failure_and_no_gold_read(tmp_path, monkeypatch):
    import score_scifact_utility8 as score
    prepared = tmp_path / "prepared"
    prepared.mkdir()
    ordered_write(tmp_path / "worker-exit.json", {"child_reaped": True, "returncode": 0, "parent_wait_interrupted": None})
    ordered_write(prepared / "selection.json", {"selected": [{"id": i} for i in range(8)]})
    ordered_write(prepared / "preparation.json", {"selection_sha256": hashlib.sha256((prepared / "selection.json").read_bytes()).hexdigest()})
    for i in range(8):
        for arm in ("A", "B", "C"):
            d = tmp_path / "inference" / f"{i}-{arm}"
            d.mkdir(parents=True)
            ordered_write(d / "result.json", {})
    monkeypatch.setattr(score, "checked", lambda *a: b'{"doc_id":100,"title":"Fixture","abstract":["Sentence"],"structured":false}\n')
    def broken(*args):
        assert (tmp_path / "cost-before-gold.json").is_file()
        raise ValueError("invalid receipt")
    monkeypatch.setattr(score, "audit_rows", broken)
    monkeypatch.setattr(score, "select_complete_fit", lambda *a: pytest.fail("must not open gold"))
    result = score_after_exit(tmp_path)
    assert result["status"] == "no_quality" and result["gold_read"] is False
    assert not (tmp_path / "quality.json").exists()
