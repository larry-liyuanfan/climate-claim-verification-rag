"""Synthetic train-sampling/runtime tests; no real train/dev/model execution."""

import copy
import importlib.util
import json
import subprocess
from pathlib import Path

import pytest

from climate_rag.scifact_diagnostic_runtime import ROUTES, run_slot
from climate_rag.scifact_diagnostic_scoring import score_diagnostic
from climate_rag.scifact_grounding import Abstract, GoldClaim, Rationale
from climate_rag.scifact_train_diagnostic import (
    STRATA, classify_opportunity, coverage, document_opportunities,
    packing_snapshot, select_component_distinct,
)
from climate_rag.scifact_terminal import source_from_abstract
from test_scifact_terminal import ABSTAIN, FixtureProvider, corpus_fixture


def gold(evidence=None, claim_id=42):
    return GoldClaim(claim_id, "A fixture claim", evidence or {}, ())


def rat(*indices):
    return Rationale("SUPPORT", tuple(indices))


def snapshot(candidates=range(10, 16), visible=None):
    return {"candidate_doc_ids": list(candidates), "visible": visible or {}}


def no_probe(ids):
    raise AssertionError("unexpected oracle read")


def test_document_opportunity_not_all_claim_and_alternatives_or():
    g = gold({10: (rat(1, 2), rat(4, 5)), 11: (rat(0),)})
    assert document_opportunities(g, {10: [1, 2]}) == [10]
    assert not coverage(g, {10: [1, 2]})["complete"]
    assert classify_opportunity(g, snapshot(visible={10: [1, 2]}), no_probe)["stratum"] == STRATA[0]
    assert document_opportunities(g, {10: [1, 4]}) == []


def test_long_rationale_not_first3_opportunity():
    g = gold({10: (rat(0, 1, 2, 3),)})
    assert coverage(g, {10: [0, 1, 2, 3]})["complete"]
    assert not document_opportunities(g, {10: [0, 1, 2, 3]})
    result = classify_opportunity(g, snapshot(visible={10: [0, 1, 2, 3]}), no_probe)
    assert result["stratum"] is None


def test_replenishable_is_actual_legal_read_witness():
    g = gold({15: (rat(2),), 19: (rat(1),)})
    calls = []

    def probe(ids):
        calls.append(ids)
        return snapshot(visible={15: [0, 1, 2]})

    result = classify_opportunity(g, snapshot(), probe)
    assert result["stratum"] == STRATA[1]
    assert calls == [["c5"]]
    assert result["total_gold_doc_count"] == 2


def test_candidates_with_gold_but_budget_unreachable_is_unclassified():
    result = classify_opportunity(gold({15: (rat(3),)}), snapshot(),
                                  lambda ids: snapshot(visible={15: [0]}))
    assert result["stratum"] is None
    assert result["oracle_read_probe_count"] == 1


def test_nei_precedes_empty_set_and_absent_uses_evidence_not_cited_ids():
    g = GoldClaim(42, "Claim", {}, (10,))
    assert classify_opportunity(g, snapshot(), no_probe)["stratum"] == "nei"
    assert classify_opportunity(gold({99: (rat(0),)}), snapshot(), no_probe)["stratum"] == STRATA[2]


def test_matching_stable_and_maximal_despite_cross_stratum_components():
    rows = []
    components = {}
    for i, (s, c) in enumerate([(STRATA[0], "shared"), (STRATA[1], "shared"),
                               (STRATA[0], "a"), (STRATA[0], "b"), (STRATA[0], "c"),
                               (STRATA[1], "d"), (STRATA[1], "e")]):
        rows.append({"id": i, "stratum": s})
        components[i] = c
    selected, summary = select_component_distinct(rows, components)
    assert len(selected) == 6
    assert len({r["component"] for r in selected}) == 6
    assert select_component_distinct(list(reversed(rows)), components) == (selected, summary)
    assert summary["shortfall"]["nei"] == 3


class TinyTokenizer:
    def apply_chat_template(self, messages, **kwargs):
        return json.dumps(messages)

    def encode(self, value, **kwargs):
        return list(range(len(value) // 4))


def test_real_controller_pack_probe_preserves_original_indices_and_read():
    corpus = corpus_fixture()
    sources = [source_from_abstract(d) for d in corpus.values()]
    first = packing_snapshot("Claim", lambda q, k: sources[:k], TinyTokenizer())
    assert 15 not in first["visible"]
    after = packing_snapshot("Claim", lambda q, k: sources[:k], TinyTokenizer(), ["c5"])
    assert list(after["visible"]) == [15]
    assert after["visible"][15] == list(range(10))
    assert after["fixture_only"]


def test_actual_pack_unreachable_long_first_sentence_is_not_replenishable():
    corpus = corpus_fixture()
    corpus[15] = Abstract(15, "long", ("x" * 50000, "short gold sentence"), False)
    sources = [source_from_abstract(d) for d in corpus.values()]
    def retrieve(q, k):
        return sources[:k]
    initial = packing_snapshot("Claim", retrieve, TinyTokenizer())
    g = GoldClaim(42, "Claim", {15: (rat(1),)}, ())
    result = classify_opportunity(g, initial,
                                  lambda ids: packing_snapshot("Claim", retrieve, TinyTokenizer(), ids))
    assert result["stratum"] is None
    assert result["oracle_read_probe_count"] == 1


def slot(actions, rerank=None):
    corpus = corpus_fixture()
    sources = [source_from_abstract(d) for d in corpus.values()]
    provider = FixtureProvider(actions)
    result = run_slot(42, "A fixture claim", "adaptive", provider,
                      lambda q, k: sources[:k], rerank or (lambda q, c: c), corpus)
    return result, corpus


@pytest.mark.parametrize("action", [
    {"action": "rewrite", "query": "A fixture claim"},
    {"action": "read", "source_ids": ["c0", "c1", "c2", "c3", "c4"]},
])
def test_rejected_tool_proposal_keeps_cost_without_event(action):
    result, _ = slot([action, ABSTAIN])
    assert result["prediction"]["evidence"] == {}
    trace = result["result"]
    assert trace["model_calls"] == 2 and trace["usage"]["output_tokens"] == 100
    assert trace["model_tool_links"][0]["event_index"] is None
    assert trace["model_tool_links"][0]["proposal_status"] == "validation_failed"
    assert not trace["trace_unlinked_events"]


def test_rejected_read_then_successful_read_links_only_execution():
    bad = {"action": "read", "source_ids": ["c0", "c1", "c2", "c3", "c4"]}
    result, _ = slot([bad, {"action": "read", "source_ids": ["c5"]}, ABSTAIN])
    links = result["result"]["model_tool_links"]
    assert links[0]["event_index"] is None and links[1]["event_index"] == 1
    assert links[1]["subsequent_model_attempt"] == 2
    assert {v["doc_id"] for v in result["result"]["visible_attempts"][-1]["visible"]} == {15}


def test_real_tool_failure_retains_event_and_cost():
    def failure(q, c):
        raise RuntimeError("synthetic tool failure")
    result, _ = slot([{"action": "rerank"}], failure)
    trace = result["result"]
    assert trace["events"][1]["status"] == "failed"
    assert trace["model_tool_links"][0]["event_index"] == 1
    assert trace["usage"]["output_tokens"] == 50
    assert result["prediction"]["evidence"] == {}


def test_conversion_failure_never_discards_completed_cost(monkeypatch):
    def fail(*args):
        raise ValueError("synthetic conversion failure")
    monkeypatch.setattr("climate_rag.scifact_diagnostic_runtime.to_original_prediction", fail)
    result, _ = slot([ABSTAIN])
    assert result["export_error"] == "ValueError"
    assert result["result"]["usage"]["output_tokens"] == 50
    assert result["prediction"] is None


def test_point_scorer_retains_four_routes_empty_failures_and_no_bootstrap():
    row, corpus = slot([ABSTAIN])
    rows = []
    for route in ROUTES:
        r = copy.deepcopy(row)
        r["route"] = r["result"]["route"] = route
        rows.append(r)
    report = score_diagnostic([gold({10: (rat(0),)})], corpus, rows,
                              [{"id": 42, "stratum": STRATA[0], "component": "a"}])
    assert report["bootstrap_replicates"] == 0
    for route in ROUTES:
        result = report["routes"][route]
        assert result["official_point_score"]["metrics"]["abstract_rationalized"]["relevant"] == 1
        assert result["known_tokens_including_failures"]["output_tokens"] == 50
        assert result["outcomes"]["model_abstention:insufficient_evidence"] == 1
    with pytest.raises(ValueError):
        score_diagnostic([gold()], corpus, rows[:-1], [{"id": 42, "stratum": "nei", "component": "a"}])


def test_untracked_source_rejected_before_gold_read(tmp_path):
    # A small real git fixture, no existing repository writes.
    subprocess.run(["git", "init", str(tmp_path)], check=True, capture_output=True)
    (tmp_path / "new.py").write_text("pass\n")
    script = Path(__file__).resolve().parents[1] / "scripts/prepare_scifact_train_diagnostic.py"
    spec = importlib.util.spec_from_file_location("prep_train_diag", script)
    module = importlib.util.module_from_spec(spec)
    import sys
    sys.path.insert(0, str(script.parent))
    try:
        spec.loader.exec_module(module)
        with pytest.raises(ValueError, match="untracked"):
            module.require_clean_source(tmp_path)
    finally:
        sys.path.pop(0)


def test_bm25_shared_entry_and_candidate_width():
    from climate_rag.scifact_retrieval import SciFactBM25
    corpus = {i: Abstract(i, "title", ("fixture evidence",), False) for i in range(25)}
    retrieve = SciFactBM25(corpus)
    assert len(retrieve("fixture evidence", 20)) == 20
    assert retrieve("fixture evidence", 5) == retrieve("fixture evidence", 20)[:5]


def test_raw_persistence_precedes_export_and_preserves_known_cost(monkeypatch):
    captured = []
    corpus = corpus_fixture()
    sources = [source_from_abstract(d) for d in corpus.values()]

    def broken(*args):
        assert captured and captured[0]["usage"]["output_tokens"] == 50
        raise ValueError("synthetic")

    monkeypatch.setattr("climate_rag.scifact_diagnostic_runtime.to_original_prediction", broken)
    row = run_slot(42, "Claim", "adaptive", FixtureProvider([ABSTAIN]),
                   lambda q, k: sources[:k], lambda q, c: c, corpus,
                   persist_raw=lambda raw: captured.append(copy.deepcopy(raw)))
    assert row["export_error"] == "ValueError"
    assert "model_tool_links" not in captured[0]


def test_operator_rejects_extra_scoring_file_before_inference_extract(tmp_path):
    import io
    import sys
    import tarfile
    import hashlib
    scripts = Path(__file__).resolve().parents[1] / "scripts"
    sys.path.insert(0, str(scripts))
    try:
        from run_scifact_train_operator import extract_exact
        archive = tmp_path / "bad.tar"
        with tarfile.open(archive, "w") as bundle:
            for name in ("claims.jsonl", "corpus.jsonl", "protocol.json", "gold.jsonl"):
                info = tarfile.TarInfo(name)
                info.size = 2
                bundle.addfile(info, io.BytesIO(b"{}"))
        sha = hashlib.sha256(archive.read_bytes()).hexdigest()
        with pytest.raises(ValueError, match="allowlist"):
            extract_exact(archive, tmp_path / "never-created", sha,
                          {"claims.jsonl", "corpus.jsonl", "protocol.json"})
        assert not (tmp_path / "never-created").exists()
    finally:
        sys.path.pop(0)


@pytest.mark.parametrize("corrupt", ["aggregate", "unknown", "identity"])
def test_scoring_reconciles_corrupt_cost_or_identity(corrupt):
    row, corpus = slot([ABSTAIN])
    rows = []
    for route in ROUTES:
        r = copy.deepcopy(row)
        r["route"] = r["result"]["route"] = route
        rows.append(r)
    if corrupt == "aggregate":
        rows[0]["result"]["usage"]["input_tokens"] += 1
    elif corrupt == "unknown":
        rows[0]["result"]["unknown_usage_attempts"] += 1
    else:
        rows[0]["result"]["claim_id"] += 1
    with pytest.raises(ValueError, match="mismatch"):
        score_diagnostic([gold({10: (rat(0),)})], corpus, rows,
                         [{"id": 42, "stratum": STRATA[0], "component": "a"}])


def test_controller_failure_keeps_gold_denominator_and_known_cost_lower_bound():
    bad = {"action": "rewrite", "query": "A fixture claim"}
    row, corpus = slot([bad, RuntimeError("synthetic generation failure")])
    rows = []
    for route in ROUTES:
        r = copy.deepcopy(row)
        r["route"] = r["result"]["route"] = route
        rows.append(r)
    report = score_diagnostic([gold({10: (rat(0),)})], corpus, rows,
                              [{"id": 42, "stratum": STRATA[0], "component": "a"}])
    result = report["routes"]["adaptive"]
    assert result["official_point_score"]["metrics"]["abstract_rationalized"]["relevant"] == 1
    assert result["known_tokens_including_failures"]["output_tokens"] == 50
    assert result["unknown_usage_attempts"] == 1 and result["token_totals_are_lower_bounds"]
