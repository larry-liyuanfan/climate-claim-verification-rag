"""Synthetic implementation contracts only, not real-model accuracy."""
import copy

import pytest

from climate_rag.fair_acquisition import COVERAGE_PROTOCOL

from climate_rag.semantic_proxy import (
    PROTOCOL, ROUTES, aggregate, cited_pair, classify, matrix, text_sha,
)
from score_saved_citation_nli import score_saved


def row(arm, label="SUPPORTS"):
    text, claim = "Synthetic ice shrinks.", "Synthetic ice shrinks."
    return {"route": arm, "task_id": "synthetic", "initial_frame": {"immutable_claim": claim},
        "immutable_claim_sha256": text_sha(claim), "visible_source_ids": {"c1": "source"},
        "outcome": "ids_validated_semantics_unmeasured",
        "answer": {"label": label, "citations": [{"text": text, "source_id": "source",
            "sentence_index": 0, "text_sha256": text_sha(text)}]},
        "generation_attempts": [{"stage": "verdict", "status": "valid_decision",
            "decision": {"action": "answer", "label": label, "sentence_ids": ["c1:0"]},
            "visible_sentence_sha256": {"c1:0": text_sha(text)},
            "observation": {"immutable_claim": claim, "current_citable": [{"sentence_id": "c1:0", "text": text}]}}]}


def test_mapping_no_negation_and_actual_sentences():
    pair = cited_pair(row(ROUTES[0], "REFUTES"))
    assert pair["premise"] == pair["hypothesis"] == "Synthetic ice shrinks."
    assert pair["target"] == "contradiction"
    assert classify([0.1, 0.85, 0.05], "entailment")["passes_proxy"]
    assert not classify([0.1, 0.7, 0.2], "entailment")["passes_proxy"]


@pytest.mark.parametrize("change", ["text", "hash", "index", "source", "label", "phase", "claim"])
def test_provenance_tampering_fails(change):
    value = row(ROUTES[0])
    if change in {"text", "source", "index", "hash"}:
        key = {"source": "source_id", "index": "sentence_index", "hash": "text_sha256"}.get(change, change)
        value["answer"]["citations"][0][key] = 3 if change == "index" else "changed"
    elif change == "label":
        value["answer"]["label"] = "REFUTES"
    elif change == "claim":
        value["generation_attempts"][-1]["observation"]["immutable_claim"] = "different claim"
    else:
        value["generation_attempts"][-1]["stage"] = "gate"
    with pytest.raises(ValueError):
        cited_pair(value)


def test_matrix_and_all_task_denominator_with_errors_and_abstention():
    run = {"protocol": PROTOCOL, "runs": [row(arm) for arm in ROUTES]}
    run["runs"][1].update(answer=None, outcome="model_abstention:insufficient_evidence")
    run["runs"][2].update(answer=None, outcome="validation_repair_exhausted")

    class Judge:
        def score(self, premise, hypothesis):
            assert premise == hypothesis
            return [0.1, 0.85, 0.05], 8

    report, scores = score_saved(run, Judge())
    assert report["measurement_cost"]["nli_calls"] == 1
    assert report["routes"][ROUTES[0]]["joint_citation_proxy_pass_all_tasks"] == 1
    assert report["routes"][ROUTES[1]]["status"] == {"abstained": 1}
    assert report["routes"][ROUTES[2]]["status"] == {"execution_failure": 1}
    assert all(report["routes"][r]["denominator_all_tasks"] == 1 for r in ROUTES)
    assert scores[2]["status"] != "abstained"
    bad = copy.deepcopy(run)
    bad["runs"].pop()
    with pytest.raises(ValueError, match="incomplete"):
        matrix(bad)
    with pytest.raises(ValueError, match="denominator"):
        matrix(run, expected_tasks=32)
    bad = copy.deepcopy(run)
    bad["runs"][1]["immutable_claim_sha256"] = "changed"
    with pytest.raises(ValueError, match="cross_route"):
        matrix(bad)


def test_overflow_is_not_neutral_or_correct_abstention():
    class Judge:
        def score(self, premise, hypothesis):
            raise OverflowError()

    report, scores = score_saved({"protocol": PROTOCOL, "runs": [row(a) for a in ROUTES]}, Judge())
    assert all(r["status"] == "overlength" and "joint_citations" not in r for r in scores)
    assert all(r["joint_citation_proxy_pass_all_tasks"] == 0 for r in report["routes"].values())


def test_invalid_probabilities_and_roster_refused():
    with pytest.raises(ValueError):
        classify([0.1, float("nan"), 0.2], "entailment")
    with pytest.raises(ValueError, match="roster"):
        aggregate([{"route": r, "slot_position": 1, "status": "overlength"} for r in ROUTES], tasks=1)


def test_coverage_protocol_requires_explicit_binding_and_reaches_scoring():
    run = {"protocol": COVERAGE_PROTOCOL, "runs": [row(a) for a in ROUTES]}
    with pytest.raises(ValueError, match="protocol"):
        matrix(run)
    with pytest.raises(ValueError, match="protocol"):
        matrix(run, expected_protocol="arbitrary")

    class Judge:
        def score(self, premise, hypothesis):
            assert premise == hypothesis
            return [0.1, 0.85, 0.05], 8

    report, records = score_saved(run, Judge(), expected_protocol=COVERAGE_PROTOCOL)
    assert len(records) == 3
    assert report["measurement_cost"]["nli_calls"] == 3
