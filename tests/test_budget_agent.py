from __future__ import annotations

from typing import Any

import pytest
from pydantic import ValidationError

from climate_rag.budget_agent import (
    AgentBudget, BudgetedEvidenceAgent, HeuristicAbstainingProvider, evidence_ledger,
)
from climate_rag.local_agent_model import GeneratedResponseError, verify_model_files
from climate_rag.rerank import DeterministicFeatureReranker


def item(key: str = "a", text: str = "Arctic ice declined in 2010.") -> dict[str, Any]:
    return {"evidence_id": key, "citation_id": "sha256:corpus:" + key, "text": text,
            "retrieval": {"score": 1.0, "rank": 1, "route": "fixture"}}


class Scripted:
    name = "scripted-contract-fixture"
    kind = "fixture"

    def __init__(self, decisions: list[dict[str, Any]]) -> None:
        self.decisions = iter(decisions)
        self.observed: list[dict[str, Any]] = []

    def decide(self, observation: dict[str, Any], budget: AgentBudget) -> dict[str, Any]:
        self.observed.append(observation)
        return next(self.decisions)


def answer(**changes: Any) -> dict[str, Any]:
    return {"action": "answer", "reason": "fixture, not entailment evaluation",
            "evidence_assessment": "sufficient", "label": "SUPPORTS",
            "statements": [{"text": "Arctic ice declined in 2010.", "evidence_id": "a",
                            "quote": "Arctic ice declined in 2010.", **changes}]}


def test_request_rejects_gold_before_any_side_effect() -> None:
    provider = Scripted([])
    agent = BudgetedEvidenceAgent(lambda _: pytest.fail("must not retrieve"), provider)
    with pytest.raises(ValidationError):
        agent.run({"claim_text": "claim", "gold_evidence_ids": ["a"]})
    assert provider.observed == []


def test_context_ledger_excludes_outside_top_k() -> None:
    ledger = evidence_ledger("Arctic 2010 50%", [item(text="Arctic 2011 150%")])
    assert ledger["missing"]["years"] == ["2010"]
    assert "50%" in ledger["missing"]["numbers"]
    provider = Scripted([{"action": "abstain", "reason": "missing"}])
    agent = BudgetedEvidenceAgent(lambda _: {"items": [item(text="unrelated"), item("b")]},
                                 provider, budget=AgentBudget(context_k=1))
    agent.run({"claim_text": "Arctic ice declined in 2010."})
    assert provider.observed[0]["ledger"]["missing"]["years"] == ["2010"]


@pytest.mark.parametrize("changes,reason", [
    ({"evidence_id": "unknown"}, "unknown_citation"),
    ({"quote": "Arctic ice declined in 2011."}, "quote_not_exact"),
    ({"text": "Arctic ice declined by 90%."}, "uncited_numeric_value"),
])
def test_answer_fail_closed(changes: dict[str, Any], reason: str) -> None:
    result = BudgetedEvidenceAgent(lambda _: {"items": [item()]},
                                  Scripted([answer(**changes)])).run({"claim_text": "ice"})
    assert result["answer"] is None and reason in result["reason"]


def test_exact_citation_not_semantic_success() -> None:
    result = BudgetedEvidenceAgent(lambda _: {"items": [item()]}, Scripted([answer()])).run(
        {"claim_text": "Arctic ice declined in 2010."})
    assert result["answer"] and result["model_calls"] == 0
    assert result["semantic_supportability"] == "not_independently_evaluated"


@pytest.mark.parametrize("changes", [{"quote": " "}, {"text": "\t"}])
def test_whitespace_citation_rejected(changes: dict[str, Any]) -> None:
    result = BudgetedEvidenceAgent(lambda _: {"items": [item()]},
                                  Scripted([answer(**changes)])).run({"claim_text": "ice"})
    assert result["answer"] is None and result["reason"] == "stage_failed"


def test_without_polarity_cannot_be_removed() -> None:
    result = BudgetedEvidenceAgent(lambda _: {"items": []}, Scripted([
        {"action": "rewrite", "query": "Volcanic eruptions cool temperatures reducing sunlight",
         "reason": "fixture"},
    ])).run({"claim_text": "Volcanic eruptions cool temperatures without reducing sunlight"})
    assert result["reason"] == "rewrite_changed_qualifier"


@pytest.mark.parametrize("query,reason", [
    ("Arctic ice declined", "rewrite_dropped_numeric_constraint"),
    ("Antarctic ice declined in 2010", "rewrite_dropped_entity"),
    ("Arctic ice never declined in 2010", "rewrite_changed_qualifier"),
    ("Arctic ice declined in 2010", "query_loop"),
])
def test_rewrite_constraint_guards(query: str, reason: str) -> None:
    result = BudgetedEvidenceAgent(lambda _: {"items": []}, Scripted([
        {"action": "rewrite", "query": query, "reason": "fixture"},
    ])).run({"claim_text": "Arctic ice declined in 2010"})
    assert result["reason"] == reason and result["tool_calls"] == 1


def test_one_rewrite_then_rerank_then_answer() -> None:
    provider = Scripted([
        {"action": "rewrite", "query": "Arctic ice declined in 2010 evidence", "reason": "gap"},
        {"action": "rerank", "reason": "rank"}, answer(),
    ])
    packets = iter([{"items": []}, {"items": [item()]}])
    result = BudgetedEvidenceAgent(lambda _: next(packets), provider,
                                  reranker=DeterministicFeatureReranker()).run(
        {"claim_text": "Arctic ice declined in 2010"})
    assert result["answer"] and result["tool_calls"] == 3
    assert result["policy_calls"] == 3 and result["model_calls"] == 0
    assert "rewrite" not in provider.observed[1]["allowed_actions"]
    assert provider.observed[2]["allowed_actions"] == ["abstain", "answer"]


def test_no_new_evidence_stops_without_loop() -> None:
    result = BudgetedEvidenceAgent(lambda _: {"items": []}, Scripted([
        {"action": "rewrite", "query": "Arctic ice evidence", "reason": "gap"},
    ])).run({"claim_text": "Arctic ice"})
    assert result["reason"] == "no_new_evidence" and result["tool_calls"] == 2


def test_disallowed_action_no_second_tool() -> None:
    result = BudgetedEvidenceAgent(lambda _: {"items": []}, Scripted([
        {"action": "rewrite", "query": "ice evidence", "reason": "gap"},
    ]), budget=AgentBudget(max_tool_calls=1)).run({"claim_text": "ice"})
    assert result["reason"] == "disallowed_action" and result["tool_calls"] == 1


def test_retrieval_failure_counts_attempt_and_redacts_error() -> None:
    def fail(_: str) -> dict[str, Any]:
        raise RuntimeError("sensitive-path-must-not-escape")
    result = BudgetedEvidenceAgent(fail, Scripted([])).run({"claim_text": "ice"})
    assert result["tool_calls"] == 1 and result["model_calls"] == 0
    assert "sensitive-path" not in str(result)


@pytest.mark.parametrize("known", [True, False])
def test_failed_generation_is_not_free(known: bool) -> None:
    class FailedModel:
        name = "local-test-double"
        kind = "local_model"

        def decide(self, observation: dict[str, Any], budget: AgentBudget) -> dict[str, Any]:
            if known:
                raise GeneratedResponseError({"input_tokens": 120, "output_tokens": 25})
            raise RuntimeError("failed before returned accounting")
    result = BudgetedEvidenceAgent(lambda _: {"items": [item()]}, FailedModel()).run(
        {"claim_text": "ice"})
    assert result["model_calls"] == 1 and result["usage_known"] is known
    if known:
        assert result["usage"] == {"input_tokens": 120, "output_tokens": 25}


def test_deadline_rejects_late_tool_result() -> None:
    ticks = iter([0.0, 0.0, 2.0, 2.0, 2.0])
    provider = Scripted([])
    result = BudgetedEvidenceAgent(lambda _: {"items": [item()]}, provider,
        budget=AgentBudget(timeout_seconds=1.0), clock=lambda: next(ticks)).run(
            {"claim_text": "ice"})
    assert result["reason"] == "deadline_exceeded" and not provider.observed
    assert result["candidate_evidence_ids"] == ["a"]
    assert result["delivered_evidence_ids"] == []


def test_no_semantic_model_means_abstention() -> None:
    result = BudgetedEvidenceAgent(lambda _: {"items": [item()]},
        HeuristicAbstainingProvider()).run({"claim_text": "Arctic ice declined in 2010."})
    assert result["answer"] is None and result["model_calls"] == 0


def test_model_digest_gate_before_import(tmp_path: Any) -> None:
    with pytest.raises(ValueError, match="weights"):
        verify_model_files(tmp_path, {})
