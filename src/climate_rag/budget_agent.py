"""Bounded observation/action controller. Heuristic coverage is NOT entailment."""

from __future__ import annotations

import re
import time
from collections.abc import Callable, Sequence
from typing import Any, Literal, Protocol

from pydantic import BaseModel, ConfigDict, Field, model_validator

from .models import RankedDocument
from .rerank import Reranker
from .tokenize import NEGATIONS, climate_tokenize
from .verification import decompose_claim, extract_constraints, normalise_claim


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)


class AgentRequest(StrictModel):
    # No benchmark label, gold IDs, or evaluation object is accepted here.
    claim_text: str = Field(min_length=1, max_length=2000, pattern=r"\S")


class AgentBudget(StrictModel):
    max_model_calls: int = Field(default=3, ge=1, le=3)
    max_tool_calls: int = Field(default=3, ge=1, le=3)
    max_output_tokens_per_call: int = Field(default=512, ge=128, le=1024)
    max_input_tokens_per_call: int = Field(default=8192, ge=1024, le=16384)
    candidate_k: int = Field(default=20, ge=5, le=100)
    context_k: int = Field(default=5, ge=1, le=10)
    timeout_seconds: float = Field(default=120.0, gt=0, le=600)

    @model_validator(mode="after")
    def widths(self) -> AgentBudget:
        if self.context_k > self.candidate_k:
            raise ValueError("context_k exceeds candidate_k")
        return self


class CitedStatement(StrictModel):
    text: str = Field(min_length=1, max_length=800, pattern=r"\S")
    evidence_id: str = Field(min_length=1, max_length=300)
    quote: str = Field(min_length=1, max_length=1500, pattern=r"\S")


class AgentDecision(StrictModel):
    action: Literal["rewrite", "rerank", "answer", "abstain"]
    reason: str = Field(min_length=1, max_length=500)
    evidence_assessment: Literal["sufficient", "insufficient", "conflicting", "unknown"] = "unknown"
    query: str | None = Field(default=None, max_length=2000)
    label: Literal["SUPPORTS", "REFUTES"] | None = None
    statements: list[CitedStatement] = Field(default_factory=list, max_length=3)

    @model_validator(mode="after")
    def payload(self) -> AgentDecision:
        if self.action == "rewrite":
            if not self.query or not self.query.strip():
                raise ValueError("rewrite requires query")
        elif self.query is not None:
            raise ValueError("query is only allowed for rewrite")
        if self.action == "answer":
            if not self.statements or self.label is None or self.evidence_assessment != "sufficient":
                raise ValueError("answer requires label and cited statements")
        elif self.statements or self.label is not None:
            raise ValueError("only answer may contain statements/label")
        return self


class DecisionProvider(Protocol):
    name: str
    kind: str

    def decide(self, observation: dict[str, Any], budget: AgentBudget) -> dict[str, Any]: ...


def evidence_ledger(claim: str, items: Sequence[dict[str, Any]]) -> dict[str, Any]:
    """Exact constraint occurrence + lexical union; no labels or gold in the API."""
    constraints = extract_constraints(claim)
    # Sentence-initial function words are not useful entity constraints.
    constraints["entities"] = [
        x for x in constraints["entities"] if x.casefold() not in
        {"the", "this", "that", "there", "what", "how", "does", "since"}
    ]
    checks: dict[str, list[dict[str, Any]]] = {}
    for kind, values in constraints.items():
        checks[kind] = []
        for value in values:
            pattern = re.compile(r"(?<!\w)" + re.escape(value) + r"(?!\w)", re.I)
            checks[kind].append({
                "value": value,
                "evidence_ids": [x["evidence_id"] for x in items
                                 if pattern.search(x["text"])],
            })
    tokens = set(climate_tokenize(claim))
    seen = set().union(*(set(climate_tokenize(x["text"])) for x in items))
    missing = {kind: [x["value"] for x in rows if not x["evidence_ids"]]
               for kind, rows in checks.items()}
    coverage = len(tokens & seen) / len(tokens) if tokens else 0.0
    return {
        "method": "heuristic_regex_exact_occurrence_and_lexical_union",
        "constraints": checks, "missing": missing, "lexical_coverage": coverage,
        "retrieved_count": len(items),
        "low_coverage": not items or coverage < 0.65 or any(missing.values()),
        "semantic_sufficiency": "not_established_by_ledger",
        "contradiction": "not_established_by_token_presence",
    }


def check_answer(
    decision: AgentDecision, items: Sequence[dict[str, Any]],
) -> list[str]:
    """Check source IDs, exact quotes and numbers, not semantic entailment."""
    by_id = {row["evidence_id"]: row for row in items}
    errors: list[str] = []
    for statement in decision.statements:
        row = by_id.get(statement.evidence_id)
        if row is None:
            errors.append("unknown_citation")
            continue
        if statement.quote not in row["text"]:
            errors.append("quote_not_exact")
        numbers = set(extract_constraints(statement.text)["numbers"])
        quoted_numbers = set(extract_constraints(statement.quote)["numbers"])
        if numbers - quoted_numbers:
            errors.append("uncited_numeric_value")
    return sorted(set(errors))


class HeuristicAbstainingProvider:
    """CPU control exercise only. Never claims model reasoning or truth."""

    name = "heuristic-abstaining-control"
    kind = "heuristic_no_model"

    def decide(self, observation: dict[str, Any], budget: AgentBudget) -> dict[str, Any]:
        del budget
        if "rewrite" in observation["allowed_actions"]:
            parts = decompose_claim(observation["claim_text"], max_queries=1)
            constraints = extract_constraints(observation["claim_text"])
            suffix = " ".join(sorted(set(sum(constraints.values(), []))))
            query = normalise_claim((parts[0] if parts else "") + " " + suffix)
            if query and query.casefold() not in observation["queries_casefold"]:
                return {"action": "rewrite", "query": query,
                        "reason": "Heuristic coverage gap; one bounded query."}
        if "rerank" in observation["allowed_actions"]:
            return {"action": "rerank", "reason": "Heuristic optional ranking."}
        return {"action": "abstain", "reason": "No semantic model configured."}


class BudgetedEvidenceAgent:
    def __init__(
        self, retrieve: Callable[[str], dict[str, Any]], provider: DecisionProvider,
        *, budget: AgentBudget | None = None, reranker: Reranker | None = None,
        clock: Callable[[], float] = time.perf_counter,
    ) -> None:
        self.retrieve = retrieve
        self.provider = provider
        self.budget = budget or AgentBudget()
        self.reranker = reranker
        self.clock = clock

    def run(
        self, request: dict[str, Any], *,
        strategy: Literal["fixed_retrieval", "fixed_rerank", "adaptive"] = "adaptive",
    ) -> dict[str, Any]:
        claim = normalise_claim(AgentRequest.model_validate(request).claim_text)
        if strategy not in {"fixed_retrieval", "fixed_rerank", "adaptive"}:
            raise ValueError("unknown strategy")
        started = self.clock()
        items: list[dict[str, Any]] = []
        queries = [claim]
        events: list[dict[str, Any]] = []
        tool_calls = model_calls = policy_calls = 0
        retrieval_calls = rerank_calls = rerank_pairs = 0
        usage = {"input_tokens": 0, "output_tokens": 0}
        usage_known = True
        pending_model_usage = False
        rewritten = reranked = False
        answer: dict[str, Any] | None = None
        reason = "budget_exhausted"

        def time_left() -> float:
            return self.budget.timeout_seconds - (self.clock() - started)

        def rank() -> None:
            nonlocal items, tool_calls, reranked, rerank_calls, rerank_pairs
            assert self.reranker is not None
            before = {x["evidence_id"]: x for x in items}
            candidates = [RankedDocument(x["evidence_id"],
                          float(x["retrieval"]["score"]), i, x["text"],
                          x["retrieval"]["route"]) for i, x in enumerate(items, 1)]
            tool_calls += 1
            rerank_calls += 1
            rerank_pairs += len(candidates)
            begin = self.clock()
            rows = self.reranker.rerank(claim, candidates, self.budget.candidate_k)
            if len({x.evidence_id for x in rows}) != len(rows):
                raise ValueError("reranker returned duplicate IDs")
            result = []
            for row in rows:
                original = before[row.evidence_id]
                if row.text != original["text"]:
                    raise ValueError("reranker changed source text")
                result.append({**original, "retrieval": {
                    "route": self.reranker.name, "rank": row.rank, "score": row.score,
                }})
            items = result
            reranked = True
            events.append({"stage": "rerank", "candidate_count": len(candidates),
                           "elapsed_ms": (self.clock() - begin) * 1000})

        try:
            tool_calls += 1
            retrieval_calls += 1
            begin = self.clock()
            packet = self.retrieve(claim)
            items = packet["items"][:self.budget.candidate_k]
            events.append({"stage": "retrieve", "query": claim,
                           "elapsed_ms": (self.clock() - begin) * 1000})
            if time_left() <= 0:
                raise TimeoutError("deadline")
            if strategy == "fixed_rerank":
                if self.reranker is None or tool_calls >= self.budget.max_tool_calls:
                    raise ValueError("fixed rerank unavailable within budget")
                rank()
            for _ in range(self.budget.max_model_calls):
                if time_left() <= 0:
                    raise TimeoutError("deadline")
                context = items[:self.budget.context_k]
                ledger = evidence_ledger(claim, context)
                allowed = ["abstain"] + (["answer"] if context else [])
                can_tool = tool_calls < self.budget.max_tool_calls
                # Reserve a final observation/answer turn after any tool action.
                if strategy == "adaptive" and policy_calls + 1 < self.budget.max_model_calls:
                    if can_tool and not rewritten and not reranked and ledger["low_coverage"]:
                        allowed.append("rewrite")
                    if can_tool and not reranked and self.reranker and items:
                        allowed.append("rerank")
                observation = {
                    "claim_text": claim, "ledger": ledger,
                    "evidence": [{"evidence_id": x["evidence_id"], "text": x["text"],
                                  "citation_id": x["citation_id"]} for x in context],
                    "allowed_actions": allowed,
                    "queries_casefold": [x.casefold() for x in queries],
                    "remaining_seconds": time_left(),
                    "remaining_tool_calls": self.budget.max_tool_calls - tool_calls,
                }
                policy_calls += 1
                if self.provider.kind == "local_model":
                    model_calls += 1
                    pending_model_usage = True
                begin = self.clock()
                raw = self.provider.decide(observation, self.budget)
                response_usage = raw.get("usage", {})
                for key in usage:
                    usage[key] += int(response_usage.get(key, 0))
                if pending_model_usage and not all(key in response_usage for key in usage):
                    usage_known = False
                pending_model_usage = False
                decision = AgentDecision.model_validate(raw.get("decision", raw))
                events.append({"stage": "decision", "decision": decision.model_dump(),
                               "elapsed_ms": (self.clock() - begin) * 1000,
                               "ledger": ledger})
                if self.provider.kind == "local_model" and raw.get("diagnostics"):
                    events[-1]["generation_diagnostics"] = raw["diagnostics"]
                if time_left() <= 0:
                    raise TimeoutError("deadline")
                if decision.action not in allowed:
                    reason = "disallowed_action"
                    break
                if decision.action == "abstain":
                    reason = decision.reason
                    break
                if decision.action == "answer":
                    errors = check_answer(decision, context)
                    if errors:
                        reason = "answer_validation:" + ",".join(errors)
                    else:
                        answer = decision.model_dump()
                        reason = "source_and_numeric_checks_passed_semantics_unverified"
                    break
                if decision.action == "rerank":
                    rank()
                    continue
                query = normalise_claim(decision.query or "")
                if query.casefold() in [x.casefold() for x in queries]:
                    reason = "query_loop"
                    break
                retained = extract_constraints(query)
                original = extract_constraints(claim)
                if any(set(original[k]) - set(retained[k]) for k in ("years", "numbers")):
                    reason = "rewrite_dropped_numeric_constraint"
                    break
                if any(not re.search(r"(?<!\w)" + re.escape(row["value"]) + r"(?!\w)", query, re.I)
                       for row in ledger["constraints"]["entities"]):
                    reason = "rewrite_dropped_entity"
                    break
                qualifiers = set(NEGATIONS) | {"never", "every", "all", "only", "more", "less"}
                source_tokens, query_tokens = set(climate_tokenize(claim)), set(climate_tokenize(query))
                if (source_tokens & qualifiers) != (query_tokens & qualifiers):
                    reason = "rewrite_changed_qualifier"
                    break
                if len(source_tokens & query_tokens) / max(1, len(source_tokens)) < 0.5:
                    reason = "rewrite_lexical_drift"
                    break
                rewritten = True
                queries.append(query)
                tool_calls += 1
                retrieval_calls += 1
                begin = self.clock()
                extra = self.retrieve(query)["items"][:self.budget.candidate_k]
                if not ({x["evidence_id"] for x in extra} - {x["evidence_id"] for x in items}):
                    events.append({"stage": "rewrite_retrieve", "query": query,
                                   "elapsed_ms": (self.clock() - begin) * 1000,
                                   "new_evidence_count": 0})
                    reason = "no_new_evidence"
                    break
                # Equal-weight rank fusion, never mix raw BM25 and model scores.
                merged = {x["evidence_id"]: x for x in [*items, *extra]}
                scores: dict[str, float] = {}
                for ranking in (items, extra):
                    for i, item in enumerate(ranking, 1):
                        key = item["evidence_id"]
                        scores[key] = scores.get(key, 0.0) + 1 / (60 + i)
                ordered = sorted(merged, key=lambda key: (-scores[key], key))
                items = [{**merged[key], "retrieval": {
                    "route": "multi-query-rrf", "rank": i, "score": scores[key],
                }} for i, key in enumerate(ordered[:self.budget.candidate_k], 1)]
                events.append({"stage": "rewrite_retrieve", "query": query,
                               "elapsed_ms": (self.clock() - begin) * 1000})
        except Exception as exc:
            if pending_model_usage:
                failed_usage = getattr(exc, "usage", None)
                if failed_usage is None:
                    usage_known = False
                else:
                    for key in usage:
                        usage[key] += int(failed_usage[key])
            # No provider response/paths/credentials in public failure details.
            reason = "deadline_exceeded" if isinstance(exc, TimeoutError) else "stage_failed"
            failure: dict[str, Any] = {"stage": "failure", "error_type": type(exc).__name__}
            if type(exc).__name__ == "GeneratedResponseError" and getattr(exc, "diagnostics", None):
                failure["generation_diagnostics"] = getattr(exc, "diagnostics")
            events.append(failure)
        return {
            "schema_version": "1.0", "claim_text": claim, "strategy": strategy,
            "provider": self.provider.name, "provider_kind": self.provider.kind,
            "answer": answer, "status": "answered" if answer else "abstained",
            "reason": reason, "queries": queries,
            "candidate_evidence_ids": [x["evidence_id"] for x in items],
            "delivered_evidence_ids": [] if reason in {"deadline_exceeded", "stage_failed"}
            else [x["evidence_id"] for x in items],
            "context_evidence_ids": [x["evidence_id"] for x in items[:self.budget.context_k]],
            "ledger": evidence_ledger(claim, items[:self.budget.context_k]),
            "tool_calls": tool_calls, "model_calls": model_calls,
            "policy_calls": policy_calls, "usage": usage, "usage_known": usage_known,
            "generation_calls": model_calls,
            "model_calls_scope": "generation provider only; see separate retrieval/rerank work counters",
            "retrieval_calls": retrieval_calls, "rerank_calls": rerank_calls,
            "rerank_candidate_pairs": rerank_pairs,
            "elapsed_ms": (self.clock() - started) * 1000, "events": events,
            "semantic_supportability": "not_independently_evaluated",
            "rewrite_equivalence": "heuristic_constraint_guard_not_semantic_proof",
            "timeout_semantics": "late results rejected; local generation has max_time; no thread kill",
        }
