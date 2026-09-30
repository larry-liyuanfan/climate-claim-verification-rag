"""Versioned bounded comparison controller, copied from frozen source80 v1.

Only common prompt-fitting hook, concrete envelope-error feedback, and protocol
receipt differ. Route policies, actual usage accounting and tools are unchanged.
"""

from __future__ import annotations

import hashlib
import json
import time
from collections.abc import Callable, Sequence
from typing import Any

from pydantic import ValidationError

from .agent_protocol import ModelResponseValidationError
from .agent_v3 import (
    Source,
    V3Budget,
    V3Provider,
    deterministic_extra_query,
    valid_search_query,
)
from .verification import normalise_claim
from .scifact_terminal import PROTOCOL, action_schema, parse_action, render_answer


class SciFactBoundedAgent:
    def __init__(
        self,
        provider: V3Provider,
        retrieve: Callable[[str, int], Sequence[Source]],
        *,
        rerank: Callable[[str, Sequence[Source]], Sequence[Source]] | None = None,
        budget: V3Budget | None = None,
        clock: Callable[[], float] = time.monotonic,
        packing_count: Callable[[dict[str, Any], dict[str, Any]], int] | None = None,
    ):
        if getattr(provider, "terminal_protocol", None) != PROTOCOL:
            raise ValueError("SciFact requires its own provider prompt contract")
        self.provider, self.retrieve, self.rerank = provider, retrieve, rerank
        self.budget, self.clock = budget or V3Budget(), clock
        self.packing_count = packing_count or provider.count_prompt

    def run(self, claim: str, route: str = "adaptive") -> dict[str, Any]:
        if route not in {
            "fixed_retrieval",
            "fixed_rerank",
            "deterministic_extra",
            "adaptive",
        }:
            raise ValueError("unknown route")
        claim = normalise_claim(claim)
        if not claim or len(claim) > 2000:
            raise ValueError("invalid claim")
        b = self.budget
        start = self.clock()
        sources: dict[str, Source] = {}
        aliases: dict[str, str] = {}
        candidates: list[str] = []
        selected: list[str] = []
        events: list[dict[str, Any]] = []
        attempts: list[dict[str, Any]] = []
        usage = {"input_tokens": 0, "output_tokens": 0}
        tool_calls = rerank_pairs = read_input_tokens = repairs = 0
        rewritten = reranked = False
        feedback: str | None = None
        reads: set[tuple[str, ...]] = set()
        answer: dict[str, Any] | None = None
        outcome = "generation_budget_exhausted"

        def left() -> float:
            return b.timeout_seconds - (self.clock() - start)

        def register(rows: Sequence[Source]) -> list[str]:
            if len({r.source_id for r in rows}) != len(rows):
                raise RuntimeError("duplicate_source")
            result = []
            for row in rows:
                if (
                    row.source_id in sources
                    and sources[row.source_id].text_sha256 != row.text_sha256
                ):
                    raise RuntimeError("source_text_mutation")
                if row.source_id not in sources:
                    aliases[row.source_id] = f"c{len(aliases)}"
                sources[row.source_id] = row
                result.append(aliases[row.source_id])
            return result

        def source(alias: str) -> Source:
            return sources[
                next(key for key, value in aliases.items() if value == alias)
            ]

        def tool(
            kind: str,
            ids: list[str] | None = None,
            query: str | None = None,
            model_selected: bool = False,
        ) -> None:
            nonlocal candidates, selected, tool_calls, rerank_pairs, rewritten, reranked
            if tool_calls >= b.max_tools or left() <= 0:
                raise RuntimeError("tool_budget_or_deadline")
            before = list(selected)
            tool_calls += 1
            began = self.clock()
            event: dict[str, Any] = {
                "tool": kind,
                "model_selected": model_selected,
                "before_context": before,
                "status": "running",
                "started": began,
                "query_sha256": hashlib.sha256((query or claim).encode()).hexdigest()
                if kind in {"retrieve", "rewrite"}
                else None,
            }
            events.append(event)
            if kind in {"retrieve", "rewrite"}:
                try:
                    rows = list(self.retrieve(query or claim, b.candidate_k))[
                        : b.candidate_k
                    ]
                except Exception as exc:
                    raise RuntimeError("retrieval_failure") from exc
                extra = register(rows)
                if kind == "rewrite":
                    rewritten = True
                    prior_rank = {s: 1 / (60 + i + 1) for i, s in enumerate(candidates)}
                    for i, s in enumerate(extra):
                        prior_rank[s] = prior_rank.get(s, 0) + 1 / (60 + i + 1)
                    candidates = sorted(prior_rank, key=lambda s: (-prior_rank[s], s))[
                        : b.candidate_k
                    ]
                else:
                    candidates = extra
                selected = candidates[: b.context_k]
            elif kind == "rerank":
                if self.rerank is None:
                    raise RuntimeError("reranker_unavailable")
                rows = [source(s) for s in candidates]
                rerank_pairs += len(rows)
                try:
                    result = register(list(self.rerank(claim, rows)))
                except Exception as exc:
                    raise RuntimeError("rerank_failure") from exc
                if set(result) != set(candidates) or len(result) != len(candidates):
                    raise RuntimeError("reranker_candidate_mismatch")
                candidates, selected = result, result[: b.context_k]
                reranked = True
            elif kind == "read":
                if (
                    ids is None
                    or not ids
                    or len(ids) > b.context_k
                    or len(set(ids)) != len(ids)
                    or not set(ids) <= set(candidates)
                ):
                    raise ValueError("read_outside_candidate_or_duplicate")
                if tuple(ids) in reads or ids == selected:
                    raise ValueError("read_loop")
                selected = list(ids)
                reads.add(tuple(ids))
            else:
                raise ValueError("unknown_tool")
            event.update(
                {
                    "status": "completed",
                    "requested_context": list(selected),
                    "candidate_ids": list(candidates),
                    "elapsed_ms": (self.clock() - began) * 1000,
                    "source_sha256": {s: source(s).text_sha256 for s in candidates},
                }
            )
            event.pop("started")
            if left() <= 0:
                raise RuntimeError("deadline_after_tool")

        def pack(
            allowed: list[str],
        ) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any], int]:
            # Fit the FULL prompt, including system/schema/previews/repair text.
            observation: dict[str, Any] = {
                "immutable_claim": claim,
                "allowed_actions": allowed,
                "current_citable": [],
                "preview_only": [],
                "feedback": feedback,
                "remaining_calls": b.max_calls - len(attempts),
                "remaining_tools": b.max_tools - tool_calls,
            }
            visible: dict[str, Any] = {}
            bare = [a for a in allowed if a != "answer"]
            schema = action_schema(bare, candidates, [], b.context_k)
            if self.packing_count(observation, schema) > b.max_input_tokens:
                raise ValueError("prompt_overhead_exceeds_budget")
            for alias in selected:
                doc = source(alias)
                for index, sentence in enumerate(doc.sentences):
                    sid = f"{alias}:{index}"
                    entry = {"sentence_id": sid, "text": sentence}
                    observation["current_citable"].append(entry)
                    next_schema = action_schema(
                        allowed, candidates, [*visible, sid], b.context_k
                    )
                    if (
                        self.packing_count(observation, next_schema)
                        > b.max_input_tokens
                    ):
                        observation["current_citable"].pop()
                        break
                    schema = next_schema
                    visible[sid] = {
                        "source_id": doc.source_id,
                        "sentence_index": index,
                        "text": sentence,
                        "text_sha256": hashlib.sha256(sentence.encode()).hexdigest(),
                        "source_text_sha256": doc.text_sha256,
                    }
            if not visible:
                observation["allowed_actions"] = bare
                schema = action_schema(bare, candidates, [], b.context_k)
            # Previews get only remaining capacity, never displace full context.
            for alias in candidates:
                if alias in selected:
                    continue
                doc = source(alias)
                preview: dict[str, Any] = {
                    "source_id": alias,
                    "preview": (doc.title + " " + doc.sentences[0])[:128],
                    "citable": False,
                }
                observation["preview_only"].append(preview)
                if self.packing_count(observation, schema) > b.max_input_tokens:
                    observation["preview_only"].pop()
                    break
            count = self.provider.count_prompt(observation, schema)
            if count > b.max_input_tokens:
                raise ValueError("packed_prompt_exceeds_budget")
            return observation, schema, visible, count

        try:
            tool("retrieve")
            if route == "deterministic_extra":
                # More retrieval work, not replacement by a lower rank window.
                extra_query = deterministic_extra_query(claim)
                if extra_query:
                    tool("rewrite", query=extra_query)
                else:
                    events.append(
                        {
                            "status": "skipped",
                            "tool": "deterministic_extra_query",
                            "model_selected": False,
                            "reason": "no_query_within_shared_contract",
                        }
                    )
            if route in {"fixed_rerank", "deterministic_extra"}:
                tool("rerank")
            for _ in range(b.max_calls):
                if left() <= 0:
                    outcome = "deadline"
                    break
                allowed = ["abstain"] + (["answer"] if selected else [])
                if (
                    route == "adaptive"
                    and len(attempts) + 1 < b.max_calls
                    and tool_calls < b.max_tools
                ):
                    if candidates:
                        allowed.append("read")
                    if not rewritten:
                        allowed.append("rewrite")
                    if candidates and self.rerank is not None and not reranked:
                        allowed.append("rerank")
                observation, schema, visible, prompt_tokens = pack(allowed)
                if left() <= 0:
                    outcome = "deadline_during_prompt_assembly"
                    break
                allowed = observation["allowed_actions"]
                read_input_tokens += prompt_tokens
                record: dict[str, Any] = {
                    "status": "pending",
                    "input_prompt_tokens": prompt_tokens,
                    "visible_sentence_sha256": {
                        k: v["text_sha256"] for k, v in visible.items()
                    },
                    "allowed_actions": list(allowed),
                    "usage_known": False,
                    "usage": {"input_tokens": 0, "output_tokens": 0},
                    "displayed_source_count": len({k.split(":")[0] for k in visible}),
                    "displayed_sentence_count": len(visible),
                    "requested_context": list(selected),
                    "visible_text_characters": sum(
                        len(v["text"]) for v in visible.values()
                    ),
                    "context_payload_tokens": self.provider.count_text(
                        json.dumps(
                            observation["current_citable"],
                            ensure_ascii=False,
                            separators=(",", ":"),
                        )
                    ),
                    "preview_payload_tokens": self.provider.count_text(
                        json.dumps(
                            observation["preview_only"],
                            ensure_ascii=False,
                            separators=(",", ":"),
                        )
                    ),
                }
                attempts.append(record)
                try:
                    response = self.provider.generate(
                        observation, schema, b.max_output_tokens, left()
                    )
                    record["usage"] = response["usage"]
                    record["usage_known"] = True
                    record["status"] = "response_received"
                    record["diagnostics"] = response.get("diagnostics", {})
                    raw = response["raw"]
                    for key in usage:
                        value = response["usage"][key]
                        if type(value) is not int or value < 0:
                            raise ValueError("invalid_usage_ledger")
                        usage[key] += value
                    if (
                        response["usage"]["input_tokens"] != prompt_tokens
                        or response["usage"]["output_tokens"] > b.max_output_tokens
                    ):
                        raise RuntimeError("provider_token_budget_mismatch")
                    if left() <= 0:
                        raise RuntimeError("deadline_after_generation")
                    decision = parse_action(
                        json.loads(raw), allowed, visible, candidates, b.context_k
                    )
                    record.update(status="valid_decision", action=decision["action"])
                    feedback = None
                    if decision["action"] == "abstain":
                        outcome = "model_abstention:" + decision["reason"]
                        break
                    if decision["action"] == "answer":
                        answer = render_answer(decision, visible)
                        outcome = "ids_validated_semantics_unmeasured"
                        break
                    if decision["action"] == "rewrite":
                        query = normalise_claim(decision["query"])
                        if not valid_search_query(claim, query):
                            raise ValueError("rewrite_constraint_or_loop")
                        tool("rewrite", query=query, model_selected=True)
                    else:
                        if decision["action"] == "read" and (
                            tuple(decision["source_ids"]) in reads
                            or decision["source_ids"] == selected
                        ):
                            raise ValueError("read_loop")
                        tool(
                            decision["action"],
                            decision.get("source_ids"),
                            model_selected=True,
                        )
                    feedback = "tool_completed: inspect current_citable; no semantic conclusion implied"
                except ModelResponseValidationError as exc:
                    record.update(
                        status="provider_response_invalid",
                        usage=exc.usage,
                        usage_known=all(k in exc.usage for k in usage)
                        and not exc.diagnostics.get("output_usage_unknown", False),
                        diagnostics=exc.diagnostics,
                    )
                    for key in usage:
                        usage[key] += exc.usage.get(key, 0)
                    category = exc.diagnostics.get("category")
                    feedback = ("invalid_model_output: gap_claim_span must be empty or an exact immutable_claim substring"
                                if category == "gap_span_invalid" else "invalid_model_output")
                except (json.JSONDecodeError, ValidationError, ValueError) as exc:
                    record["status"] = "validation_failed"
                    feedback = (
                        "invalid_schema"
                        if isinstance(exc, ValidationError)
                        else (
                            "invalid_json"
                            if isinstance(exc, json.JSONDecodeError)
                            else str(exc)
                        )
                    )
                if record["status"] in {
                    "validation_failed",
                    "provider_response_invalid",
                }:
                    record["error_code"] = feedback
                    if repairs >= b.max_repairs:
                        outcome = "validation_repair_exhausted"
                        break
                    repairs += 1
                    feedback = (
                        (feedback or "validation_failed")
                        + "; choose a new legal action; read known preview candidates before citation"
                    )
        except Exception as exc:
            # Preserve failure + attempts/usage; no silent success or free retry.
            outcome = "controller_failure:" + type(exc).__name__
            for event in events:
                if event.get("status") == "running":
                    event.update(
                        status="failed",
                        elapsed_ms=(self.clock() - event.pop("started")) * 1000,
                    )
            events.append({"failure_type": type(exc).__name__})
            if attempts and attempts[-1]["status"] in {"pending", "response_received"}:
                attempts[-1]["status"] = "terminal_failure"
        return {
            "protocol": PROTOCOL,
            "controller_version": "scifact-common-packing-v1",
            "route": route,
            "provider": self.provider.name,
            "provider_kind": self.provider.kind,
            "answer": answer,
            "outcome": outcome,
            "generation_attempts": attempts,
            "events": events,
            "usage": usage,
            "model_calls": len(attempts),
            "tool_calls": tool_calls,
            "validation_repairs": repairs,
            "rerank_pairs": rerank_pairs,
            "cumulative_model_prompt_tokens": read_input_tokens,
            "context_payload_tokens_sum": sum(
                a["context_payload_tokens"] for a in attempts
            ),
            "preview_payload_tokens_sum": sum(
                a["preview_payload_tokens"] for a in attempts
            ),
            "elapsed_ms": (self.clock() - start) * 1000,
            "budget": b.model_dump(),
            "equal_caps_not_equal_actual_cost": True,
            "unknown_usage_attempts": sum(not a["usage_known"] for a in attempts),
        }
