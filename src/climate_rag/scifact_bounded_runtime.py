"""Gold-free F/F+G runtime with shared initial packing and separate actual cost."""
from __future__ import annotations

import copy
from collections.abc import Mapping
from typing import Any

from .agent_v3 import V3Budget
from .evidence_gap_candidate import CANDIDATE_PROTOCOL, GapProviderAdapter, render_gap_prompt
from .local_bounded_scifact_provider import observation_identity
from .scifact_bounded_agent import SciFactBoundedAgent
from .scifact_diagnostic_runtime import TraceProvider
from .scifact_terminal import render_scifact_prompt, to_original_prediction
from .scifact_utility_contract import UtilityDiagnostic
from .agent_v3 import Source

ARMS = ("format_repaired", "format_repaired_gap")
PAIR_PROTOCOL = "scifact-bounded-gap-paired-v1"


class GapTemplateCounter:
    name, kind, wire_protocol = "template-counter-no-model", "fixture", CANDIDATE_PROTOCOL

    def __init__(self, tokenizer: Any) -> None:
        self.tokenizer = tokenizer

    def count_text(self, text: str) -> int:
        return len(self.tokenizer.encode(text, add_special_tokens=False))

    def count_prompt(self, observation: Mapping[str, Any], schema: Mapping[str, Any]) -> int:
        return self.count_text(render_gap_prompt(self.tokenizer, observation, schema))

    def generate(self, *args: Any, **kwargs: Any) -> dict[str, Any]:
        raise RuntimeError("template_counter_never_generates")


class CommonPacking:
    """Max(F, empty-history G, active actual prompt), pure during packing.

The two arms' initial observations therefore pack identically for a given
candidate order. Later model choices/history may differ; equality is NOT forced
after tool decisions. count_prompt and the usage ledger remain actual, not max.
"""
    def __init__(self, active: Any, tokenizer: Any) -> None:
        self.active, self.tokenizer = active, tokenizer
        self.empty_gap = GapProviderAdapter(GapTemplateCounter(tokenizer))

    def __call__(self, observation: dict[str, Any], schema: dict[str, Any]) -> int:
        plain = len(self.tokenizer.encode(render_scifact_prompt(self.tokenizer, observation, schema),
                                          add_special_tokens=False))
        return max(plain, self.empty_gap.count_prompt(observation, schema),
                   int(self.active.count_prompt(observation, schema)))


class BoundedTraceProvider(TraceProvider):
    def generate(self, observation: dict[str, Any], schema: dict[str, Any],
                 max_output_tokens: int, remaining_seconds: float) -> dict[str, Any]:
        identity = observation_identity(observation)
        count = len(self.visible_attempts)
        try:
            return super().generate(observation, schema, max_output_tokens, remaining_seconds)
        finally:
            if len(self.visible_attempts) == count + 1:
                self.visible_attempts[-1].update(actual_observation=identity,
                                                 feedback=observation.get("feedback"))


def run_bounded_slot(claim_id: int, claim: str, route: str, arm: str,
                     backend: Any, retrieve: Any, rerank: Any, corpus: Any,
                     persist_raw: Any = None, *, diagnostic: UtilityDiagnostic | None = None) -> dict[str, Any]:
    if arm not in ARMS or bool(backend.gap) != (arm == ARMS[1]):
        raise ValueError("arm_backend_mismatch")
    provider = GapProviderAdapter(backend) if backend.gap else backend
    if diagnostic is not None and backend.gap:
        raise ValueError("utility_diagnostic_is_base_only_no_gap_or_adapter")
    aliases: dict[str, Any] = {}
    seen: dict[str, str] = {}

    def tracked_retrieve(query: str, width: int) -> Any:
        rows = list(retrieve(query, width))[:width]
        for source in rows:
            if source.source_id not in seen:
                alias = f"c{len(seen)}"
                seen[source.source_id], aliases[alias] = alias, source
            elif aliases[seen[source.source_id]].text_sha256 != source.text_sha256:
                raise ValueError("trace source mutation")
        return rows

    traced = BoundedTraceProvider(provider, aliases)
    if diagnostic is not None and diagnostic.prefix is not None:
        p = diagnostic.prefix["payload"]
        for row in p["sources"]:
            alias = p["aliases"][row["source_id"]]
            aliases[alias] = Source(row["source_id"], row["title"], tuple(row["sentences"]))
            seen[row["source_id"]] = alias
        frame = p["frame"]
        traced.visible_attempts.append({"attempt_index": 0, "shared_prefix_reference": True,
            "visible": [{"alias": sid.split(":")[0], "doc_id": int(v["source_id"]),
                         "sentence_index": v["sentence_index"], "source_text_sha256": v["source_text_sha256"],
                         "text_sha256": v["text_sha256"]} for sid, v in frame["visible"].items()],
            "actual_observation": observation_identity(frame["observation"]),
            "feedback": frame["observation"]["feedback"]})
    result = SciFactBoundedAgent(
        traced, tracked_retrieve, rerank=rerank, budget=V3Budget(),
        packing_count=CommonPacking(provider, backend.base.tokenizer)).run(claim, route, diagnostic=diagnostic)
    result.update(claim_id=claim_id, arm=arm, comparison_protocol=PAIR_PROTOCOL,
                  visible_attempts=traced.visible_attempts)
    if backend.gap:
        result["candidate_wire_audit"] = copy.deepcopy(provider.records)
    return finalize_bounded_slot(claim_id, route, arm, result, traced.visible_attempts, corpus, persist_raw)


def finalize_bounded_slot(claim_id: int, route: str, arm: str, result: dict[str, Any],
                          visible_attempts: list[dict[str, Any]], corpus: Any,
                          persist_raw: Any = None) -> dict[str, Any]:
    """Shared unchanged event linkage, durable cost, and fail-closed export."""
    if persist_raw is not None:
        persist_raw(result)  # durable full cost/response references before auxiliary export
    remaining = [i for i, e in enumerate(result["events"]) if e.get("model_selected")]
    audit = []
    attempts = result["generation_attempts"]
    for i, attempt in enumerate(attempts):
        event_index = None
        if (attempt.get("action") in {"read", "rewrite", "rerank"}
                and attempt["status"] == "valid_decision" and remaining
                and result["events"][remaining[0]]["tool"] == attempt["action"]):
            event_index = remaining.pop(0)
        next_attempt = attempts[i + 1] if i + 1 < len(attempts) else None
        audit.append({
            "attempt_index": i,
            "raw_wire_action": attempt.get("diagnostics", {}).get("raw_wire_action"),
            "strict_action": attempt.get("action"), "strict_status": attempt["status"],
            "actual_event_index": event_index,
            "actual_event_status": result["events"][event_index]["status"] if event_index is not None else None,
            "next_attempt_index": i + 1 if next_attempt is not None else None,
            "next_feedback": visible_attempts[i + 1]["feedback"] if next_attempt is not None else None,
            "next_strict_action": next_attempt.get("action") if next_attempt is not None else None,
            "next_strict_status": next_attempt["status"] if next_attempt is not None else None})
    result.update(decision_execution_audit=audit, trace_unlinked_events=remaining,
                  model_tool_links=[{"attempt_index": row["attempt_index"],
                                     "event_index": row["actual_event_index"],
                                     "proposal_status": row["strict_status"],
                                     "subsequent_model_attempt": row["next_attempt_index"]}
                                    for row in audit if row["strict_action"] in {"read", "rewrite", "rerank"}],
                  initial_context_identity=(visible_attempts[0]["actual_observation"]
                                            if visible_attempts else None))
    try:
        converted = to_original_prediction(claim_id, result, corpus)
    except Exception as exc:
        converted = {"prediction": None, "termination_reason": result["outcome"],
                     "export_error": type(exc).__name__}
    return {"claim_id": claim_id, "route": route, "arm": arm, "result": result, **converted}
