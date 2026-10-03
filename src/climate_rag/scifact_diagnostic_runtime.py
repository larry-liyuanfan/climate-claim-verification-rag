"""Gold-free bounded train runner primitives; no file loading or scheduling."""

from __future__ import annotations

import hashlib
from collections.abc import Callable, Mapping, Sequence
from typing import Any

from .agent_v3 import Source, V3Budget, V3Provider
from .scifact_agent_v1 import SciFactDocumentAgentV1
from .scifact_grounding import Abstract
from .scifact_terminal import PROTOCOL, to_original_prediction

ROUTES = ("fixed_retrieval", "fixed_rerank", "deterministic_extra", "adaptive")
RELEASE = "climate-scifact-train-diagnostic-20260930-r1"


def validate_protocol(protocol: Mapping[str, Any], claims: Sequence[Mapping[str, Any]]) -> None:
    if (protocol["release_id"] != RELEASE
            or protocol["scope"] != "biased_eligible_train_diagnostic_not_dev_test"
            or tuple(protocol["routes"]) != ROUTES
            or V3Budget.model_validate(protocol["budget"]) != V3Budget()
            or protocol["corpus_documents"] != 5183 or protocol["pythonhashseed"] != "0"
            or protocol["official_dev_read"] is not False
            or protocol["model_calls"] != 0
            or not 1 <= len(claims) <= 12 or len(claims) != protocol["claims"]
            or any(set(r) != {"id", "claim"} or type(r["id"]) is not int
                   or not isinstance(r["claim"], str) or not r["claim"].strip() for r in claims)
            or len({r["id"] for r in claims}) != len(claims)
            or protocol["slots"] != [{"claim_id": r["id"], "route": route}
                                     for r in claims for route in ROUTES]):
        raise ValueError("unreleased/invalid train diagnostic matrix")


class TraceProvider:
    """Capture ID/hash provenance without changing prompts, outputs or budgets."""
    terminal_protocol = PROTOCOL

    def __init__(self, base: V3Provider, aliases: dict[str, Source]) -> None:
        if getattr(base, "terminal_protocol", None) != PROTOCOL:
            raise ValueError("underlying provider terminal protocol mismatch")
        self.base, self.aliases = base, aliases
        self.name, self.kind = base.name, base.kind
        self.visible_attempts: list[dict[str, Any]] = []

    def count_text(self, text: str) -> int:
        return self.base.count_text(text)

    def count_prompt(self, observation: Mapping[str, Any], schema: Mapping[str, Any]) -> int:
        return self.base.count_prompt(observation, schema)

    def generate(self, observation: dict[str, Any], schema: dict[str, Any],
                 max_output_tokens: int, remaining_seconds: float) -> dict[str, Any]:
        visible = []
        for entry in observation["current_citable"]:
            alias, raw_index = entry["sentence_id"].split(":")
            source, index = self.aliases[alias], int(raw_index)
            if source.sentences[index] != entry["text"]:
                raise ValueError("trace alias/original sentence mismatch")
            visible.append({"alias": alias, "doc_id": int(source.source_id),
                            "sentence_index": index,
                            "source_text_sha256": source.text_sha256,
                            "text_sha256": hashlib.sha256(entry["text"].encode()).hexdigest()})
        self.visible_attempts.append({"attempt_index": len(self.visible_attempts),
                                      "visible": visible})
        return self.base.generate(observation, schema, max_output_tokens, remaining_seconds)


def run_slot(
    claim_id: int, claim: str, route: str, provider: V3Provider,
    retrieve: Callable[[str, int], Sequence[Source]],
    rerank: Callable[[str, Sequence[Source]], Sequence[Source]],
    corpus: Mapping[int, Abstract],
    persist_raw: Callable[[dict[str, Any]], None] | None = None,
) -> dict[str, Any]:
    aliases: dict[str, Source] = {}
    seen: dict[str, str] = {}

    def tracked_retrieve(query: str, width: int) -> Sequence[Source]:
        rows = list(retrieve(query, width))[:width]
        for source in rows:
            if source.source_id not in seen:
                alias = f"c{len(seen)}"
                seen[source.source_id], aliases[alias] = alias, source
            elif aliases[seen[source.source_id]].text_sha256 != source.text_sha256:
                raise ValueError("trace source mutation")
        return rows

    traced = TraceProvider(provider, aliases)
    result = SciFactDocumentAgentV1(traced, tracked_retrieve, rerank=rerank,
                                   budget=V3Budget()).run(claim, route)
    if persist_raw is not None:
        persist_raw(result)  # First durable record, before any observation/export work.
    result.update(claim_id=claim_id, visible_attempts=traced.visible_attempts)
    # Rejected proposals are not tool events. In particular a parsed read_loop
    # retains its action field, but never executes. Never lose an expensive slot
    # because auxiliary trace conversion fails.
    tool_events = [i for i, e in enumerate(result["events"]) if e.get("model_selected")]
    links = []
    remaining = list(tool_events)
    for i, attempt in enumerate(result["generation_attempts"]):
        if attempt.get("action") not in {"read", "rewrite", "rerank"}:
            continue
        event_index = None
        if (attempt["status"] == "valid_decision" and remaining
                and result["events"][remaining[0]]["tool"] == attempt["action"]):
            event_index = remaining.pop(0)
        links.append({"attempt_index": i, "event_index": event_index,
                      "proposal_status": attempt["status"],
                      "subsequent_model_attempt": i + 1 if i + 1 < result["model_calls"] else None})
    result["model_tool_links"] = links
    result["trace_unlinked_events"] = remaining
    try:
        converted = to_original_prediction(claim_id, result, corpus)
    except Exception as exc:
        converted = {"prediction": None, "termination_reason": result["outcome"],
                     "export_error": type(exc).__name__}
    return {"claim_id": claim_id, "route": route, "result": result, **converted}
