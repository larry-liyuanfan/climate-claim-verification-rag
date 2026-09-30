"""Gold-free same-packing semantic pair; CPU fixtures never instantiate models."""
from __future__ import annotations

import copy
import hashlib
import json
from collections.abc import Mapping, Sequence
from typing import Any

from .agent_v3 import Source, V3Budget
from .evidence_gap_candidate import CANDIDATE_PROTOCOL, GapProviderAdapter
from .scifact_bounded_agent import SciFactBoundedAgent
from .scifact_bounded_runtime import BoundedTraceProvider, finalize_bounded_slot
from .scifact_semantic_policy import OLD_G, PAIR, POLICIES, policy_sha, render_policy_prompt


class PolicyCounter:
    name, kind, wire_protocol = "semantic-template-counter", "fixture", CANDIDATE_PROTOCOL

    def __init__(self, tokenizer: Any, policy: str) -> None:
        policy_sha(policy)
        self.tokenizer, self.policy = tokenizer, policy

    def count_text(self, text: str) -> int:
        return len(self.tokenizer.encode(text, add_special_tokens=False))

    def count_prompt(self, observation: Mapping[str, Any], schema: Mapping[str, Any]) -> int:
        return self.count_text(render_policy_prompt(self.tokenizer, observation, schema, self.policy))

    def generate(self, *args: Any, **kwargs: Any) -> dict[str, Any]:
        raise RuntimeError("template_counter_never_generates")


class SemanticCommonPacking:
    """Max both empty histories and both renders of the active actual history.

    Pure during fitting. Initial contexts match across arms; later histories may
    differ. Costs still use only the active prompt. This is no trajectory oracle.
    """

    def __init__(self, active: GapProviderAdapter, tokenizer: Any) -> None:
        self.active = active
        self.counters = [PolicyCounter(tokenizer, p) for p in POLICIES]
        self.empty = [GapProviderAdapter(c) for c in self.counters]

    def __call__(self, observation: dict[str, Any], schema: dict[str, Any]) -> int:
        decorated, envelope = self.active.prepare(observation, schema)
        return max(*(c.count_prompt(observation, schema) for c in self.empty),
                   *(c.count_prompt(decorated, envelope) for c in self.counters),
                   self.active.count_prompt(observation, schema))


def run_semantic_slot(claim_id: int, claim: str, route: str, backend: Any,
                      retrieve: Any, rerank: Any, corpus: Any, source_git: str,
                      persist_raw: Any = None) -> dict[str, Any]:
    if backend.policy not in POLICIES or not backend.gap:
        raise ValueError("semantic_pair_requires_explicit_gap_policy")
    provider = GapProviderAdapter(backend)
    aliases: dict[str, Any] = {}
    seen: dict[str, str] = {}

    def tracked(query: str, width: int) -> Any:
        rows = list(retrieve(query, width))[:width]
        for source in rows:
            if source.source_id not in seen:
                alias = f"c{len(seen)}"
                seen[source.source_id], aliases[alias] = alias, source
            elif aliases[seen[source.source_id]].text_sha256 != source.text_sha256:
                raise ValueError("trace_source_mutation")
        return rows

    traced = BoundedTraceProvider(provider, aliases)
    result = SciFactBoundedAgent(traced, tracked, rerank=rerank, budget=V3Budget(),
        packing_count=SemanticCommonPacking(provider, backend.base.tokenizer)).run(claim, route)
    result.update(claim_id=claim_id, arm=backend.policy, comparison_protocol=PAIR,
                  source_git=source_git, prompt_sha256=policy_sha(backend.policy),
                  candidate_wire_audit=copy.deepcopy(provider.records),
                  visible_attempts=traced.visible_attempts)
    row = finalize_bounded_slot(claim_id, route, backend.policy, result, traced.visible_attempts, corpus, persist_raw)
    return row | {
            "comparison_protocol": PAIR, "source_git": source_git,
            "prompt_sha256": policy_sha(backend.policy)}


class SemanticPackingFixture(PolicyCounter):
    """Explicit read/abstain wire with real adapter history; not model behavior."""
    gap = True

    def __init__(self, tokenizer: Any, policy: str, read_ids: Sequence[str]) -> None:
        super().__init__(tokenizer, policy)
        self.read_ids = list(read_ids)
        self.observations: list[dict[str, Any]] = []

    def generate(self, observation: dict[str, Any], schema: dict[str, Any],
                 max_output_tokens: int, remaining_seconds: float) -> dict[str, Any]:
        self.observations.append(copy.deepcopy(observation))
        decision = ({"action": "read", "source_ids": self.read_ids}
                    if self.read_ids and len(self.observations) == 1
                    else {"action": "abstain", "reason": "insufficient_evidence"})
        wire = {"evidence_state": {"retrieval_need": "uncertain", "relevance": "unknown", "support": "unknown"},
                "gap_claim_span": "", "decision": decision}
        raw = json.dumps(wire, separators=(",", ":"))
        size = len(raw.encode())
        return {"raw": raw, "usage": {"input_tokens": self.count_prompt(observation, schema),
                                       "output_tokens": self.count_text(raw)},
                "diagnostics": {"raw_wire_action": decision["action"], "private_attachment": {"sha256": hashlib.sha256(raw.encode()).hexdigest(),
                    "attempted_bytes": size, "stored_bytes": size, "truncated": False, "io_failed": False}}}


def unused_reranker(query: str, candidates: Sequence[Source]) -> Sequence[Source]:
    raise AssertionError("CPU fixture exposes rerank schema but must never execute it")


def semantic_packing_snapshot(claim: str, retrieve: Any, tokenizer: Any,
                              read_ids: Sequence[str] = (), policy: str = OLD_G) -> dict[str, Any]:
    backend = SemanticPackingFixture(tokenizer, policy, read_ids)
    adapter = GapProviderAdapter(backend)
    result = SciFactBoundedAgent(adapter, retrieve, rerank=unused_reranker, budget=V3Budget(),
        packing_count=SemanticCommonPacking(adapter, tokenizer)).run(claim, "adaptive")
    expected = 2 if read_ids else 1
    if (len(backend.observations) != expected or result["outcome"] != "model_abstention:insufficient_evidence"
            or (read_ids and not any(e["tool"] == "read" and e["status"] == "completed"
                                    and e["model_selected"] for e in result["events"]))):
        raise ValueError("semantic_fixture_did_not_execute_declared_history")
    from .local_bounded_scifact_provider import observation_identity
    candidates: list[Source] = list(retrieve(claim, 20))
    aliases = {f"c{i}": int(s.source_id) for i, s in enumerate(candidates)}
    visible: dict[int, list[int]] = {}
    for entry in backend.observations[-1]["current_citable"]:
        alias, index = entry["sentence_id"].split(":")
        visible.setdefault(aliases[alias], []).append(int(index))
    return {"candidate_doc_ids": [int(s.source_id) for s in candidates], "visible": visible,
            "initial_identity": observation_identity(backend.observations[0]),
            "probe_calls": expected, "model_calls": 0, "fixture_only": True,
            "policy": policy, "prompt_sha256": policy_sha(policy),
            "prompt_tokens": result["generation_attempts"][-1]["input_prompt_tokens"],
            "fixture_history": {"evidence_state": {"retrieval_need": "uncertain", "relevance": "unknown", "support": "unknown"},
                                "gap_claim_span": "", "read_ids": list(read_ids), "terminal": "abstain"},
            "history_scope": "this scripted history only; not arbitrary model trajectories"}


def validate_pair_rows(rows: Sequence[Mapping[str, Any]], source_git: str,
                       ids: Sequence[int], routes: Sequence[str]) -> None:
    expected = {(i, r, p) for i in ids for r in routes for p in POLICIES}
    if len(rows) != len(expected) or {(r["claim_id"], r["route"], r["arm"]) for r in rows} != expected:
        raise ValueError("new_semantic_pair_matrix_required")
    for r in rows:
        if (r.get("comparison_protocol") != PAIR or r.get("source_git") != source_git
                or r.get("prompt_sha256") != policy_sha(r["arm"])
                or any(r["result"].get(k) != r[k] for k in
                       ("claim_id", "route", "arm", "comparison_protocol", "source_git", "prompt_sha256"))):
            raise ValueError("old_results_cannot_substitute_for_new_baseline")
    for i in ids:
        for route in routes:
            pair = [r["result"]["initial_context_identity"] for r in rows if r["claim_id"] == i and r["route"] == route]
            if pair[0] is None or pair[0] != pair[1]:
                raise ValueError("semantic_initial_packing_incomparable")
