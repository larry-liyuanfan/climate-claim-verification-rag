"""Gold-aware TRAIN-only sampling; never imported by the inference entry point.

Strata are biased diagnostic opportunities, not generalization estimates.
Packing probes use a scripted fixture, not model inference or policy success.
"""

from __future__ import annotations

import copy
import hashlib
import json
from collections import Counter
from collections.abc import Callable, Collection, Mapping, Sequence
from typing import Any

from .agent_v3 import Source, V3Budget
from .scifact_agent_v1 import SciFactDocumentAgentV1
from .scifact_grounding import GoldClaim
from .scifact_terminal import (
    MAX_DOCUMENTS, MAX_SENTENCES_PER_DOCUMENT, MAX_TOTAL_SENTENCES,
    PROTOCOL, render_scifact_prompt,
)

SALT = "scifact-eligible-train-diagnostic-20260930-v1"
STRATA = ("initial_doc_opportunity", "top20_doc_replenishable", "gold_absent_top20", "nei")
ROUTES = ("fixed_retrieval", "fixed_rerank", "deterministic_extra", "adaptive")


def rank_key(value: str | int) -> str:
    return hashlib.sha256(f"{SALT}:{value}".encode()).hexdigest()


def coverage(claim: GoldClaim, visible: Mapping[int, Sequence[int]]) -> dict[str, bool]:
    """Alternatives are OR within each doc; all docs needed for this stratum.

    This is NOT the official metric: its abstract rationalized rule uses first3,
    while its sentence denominator includes the union of alternative rationales.
    """
    if not claim.evidence:
        return {"complete": False, "first3_reachable": False, "all_gold_sentences": False}
    minimum = []
    first3 = True
    union = True
    for doc, rationales in claim.evidence.items():
        shown = set(visible.get(doc, ()))
        fits = [len(r.sentences) for r in rationales
                if set(r.sentences) <= shown and len(r.sentences) <= MAX_SENTENCES_PER_DOCUMENT]
        minimum.append(min(fits) if fits else MAX_TOTAL_SENTENCES + 1)
        first3 &= any(set(r.sentences) <= shown and len(r.sentences) <= 3 for r in rationales)
        union &= {i for r in rationales for i in r.sentences} <= shown
    complete = len(claim.evidence) <= MAX_DOCUMENTS and sum(minimum) <= MAX_TOTAL_SENTENCES
    return {"complete": complete, "first3_reachable": complete and first3,
            "all_gold_sentences": union}


def document_opportunities(claim: GoldClaim, visible: Mapping[int, Sequence[int]]) -> list[int]:
    """Original abstract-rationalized opportunity, not an observed prediction.

    One full alternative of <=3 original sentences could be placed first. Each
    document is independent; this does not require all gold documents covered.
    """
    return sorted(doc for doc, rationales in claim.evidence.items()
                  if any(len(r.sentences) <= 3 and set(r.sentences) <= set(visible.get(doc, ()))
                         for r in rationales))


class PackingProbe:
    """Actual frozen renderer/tokenizer; scripted decisions, no weights or gold."""
    name = "cpu-packing-fixture-not-model"
    kind = "fixture"
    terminal_protocol = PROTOCOL

    def __init__(self, tokenizer: Any, read_ids: Sequence[str] = ()) -> None:
        self.tokenizer, self.read_ids = tokenizer, list(read_ids)
        self.observations: list[dict[str, Any]] = []

    def count_text(self, text: str) -> int:
        return len(self.tokenizer.encode(text, add_special_tokens=False))

    def count_prompt(self, observation: Mapping[str, Any], schema: Mapping[str, Any]) -> int:
        return self.count_text(render_scifact_prompt(self.tokenizer, observation, schema))

    def generate(self, observation: dict[str, Any], schema: dict[str, Any],
                 max_output_tokens: int, remaining_seconds: float) -> dict[str, Any]:
        self.observations.append(copy.deepcopy(observation))
        value = ({"action": "read", "source_ids": self.read_ids}
                 if self.read_ids and len(self.observations) == 1
                 else {"action": "abstain", "reason": "insufficient_evidence"})
        raw = json.dumps(value)
        return {"raw": raw, "usage": {"input_tokens": self.count_prompt(observation, schema),
                                      "output_tokens": self.count_text(raw)}}


def unused_reranker(query: str, candidates: Sequence[Source]) -> Sequence[Source]:
    raise AssertionError("CPU packing probe must not run reranker")


def packing_snapshot(
    claim: str, retrieve: Callable[[str, int], Sequence[Source]], tokenizer: Any,
    read_ids: Sequence[str] = (),
) -> dict[str, Any]:
    provider = PackingProbe(tokenizer, read_ids)
    result = SciFactDocumentAgentV1(provider, retrieve, rerank=unused_reranker,
                                   budget=V3Budget()).run(claim, "adaptive")
    expected_calls = 2 if read_ids else 1
    if (len(provider.observations) != expected_calls
            or result["outcome"] != "model_abstention:insufficient_evidence"):
        raise ValueError("packing probe failed; never treat as missing gold")
    # No rewrite is allowed in this probe, hence aliases follow first Top20 order.
    candidates = list(retrieve(claim, 20))
    aliases = {f"c{i}": int(s.source_id) for i, s in enumerate(candidates)}
    visible: dict[int, list[int]] = {}
    for entry in provider.observations[-1]["current_citable"]:
        alias, index = entry["sentence_id"].split(":")
        visible.setdefault(aliases[alias], []).append(int(index))
    return {"candidate_doc_ids": [int(s.source_id) for s in candidates],
            "visible": visible, "probe_calls": expected_calls,
            "prompt_tokens": result["generation_attempts"][-1]["input_prompt_tokens"],
            "fixture_only": True}


def classify_opportunity(
    claim: GoldClaim, initial: Mapping[str, Any],
    probe_read: Callable[[Sequence[str]], Mapping[str, Any]],
) -> dict[str, Any]:
    candidates = list(initial["candidate_doc_ids"])
    if len(candidates) != len(set(candidates)) or len(candidates) > 20:
        raise ValueError("invalid initial candidate contract")
    initial_coverage = coverage(claim, initial["visible"])
    opportunities = document_opportunities(claim, initial["visible"])
    result: dict[str, Any] = {"id": claim.claim_id, "initial": dict(initial),
                              "initial_coverage": initial_coverage, "stratum": None,
                              "initial_eligible_gold_doc_ids": opportunities,
                              "initial_eligible_gold_doc_count": len(opportunities),
                              "total_gold_doc_count": len(claim.evidence),
                              "oracle_read_probe_count": 0}
    gold_docs = set(claim.evidence)
    if not gold_docs:
        result["stratum"] = "nei"
    elif opportunities:
        result["stratum"] = "initial_doc_opportunity"
    elif not gold_docs.intersection(candidates):
        result["stratum"] = "gold_absent_top20"
    else:
        # Single-document reads suffice for this per-document opportunity.
        # This offline oracle witness is never a model instruction.
        for doc in sorted(gold_docs.intersection(candidates)):
            if not any(len(r.sentences) <= 3 for r in claim.evidence[doc]):
                continue
            aliases = [f"c{candidates.index(doc)}"]
            if [doc] == candidates[:5]:
                continue  # Controller correctly rejects identical current read.
            witness = dict(probe_read(aliases))
            result["oracle_read_probe_count"] += 1
            if document_opportunities(claim, witness["visible"]):
                result.update(stratum="top20_doc_replenishable", witness=witness,
                              witness_read_doc_ids=[doc])
                break
        if result["stratum"] is None:
            result["excluded_reason"] = "gold_in_candidates_but_not_packable_within_contract"
    return result


def select_component_distinct(
    rows: Sequence[Mapping[str, Any]], components: Mapping[int, str],
    quota: int = 3,
    *, excluded_components: Collection[str] = (),
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Deterministic maximum bipartite matching, not order-dependent greedy fill."""
    if quota != 3 or len({r["id"] for r in rows}) != len(rows):
        raise ValueError("fixed quota or duplicate claim IDs")
    slots = [(s, i) for s in STRATA for i in range(quota)]
    excluded = frozenset(excluded_components)
    options: dict[str, dict[str, Mapping[str, Any]]] = {s: {} for s in STRATA}
    for row in sorted(rows, key=lambda r: (rank_key(r["id"]), r["id"])):
        component = components[row["id"]]
        if component in excluded:
            continue
        if row["stratum"] in STRATA:
            options[row["stratum"]].setdefault(component, row)
    occupied: dict[str, tuple[str, int]] = {}

    def assign(slot: tuple[str, int], seen: set[str]) -> bool:
        for component in sorted(options[slot[0]], key=lambda c: (rank_key(c), c)):
            if component in seen:
                continue
            seen.add(component)
            if component not in occupied or assign(occupied[component], seen):
                occupied[component] = slot
                return True
        return False

    for slot in slots:
        assign(slot, set())
    selected = [dict(options[slot[0]][component], component=component)
                for component, slot in sorted(occupied.items(), key=lambda x: x[1])]
    counts = Counter(r["stratum"] for r in selected)
    return selected, {"selected_counts": {s: counts[s] for s in STRATA},
                      "eligible_component_counts": {s: len(options[s]) for s in STRATA},
                      "shortfall": {s: quota - counts[s] for s in STRATA},
                      "selection": "fixed-hash maximum component-slot matching",
                      "shortfall_policy": "stop at actual matching; never relax or backfill"}
