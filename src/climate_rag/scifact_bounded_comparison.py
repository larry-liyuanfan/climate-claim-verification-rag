"""Pair comparability is established before any gold-backed difference report."""
from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

from .scifact_bounded_runtime import ARMS
from .scifact_diagnostic_runtime import ROUTES
from .scifact_diagnostic_scoring import score_diagnostic


def compare_initial_contexts(rows: Sequence[Mapping[str, Any]], claim_ids: Sequence[int]) -> dict[str, Any]:
    expected = {(c, r, a) for c in claim_ids for r in ROUTES for a in ARMS}
    keys = [(r["claim_id"], r["route"], r["arm"]) for r in rows]
    if len(keys) != len(expected) or set(keys) != expected:
        raise ValueError("incomplete/duplicate paired matrix")
    keyed = dict(zip(keys, rows, strict=True))
    comparisons = []
    for c in claim_ids:
        for route in ROUTES:
            a, b = (keyed[c, route, arm]["result"] for arm in ARMS)
            x, y = a.get("initial_context_identity"), b.get("initial_context_identity")
            comparable = x is not None and y is not None and x == y
            comparisons.append({"claim_id": c, "route": route, "comparable": comparable,
                                "reason": None if comparable else "missing_or_different_initial_actual_context",
                                "costs": {arm: keyed[c, route, arm]["result"]["usage"] for arm in ARMS}})
    return {"pairs": comparisons, "all_comparable": all(r["comparable"] for r in comparisons),
            "subsequent_context_equality_asserted": False}


def score_bounded_pair(gold: Any, corpus: Any, rows: Sequence[Mapping[str, Any]], selection: Any) -> dict[str, Any]:
    gate = compare_initial_contexts(rows, [g.claim_id for g in gold])
    report: dict[str, Any] = {"comparability": gate, "independent_test": False,
                              "scope": "same biased eligible-train components; diagnostic only",
                              "bootstrap_replicates": 0, "paired_quality": None}
    if not gate["all_comparable"]:
        report["status"] = "not_comparable_costs_retained"
        return report
    report["paired_quality"] = {arm: score_diagnostic(gold, corpus, [r for r in rows if r["arm"] == arm], selection)
                                 for arm in ARMS}
    report["status"] = "paired_diagnostic_only"
    report["attribution"] = "G package includes self-report/prompt overhead/history; no isolated reasoning-causality claim"
    return report
