"""Separate train-gold point diagnosis; never a small-sample bootstrap report."""

from __future__ import annotations

from collections import Counter
from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np

from .scifact_diagnostic_runtime import ROUTES
from .scifact_grounding import Abstract, GoldClaim
from .scifact_scoring import parse_prediction, score_original
from .scifact_terminal import to_original_prediction
from .scifact_train_diagnostic import document_opportunities


def score_diagnostic(
    gold: Sequence[GoldClaim], corpus: Mapping[int, Abstract],
    rows: Sequence[Mapping[str, Any]], selection: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    expected = {(g.claim_id, r) for g in gold for r in ROUTES}
    actual = [(r["claim_id"], r["route"]) for r in rows]
    strata = {r["id"]: r["stratum"] for r in selection}
    if (not 1 <= len(gold) <= 12 or len({g.claim_id for g in gold}) != len(gold)
            or len(actual) != len(expected) or set(actual) != expected
            or len(selection) != len(gold) or set(strata) != {g.claim_id for g in gold}
            or len({r["component"] for r in selection}) != len(gold)):
        raise ValueError("incomplete/duplicate/extra diagnostic matrix or components")
    by_id = {g.claim_id: g for g in gold}
    output: dict[str, Any] = {}
    for route in ROUTES:
        rr = [r for r in rows if r["route"] == route]
        predictions = []
        offers: Counter[str] = Counter()
        proposals: Counter[str] = Counter()
        executed: Counter[str] = Counter()
        outcomes: Counter[str] = Counter()
        unknown = tokens_in = tokens_out = 0
        offered_slots = executed_slots = 0
        claim_diagnostics = []
        elapsed = []
        for row in rr:
            result = row["result"]
            if result["route"] != route or result["claim_id"] != row["claim_id"]:
                raise ValueError("slot/controller identity mismatch")
            if row.get("export_error") or result.get("trace_unlinked_events"):
                raise ValueError("trace/export audit failed; raw costs retained, no quality report")
            canonical = to_original_prediction(row["claim_id"], result, corpus)
            if canonical["prediction"] != row["prediction"]:
                raise ValueError("prediction conversion mismatch")
            predictions.append(parse_prediction(row["prediction"], corpus))
            outcomes[result["outcome"]] += 1
            attempts = result["generation_attempts"]
            if (result["model_calls"] != len(attempts)
                    or type(result["unknown_usage_attempts"]) is not int
                    or any(type(a["usage_known"]) is not bool for a in attempts)
                    or result["unknown_usage_attempts"] != sum(not a["usage_known"] for a in attempts)):
                raise ValueError("attempt/unknown usage ledger mismatch")
            for key in ("input_tokens", "output_tokens"):
                value = result["usage"][key]
                if type(value) is not int or value < 0:
                    raise ValueError("invalid cost")
                parts = [a["usage"].get(key, 0) for a in attempts]
                if (any(type(v) is not int or v < 0 for v in parts)
                        or sum(parts) != value
                        or any(a["usage_known"] and key not in a["usage"] for a in attempts)):
                    raise ValueError("aggregate/attempt usage mismatch")
            tokens_in += result["usage"]["input_tokens"]
            tokens_out += result["usage"]["output_tokens"]
            unknown += result["unknown_usage_attempts"]
            if not np.isfinite(result["elapsed_ms"]) or result["elapsed_ms"] < 0:
                raise ValueError("invalid latency")
            elapsed.append(result["elapsed_ms"])
            offered = False
            for a in attempts:
                for tool in {"read", "rewrite", "rerank"}.intersection(a["allowed_actions"]):
                    offers[tool] += 1
                    offered = True
                if a.get("action") in {"read", "rewrite", "rerank"}:
                    proposals[a["action"]] += 1
            offered_slots += offered
            active_events = [e for e in result["events"] if e.get("model_selected")]
            executed_slots += bool(active_events)
            for event in active_events:
                executed[event["tool"] + ":" + event["status"]] += 1
            visible: dict[int, list[int]] = {}
            if result["visible_attempts"]:
                for v in result["visible_attempts"][0]["visible"]:
                    visible.setdefault(v["doc_id"], []).append(v["sentence_index"])
            g = by_id[row["claim_id"]]
            count = len(document_opportunities(g, visible))
            claim_diagnostics.append({
                "claim_id": g.claim_id, "stratum": strata[g.claim_id],
                "route_first_actual_visible_eligible_gold_docs": count,
                "total_gold_docs": len(g.evidence),
                "model_tool_events": len(active_events),
                "post_tool_decisions": sum(link["event_index"] is not None
                                           and link["subsequent_model_attempt"] is not None
                                           for link in result["model_tool_links"]),
                "outcome": result["outcome"],
            })
        output[route] = {
            "official_point_score": score_original(gold, predictions),
            "by_stratum": {s: score_original(
                [g for g in gold if strata[g.claim_id] == s],
                [p for p in predictions if strata[p.claim_id] == s])
                for s in sorted(set(strata.values()))},
            "outcomes": dict(outcomes),
            "known_tokens_including_failures": {"input_tokens": tokens_in, "output_tokens": tokens_out},
            "unknown_usage_attempts": unknown, "token_totals_are_lower_bounds": bool(unknown),
            "whole_question_p50_ms": float(np.quantile(elapsed, .5)),
            "whole_question_p95_ms": float(np.quantile(elapsed, .95)),
            "actual_tools": {"offered_per_attempt": dict(offers), "proposed": dict(proposals),
                             "model_selected_event_status": dict(executed),
                             "slots_with_tool_opportunity": offered_slots,
                             "slots_with_model_selected_event": executed_slots},
            "claim_diagnostics": claim_diagnostics,
        }
    return {"schema_version": "scifact-train-diagnostic-point-v1", "routes": output,
            "sample_scope": "biased gold-stratified eligible train diagnostic, not benchmark population",
            "bootstrap_replicates": 0, "confidence_interval": None, "p_value": None,
            "bootstrap_not_run_reason": "predeclared small diagnostic",
            "claim_verdict_accuracy": None, "independent_test": False}
