"""One-run CPU closeout. Run on Spartan; export counts, never claim/source IDs or text."""

from __future__ import annotations

import argparse
from collections import Counter
from collections.abc import Callable
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np

EXPECTED = {
    "operator-status.json": "d483a1426fdbeb445c2e676f77183a257dfffb07bde2a0634afaff620b448ed8",
    "cost-before-quality.json": "1bc867a1eea41bb7013423405dad5cbf7b0c6da3102c09d67b9834dd6795fa6d",
    "compact.json": "a07721e8f5a9cf477b2f3052dc65bfc64ebd7da8642f212af593c15ad8c3f19b",
    "inference/run.json": "958244b0aeeae90d56b05686ea26bceda098067f843906d982aca5677646e7ae",
}
ROUTES = ("fixed_retrieval", "fixed_rerank", "deterministic_extra", "fixed_multiquery", "adaptive")
GOLD_SHA = "d2dd28422bffacf87ded2153b3bfac4ca9e1edc903e1d4e7a88d3a40f5d2fd8b"


def read_checked(path: Path, expected: str) -> Any:
    raw = path.read_bytes()
    if hashlib.sha256(raw).hexdigest() != expected:
        raise ValueError("frozen_input_hash")
    return json.loads(raw)


def visibility(row: dict[str, Any], attempt: dict[str, Any]) -> set[str]:
    aliases = row["visible_source_ids"]
    return {aliases[s["sentence_id"].rsplit(":", 1)[0]]
            for s in attempt["observation"]["current_citable"]}


def diagnose(row: dict[str, Any], gold: dict[str, Any]) -> dict[str, Any]:
    """Internal numeric features only; do not mistake alias inventory for shown full text."""
    attempts = row["generation_attempts"]
    events = [e for e in row["events"] if e.get("status") == "completed"]
    gold_ids = set(gold["evidence_ids"])
    aliases = row["visible_source_ids"]
    first = next(e for e in events if e["tool"] == "retrieve")
    initial = {aliases[s] for s in first["candidate_ids"]}
    shown = set().union(*(visibility(row, a) for a in attempts))
    answer = row["answer"]
    cited = {s["source_id"] for s in answer["citations"]} if answer else set()
    rewrites = [e for e in events if e["tool"] == "rewrite"]
    feedback_pairs = 0
    for event in rewrites:
        following = [a for a in attempts if
                     (a["observation"].get("tool_feedback") or {}).get("query_sha256")
                     == event["query_sha256"]]
        feedback_pairs += bool(following and any(a.get("decision") for a in following))
    eligible = gold.get("label") in {"SUPPORTS", "REFUTES"}
    return {
        "evidence_eligible": bool(gold_ids), "label_eligible": eligible,
        "initial_candidate_gold_hit": bool(initial & gold_ids),
        "final_candidate_top5_gold_hit": bool(set(row["delivered_evidence_ids"][:5]) & gold_ids),
        "gold_full_text_shown": bool(shown & gold_ids),
        "gold_cited": bool(cited & gold_ids),
        "correct_label": bool(eligible and answer and answer["label"] == gold["label"]),
        "answered": answer is not None,
        "extra_searches": len(rewrites),
        "model_feedback_searches": sum(e.get("selection_origin") == "model_feedback" for e in rewrites),
        "feedback_then_decision": feedback_pairs,
        "new_source_count": sum(len(e["new_source_ids"]) for e in rewrites),
        "new_source_gold_hit": any({aliases[s] for s in e["new_source_ids"]} & gold_ids for e in rewrites),
        "actions": [a.get("decision", {}).get("action", "invalid") for a in attempts],
        "errors": [a.get("error_code", "unspecified") for a in attempts if a["status"] != "valid_decision"],
        "read_same_as_current_selection": sum(
            a.get("decision", {}).get("action") == "read"
            and a["decision"]["source_ids"] == a["requested_context"] for a in attempts),
        "queries_available_first": "rewrite" in attempts[0]["allowed_actions"],
        "tools_after_initial": len(events) - 1,
        "calls": row["model_calls"],
        "outcome": row["outcome"],
    }


def interval(left: list[float], right: list[float]) -> dict[str, Any]:
    """Exploratory cost CI, not an added/changed acceptance gate."""
    delta = np.asarray(right, dtype=float) - np.asarray(left, dtype=float)
    rng = np.random.default_rng(20260929)
    values = [float(delta[rng.integers(0, len(delta), size=len(delta))].mean())
              for _ in range(5000)]
    return {"pairs": len(delta), "samples": 5000, "mean_difference": float(delta.mean()),
            "ci_lower": float(np.quantile(values, .025)), "ci_upper": float(np.quantile(values, .975))}


def summarize(run: dict[str, Any], gold: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    rows = run["runs"]
    ids = [r["task_id"] for r in rows if r["route"] == ROUTES[0]]
    if len(ids) != 32 or len(set(ids)) != 32 or len(rows) != 160:
        raise ValueError("matrix_size")
    if [(r["task_id"], r["route"]) for r in rows] != [(t, route) for t in ids for route in ROUTES]:
        raise ValueError("matrix_order")
    if set(ids) != set(gold["claims"]):
        raise ValueError("gold_ids")
    if any(r["immutable_claim_sha256"] != gold["claims"][r["task_id"]]["claim_sha256"] for r in rows):
        raise ValueError("claim_identity")
    details = {route: [diagnose(r, gold["claims"][r["task_id"]])
                      for r in rows if r["route"] == route] for route in ROUTES}
    summary: dict[str, Any] = {"matrix_slots": len(rows), "unique_tasks": len(ids), "routes": {}}
    labels = Counter(g.get("label", "MISSING") for g in gold["claims"].values())
    if not set(labels) <= {"SUPPORTS", "REFUTES", "NOT_ENOUGH_INFO", "DISPUTED", "CONFLICTING", "MISSING", None}:
        raise ValueError("unrecognized_gold_label")
    summary["gold_label_counts"] = dict(labels)
    for route, values in details.items():
        summary["routes"][route] = {
            "slots": len(values),
            **{key: sum(d[key] for d in values) for key in (
                "evidence_eligible", "label_eligible", "initial_candidate_gold_hit",
                "final_candidate_top5_gold_hit", "gold_full_text_shown", "gold_cited",
                "correct_label", "answered", "extra_searches", "model_feedback_searches",
                "feedback_then_decision", "new_source_count", "new_source_gold_hit",
                "queries_available_first", "tools_after_initial", "calls", "read_same_as_current_selection")},
            "actions": dict(Counter(a for d in values for a in d["actions"])),
            "errors": dict(Counter(e for d in values for e in d["errors"])),
            "candidate_gold_not_shown": sum(d["initial_candidate_gold_hit"] and not d["gold_full_text_shown"] for d in values),
            "shown_gold_not_cited": sum(d["gold_full_text_shown"] and not d["gold_cited"] for d in values),
            "shown_gold_wrong_label": sum(d["gold_full_text_shown"] and d["label_eligible"] and not d["correct_label"] and d["answered"] for d in values),
            "shown_gold_abstain": sum(d["gold_full_text_shown"] and not d["answered"] for d in values),
        }
    adaptive = details["adaptive"]
    criteria: dict[str, Callable[[dict[str, Any]], bool]] = {
        "candidate_not_read_terminal": lambda d: d["initial_candidate_gold_hit"] and not d["gold_full_text_shown"],
        "shown_gold_wrong_verdict": lambda d: d["gold_full_text_shown"] and d["answered"] and d["label_eligible"] and not d["correct_label"],
        "shown_gold_not_adopted": lambda d: d["gold_full_text_shown"] and not d["gold_cited"],
        "validation_exhaustion": lambda d: d["outcome"] == "validation_repair_exhausted",
        "successful_grounded_id_match": lambda d: d["correct_label"] and d["gold_cited"],
    }
    summary["cases"] = []
    private_map = {}
    used = set()
    for category, predicate in criteria.items():
        matches = [i for i, d in enumerate(adaptive) if predicate(d)]
        available = [i for i in matches if i not in used]
        if not available:
            continue
        i = available[0]  # Original order, no favorable sorting.
        used.add(i)
        case = "case_" + str(len(summary["cases"]) + 1)
        summary["cases"].append({"case": case, "category": category, "matching_cases": len(matches),
                                 "adaptive": adaptive[i],
                                 "fixed_rerank": details["fixed_rerank"][i],
                                 "fixed_multiquery": details["fixed_multiquery"][i]})
        private_map[case] = {"task_id": ids[i], "adaptive_slot": next(r["slot"] for r in rows if r["task_id"] == ids[i] and r["route"] == "adaptive")}
    def cost_values(route: str, metric: str) -> list[float]:
        return [sum(r["physical_cost"]["total_tokens"].values()) if metric == "generator_tokens"
                else r["elapsed_ms"] if metric == "elapsed_ms" else r["reranker_cost"]["known_token_lower_bound"]
                for r in rows if r["route"] == route]
    summary["exploratory_cost_intervals_not_gate_changes"] = {
        c: {m: interval(cost_values(c, m), cost_values("adaptive", m))
            for m in ("generator_tokens", "elapsed_ms", "reranker_tokens")}
        for c in ("fixed_rerank", "fixed_multiquery")}
    return summary, private_map


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--gold", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    loaded = {name: read_checked(args.run / name, digest) for name, digest in EXPECTED.items()}
    operator, cost = loaded["operator-status.json"], loaded["cost-before-quality.json"]
    if operator["status"] != "completed" or not cost["worker_proof"]["child_reaped"] or cost["completed_slots"] != 160:
        raise ValueError("terminal_complete_required")
    if len(list((args.run / "inference").glob("slot-*/finished.json"))) != 160:
        raise ValueError("finished_slot_count")
    gold = read_checked(args.gold, GOLD_SHA)
    summary, private_map = summarize(loaded["inference/run.json"], gold)
    summary.update(source_hashes=EXPECTED, operator=operator,
                   physical_cost=cost, frozen_scores=loaded["compact.json"],
                   privacy="No original claim/query/evidence text or task/document IDs; case lookup stays remote",
                   semantic_entailment="unmeasured")
    # Free-form error strings are diagnostic codes; fail if unexpected contents appear.
    allowed_errors = {"duplicate_sentence_reference", "known_candidate_not_read", "unknown_sentence_reference",
                      "invalid_schema", "invalid_json", "unspecified", "read_loop"}
    for route in summary["routes"].values():
        if not set(route["errors"]) <= allowed_errors:
            raise ValueError("unreviewed_error_code_do_not_export")
    args.output.mkdir(mode=0o700)
    (args.output / "private-case-map.json").write_text(json.dumps(private_map), encoding="utf-8")
    (args.output / "aggregate.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"matrix_slots": summary["matrix_slots"], "case_count": len(summary["cases"]), "aggregate_sha256": hashlib.sha256((args.output / "aggregate.json").read_bytes()).hexdigest()}))


if __name__ == "__main__":
    main()
