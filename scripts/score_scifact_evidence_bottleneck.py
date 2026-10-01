"""Post-exit audit, original SciFact scoring and conservative redacted export.

The load_gold callback is reached only after all physical/visibility bindings.
This CPU package exercises it with synthetic labels only, never real gold.
"""
from __future__ import annotations

from collections import Counter
import json
from pathlib import Path
from typing import Any

from audit_scifact_bounded_closeout import strict_whole_answer
from climate_rag.scifact_evidence_bottleneck import PROTOCOL, ROUTES, audit_case, costs
from climate_rag.scifact_natural_contract import require
from climate_rag.scifact_read_continuation import ordered_write
from climate_rag.scifact_scoring import ClaimPrediction, parse_prediction, score_original
from climate_rag.scifact_semantic_contract import MODEL_SHA, checked
from climate_rag.scifact_utility_contract import identity
from climate_rag.scifact_utility_runtime import ledger_cost

BASELINE_SHA = "d68fa9ace3ff7749c2f9ca5a86775d09ca7eb413214127425f109f3a0c4cb6e9"


def archived_baselines(ordered_ids_sha: str) -> dict[str, Any]:
    path = Path(__file__).resolve().parents[1] / "docs/verified-runs/scifact-paired-31980221-cpu-replay.json"
    old = json.loads(checked(path, BASELINE_SHA))
    require(old["comparison"]["ordered_ids_sha256"] == ordered_ids_sha, "archived_baseline_same_roster")
    return {"artifact_sha256": BASELINE_SHA, "new_calls": 0,
            "fixed_top1": {v: old["versions"][v]["arms"]["fixed_top1"] for v in ("v2", "v3")},
            "scope": "already_consumed_conditional_TRAIN24_not_new_test"}


def summarize(rows: Any, golds: Any, corpus: Any) -> dict[str, Any]:
    require([r["claim_id"] for r in rows] == [g.claim_id for g in golds], "ordered_complete_scoring_matrix")
    results = {}
    for route in ROUTES:
        predictions, cases = [], []
        for r, gold in zip(rows, golds, strict=True):
            cell = r["routes"][route]
            valid = cell["state"] == "valid_terminal"
            pred = parse_prediction(cell["prediction"], corpus) if valid else ClaimPrediction(gold.claim_id, {})
            predictions.append(pred)
            cases.append({"positive": bool(gold.evidence), "valid_positive_prediction": valid,
                "strict_whole_answer": valid and strict_whole_answer(pred, gold),
                "nei_false_evidence": not gold.evidence and bool(pred.evidence),
                "label": cell["label"], "failure": cell["failure"],
                "selected_count": len(cell["selected_sentence_ids"] or [])})
        results[route] = {"planned": len(golds), "cases": cases, "official_micro": score_original(golds, predictions),
            "strict_positive_correct": sum(c["strict_whole_answer"] for c in cases),
            "nei_false_evidence": sum(c["nei_false_evidence"] for c in cases),
            "failure_count": sum(c["failure"] is not None for c in cases),
            "failure_reasons": dict(Counter(c["failure"] for c in cases if c["failure"])),
            "document_insufficient": sum(c["label"] == "INSUFFICIENT" for c in cases),
            "unresolved_count": sum(not c["valid_positive_prediction"] for c in cases)}
    b, c = results["B"], results["C"]
    # The selected candidate rationale is identical, even if label projection
    # later abstains. Do not describe different positive predictions as better
    # selection/complete-rationale coverage.
    require(all(r["routes"]["B"]["selected_sentence_ids"] == r["routes"]["C"]["selected_sentence_ids"]
                and r["routes"]["B"]["selection_sha256"] == r["routes"]["C"]["selection_sha256"] for r in rows),
            "B_C_shared_rationale_changed")
    delta = {k: c["official_micro"]["metrics"][k]["f1"] - b["official_micro"]["metrics"][k]["f1"]
             for k in ("abstract_label_only", "abstract_rationalized")}
    delta["nei_false_evidence_reduction"] = b["nei_false_evidence"] - c["nei_false_evidence"]
    supported = all(v >= 0 for v in delta.values()) and any(v > 0 for v in delta.values())
    supported = supported and c["failure_count"] <= b["failure_count"]
    return {"routes": results, "C_minus_B_or_error_reduction": delta,
        "mechanism": "supported_on_consumed_conditional_TRAIN_only" if supported else "not_supported",
        "candidate_rationale_coverage_identical_by_construction": True,
        "no_global_NEI_accuracy_claim": True, "Agent_gain_established": False}


def score_after_exit(output: Path, release_path: Path, release_sha: str, frames: Any,
                     tokenizer: Any, corpus: Any, load_gold: Any, *, reports: Path) -> dict[str, Any]:
    require(not reports.resolve().is_relative_to(output.resolve())
            and not output.resolve().is_relative_to(reports.resolve()), "reports_must_be_disjoint")
    reports.mkdir(parents=True, exist_ok=False)
    # Preserve consumed cost even when release/preparation/worker checks fail.
    cost = ledger_cost(output / "ledger")
    # Real contract always plans 24 x3. A synthetic receipt may use fewer, but
    # only from its SHA-bound release, never from surviving result files.
    planned = 72
    try:
        bound = json.loads(checked(release_path, release_sha))
        if bound.get("scope") == "synthetic_fixture":
            planned = bound["planned_route_results"]
            require(type(planned) is int and 3 <= planned <= 72 and planned % 3 == 0, "synthetic_size")
    except (ValueError, OSError, KeyError, TypeError):
        pass
    ordered_write(reports / "cost-before-gold.json", {"physical": cost, "gold_read": False,
        "planned_route_results": planned, "planned_per_route": planned//3})
    try:
        release = json.loads(checked(release_path, release_sha))
        require(release["protocol"] == PROTOCOL, "release_protocol")
        plan = json.loads((output / "planned.json").read_bytes())
        ids = list(dict.fromkeys(s["claim_id"] for s in plan["slots"]))
        require(plan["slots"] == [{"claim_id": i, "route": s} for i in ids for s in ROUTES]
                and len(ids) == len(frames) and 1 <= len(ids) <= 24
                and len(ids)*3 == planned and identity(ids) == release["ordered_ids_sha256"]
                and (release["scope"] == "synthetic_fixture" or len(ids) == 24)
                and plan["release_sha256"] == release_sha and plan["source_git"] == release["source_git"]
                and plan["model_sha256"] == release["model_sha256"] == MODEL_SHA
                and release["frames_identity"] == identity(frames)
                and json.loads((output / "initial-frames.json").read_bytes()) == frames
                and json.loads((output / "release.json").read_bytes()) == release, "planned_identity")
        proof = json.loads((output / "worker-exit.json").read_bytes())
        require(proof["child_reaped"] is True and proof["returncode"] == 0 and proof["interrupted"] is None,
                "complete_worker_exit_before_gold")
        require(proof["release_sha256"] == release_sha, "exit_release_binding")
        require(cost["unknown_usage_attempts"] == 0 and cost["unique_physical_calls"] <= 96, "physical_cost_guard")
        contract = release["generation_contract"]
        require(contract is not None or release["scope"] == "synthetic_fixture", "real_generation_contract_required")
        baselines = (None if release["scope"] == "synthetic_fixture" else
                     archived_baselines(release["ordered_ids_sha256"]))
        rows = [audit_case(output / str(i), frame, output / "ledger", tokenizer, corpus,
                           release_sha=release_sha, generation_contract=contract)
                for i, frame in zip(ids, frames, strict=True)]
        require([r["claim_id"] for r in rows] == ids, "audited_claim_identity")
        accounting = costs(output / "ledger", rows)
        ordered_write(reports / "audited-cost-before-gold.json", dict(accounting, gold_read=False,
                      planned_route_results=planned, planned_per_route=planned//3))
    except (ValueError, OSError, KeyError, TypeError, RuntimeError) as exc:
        result = {"protocol": PROTOCOL, "status": "no_quality", "reason": str(exc), "gold_read": False,
                  "physical": cost, "planned_route_results": planned, "planned_per_route": planned//3,
                  "unscored_route_results": planned}
        ordered_write(reports / "no-quality.json", result)
        return result
    golds = load_gold(ids)
    result = {"protocol": PROTOCOL, "status": "scored", "scope": release["scope"],
        "gold_read": True, "source_git": release["source_git"], "release_sha256": release_sha,
        "model_sha256": plan["model_sha256"], "accounting": accounting,
        "archived_baselines": baselines,
        "baseline_comparison": "Retain archived strongest v2 top1 and v3 top1; no new baseline calls.",
        **summarize(rows, golds, corpus)}
    ordered_write(reports / "quality.json", result)
    return result


def export_compact(reports: Path, target: Path) -> dict[str, Any]:
    report = json.loads((reports / "quality.json").read_bytes())
    require(report["status"] == "scored", "scored_report_required")
    # Whitelist: no original IDs, raw claim/evidence, gold rows or responses.
    compact = {k: report[k] for k in ("protocol", "status", "scope", "source_git", "model_sha256", "release_sha256",
        "accounting", "mechanism", "C_minus_B_or_error_reduction", "candidate_rationale_coverage_identical_by_construction",
        "no_global_NEI_accuracy_claim", "Agent_gain_established", "baseline_comparison", "archived_baselines")}
    compact["routes"] = {route: {k: v for k, v in values.items() if k != "cases"}
                         for route, values in report["routes"].items()}
    require(not target.exists(), "exclusive_export")
    ordered_write(target, compact)
    return compact
