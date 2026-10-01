"""Post-exit scoring on frozen TRAIN24 only; failures never count as correct NEI."""
from __future__ import annotations

from collections import Counter
import json
import math
from pathlib import Path
import tarfile
from typing import Any

from audit_scifact_bounded_closeout import strict_whole_answer
from climate_rag.scifact_document_verifier import ARMS, PROTOCOL, audit_episode
from climate_rag.scifact_fit_selection import select_complete_fit
from climate_rag.scifact_grounding import parse_abstract
from climate_rag.scifact_read_continuation import ordered_write
from climate_rag.scifact_scoring import ClaimPrediction, parse_prediction, score_original
from climate_rag.scifact_semantic_contract import CORPUS_SHA, checked
from climate_rag.scifact_utility_runtime import ledger_cost
from prepare_scifact_document_verifier import load_frames
from prepare_scifact_natural import SELECTION_SHA
from prepare_scifact_state_supervision import member
from prepare_scifact_utility8 import ARCHIVE, ARCHIVE_SHA, PREP, TRAIN_SHA
from run_scifact_grounding_train_operator import ROOT, require, sha


def arm_costs_before_gold(ledger: Path, ids: list[int], arms: tuple[str, ...] = ARMS) -> tuple[dict[str, Any], dict[str, Any]]:
    groups: dict[str, list[str]] = {arm: [] for arm in (*arms, "unassigned")}
    slots = {f"{i}-{arm}": arm for i in ids for arm in arms}
    for path in ledger.glob("g*.reserved.json"):
        key = path.name.removesuffix(".reserved.json")
        try:
            request = json.loads(path.read_bytes())
            slot = request.get("slot")
            arm = slots.get(slot, "unassigned") if isinstance(slot, str) else "unassigned"
        except (OSError, ValueError, AttributeError):
            arm = "unassigned"  # interrupted reservation remains charged, never dropped
        groups[arm].append(key)
    result = {}
    for arm, keys in groups.items():
        elapsed, unknown = 0.0, 0
        for key in keys:
            try:
                finished = json.loads((ledger / (key+".finished.json")).read_bytes())
                ms = finished.get("response", {}).get("diagnostics", {}).get("generation_elapsed_ms")
                if type(ms) not in {float, int} or not math.isfinite(ms) or ms < 0:
                    unknown += 1
                else:
                    elapsed += ms
            except (OSError, ValueError, AttributeError):
                unknown += 1
        result[arm] = {**ledger_cost(ledger, keys), "model_backend_elapsed_ms": None if unknown else elapsed,
                       "backend_elapsed_known_lower_bound_ms": elapsed, "unknown_backend_timing_attempts": unknown}
    return {arm: result[arm] for arm in arms}, result["unassigned"]


def summarize(ids: list[int], golds: list[Any], rows: Any, corpus: Any,
              arms: tuple[str, ...] = ARMS, baseline_arm: str = "fixed") -> dict[str, Any]:
    require([g.claim_id for g in golds] == ids, "ordered_all_claims")
    report: dict[str, Any] = {"arms": {}}
    for arm in arms:
        predictions, cases = [], []
        for gold in golds:
            row = rows[gold.claim_id, arm]
            valid = row["state"] == "valid_terminal"
            prediction = parse_prediction(row["prediction"], corpus) if valid else ClaimPrediction(gold.claim_id, {})
            correct = valid and strict_whole_answer(prediction, gold)
            cases.append({"claim_id": gold.claim_id, "positive": bool(gold.evidence),
                "state": row["state"], "strict_whole_answer": correct,
                "verifier_calls": len(row["verification_feedback"]),
                "model_selected_verifier_calls": sum(f["model_selected"] for f in row["verification_feedback"]),
                "verifier_status": dict(Counter(f["status"] for f in row["verification_feedback"])),
                "physical_calls": row["physical_calls"], "elapsed_seconds": row["elapsed_seconds"]})
            predictions.append(prediction)
        report["arms"][arm] = {"planned": len(ids), "cases": cases,
            "strict_whole_answer_count": sum(c["strict_whole_answer"] for c in cases),
            "positive_count": sum(c["positive"] for c in cases),
            "positive_correct": sum(c["positive"] and c["strict_whole_answer"] for c in cases),
            "nei_correct": sum(not c["positive"] and c["strict_whole_answer"] for c in cases),
            "unresolved": sum(c["state"] != "valid_terminal" for c in cases),
            "official_micro": score_original(golds, predictions)}
    fixed, adaptive = report["arms"][baseline_arm], report["arms"]["adaptive"]
    if fixed["positive_correct"] == adaptive["positive_correct"] == 0:
        report["hypothesis"] = "not_supported_positive_grounding_remains_zero"
    elif fixed["positive_correct"] == 0:
        report["hypothesis"] = "adaptive_only_positive_recovery_requires_real_trajectory_audit_not_automatic_Agent_gain"
    else:
        report["hypothesis"] = "fixed_verifier_positive_recovery_observed_agent_gain_requires_case_audit"
    report["adaptive_minus_fixed_positive_correct"] = adaptive["positive_correct"] - fixed["positive_correct"]
    if arms == ("fixed_top1", "fixed_all", "adaptive"):
        counts = {a: report["arms"][a]["positive_correct"] for a in arms}
        recovered = [a for a in arms if counts[a] > 0]
        report["positive_recovered_arms"] = recovered
        report["fixed_baseline_arm"] = baseline_arm
        report["adaptive_minus_fixed_positive_correct_by_arm"] = {
            a: counts["adaptive"] - counts[a] for a in ("fixed_top1", "fixed_all")}
        report["hypothesis"] = ("not_supported_positive_grounding_remains_zero" if not recovered else
            "positive_recovery_observed_in_named_arms_requires_trajectory_and_both_baselines_not_automatic_Agent_gain")
    report["limits"] = "Previously consumed TRAIN24 diagnostic; not independent test, SFT evidence or causal Agent proof."
    return report


def score_after_exit(output: Path, load_tokenizer: Any, release: Any, root: Path = ROOT, *,
                     protocol: str = PROTOCOL, arms: tuple[str, ...] = ARMS,
                     audit_fn: Any = None, baseline_arm: str = "fixed",
                     comparison_limits: dict[str, Any] | None = None,
                     input_adapter: Any = None) -> dict[str, Any]:
    proof = json.loads((output / "worker-exit.json").read_bytes())
    require(proof["child_reaped"] is True, "exit_before_cost_and_gold")
    inference = output / "inference"
    cost = ledger_cost(inference / "ledger")  # preserve physical cost independently of prepared inputs
    selected_sha = SELECTION_SHA
    preparation_cost: Any = "historical frozen artifacts replayed, not newly timed online retrieval"
    input_failed = False
    if input_adapter is not None:
        selected_sha = release["selection_sha256"]
        preparation_cost = {"status": "unknown_input_binding_failed", "seconds": None}
    ids: list[int] = []
    try:
        selection = json.loads(checked(output / "prepared/selection.json", selected_sha))
        ids = [r["id"] for r in selection["selected"]]
        require(len(ids) == len(set(ids)) == 24, "all_24_fixed")
    except (ValueError, KeyError, OSError, TypeError):
        if input_adapter is None:
            raise
        ids, input_failed = [], True
    if input_adapter is not None and not input_failed:
        try:
            bound, _ = input_adapter.check_prepared(output / "prepared", release)
            preparation_cost = bound["shared_preparation_cost"]
        except (ValueError, KeyError, OSError, TypeError, RuntimeError, ImportError):
            input_failed = True
    arm_costs, unassigned = arm_costs_before_gold(inference / "ledger", ids, arms)
    missing = sum(not (inference / f"{i}-{a}/result.json").exists() for i in ids for a in arms)
    planned_slots = 24 * len(arms)  # failed input identity never shrinks the denominator
    if not ids:
        missing = planned_slots
    audit_call = audit_episode if audit_fn is None else audit_fn
    ordered_write(output / "cost-before-gold.json", {"protocol": protocol, "planned_slots": planned_slots,
        "physical_generation_cost": cost, "missing_episodes": missing, "gold_read": False,
        "arm_physical_generation_cost": arm_costs,
        "unassigned_physical_generation_cost": unassigned,
        "exit_proof_sha256": sha(output / "worker-exit.json"),
        "initial_retrieval_cost": preparation_cost})
    def no_quality(reason: str) -> dict[str, Any]:
        report = {"protocol": protocol, "status": "no_quality", "reason": reason, "planned_slots": planned_slots,
                  "gold_read": False, "cost_sha256": sha(output / "cost-before-gold.json")}
        ordered_write(output / "no-quality.json", report)
        return report
    if input_failed:
        return no_quality("prospective_input_binding_failed_cost_preserved")
    if (proof["returncode"] != 0 or proof["interrupted"] is not None or missing
            or cost["unknown_usage_attempts"] or cost["unique_physical_calls"] > 240
            or unassigned["unique_physical_calls"]):
        return no_quality("worker_failure_missing_episode_or_unknown_cost")
    try:
        tokenizer = load_tokenizer()
        corpus = {d.doc_id: d for d in (parse_abstract(json.loads(r)) for r in
            checked(output / "prepared/inference/corpus.jsonl", CORPUS_SHA).splitlines())}
        receipt = json.loads((output / "prepared/preparation.json").read_bytes())
        claims = json.loads(checked(output / "prepared/inference/claims.json", receipt["claims_sha256"]))
        frames = (input_adapter.load_frames(output / "prepared", claims, corpus, tokenizer, release)
                  if input_adapter is not None else
                  load_frames(claims, corpus, tokenizer, release["initial_inventory_sha256"]))
        require(json.loads((inference / "initial-frames.json").read_bytes()) == frames, "input_copy_changed")
        rows = {}
        for i, frame in zip(ids, frames, strict=True):
            for arm in arms:
                directory = inference / f"{i}-{arm}"
                row = json.loads((directory / "result.json").read_bytes())
                require(row["claim_id"] == i and row["arm"] == arm, "slot_identity")
                rows[i, arm] = audit_call(row, frame, inference / "ledger", directory / "private-responses", tokenizer, corpus)
        require(sum(r["physical_calls"] for r in rows.values()) == cost["unique_physical_calls"], "no_unassigned_calls")
    except (ValueError, KeyError, OSError, TypeError, RuntimeError, ImportError):
        return no_quality("physical_or_feedback_identity_failed")
    require(sha(root / "envs" / ARCHIVE) == ARCHIVE_SHA, "original_train_archive")
    with tarfile.open(root / "envs" / ARCHIVE) as bundle:
        with member(bundle, PREP + "/gold/claims_train.jsonl") as stream:
            gold = select_complete_fit(stream, TRAIN_SHA, ids, corpus)
    report = {"protocol": protocol, "status": "scored", "selection_sha256": selected_sha,
        "scoring_train_member_sha256": TRAIN_SHA, "cost_sha256": sha(output / "cost-before-gold.json"),
        **summarize(ids, list(gold.values()), rows, corpus, arms, baseline_arm), "arm_physical_generation_cost": arm_costs,
        "generation_time_note": "ledger elapsed includes prompt/journal/decoding; not end-to-end online SLA",
        "training_authorized": False}
    if comparison_limits is not None:
        report["comparison_limits"] = comparison_limits
    if input_adapter is not None:
        report["limits"] = release["input_scope"]
        report["initial_retrieval_cost"] = preparation_cost
    ordered_write(output / "quality.json", report)
    return report
