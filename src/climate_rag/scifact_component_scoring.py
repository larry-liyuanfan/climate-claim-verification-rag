"""Private component scoring. Never import this module from an inference worker."""
from __future__ import annotations

import json
from typing import Any

from .scifact_component_contract import parse, require


def scored_row(prepared: dict[str, Any], execution: dict[str, Any]) -> dict[str, Any]:
    """One join point; all failure rows retain their preparation stratum."""
    require(prepared["component"] == execution["component"], "scoring_component_mismatch")
    require(prepared["packing"] == execution.get("packing"), "scoring_input_identity_mismatch")
    row = dict(execution)
    require("score" not in row, "execution_must_not_supply_score")
    row["claim_id"] = prepared["claim_id"]
    row["target_doc_id"] = prepared["target"].get("target_doc_id")
    if row["component"] == "relation":
        nei = prepared["target"].get("nei_control")
        require(type(nei) is bool, "relation_stratum_missing")
        row["nei_control"] = nei
    if row["status"] == "valid":
        require(prepared["packing"]["status"] == "prepared", "unprepared_valid_execution")
        prediction = parse(json.dumps(row["prediction"]), prepared["input"])
        row["score"] = score(row["component"], prediction, prepared["target"])
    return row


def score(component: str, prediction: dict[str, Any], target: dict[str, Any]) -> dict[str, Any]:
    abstain = prediction["decision"] == "abstain"
    if component == "screening":
        relevant, pool, selected = (set(target["relevant_doc_ids"]), set(target["pool_doc_ids"]), set(prediction["document_ids"]))
        available = relevant & pool
        return {"correct": selected == available, "explicit_abstention": abstain,
                "correct_selected": len(selected & relevant), "selected": len(selected),
                "gold_documents": len(relevant), "gold_in_pool": len(available),
                "candidate_missing": len(relevant - pool), "model_omitted_from_full_context": len(available - selected),
                "selected_unannotated_documents": len(selected - relevant),
                "recall_in_pool": len(selected & relevant) / len(available) if available else None,
                "nei_control": not relevant}
    if component == "relation":
        return {"correct": not abstain and prediction["relation"] == target["relation"],
                "explicit_abstention": abstain, "nei_control": target["nei_control"],
                "label_source": target["label_source"]}
    if component != "rationale":
        raise ValueError("unknown_component")
    ordered = prediction["sentence_ids"]
    selected, first3 = set(ordered), set(ordered[:3])
    alternatives = [set(a) for a in target["alternatives"]]
    union = set().union(*alternatives)
    complete = any(a <= selected for a in alternatives)
    first_complete = any(a <= first3 for a in alternatives)
    return {"correct": first_complete, "explicit_abstention": abstain,
            "alternative_complete": complete, "first3_complete": first_complete,
            "exact_any_alternative": any(a == selected for a in alternatives),
            "nonannotated_extra_sentences": len(selected - union),
            "minimum_redundancy_after_complete_alternative": min(len(selected - a) for a in alternatives if a <= selected) if complete else None,
            "first3_reachable": target["first3_reachable"], "oracle_conditioned": True}


def localization(relation: dict[str, Any], rationale: dict[str, Any], screening: dict[str, Any] | None) -> str:
    require(type(relation.get("claim_id")) is int and relation["claim_id"] == rationale.get("claim_id"), "localization_claim_pair")
    if screening:
        require(screening.get("claim_id") == relation["claim_id"], "localization_screening_pair")
    if relation.get("status") != "valid" or rationale.get("status") != "valid":
        return "format_coverage_or_provider_gap_not_semantic_localization"
    require(type(relation.get("target_doc_id")) is int and relation["target_doc_id"] == rationale.get("target_doc_id"),
            "localization_document_pair")
    rc, sc = relation["score"]["correct"], rationale["score"]["correct"]
    if not rc and sc:
        return "relation_candidate_not_proven_training_fix"
    if rc and not sc:
        return "sentence_selection_candidate_not_proven_training_fix"
    if not rc and not sc:
        return "joint_grounding_candidate"
    if not rationale["score"].get("exact_any_alternative", False):
        return "rationale_overselection_despite_first3_coverage"
    if screening and screening.get("status") == "valid" and not screening["score"]["correct"]:
        return "document_discrimination_or_negative_label_quality_candidate"
    return "oracle_success_not_Agent_benefit_or_action_SFT_authorization"


def denominator_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    for row in rows:
        require(row.get("component") in {"screening", "relation", "rationale"}, "summary_component")
        require(row.get("status") in {"valid", "preparation_gap", "schema_failed", "provider_failed", "not_attempted_after_stop"}, "summary_status")
        if row["component"] == "relation":
            require(type(row.get("nei_control")) is bool, "relation_stratum_missing")
        if row["status"] == "valid":
            value = row.get("score", {})
            require(type(value.get("correct")) is bool and type(value.get("explicit_abstention")) is bool
                    and row.get("attempted") is True, "valid_score_contract")
        else:
            require("score" not in row, "failed_row_must_not_have_score")
        require(row["status"] != "preparation_gap" or not row.get("attempted", False), "preparation_gap_attempted")
        require(row["status"] != "not_attempted_after_stop" or not row.get("attempted", False), "stopped_slot_attempted")
    def summarize(subset: list[dict[str, Any]]) -> dict[str, Any]:
        valid = sum(r["status"] == "valid" for r in subset)
        correct = sum(r["score"]["correct"] for r in subset if r["status"] == "valid")
        return {"planned": len(subset), "prepared": sum(r["status"] != "preparation_gap" for r in subset),
                "attempted": sum(r.get("attempted", False) for r in subset), "valid": valid,
                "schema_failed": sum(r["status"] == "schema_failed" for r in subset),
                "provider_failed": sum(r["status"] == "provider_failed" for r in subset),
                "not_attempted_after_stop": sum(r["status"] == "not_attempted_after_stop" for r in subset),
                "explicit_abstention": sum(r["score"]["explicit_abstention"] for r in subset if r["status"] == "valid"),
                "correct": correct, "accuracy_given_valid": correct / valid if valid else None,
                "task_success_all_planned": correct / len(subset) if subset else None}
    result = {c: summarize([r for r in rows if r["component"] == c]) for c in ("screening", "relation", "rationale")}
    for nei in (False, True):
        result["relation_nei_control" if nei else "relation_gold_document"] = summarize(
            [r for r in rows if r["component"] == "relation" and r.get("nei_control") is nei])
    return result
