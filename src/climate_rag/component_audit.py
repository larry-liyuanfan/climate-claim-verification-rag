"""Post-exit private wire reconciliation and aggregate-only scoring."""
from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import Any

from .component_decoder import decoder_identity, decoder_schema
from .component_execution import RELEASE, complete_usage, skeleton
from .component_preflight import preflight_cases
from .scifact_component_contract import ContractError, parse, require, schema_for
from .scifact_component_runtime import known_usage
from .scifact_component_scoring import denominator_summary, localization, scored_row
from .scifact_semantic_contract import checked, sha


def attachment(root: Path, kind: str, receipt: dict[str, Any], raw: str | None = None) -> None:
    require(receipt.get("truncated") is False and receipt.get("io_failed") is False
            and receipt.get("attempted_bytes") == receipt.get("stored_bytes")
            and receipt.get("dropped_bytes") == 0, "physical_receipt_incomplete")
    files = list(root.glob(f"*-{kind}.txt"))
    require(all(not p.is_symlink() for p in root.iterdir()) and len(files) <= 1, "private_receipt_files")
    blob = files[0].read_bytes() if files else b""
    require(len(blob) == receipt["stored_bytes"] and sha(blob) == receipt.get("sha256")
            == receipt.get("stored_prefix_sha256"), "physical_receipt_hash")
    if raw is not None:
        require(blob == raw.encode(), "raw_attachment_mismatch")
    if kind == "grammar":
        require(not blob, "grammar_backend_error")


def reconcile(slot: dict[str, Any], root: Path, phase: str) -> dict[str, Any]:
    directory = root / f'{phase}-{slot["slot"]:02d}'
    initial = skeleton(slot, phase)
    started, completed = directory / "started.json", directory / "completed.json"
    if not started.exists():
        require(not completed.exists() or initial["status"] == "preparation_gap", "completed_without_attempt")
        return initial
    reservation = json.loads(started.read_bytes())
    require(reservation["slot"] == slot["slot"] and reservation["phase"] == phase and
            reservation["packing"] == slot["packing"] and reservation["attempted"] is True, "attempt_identity")
    # A completed write can fail after response was fsynced. Recover verified
    # cost but NEVER promote its quality or retry an interrupted invocation.
    try:
        record = json.loads(completed.read_bytes())
        recovered = False
    except (FileNotFoundError, json.JSONDecodeError):
        record = reservation | {"status": "provider_failed", "error_category": "interrupted_unknown_cost"}
        recovered = True
    require(record["packing"] == slot["packing"] and record["slot"] == slot["slot"] and
            record["phase"] == phase and record["attempted"] is True, "completion_identity")
    spec, packed = slot["input"], slot["packing"]
    wire = json.loads(checked(directory / "wire.json", reservation["wire_sha256"]))
    require(record["wire_sha256"] == reservation["wire_sha256"] and wire["input"] == spec
            and wire["packing"] == packed and wire["canonical_schema"] == schema_for(spec)
            and wire["decoder_schema"] == decoder_schema(schema_for(spec))
            and wire["decoder"] == decoder_identity(schema_for(spec))
            and sha(wire["prompt"].encode()) == packed["prompt_sha256"], "physical_wire_identity")
    require("score" not in record, "inference_cannot_supply_score")
    response_path = directory / "response.json"
    response = None
    if response_path.exists():
        try:
            blob = response_path.read_bytes()
            response = json.loads(blob)
        except json.JSONDecodeError:
            require(recovered or record["status"] == "provider_failed", "incomplete_response")
        if response is not None:
            if "response_sha256" in record:
                require(sha(blob) == record["response_sha256"], "response_hash")
            diagnostics, usage = response["diagnostics"], response["usage"]
            require(complete_usage(usage, diagnostics), "physical_usage")
            raw = response["raw"]
            require(isinstance(raw, str) and diagnostics.get("output_tokens") == usage["output_tokens"]
                    and diagnostics.get("output_sha256") == sha(raw.encode())
                    and diagnostics.get("output_bytes") == len(raw.encode()), "physical_raw_diagnostic_join")
            require(diagnostics["actual_component_packing"] == packed and diagnostics["decoder"] == wire["decoder"], "physical_packing")
            for kind, key in (("response", "private_attachment"), ("grammar", "grammar_log")):
                attachment(directory / "private", kind, diagnostics[key], raw if kind == "response" else None)
            if not recovered:
                require(record["usage_known"] is True and record["usage"] == usage
                        and record["known_usage_lower_bound"] == usage, "response_cost_mismatch")
            else:
                record.update(usage=usage, usage_known=True, known_usage_lower_bound=usage,
                              response_sha256=sha(blob), error_category="interrupted_quality_unscored_cost_recovered")
    if record["status"] in {"valid", "schema_failed"}:
        if response is None:
            raise ContractError("missing_success_response")
        diagnostics, usage = response["diagnostics"], response["usage"]
        require(usage["input_tokens"] == packed["input_tokens"] and usage["output_tokens"] <= 512,
                "physical_usage")
        require(diagnostics.get("eos_observed") is True and diagnostics.get("reached_max_new_tokens") is False
                and record["elapsed_ms"] <= 120000, "physical_generation_contract")
        try:
            prediction = parse(response["raw"], spec)
        except ContractError:
            require(record["status"] == "schema_failed" and "prediction" not in record, "schema_status_disagrees")
        else:
            require(record["status"] == "valid" and record["prediction"] == prediction, "raw_prediction_disagrees")
    else:
        require(record["status"] == "provider_failed" and "prediction" not in record, "completion_status")
        failure_path = directory / "failure.json"
        if response is None and failure_path.exists():
            blob = failure_path.read_bytes()
            if "failure_sha256" in record:
                require(sha(blob) == record["failure_sha256"], "exception_cost_hash")
            try:
                failure = json.loads(blob)
            except json.JSONDecodeError:
                require(recovered, "partial_exception_cost")
            else:
                usage, diagnostics = failure["usage"], failure["diagnostics"]
                lower, known = known_usage(usage), complete_usage(usage, diagnostics)
                require(usage is None or (isinstance(usage, dict) and lower == usage), "exception_usage_shape")
                if recovered:
                    record.update(usage=usage if known else None, usage_known=known, known_usage_lower_bound=lower)
                else:
                    require(record["usage_known"] is known and record["usage"] == (usage if known else None)
                            and record.get("known_usage_lower_bound", {}) == lower, "exception_cost_consistency")
        require(type(record["usage_known"]) is bool and
                ((record["usage_known"] and complete_usage(record["usage"], {}))
                 or (not record["usage_known"] and record["usage"] is None)), "failed_usage_contract")
    return dict(record)


def aggregate(slots: list[dict[str, Any]], targets: list[dict[str, Any]], records: list[dict[str, Any]],
              preflight: list[dict[str, Any]], *, preflight_policy: str = "semantic-exact-v1"
              ) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    require(preflight_policy in {"semantic-exact-v1", "technical-preflight-semantic-measurement-v1"},
            "unknown_preflight_policy")
    require(len(slots) == len(targets) == len(records) == 33, "audit_matrix")
    require([r["slot"] for r in slots] == [r["slot"] for r in targets] == [r["slot"] for r in records]
            == list(range(33)), "audit_slot_order")
    rows = []
    for slot, target, record in zip(slots, targets, records, strict=True):
        prepared = {"input": slot["input"], "packing": slot["packing"], **target}
        if record["component"] is None:
            require(record["status"] == "preparation_gap" and not record["attempted"], "missing_component")
            record = record | {"component": target["component"]}
        rows.append(scored_row(prepared, record))
    reasons: Counter[str] = Counter()
    pairs: Counter[str] = Counter()
    for rationale in (r for r in rows if r["component"] == "rationale"):
        same = [r for r in rows if r["claim_id"] == rationale["claim_id"]]
        relation = next(r for r in same if r["component"] == "relation")
        screening = next(r for r in same if r["component"] == "screening")
        reasons[localization(relation, rationale, screening)] += 1
        if relation["status"] == rationale["status"] == "valid":
            pairs[f'relation_{int(relation["score"]["correct"])}_first3_{int(rationale["score"]["correct"])}'] += 1
        else:
            pairs["unscorable"] += 1
    all_records = [*preflight, *records]
    total = sum(bool(r["attempted"]) for r in all_records)
    require(total <= 37 and len(preflight) == 4, "attempt_ceiling")
    stopped, preflight_passed = False, 0
    for record, (_, expected) in zip(preflight, preflight_cases(), strict=True):
        if stopped:
            require(record["status"] in {"not_attempted_after_stop", "preparation_gap"} and not record["attempted"], "preflight_after_stop")
        passed = record["status"] == "valid" and record.get("prediction") == expected
        preflight_passed += passed
        ready = (passed if preflight_policy == "semantic-exact-v1" else
                 record["status"] == "valid" and record["usage_known"] and complete_usage(record["usage"], {}))
        stopped = stopped or not ready
    for record in records:
        if stopped:
            require(record["status"] in {"not_attempted_after_stop", "preparation_gap"} and not record["attempted"], "diagnostic_after_stop")
        stopped = stopped or record["status"] == "provider_failed"
    screens = [r["score"] for r in rows if r["component"] == "screening" and r["status"] == "valid"]
    rationales = [r["score"] for r in rows if r["component"] == "rationale" and r["status"] == "valid"]
    redundancy = [s["minimum_redundancy_after_complete_alternative"] for s in rationales
                  if s["minimum_redundancy_after_complete_alternative"] is not None]
    screening_detail = {key: sum(s[key] for s in screens) for key in (
        "gold_documents", "gold_in_pool", "candidate_missing", "correct_selected", "selected",
        "model_omitted_from_full_context", "selected_unannotated_documents")}
    screening_detail.update(valid_samples=len(screens), valid_gold_samples=sum(s["gold_documents"] > 0 for s in screens),
        all_annotated_gold_selected=sum(s["gold_documents"] > 0 and s["correct_selected"] == s["gold_documents"] for s in screens))
    rationale_detail = {key: sum(s[key] for s in rationales) for key in (
        "alternative_complete", "first3_complete", "exact_any_alternative", "nonannotated_extra_sentences", "first3_reachable")}
    rationale_detail.update(valid_samples=len(rationales), redundancy_samples=len(redundancy), redundancy_sum=sum(redundancy))
    summary = {"release": RELEASE, "scope": "already_consumed_train_oracle_diagnostic_not_generalization",
        "status": "stopped_no_semantic_effect_claim" if any(r["status"] in {"provider_failed", "not_attempted_after_stop"} for r in all_records) else "complete",
        "denominators": denominator_summary(rows), "localization": dict(reasons), "relation_rationale_pairs": dict(pairs),
        "screening_detail": screening_detail, "rationale_detail": rationale_detail,
        "screening_boundary": "conditional_pool_screening_empty_success_is_not_full_retrieval_success",
        "annotation_boundary": "unannotated_documents_and_sentences_not_proven_factually_wrong",
        "planned_preflight": 4, "preflight_valid": sum(r["status"] == "valid" for r in preflight),
        "preflight_passed": preflight_passed,
        "planned_diagnostic": 33, "total_attempted": total,
        "unknown_cost_attempts": sum(r["attempted"] and not r["usage_known"] for r in all_records),
        "known_token_lower_bound": {key: sum((r.get("usage") or r.get("known_usage_lower_bound") or {}).get(key, 0)
                                            for r in all_records) for key in ("input_tokens", "output_tokens")},
        "elapsed_ms_sum": sum(r.get("elapsed_ms", 0) for r in all_records),
        "scoring_targets_loaded_only_after_inference_exit": True, "official_dev_read": False,
        "raw_exported": False, "retry_calls": 0, "not_Agent_improvement": True}
    if preflight_policy != "semantic-exact-v1":
        summary["preflight_policy"] = preflight_policy
        summary["preflight_semantic_matches"] = summary.pop("preflight_passed")
        summary["preflight_technical_ready"] = sum(r["status"] == "valid" and r["usage_known"]
            and complete_usage(r["usage"], {}) for r in preflight)
    return summary, rows
