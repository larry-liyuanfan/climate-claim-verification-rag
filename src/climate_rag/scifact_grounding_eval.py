"""Single-attempt paired grounding evaluation; no scorer/gold in inference."""
from __future__ import annotations

import json
import time
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from .scifact_grounding import Abstract, GoldClaim
from .scifact_component_runtime import known_usage
from .component_execution import complete_usage, durable as write_once
from .scifact_grounding_sft import CONFIG, canonical_context, prediction, require, token_record
from .scifact_scoring import parse_prediction, score_original
from .scifact_semantic_contract import encoded, sha


class GroundingPersistenceError(RuntimeError):
    def __init__(self, record: dict[str, Any]) -> None:
        super().__init__("grounding_paid_cost_not_durable_no_retry")
        self.partial_report = record


def evaluate_arm(rows: Sequence[Mapping[str, Any]], corpus: Mapping[int, Abstract],
                 provider: Any, output: Path, arm: str) -> dict[str, Any]:
    """A failed/unknown attempt retains its reservation and cannot be retried.

    Only the caller switches adapter state. Both arms get exactly the same
    rows, original full contexts and terminal schema, with at most 12 calls.
    """
    require(arm in {"base", "adapted"} and len(rows) == 12, "paired_arm_shape")
    require(len({r["claim_id"] for r in rows}) == 12, "duplicate_eval_claim")
    output.mkdir(mode=0o700)
    identity = sha(encoded(rows))
    write_once(output / "initialized.json", {"arm": arm, "input_identity": identity})
    records = []
    for index, row in enumerate(rows):
        ctx = canonical_context(row["context"], corpus)
        packed = token_record(provider.base.tokenizer, ctx)
        require(packed == row["packing"], "frozen_prompt_mismatch")
        slot = output / f"slot-{index:02d}"
        slot.mkdir(mode=0o700)
        if hasattr(provider, "start_slot"):
            provider.start_slot(slot / "private")
        record: dict[str, Any] = {"claim_id": row["claim_id"], "arm": arm,
            "input_identity": sha(encoded(row)), "status": "unknown",
            "attempted": True, "usage_known": False, "usage": None,
            "prediction": None}
        write_once(slot / "started.json", record)
        started = time.perf_counter()
        stage = "provider"
        try:
            response = provider.generate(ctx["observation"], ctx["schema"],
                CONFIG["max_output_tokens"], CONFIG["max_seconds_per_call"])
            usage = response.get("usage")
            record["usage"] = usage
            record["usage_known"] = complete_usage(usage, response.get("diagnostics", {}))
            # Paid usage exists before persistence; a write error must retain it.
            try:
                write_once(slot / "response.json", response)
            except Exception as exc:
                record["persistence_failed"] = True
                raise GroundingPersistenceError(record) from exc
            stage = "output_contract"
            require(isinstance(usage, dict) and set(usage) == {"input_tokens", "output_tokens"}
                    and all(type(v) is int and v >= 0 for v in usage.values()), "unknown_usage")
            require(record["usage_known"], "unknown_output_usage")
            require(usage["input_tokens"] == packed["input_tokens"]
                    and usage["output_tokens"] <= CONFIG["max_output_tokens"], "actual_token_budget")
            diagnostics = response.get("diagnostics", {})
            require(diagnostics.get("eos_observed") is True
                    and diagnostics.get("reached_max_new_tokens") is False, "incomplete_output")
            require(time.perf_counter() - started <= CONFIG["max_seconds_per_call"], "deadline")
            stage = "parse"
            from .scifact_component_contract import unique_pairs
            action = json.loads(response["raw"], object_pairs_hook=unique_pairs)
            record.update(prediction=prediction(row["claim_id"], action, ctx, corpus), status="valid")
        except Exception as exc:
            record.update(status="failed", error_type=type(exc).__name__,
                          stop_required=stage != "parse")
            if hasattr(exc, "usage"):
                record["usage_lower_bound"] = exc.usage
                record["usage"] = exc.usage
                record["usage_known"] = complete_usage(exc.usage, getattr(exc, "diagnostics", {}))
            failure = {"error_type": type(exc).__name__, "usage": record["usage"],
                       "usage_known": record["usage_known"],
                       "persistence_failed": record.get("persistence_failed", False)}
            try:
                write_once(slot / "failure.json", failure)
            except Exception as persist:
                raise GroundingPersistenceError(record) from persist
        record["elapsed_ms"] = (time.perf_counter() - started) * 1000
        try:
            write_once(slot / "completed.json", record)
        except Exception as exc:
            raise GroundingPersistenceError(record) from exc
        records.append(record)
        # Unknown cost stops the arm. Missing matrix cannot open validation.
        if not record["usage_known"] or record.get("stop_required"):
            break
    result = {"arm": arm, "input_identity": identity, "attempts": len(records),
              "records": records, "gold_loaded": False,
              "execution_kind": getattr(provider, "kind", "fixture")}
    result["physical_files_sha256"] = {p.relative_to(output).as_posix(): sha(p.read_bytes())
                                        for p in output.rglob("*") if p.is_file()}
    write_once(output / "complete.json", result)
    return result


def audit_arm(directory: Path, rows: Sequence[Mapping[str, Any]],
              corpus: Mapping[int, Abstract]) -> dict[str, Any]:
    """Reconcile physical raw responses with production parsing, no generation."""
    require(not directory.is_symlink(), "audit_directory_symlink")
    result = json.loads((directory / "complete.json").read_bytes())
    require(result["input_identity"] == sha(encoded(rows)), "audit_frozen_input_identity")
    require(len(rows) == 12 and len(result["records"]) == result["attempts"] <= 12, "audit_matrix")
    physical = {p.relative_to(directory).as_posix(): sha(p.read_bytes())
                for p in directory.rglob("*") if p.is_file() and p != directory / "complete.json"}
    require(not any(p.is_symlink() for p in directory.rglob("*"))
            and physical == result["physical_files_sha256"], "audit_physical_files")
    require({p.name for p in directory.iterdir()} == {"initialized.json", "complete.json"}
            | {f"slot-{i:02d}" for i in range(result["attempts"])}, "audit_orphan_attempt")
    for index, (row, record) in enumerate(zip(rows[:result["attempts"]], result["records"], strict=True)):
        slot = directory / f"slot-{index:02d}"
        start = json.loads((slot / "started.json").read_bytes())
        finished = json.loads((slot / "completed.json").read_bytes())
        require(finished == record and record["claim_id"] == row["claim_id"]
                and start["input_identity"] == record["input_identity"] == sha(encoded(row))
                and start["attempted"] is True, "audit_slot_identity")
        response_file = slot / "response.json"
        if record.get("persistence_failed"):
            # A partial physical file is still hash-bound above, but is not a
            # complete response or quality evidence. Durable failure + usage
            # are the only cost record; never parse/replay the truncated bytes.
            require(record["status"] == "failed" and record["prediction"] is None,
                    "audit_persistence_failure_status")
        elif response_file.exists():
            response = json.loads(response_file.read_bytes())
            require(record["usage"] == response.get("usage")
                    and record["usage_known"] == complete_usage(response.get("usage"), response.get("diagnostics", {})),
                    "audit_response_cost")
            if result["execution_kind"] == "local_model":
                from .component_audit import attachment
                diag = response["diagnostics"]
                attachment(slot / "private", "response", diag["private_attachment"], response["raw"])
                attachment(slot / "private", "grammar", diag["grammar_log"])
            if record["status"] == "valid":
                ctx = canonical_context(row["context"], corpus)
                from .scifact_component_contract import unique_pairs
                parsed = json.loads(response["raw"], object_pairs_hook=unique_pairs)
                require(record["prediction"] == prediction(row["claim_id"], parsed, ctx, corpus), "audit_raw_prediction")
                require(response["usage"]["input_tokens"] == row["packing"]["input_tokens"]
                        and response["usage"]["output_tokens"] <= CONFIG["max_output_tokens"]
                        and response["diagnostics"].get("eos_observed") is True
                        and response["diagnostics"].get("reached_max_new_tokens") is False,
                        "audit_valid_output_budget")
        else:
            require(record["status"] == "failed", "missing_raw_valid_prediction")
        if record["status"] == "failed":
            failure = json.loads((slot / "failure.json").read_bytes())
            require(failure["usage"] == record["usage"] and failure["usage_known"] == record["usage_known"],
                    "audit_failure_cost")
            require(failure.get("persistence_failed", False) == record.get("persistence_failed", False),
                    "audit_failure_persistence_status")
    return dict(result)


def score_arm(result: Mapping[str, Any], gold: Sequence[GoldClaim],
              corpus: Mapping[int, Abstract]) -> dict[str, Any]:
    """Separate post-inference scoring; failure-empty != valid NEI abstention."""
    records = result["records"]
    require(len(gold) == 12 and len({g.claim_id for g in gold}) == 12, "gold_matrix")
    require({r["claim_id"] for r in records} <= {g.claim_id for g in gold}, "unknown_result_claim")
    by_id = {r["claim_id"]: r for r in records}
    require(len(by_id) == len(records), "duplicate_result_claim")
    preds = []
    failed = unknown = unjudged = complete_any = exact = valid_nei_abstention = 0
    for g in gold:
        row = by_id.get(g.claim_id)
        valid = row is not None and row["status"] == "valid"
        failed += not valid
        unknown += row is not None and not row["usage_known"]
        raw = row["prediction"]["prediction"] if valid and row is not None else {"id": g.claim_id, "evidence": {}}
        pred = parse_prediction(raw, corpus)
        preds.append(pred)
        if valid and not g.evidence and not pred.evidence:
            valid_nei_abstention += 1
        for doc, item in pred.evidence.items():
            if doc not in g.evidence:
                unjudged += 1
                continue
            alternatives = g.evidence[doc]
            label_ok = item.label == alternatives[0].label
            complete_any += label_ok and any(set(r.sentences) <= set(item.sentences) for r in alternatives)
            exact += label_ok and any(set(r.sentences) == set(item.sentences) for r in alternatives)
    official = score_original(gold, preds)
    return {"input_identity": result["input_identity"], "claims": len(gold),
        "attempts": result["attempts"], "not_attempted": len(gold) - result["attempts"],
        "planned_unsuccessful": failed, "unknown_usage": unknown,
        "stop_required": sum(bool(r.get("stop_required")) for r in records),
        "correctly_rationalized_documents": official["metrics"]["abstract_rationalized"]["correct"],
        "nei_false_evidence": official["separate_diagnostics_not_official_f1"]["nei_claims_with_false_evidence"],
        "valid_nei_abstention": valid_nei_abstention, "complete_alternative_any_position": complete_any,
        "exact_alternative": exact, "unjudged_extra_documents_not_human_false": unjudged,
        "official": official, "cost_including_failures": {
            "input_tokens_known_lower_bound": sum(known_usage(r["usage"] or r.get("usage_lower_bound", {})).get("input_tokens", 0) for r in records),
            "output_tokens_known_lower_bound": sum(known_usage(r["usage"] or r.get("usage_lower_bound", {})).get("output_tokens", 0) for r in records),
            "totals_are_lower_bounds": unknown > 0,
            "elapsed_ms": sum(r["elapsed_ms"] for r in records)},
        "boundary": "TRAIN_internal_component_split_not_Agent_or_external_test_gain"}
