"""Post-exit TRAIN-only utility scoring; scripted candidates are program-teacher."""
from __future__ import annotations

from collections import Counter
import json
from pathlib import Path
import tarfile
from typing import Any

from climate_rag.scifact_fit_selection import select_complete_fit
from climate_rag.scifact_grounding import parse_abstract
from climate_rag.scifact_natural_contract import audit_attempt, valid_physical_prefix
from climate_rag.scifact_natural_selection import PROTOCOL, SCOPE
from climate_rag.scifact_read_continuation import ordered_write
from climate_rag.scifact_scoring import ClaimPrediction, parse_prediction, score_original
from climate_rag.scifact_semantic_contract import CORPUS_SHA, checked
from climate_rag.scifact_utility_contract import ARMS
from climate_rag.scifact_utility_runtime import direct_prediction, ledger_cost, reranker_cost
from prepare_scifact_natural import SELECTION_SHA
from prepare_scifact_state_supervision import member
from prepare_scifact_utility8 import ARCHIVE, ARCHIVE_SHA, PREP, TRAIN_SHA
from run_scifact_grounding_train_operator import ROOT, require, sha
from score_scifact_utility8 import reconstruct, strict_whole_answer, tool_changes


def audit_rows(output: Path, ids: list[int], corpus: Any, tokenizer: Any) -> dict[tuple[int, str], dict[str, Any]]:
    rows = {}
    ledger = output / "inference/ledger"
    for i in ids:
        natural = output / "inference" / f"{i}-A"
        prefix_path = natural / "prefix.json"
        direct, first = None, None
        if prefix_path.exists():
            prefix = json.loads(prefix_path.read_bytes())
            try:
                valid_physical_prefix(prefix, ledger, natural / "private-responses", tokenizer, f"{i}-A")
                first = prefix["payload"]["decision"]["action"]
                direct = direct_prediction(prefix, i, corpus)
                direct = direct["prediction"] if direct else None
            except (ValueError, KeyError, OSError, TypeError):
                pass  # No physical initial prefix: no alternate step or replacement.
        for arm in ARMS:
            directory = output / "inference" / f"{i}-{arm}"
            row = json.loads((directory / "result.json").read_bytes())
            require(row["claim_id"] == i and row["arm"] == arm, "slot_identity")
            own = [p.stem.removesuffix(".reserved") for p in sorted(ledger.glob("g*.reserved.json"))
                   if json.loads(p.read_bytes())["slot"] == f"{i}-{arm}"]
            require(set(own) == set(row["physical_generation_ids"]) and len(own) <= (5 if arm == "A" else 1),
                    "slot_physical_capacity_or_identity")
            require(not ledger_cost(ledger, own)["unknown_usage_attempts"], "unknown_slot_cost")
            if arm != "A" and first is None:
                require(not own and row["prediction"] is None, "branch_requires_valid_initial_prefix")
            result = row.get("result") or {}
            state, prediction = "unresolved", None
            attempts = result.get("generation_attempts", [])
            frames = sorted(directory.glob("frame-*.json"), key=lambda p: int(p.stem.split("-")[-1]))
            if row["audit_status"] == "valid_terminal":
                require(bool(attempts and frames), "terminal_physical_trace_missing")
                frame = json.loads(frames[-1].read_bytes())
                decision = audit_attempt(attempts[-1], frame, ledger, directory / "private-responses", tokenizer, f"{i}-{arm}")
                require(decision["action"] in {"answer", "abstain"}, "terminal_action_required")
                key = attempts[-1]["diagnostics"]["physical_attempt_id"]
                response = json.loads((ledger / (key + ".finished.json")).read_bytes())["response"]
                prediction = reconstruct(i, response["raw"], frame, corpus)
                require(prediction == row["prediction"] and prediction is not None, "audited_prediction_identity")
                state = "valid_terminal"
            else:
                require(row["prediction"] is None, "unresolved_must_not_publish_prediction")
            rows[i, arm] = dict(row, audit_status=state, prediction=prediction,
                                audited_direct=direct, initial_action=first)
    return rows


def summarize(ids: list[int], golds: list[Any], rows: dict[tuple[int, str], dict[str, Any]], corpus: Any) -> dict[str, Any]:
    require(1 <= len(ids) <= 24 and len(set(ids)) == len(ids) and [g.claim_id for g in golds] == ids,
            "fixed_planned_denominator")
    result: dict[str, Any] = {"arms": {}, "program_teacher_candidates": []}
    for arm in ARMS:
        predictions, cases = [], []
        states: Counter[str] = Counter()
        for g in golds:
            row = rows[g.claim_id, arm]
            valid = row["audit_status"] == "valid_terminal"
            states[row["audit_status"]] += 1
            pred = parse_prediction(row["prediction"], corpus) if valid else ClaimPrediction(g.claim_id, {})
            strict = valid and strict_whole_answer(pred, g)
            natural = rows[g.claim_id, "A"]
            natural_ok = natural["audit_status"] == "valid_terminal"
            natural_strict = natural_ok and strict_whole_answer(parse_prediction(natural["prediction"], corpus), g)
            direct = row.get("audited_direct")
            direct_strict = strict_whole_answer(parse_prediction(direct, corpus), g) if direct is not None else None
            recoverable = arm != "A" and direct_strict is False and strict
            case = {"claim_id": g.claim_id, "state": row["audit_status"], "strict_whole_answer": strict,
                "initial_action": row.get("initial_action"), "direct_strict": direct_strict,
                "natural_endpoint_strict": natural_strict, "natural_endpoint_resolved": natural_ok,
                "recoverable_direct_terminal_opportunity": recoverable,
                "initial_tool_strategy_comparison_only": row.get("initial_action") in {"read", "rewrite", "rerank"},
                "nei_failure_not_correct": not g.evidence and not valid,
                "nei_valid_abstention": not g.evidence and valid and not pred.evidence}
            cases.append(case)
            if recoverable:
                result["program_teacher_candidates"].append({"claim_id": g.claim_id, "arm": arm,
                    "origin": "scripted_tool_program_teacher_not_human_or_autonomous_success",
                    "slot_reference": f"inference/{g.claim_id}-{arm}", "training_authorized": False})
            predictions.append(pred)
        result["arms"][arm] = {"planned": len(ids), "states": dict(states), "cases": cases,
            "strict_whole_answer_count": sum(c["strict_whole_answer"] for c in cases),
            "official_micro": score_original(golds, predictions),
            "failure_policy": "all planned cases retained; unresolved never correct NEI"}
    return result


def score_after_exit(output: Path, tokenizer: Any, root: Path = ROOT) -> dict[str, Any]:
    proof = json.loads((output / "worker-exit.json").read_bytes())
    require(proof["child_reaped"] is True, "inference_exit_required")
    prepared = output / "prepared"
    receipt = json.loads((prepared / "preparation.json").read_bytes())
    require(receipt["selection_sha256"] == SELECTION_SHA, "preparation_selection_binding")
    selection = json.loads(checked(prepared / "selection.json", SELECTION_SHA))
    ids = [r["id"] for r in selection["selected"]]
    require(len(ids) == len(set(ids)) == 24, "frozen_24_denominator")
    cost, rerank = ledger_cost(output / "inference/ledger"), reranker_cost(output / "reranker-ledger")
    missing = sum(not (output / "inference" / f"{i}-{a}" / "result.json").exists() for i in ids for a in ARMS)
    tool_events = []
    unknown_tool_slots = 0
    for i in ids:
        for arm in ARMS:
            path = output / "inference" / f"{i}-{arm}" / "result.json"
            if path.exists():
                try:
                    row = json.loads(path.read_bytes())
                    events = (row.get("result") or {}).get("events", [])
                    require(isinstance(events, list) and all(isinstance(e, dict) and "status" in e for e in events),
                            "tool_event_receipt_shape")
                    tool_events.extend(e for e in events if "tool" in e and e["status"] != "skipped"
                                       and not e.get("shared_prefix_reference"))
                except (ValueError, OSError, TypeError, AttributeError):
                    unknown_tool_slots += 1
    ordered_write(output / "cost-before-gold.json", {"protocol": PROTOCOL, "physical_generation_cost": cost,
        "reranker_cost": rerank, "exit_proof_sha256": sha(output / "worker-exit.json"),
        "planned_slots": 72, "missing_slot_receipts": missing,
        "unique_tool_events_including_initial_retrieval": None if unknown_tool_slots or missing else len(tool_events),
        "known_tool_event_lower_bound": len(tool_events), "unknown_tool_slots": unknown_tool_slots + missing,
        "tool_status_counts": dict(Counter(e["status"] for e in tool_events)), "gold_read": False})
    def no_quality(reason: str) -> dict[str, Any]:
        report = {"protocol": PROTOCOL, "status": "no_quality", "reason": reason, "planned_slots": 72,
                  "cost_sha256": sha(output / "cost-before-gold.json"), "gold_read": False}
        ordered_write(output / "no-quality.json", report)
        return report
    if (proof["returncode"] != 0 or proof["parent_wait_interrupted"] is not None or missing
            or unknown_tool_slots or cost["unknown_usage_attempts"] or rerank["unknown_requests"]):
        return no_quality("worker_failure_missing_receipt_or_unknown_cost")
    if cost["unique_physical_calls"] > 168 or rerank["physical_requests"] > 48 or rerank["requested_pairs"] > 960:
        return no_quality("physical_capacity_exceeded")
    try:
        tokenizer = tokenizer() if callable(tokenizer) else tokenizer
        corpus = {d.doc_id: d for d in (parse_abstract(json.loads(r)) for r in
                  checked(prepared / "inference/corpus.jsonl", CORPUS_SHA).splitlines())}
        rows = audit_rows(output, ids, corpus, tokenizer)
        changes = tool_changes(output, rows)
    except (ValueError, KeyError, OSError, TypeError, RuntimeError, ImportError):
        return no_quality("physical_identity_audit_failed")
    ordered_write(output / "tool-changes-before-gold.json", changes)
    require(sha(root / "envs" / ARCHIVE) == ARCHIVE_SHA, "original_train_archive")
    with tarfile.open(root / "envs" / ARCHIVE) as bundle:
        with member(bundle, PREP + "/gold/claims_train.jsonl") as stream:
            gold = select_complete_fit(stream, TRAIN_SHA, ids, corpus)
    report = {"protocol": PROTOCOL, "scope": SCOPE, "scoring_train_member_sha256": TRAIN_SHA,
        "selection_sha256": SELECTION_SHA, "cost_sha256": sha(output / "cost-before-gold.json"),
        **summarize(ids, list(gold.values()), rows, corpus), "training_authorized": False,
        "multi_step_strategy_differences_are_not_single_action_causal_effects": True}
    ordered_write(output / "quality.json", report)
    return report
