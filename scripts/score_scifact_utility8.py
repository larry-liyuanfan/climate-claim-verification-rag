"""Score only the eight selected TRAIN claims after inference has fully exited."""
from __future__ import annotations

from collections import Counter
import json
from pathlib import Path
import tarfile
from typing import Any

from audit_scifact_bounded_closeout import strict_whole_answer
from climate_rag.scifact_fit_selection import select_complete_fit
from climate_rag.scifact_grounding import parse_abstract
from climate_rag.scifact_read_continuation import ordered_write
from climate_rag.scifact_scoring import ClaimPrediction, parse_prediction, score_original
from climate_rag.scifact_semantic_contract import CORPUS_SHA, checked
from climate_rag.scifact_terminal import PROTOCOL as TERMINAL, parse_action, render_answer, to_original_prediction
from climate_rag.scifact_utility_contract import ARMS, PROTOCOL, identity
from climate_rag.scifact_utility_runtime import ledger_cost, reranker_cost
from prepare_scifact_state_supervision import member
from prepare_scifact_utility8 import ARCHIVE, ARCHIVE_SHA, PREP, TRAIN_SHA
from run_scifact_grounding_train_operator import ROOT, require, sha


def evidence_sets(pred: Any, gold: Any) -> dict[str, set[Any]]:
    result: dict[str, set[Any]] = {k: set() for k in ("complete_documents", "wrong_label", "wrong_document", "wrong_sentence")}
    for doc_id, value in pred.evidence.items():
        rats = gold.evidence.get(doc_id)
        if not rats:
            result["wrong_document"].add(doc_id)
            result["wrong_sentence"].update((doc_id, i) for i in value.sentences)
            continue
        if value.label != rats[0].label:
            result["wrong_label"].add(doc_id)
        elif any(set(r.sentences) <= set(value.sentences[:3]) for r in rats):
            result["complete_documents"].add(doc_id)
        result["wrong_sentence"].update((doc_id, i) for i in value.sentences
                                       if i not in {s for r in rats for s in r.sentences})
    return result


def summarize(ids: list[int], golds: list[Any], rows: dict[tuple[int, str], dict[str, Any]], corpus: Any) -> dict[str, Any]:
    require(len(ids) == len(set(ids)) == 8 and [g.claim_id for g in golds] == ids, "fixed_scoring_denominator")
    summary: dict[str, Any] = {}
    for arm in ARMS:
        predictions, cases = [], []
        states: Counter[str] = Counter()
        for g in golds:
            row = rows.get((g.claim_id, arm))
            state = row.get("audit_status", "unresolved") if row else "unresolved"
            states[state] += 1
            raw = row.get("prediction") if row else None
            pred = parse_prediction(raw, corpus) if raw is not None else ClaimPrediction(g.claim_id, {})
            valid = state == "valid_terminal"
            sets = evidence_sets(pred, g)
            direct = row.get("audited_direct") if row else None
            comparison = None
            if direct is not None:
                baseline = evidence_sets(parse_prediction(direct, corpus), g)
                added_errors = {k: len(sets[k] - baseline[k]) for k in ("wrong_document", "wrong_label", "wrong_sentence")}
                gained = sets["complete_documents"] - baseline["complete_documents"]
                comparison = {"gained_complete_documents": len(gained),
                    "lost_complete_documents": len(baseline["complete_documents"] - sets["complete_documents"]),
                    "added_errors": added_errors,
                    "tradeoff_lost_complete_documents": bool(baseline["complete_documents"] - sets["complete_documents"]),
                    "candidate_for_manual_trajectory_review": valid and bool(gained) and not any(added_errors.values())
                        and not (baseline["complete_documents"] - sets["complete_documents"])}
            cases.append({"claim_id": g.claim_id, "valid_terminal": valid,
                "strict_whole_answer": valid and strict_whole_answer(pred, g),
                "official_nei": not g.evidence, "nei_false_evidence": not g.evidence and bool(pred.evidence),
                "nei_valid_abstention": not g.evidence and valid and not pred.evidence,
                "nei_failure_empty_prediction": not g.evidence and not valid and not pred.evidence,
                **{k: len(v) for k, v in sets.items()}, "direct_comparison": comparison})
            predictions.append(pred)
        summary[arm] = {"planned": 8, "states": dict(states), "quality_finalized": states["unresolved"] == 0,
            "official_micro": score_original(golds, predictions), "cases": cases,
            "strict_whole_answer_count": sum(c["strict_whole_answer"] for c in cases),
            "note": "Empty failure fallbacks retain micro denominators, never count as correct NEI."}
    return summary


def reconstruct(claim_id: int, raw: str, frame: dict[str, Any], corpus: Any) -> dict[str, Any] | None:
    aliases = list(frame["alias_to_source"])
    d = parse_action(json.loads(raw), frame["observation"]["allowed_actions"], frame["visible"], aliases, 5)
    if d["action"] not in {"answer", "abstain"}:
        return None
    answer = render_answer(d, frame["visible"]) if d["action"] == "answer" else None
    result = {"protocol": TERMINAL, "answer": answer,
              "outcome": "ids_validated_semantics_unmeasured" if answer else "model_abstention:" + d["reason"]}
    return dict(to_original_prediction(claim_id, result, corpus)["prediction"])


def audit_rows(output: Path, ids: list[int], corpus: Any) -> dict[tuple[int, str], dict[str, Any]]:
    rows = {}
    for i in ids:
        direct = None
        prefix_path = output / "inference" / f"{i}-A" / "prefix.json"
        if prefix_path.exists():
            prefix = json.loads(prefix_path.read_bytes())
            p = prefix["payload"]
            require(prefix["sha256"] == identity(p), "prefix_integrity")
            if p["attempt"]["status"] == "valid_decision" and p["response"]:
                direct = reconstruct(i, p["response"]["raw"], p["frame"], corpus)
        for arm in ARMS:
            slot = output / "inference" / f"{i}-{arm}"
            if not (slot / "result.json").is_file():
                rows[(i, arm)] = {"prediction": None, "audit_status": "unresolved", "audited_direct": direct}
                continue
            row = json.loads((slot / "result.json").read_bytes())
            result = row.get("result")
            outcome = (result or {}).get("outcome", "")
            terminal = outcome == "ids_validated_semantics_unmeasured" or outcome.startswith("model_abstention:")
            state = "invalid_or_nonterminal"
            if terminal:
                attempt = result["generation_attempts"][-1]
                key = attempt["diagnostics"]["physical_attempt_id"]
                receipt = output / "inference" / "ledger" / (key + ".finished.json")
                frame_index = result.get("new_model_calls", result["model_calls"]) - 1
                frame_path = slot / f"frame-{frame_index}.json"
                if not receipt.is_file() or not frame_path.is_file():
                    state = "unresolved"
                else:
                    physical = json.loads(receipt.read_bytes())
                    require(physical["status"] == "returned", "terminal_requires_real_response")
                    rebuilt = reconstruct(i, physical["response"]["raw"], json.loads(frame_path.read_bytes()), corpus)
                    require(rebuilt is not None and rebuilt == row["prediction"], "raw_prediction_identity")
                    state = "valid_terminal"
            rows[(i, arm)] = dict(row, audit_status=state, audited_direct=direct)
    return rows


def tool_changes(output: Path, rows: dict[tuple[int, str], dict[str, Any]]) -> list[dict[str, Any]]:
    changes = []
    for (i, arm), row in rows.items():
        result = row.get("result") or {}
        directory = output / "inference" / f"{i}-{arm}"
        frame_files = sorted(directory.glob("frame-*.json"), key=lambda p: int(p.stem.split("-")[-1]))
        frames = [json.loads(p.read_bytes()) for p in frame_files]
        if arm != "A" and (directory / "initial-frame.json").exists():
            frames.insert(0, json.loads((directory / "initial-frame.json").read_bytes())["frame"])
        audit = result.get("decision_execution_audit", [])
        for index, event in enumerate(result.get("events", [])):
            scripted = event.get("origin") == "scripted_intervention"
            if not (scripted or event.get("model_selected")):
                continue
            link = next((a for a in audit if a["actual_event_index"] == index), None)
            before_index = 0 if scripted else link["attempt_index"] if link else None
            after_index = 1 if scripted else link["next_attempt_index"] if link else None
            before = frames[before_index] if before_index is not None and before_index < len(frames) else None
            after = frames[after_index] if after_index is not None and after_index < len(frames) else None
            changed = None
            if before is not None and after is not None:
                before_ids, after_ids = list(before["visible"]), list(after["visible"])
                changed = {"newly_visible_sentence_ids": sorted(set(after_ids) - set(before_ids)),
                    "lost_visible_sentence_ids": sorted(set(before_ids) - set(after_ids)),
                    "visible_order_changed": before_ids != after_ids,
                    "same_visible_set": set(before_ids) == set(after_ids),
                    "feedback": after["observation"]["feedback"],
                    "post_tool_observation_sha256": identity(after["observation"])}
            changes.append({"claim_id": i, "arm": arm, "tool": event["tool"], "status": event["status"],
                "origin": "scripted" if scripted else "model_selected", "elapsed_ms": event.get("elapsed_ms"),
                "before_requested_context": event.get("before_context"),
                "after_requested_context": event.get("requested_context"), "next_frame_change": changed})
    return changes


def score_after_exit(output: Path, root: Path = ROOT) -> dict[str, Any]:
    proof = json.loads((output / "worker-exit.json").read_bytes())
    require(proof["child_reaped"] is True and proof["parent_wait_interrupted"] is None, "inference_exit_required")
    prepared = output / "prepared"
    selection = json.loads((prepared / "selection.json").read_bytes())
    receipt = json.loads((prepared / "preparation.json").read_bytes())
    require(receipt["selection_sha256"] == sha(prepared / "selection.json"), "selection_frozen")
    ids = [r["id"] for r in selection["selected"]]
    cost = ledger_cost(output / "inference/ledger")
    rerank = reranker_cost(output / "reranker-ledger")
    missing = sum(not (output / "inference" / f"{i}-{arm}" / "result.json").is_file() for i in ids for arm in ARMS)
    ordered_write(output / "cost-before-gold.json", {"protocol": PROTOCOL, "physical_generation_cost": cost,
        "reranker_cost": rerank,
        "exit_proof_sha256": sha(output / "worker-exit.json"), "planned_slots": 24,
        "missing_slot_receipts": missing, "quality_audit_not_yet_executed": True})
    def no_quality(reason: str) -> dict[str, Any]:
        report = {"protocol": PROTOCOL, "status": "no_quality", "reason": reason,
                  "planned_slots": 24, "cost_sha256": sha(output / "cost-before-gold.json"), "gold_read": False}
        ordered_write(output / "no-quality.json", report)
        return report
    if proof["returncode"] != 0 or missing or cost["unknown_usage_attempts"] or rerank["unknown_requests"]:
        return no_quality("worker_failure_missing_receipt_or_unknown_cost")
    try:
        corpus = {d.doc_id: d for d in (parse_abstract(json.loads(r)) for r in
                  checked(prepared / "inference/corpus.jsonl", CORPUS_SHA).splitlines())}
        rows = audit_rows(output, ids, corpus)
        changes = tool_changes(output, rows)
    except (ValueError, KeyError, OSError, TypeError):
        return no_quality("raw_identity_audit_failed")
    if any(r["audit_status"] == "unresolved" for r in rows.values()):
        return no_quality("unresolved_attempt_no_gold_or_quality")
    ordered_write(output / "tool-changes-before-gold.json", changes)
    require(sha(root / "envs" / ARCHIVE) == ARCHIVE_SHA, "original_train_archive")
    with tarfile.open(root / "envs" / ARCHIVE) as bundle:
        with member(bundle, PREP + "/gold/claims_train.jsonl") as stream:
            gold = select_complete_fit(stream, TRAIN_SHA, ids, corpus)
    result = {"protocol": PROTOCOL, "scoring_train_member_sha256": TRAIN_SHA,
        "selection_sha256": receipt["selection_sha256"], "cost": cost, "reranker_cost": rerank,
        "arms": summarize(ids, list(gold.values()), rows, corpus),
        "tool_changes": changes,
        "scope": "outcome_selected_exposed_TRAIN_not_test_or_causal_effect", "training_authorized": False}
    ordered_write(output / "quality.json", result)
    return result
