"""Read-only posthoc attribution; never replay controllers, pack oracle reads or infer."""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
import os
from pathlib import Path
import subprocess
import tarfile
from typing import Any

from audit_scifact_bounded_closeout import strict_whole_answer
from climate_rag.agent_v3 import V3Budget
from climate_rag.local_bounded_scifact_provider import observation_identity
from climate_rag.scifact_fit_selection import select_complete_fit
from climate_rag.scifact_grounding import Abstract, GoldClaim, parse_abstract
from climate_rag.scifact_natural_contract import audit_attempt, ranked_preview_read, valid_physical_prefix
from climate_rag.scifact_observation_receipts import _transition
from climate_rag.scifact_read_continuation import ordered_write
from climate_rag.scifact_scoring import parse_prediction
from climate_rag.scifact_semantic_contract import CORPUS_SHA, TOKENIZER_SHA, checked
from climate_rag.scifact_terminal import source_from_abstract
from climate_rag.scifact_train_diagnostic import coverage, document_opportunities
from climate_rag.scifact_utility_contract import ARMS, identity, validate_prefix
from climate_rag.scifact_utility_runtime import ledger_cost, reranker_cost
from prepare_scifact_natural import SELECTION_SHA
from prepare_scifact_state_supervision import member
from prepare_scifact_utility8 import ARCHIVE, ARCHIVE_SHA, PREP, TRAIN_SHA
from run_scifact_grounding_train_operator import ROOT, require, sha
from score_scifact_utility8 import reconstruct, tool_changes

EXECUTION_SHA = "723c6a8fc4f0956ba91497b278d5f8fe65dc9fee"
EXECUTION_RELEASE = "3abdaa9ef67d8a3d026d1d9e6494b99d56670e7abcc6d1e4489ff6171fd24a3a"
JOB = "31895661"
VERSION = "scifact-natural-posthoc-attribution-v1"
TERMINAL_STATES = {"COMPLETED", "FAILED", "CANCELLED", "TIMEOUT", "OUT_OF_MEMORY", "NODE_FAIL", "PREEMPTED", "BOOT_FAIL"}


def frame_visible(frame: dict[str, Any], attempt: dict[str, Any], corpus: dict[int, Abstract],
                  source_hashes: dict[str, str]) -> dict[int, list[int]]:
    """Original IDs/order, exact citable text and current source hashes; previews excluded."""
    aliases = frame["alias_to_source"]
    visible = frame["visible"]
    require(len(set(aliases.values())) == len(aliases), "alias_source_bijection")
    require(frame["observation"]["current_citable"] == [
        {"sentence_id": sid, "text": v["text"]} for sid, v in visible.items()], "citable_order_or_text")
    require(attempt["visible_sentence_sha256"] == {sid: v["text_sha256"] for sid, v in visible.items()}
            and attempt["diagnostics"]["actual_observation"] == observation_identity(frame["observation"])
            and attempt["input_prompt_tokens"] == frame["prompt_tokens"], "generation_frame_binding")
    result: dict[int, list[int]] = {}
    for sid, v in visible.items():
        alias, index = sid.rsplit(":", 1)
        require(index.isdecimal() and str(int(index)) == index and aliases[alias] == v["source_id"], "original_alias")
        doc_id, number = int(aliases[alias]), int(index)
        source = source_from_abstract(corpus[doc_id])
        require(str(doc_id) == aliases[alias] and type(v["sentence_index"]) is int
                and number == v["sentence_index"] and 0 <= number < len(source.sentences), "original_sentence_index")
        require(v["text"] == source.sentences[number]
                and v["text_sha256"] == hashlib.sha256(v["text"].encode()).hexdigest()
                and v["source_text_sha256"] == source.text_sha256 == source_hashes[alias], "visible_source_hash")
        result.setdefault(doc_id, []).append(number)
    return result


def candidate_fulltext(prefix: dict[str, Any], corpus: dict[int, Abstract]) -> dict[int, list[int]]:
    p = validate_prefix(prefix, prefix["payload"]["claim"], V3Budget())
    require(p["frame"]["alias_to_source"] == {v: k for k, v in p["aliases"].items()}, "prefix_inverse_aliases")
    hashes = p["events"][0]["source_sha256"]
    result = {}
    for row in p["sources"]:
        source = source_from_abstract(corpus[int(row["source_id"])])
        require(row["title"] == source.title and row["sentences"] == list(source.sentences)
                and row["sha256"] == source.text_sha256 == hashes[p["aliases"][row["source_id"]]], "candidate_source_hash")
        result[int(row["source_id"])] = list(range(len(source.sentences)))
    return result


def coverage_detail(gold: GoldClaim, view: dict[int, list[int]] | None) -> dict[str, Any]:
    if not gold.evidence:
        return {"status": "NEI_not_applicable", "coverage": None, "document_opportunities": [], "gold_documents": 0}
    if view is None:
        return {"status": "unknown", "coverage": None, "document_opportunities": None, "gold_documents": len(gold.evidence)}
    return {"status": "observed", "coverage": coverage(gold, view),
            "document_opportunities": document_opportunities(gold, view), "gold_documents": len(gold.evidence)}


def initial_diagnosis(gold: GoldClaim, pool: dict[int, list[int]] | None,
                      initial: dict[str, Any] | None, corpus: dict[int, Abstract]) -> dict[str, Any]:
    pool_detail = coverage_detail(gold, pool)
    detail = coverage_detail(gold, initial["visible"] if initial else None)
    category = "unknown_initial_or_candidate_integrity"
    if not gold.evidence:
        category = "official_NEI_separate"
    elif pool is not None and not coverage(gold, pool)["complete"]:
        category = "candidate_pool_not_complete_under_frozen_contract"
    elif initial is not None and pool is not None:
        if not coverage(gold, initial["visible"])["complete"]:
            category = "candidate_complete_but_initial_context_incomplete"
        elif not coverage(gold, initial["visible"])["first3_reachable"]:
            category = "initial_semantically_complete_but_strict_first3_unreachable"
        elif initial["decision"]["action"] in {"answer", "abstain"}:
            correct = strict_whole_answer(parse_prediction(initial["prediction"], corpus), gold)
            category = ("initial_sufficient_correct_terminal" if correct else
                        "initial_sufficient_legal_abstention" if initial["decision"]["action"] == "abstain" else
                        "initial_sufficient_legal_wrong_answer")
        else:
            category = "initial_sufficient_tool_selected"
    return {"category": category, "candidate_fulltext": pool_detail, "initial_visible": detail,
            "unexecuted_read_reachability": "unknown", "unexecuted_rewrite_reachability": "unknown",
            "candidate_hit_is_not_executable_gold_read": True, "whole_corpus_opportunity": "not_measured"}


def endpoint(gold: GoldClaim, row: dict[str, Any], corpus: dict[int, Abstract]) -> str:
    if row.get("audit_status") != "valid_terminal" or row.get("prediction") is None:
        return "unresolved"
    return "correct" if strict_whole_answer(parse_prediction(row["prediction"], corpus), gold) else "wrong"


def aligned_tools(gold: GoldClaim, arm: str, row: dict[str, Any], frames: list[Any],
                  changes: list[dict[str, Any]], corpus: dict[int, Abstract]) -> list[dict[str, Any]]:
    """Only observed completed tools may establish a coverage transition, never causality."""
    result = row.get("result") or {}
    events = [(i, e) for i, e in enumerate(result.get("events", []))
              if e.get("model_selected") or e.get("origin") == "scripted_intervention"]
    links = result.get("decision_execution_audit", [])
    out = []
    for position, (index, event) in enumerate(events):
        scripted = event.get("origin") == "scripted_intervention"
        origin = "scripted" if scripted else "model_selected"
        item = {"event_index": index, "tool": event.get("tool"), "origin": origin,
                "execution_status": event.get("status"), "attribution_status": "unknown",
                "transition_insufficient_to_complete": None, "causal_effect": "not_identified",
                "transition_outcome": "unknown",
                "final_outcome": endpoint(gold, row, corpus), "final_prediction": row.get("prediction")}
        try:
            require((arm == "A") != scripted and not (scripted and event.get("model_selected")), "arm_origin")
            link = next((entry for entry in links if entry["actual_event_index"] == index), None)
            before, after = (0, 1) if scripted else (link["attempt_index"], link["next_attempt_index"]) if link else (None, None)
            require(type(before) is int and type(after) is int and 0 <= before < after < len(frames), "missing_event_frame")
            assert isinstance(before, int) and isinstance(after, int)
            prior, following = frames[before], frames[after]
            require(prior is not None and following is not None and event["status"] == "completed", "unobserved_completed_tool")
            proposal = following["executed_proposal"] if scripted else prior["decision"]
            _transition(event, prior["source_state"], proposal, scripted=scripted)
            require(identity(following["source_state"]) == identity(event), "post_tool_state_binding")
            if not scripted:
                assert link is not None
                require(prior["decision"]["action"] == link["strict_action"] == event["tool"] and link["strict_status"] == "valid_decision"
                        and link["actual_event_status"] == "completed" and link["next_feedback"] == following["feedback"],
                        "proposal_execution_feedback_link")
                if event["tool"] == "read":
                    require(prior["decision"]["source_ids"] == event["requested_context"], "read_argument_identity")
            require(position < len(changes), "tool_change_missing")
            change = changes[position]
            require(change["origin"] == origin and change["tool"] == event["tool"] and change["status"] == "completed"
                    and change["next_frame_change"] is not None
                    and change["next_frame_change"]["post_tool_observation_sha256"] == following["observation_sha256"]
                    and change["before_requested_context"] == event["before_context"]
                    and change["after_requested_context"] == event["requested_context"], "tool_change_binding")
            a, b = coverage_detail(gold, prior["visible"]), coverage_detail(gold, following["visible"])
            transition = bool(gold.evidence) and not a["coverage"]["complete"] and b["coverage"]["complete"]
            item.update(attribution_status="observed" if gold.evidence else "NEI_not_applicable",
                before_coverage=a, after_coverage=b, transition_insufficient_to_complete=transition if gold.evidence else None,
                first3_transition=(not a["coverage"]["first3_reachable"] and b["coverage"]["first3_reachable"]) if gold.evidence else None,
                before_frame_sha256=prior["frame_sha256"], next_frame_sha256=following["frame_sha256"],
                next_feedback=following["feedback"], next_decision=following["decision"],
                proposal=proposal,
                tool_change=change,
                transition_outcome=("not_applicable" if not transition else
                    "strict_first3_unreachable" if not b["coverage"]["first3_reachable"] else endpoint(gold, row, corpus)))
        except (ValueError, KeyError, TypeError, IndexError):
            pass
        out.append(item)
    return out


def require_ready(run: Path, scheduler_state: str) -> dict[str, Any]:
    require(bool(scheduler_state.split()) and scheduler_state.split()[0].rstrip("+") in TERMINAL_STATES, "job_not_terminal")
    proof_path, cost_path = run / "worker-exit.json", run / "cost-before-gold.json"
    proof, cost = json.loads(proof_path.read_bytes()), json.loads(cost_path.read_bytes())
    require(proof["child_reaped"] is True and proof["release_sha256"] == EXECUTION_RELEASE, "child_not_reaped_or_wrong_release")
    require(cost["exit_proof_sha256"] == sha(proof_path), "persisted_cost_exit_binding")
    return {"worker_exit_sha256": sha(proof_path), "persisted_cost_sha256": sha(cost_path)}


def observed_slot(run: Path, claim_id: int, arm: str, corpus: dict[int, Abstract], tokenizer: Any) -> tuple[dict[str, Any], list[Any], list[Any]]:
    directory, ledger = run / "inference" / f"{claim_id}-{arm}", run / "inference/ledger"
    fallback = {"claim_id": claim_id, "arm": arm, "audit_status": "unresolved", "prediction": None, "result": None}
    try:
        row = json.loads((directory / "result.json").read_bytes())
        require(row["claim_id"] == claim_id and row["arm"] == arm, "row_identity")
        own = [p.stem.removesuffix(".reserved") for p in sorted(ledger.glob("g*.reserved.json"))
               if json.loads(p.read_bytes())["slot"] == f"{claim_id}-{arm}"]
        declared = row["physical_generation_ids"]
        require(len(own) <= (5 if arm == "A" else 1) and len(declared) == len(set(declared))
                and set(own) == set(declared), "physical_inventory_mismatch")
        attempts = (row.get("result") or {}).get("generation_attempts", [])
        actual_ids = [a["diagnostics"]["physical_attempt_id"] for a in attempts]
        prefix, intervention = None, None
        shared: list[str] = []
        if arm != "A" and attempts:
            natural = run / "inference" / f"{claim_id}-A"
            prefix = json.loads((natural / "prefix.json").read_bytes())
            valid_physical_prefix(prefix, ledger, natural / "private-responses", tokenizer, f"{claim_id}-A")
            shared = [prefix["payload"]["attempt"]["diagnostics"]["physical_attempt_id"]]
            intervention = ranked_preview_read(prefix)[0] if arm == "B" else {"action": "rerank"}
        logical = shared + own
        declared_logical = row["logical_generation_ids"]
        require(len(actual_ids) == len(set(actual_ids)) == len(declared_logical) == len(set(declared_logical))
                and set(actual_ids) == set(declared_logical) and actual_ids[:len(shared)] == shared
                and len(actual_ids) == len(logical) and set(actual_ids) == set(logical), "logical_inventory_mismatch")
        require(not ledger_cost(ledger, logical)["unknown_usage_attempts"], "unknown_slot_cost")
    except (ValueError, KeyError, TypeError, OSError):
        return fallback, [], []
    result = row.get("result") or {}
    attempts, events = result.get("generation_attempts", []), result.get("events", [])
    frames: list[Any] = []
    active = events[0] if events and events[0].get("status") == "completed" else None
    previous_aliases: dict[str, str] = {}
    for i, attempt in enumerate(attempts):
        try:
            if i > 0:
                index = 1 if arm != "A" and i == 1 else next((entry["actual_event_index"]
                    for entry in result["decision_execution_audit"] if entry["next_attempt_index"] == i), None)
                if index is not None:
                    active = events[index] if events[index]["status"] == "completed" else None
            path = directory / ("initial-frame.json" if arm != "A" and i == 0 else f"frame-{i if arm == 'A' else i-1}.json")
            frame = json.loads(path.read_bytes())
            if arm != "A" and i == 0:
                frame = frame["frame"]
                require(prefix is not None and identity(frame) == identity(prefix["payload"]["frame"]), "shared_initial_identity")
            require(active is not None and all(frame["alias_to_source"].get(k) == v for k, v in previous_aliases.items()), "source_state_or_alias_drift")
            assert active is not None
            require(attempt["requested_context"] == active["requested_context"], "actual_requested_context")
            slot = f"{claim_id}-A" if arm != "A" and i == 0 else f"{claim_id}-{arm}"
            private = run / "inference" / slot / "private-responses"
            decision = audit_attempt(attempt, frame, ledger, private, tokenizer, slot)
            require(attempt.get("controller_legal") is not False, "illegal_proposal")
            view = frame_visible(frame, attempt, corpus, active["source_sha256"])
            previous_aliases = dict(frame["alias_to_source"])
            physical = json.loads((ledger / (attempt["diagnostics"]["physical_attempt_id"] + ".finished.json")).read_bytes())
            frames.append({"visible": view, "decision": decision,
                "source_state": dict(active), "executed_proposal": intervention if arm != "A" and i == 1 else None,
                "physical_attempt_id": attempt["diagnostics"]["physical_attempt_id"],
                "prediction": reconstruct(claim_id, physical["response"]["raw"], frame, corpus),
                "frame_sha256": identity(frame), "observation_sha256": identity(frame["observation"]),
                "feedback": frame["observation"]["feedback"]})
        except (ValueError, KeyError, TypeError, OSError, IndexError):
            frames.append(None)
    if (not frames or any(f is None for f in frames) or frames[-1]["prediction"] is None
            or row.get("audit_status") != "valid_terminal" or frames[-1]["prediction"] != row.get("prediction")):
        row = dict(row, audit_status="unresolved", prediction=None)
    try:
        changes = tool_changes(run, {(claim_id, arm): row}) if frames and all(f is not None for f in frames) else []
    except (ValueError, KeyError, TypeError, OSError, IndexError):
        row, changes = dict(row, audit_status="unresolved", prediction=None), []
    return row, frames, changes


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--release", type=Path, required=True)
    parser.add_argument("--release-sha", required=True)
    args = parser.parse_args()
    release = json.loads(checked(args.release, args.release_sha))
    source = Path(__file__).resolve().parents[1]
    audit_sha = (source / "SOURCE_REVISION").read_text().strip()
    require(release["authorization"] == "coordinator_readonly_posthoc_release"
            and release["job_id"] == JOB and release["execution_source_git"] == EXECUTION_SHA
            and release["audit_source_git"] == audit_sha and len(audit_sha) == 40, "separate_audit_release_required")
    require(os.name == "posix" and bool(os.environ.get("SLURM_JOB_ID")), "allocated_CPU_posthoc_only")
    run = ROOT / "runs/scifact-natural-fit24-v1-20261001"
    status = subprocess.run(["sacct", "-j", JOB, "-X", "-n", "-P", "--format=State"],
                            check=True, capture_output=True, text=True).stdout.strip().splitlines()
    require(len(status) == 1, "unambiguous_terminal_accounting")
    binding = require_ready(run, status[0].strip("|"))
    require(json.loads((run / "reserved.json").read_bytes())["source_git"] == EXECUTION_SHA, "execution_source_changed")
    selected = json.loads(checked(run / "prepared/selection.json", SELECTION_SHA))["selected"]
    ids = [r["id"] for r in selected]
    require(len(ids) == len(set(ids)) == 24, "fixed_24_only")
    output = ROOT / "posthoc" / ("scifact-natural-attribution-" + audit_sha[:12])
    require(release["output"] == str(output) and not output.exists(), "new_audit_output_only")
    tokenizer_path = Path(release["tokenizer_directory"]).resolve()
    require(tokenizer_path.is_relative_to(ROOT / "envs"), "existing_project_tokenizer_only")
    for name, expected in TOKENIZER_SHA.items():
        checked(tokenizer_path / name, expected)
    from transformers.models.auto.tokenization_auto import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, local_files_only=True)
    corpus = {d.doc_id: d for d in (parse_abstract(json.loads(r)) for r in
              checked(run / "prepared/inference/corpus.jsonl", CORPUS_SHA).splitlines())}
    slots: dict[tuple[int, str], tuple[dict[str, Any], list[Any], list[Any]]] = {}
    pools: dict[int, dict[int, list[int]] | None] = {}
    initial: dict[int, Any] = {}
    for i in ids:
        for arm in ARMS:
            slots[i, arm] = observed_slot(run, i, arm, corpus, tokenizer)
        try:
            prefix = json.loads((run / "inference" / f"{i}-A" / "prefix.json").read_bytes())
            pools[i] = candidate_fulltext(prefix, corpus)
            valid_physical_prefix(prefix, run / "inference/ledger", run / "inference" / f"{i}-A/private-responses", tokenizer, f"{i}-A")
            initial[i] = slots[i, "A"][1][0]
            require(initial[i] is not None
                    and initial[i]["frame_sha256"] == identity(prefix["payload"]["frame"])
                    and initial[i]["physical_attempt_id"] == prefix["payload"]["attempt"]["diagnostics"]["physical_attempt_id"],
                    "initial_physical_prefix_identity")
        except (ValueError, KeyError, TypeError, OSError, IndexError):
            pools.setdefault(i, None)
            initial[i] = None
    # Original immutable cost and physical ledger are references, never overwritten.
    physical = ledger_cost(run / "inference/ledger")
    rerank = reranker_cost(run / "reranker-ledger")
    # A global unknown cannot be assigned to a favourable slot; retain every denominator.
    if physical["unknown_usage_attempts"] or rerank["unknown_requests"]:
        slots = {key: (dict(row, audit_status="unresolved", prediction=None), frames, changes)
                 for key, (row, frames, changes) in slots.items()}
    require(sha(ROOT / "envs" / ARCHIVE) == ARCHIVE_SHA, "frozen_train_archive")
    with tarfile.open(ROOT / "envs" / ARCHIVE) as archive:
        with member(archive, PREP + "/gold/claims_train.jsonl") as stream:
            golds = select_complete_fit(stream, TRAIN_SHA, ids, corpus)
    cases: list[dict[str, Any]] = []
    model_events: list[dict[str, Any]] = []
    scripted_events: list[dict[str, Any]] = []
    for i in ids:
        gold = golds[i]
        for arm in ARMS:
            row, frames, changes = slots[i, arm]
            events = aligned_tools(gold, arm, row, frames, changes, corpus)
            (model_events if arm == "A" else scripted_events).extend(dict(e, claim_id=i, arm=arm) for e in events)
            cases.append({"claim_id": i, "arm": arm, "endpoint": endpoint(gold, row, corpus),
                "initial": initial_diagnosis(gold, pools[i], initial[i], corpus), "tool_events": events,
                "unexecuted_tool_proposals": [dict(entry, reachability="unknown", subsequent_outcome="unresolved")
                    for entry in (row.get("result") or {}).get("decision_execution_audit", [])
                    if entry["strict_action"] in {"read", "rewrite", "rerank"} and entry["actual_event_index"] is None]})
    output.mkdir(mode=0o700)
    report = {"version": VERSION, "execution_source_git": EXECUTION_SHA, "audit_source_git": audit_sha,
        "audit_script_sha256": sha(Path(__file__)), "selection_sha256": SELECTION_SHA, **binding,
        "scope": "fixed_exposed_TRAIN24_posthoc_not_new_experiment", "planned_slots": 72,
        "endpoint_counts": dict(Counter(c["endpoint"] for c in cases)), "cases": cases,
        "model_events": model_events, "scripted_events": scripted_events,
        "physical_generation_cost": physical, "reranker_cost": rerank,
        "unexecuted_tool_reachability": "unknown", "causal_effect": "not_identified",
        "model_generation_calls": 0, "controller_replays": 0, "training_authorized": False}
    ordered_write(output / "attribution.json", report)
    ordered_write(output / "receipt.json", {"audit_source_git": audit_sha, "execution_source_git": EXECUTION_SHA,
        "attribution_sha256": sha(output / "attribution.json"), "planned_slots": 72, "generation_calls": 0})


if __name__ == "__main__":
    main()
