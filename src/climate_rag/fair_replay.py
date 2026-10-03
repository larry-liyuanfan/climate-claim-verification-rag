"""Frozen fair cohort contract, trace audit and post-exit annotation-proxy scoring.

No loader, network client, gold lookup or model judge lives in this module.
Official evidence-ID concordance is NOT semantic entailment.
"""
from __future__ import annotations

from collections import Counter
import hashlib
import json
import re
from typing import Any

from . import fair_acquisition, targeted_query
from . import coverage_acquisition
from .agent_v3 import action_schema
from .metrics import paired_bootstrap, per_claim_retrieval_metrics
from .models import Claim, Prediction
from .targeted_replay import BUDGET, CORPUS_SHA, MODEL_SHA, RERANKER_SHA
from .verification import normalise_claim

STUDY = "public_v2_retrieval_exposed_development_policy_comparison_not_independent_test"
REGRESSION_STUDY = "consumed_fair32_coverage_regression_not_independent_quality_validation"
BINDING_KEYS = {"protocol", "study_kind", "task_count", "cohort_sha256", "tasks_sha256",
                "exposure_audit_sha256", "scoring_contract_sha256", "initial_contract_sha256",
                "corpus_sha256", "model_sha256", "reranker_sha256", "eligible", "selection_rule"}


def identity(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True,
                                    separators=(",", ":")).encode()).hexdigest()


def validate_binding(binding: Any, *, execution: bool = True) -> None:
    if not isinstance(binding, dict) or set(binding) != BINDING_KEYS:
        raise ValueError("unbound_fair_identity")
    expected_study = REGRESSION_STUDY if binding["protocol"] == coverage_acquisition.PROTOCOL else STUDY
    if (not fair_acquisition.is_fair(binding["protocol"]) or binding["study_kind"] != expected_study
            or type(binding["task_count"]) is not int or not 1 <= binding["task_count"] <= 32
            or type(binding["eligible"]) is not bool or not binding["selection_rule"]):
        raise ValueError("fair_study_or_cohort")
    for key in BINDING_KEYS:
        if key.endswith("sha256") and not re.fullmatch(r"[a-f0-9]{64}", str(binding[key])):
            raise ValueError("unbound_fair_identity:" + key)
    if (binding["corpus_sha256"] != CORPUS_SHA or binding["model_sha256"] != MODEL_SHA
            or binding["reranker_sha256"] != RERANKER_SHA):
        raise ValueError("fair_shared_models_or_corpus")
    if execution and not binding["eligible"]:
        raise ValueError("cohort_not_eligible")


def validate_tasks(tasks: Any, binding: Any) -> None:
    validate_binding(binding)
    if (not isinstance(tasks, list) or len(tasks) != binding["task_count"]
            or any(not isinstance(t, dict) or set(t) != {"id", "claim_text"}
                   or not isinstance(t["id"], str) or not isinstance(t["claim_text"], str)
                   or not 1 <= len(normalise_claim(t["claim_text"])) <= 2000 for t in tasks)
            or len({t["id"] for t in tasks}) != len(tasks)
            or identity(tasks) != binding["tasks_sha256"]):
        raise ValueError("fair_cohort_tasks_or_gold_leak")


def policy(binding: dict[str, Any]) -> dict[str, Any]:
    validate_binding(binding, execution=False)
    slots = 3 * binding["task_count"]
    return {"protocol": binding["protocol"], "routes": list(fair_acquisition.ROUTES),
            "planned_slots": slots, "max_generator_calls": slots * BUDGET["max_calls"],
            "max_rerank_requests": slots * (BUDGET["max_tools"] - 1),
            "budget": BUDGET, "binding": binding, "same_caps_not_equal_actual_cost": True,
            "bootstrap_samples": 5000, "seed": 20261003,
            "comparison_kind": "whole_policy_not_pure_feedback_causal_effect",
            "automatic_retry": False, "training": False}


def audit_state(row: dict[str, Any], sources: dict[str, Any]) -> None:
    """Reconstruct phase, masks, receipts and real feedback/state provenance."""
    events = row["events"]
    tools = [e for e in events if "tool" in e]
    if (len(tools) != row["tool_calls"] or not tools or tools[0]["tool"] != "retrieve"
            or tools[0]["trigger_attempt"] != -1 or any(e["status"] == "running" for e in tools)):
        raise ValueError("fair_tool_initial_or_count")
    if row["initial_frame"] is None:
        if row["answer"] is not None or row["generation_attempts"] or not row["outcome"].startswith("controller_failure:"):
            raise ValueError("fair_missing_initial_frame_not_failure")
        return
    if identity(row["initial_frame"]) != row["initial_frame_sha256"]:
        raise ValueError("fair_initial_frame_hash")
    aliases = row["visible_source_ids"]
    committed: dict[str, str] = {}
    for event in tools:
        delivery = event.get("delivery")
        if delivery is None:
            continue
        visible_map = delivery["visible_sentence_sha256"]
        expected_visible = dict(committed)
        for alias in event["requested_context"]:
            for index, sentence in enumerate(sources[aliases[alias]].sentences):
                expected_visible[f"{alias}:{index}"] = hashlib.sha256(sentence.encode()).hexdigest()
        if visible_map != expected_visible:
            raise ValueError("fair_delivery_not_complete_context_commit")
        if event["tool"] == "retrieve" and visible_map != {
                v["sentence_id"]: hashlib.sha256(v["text"].encode()).hexdigest()
                for v in row["initial_frame"]["current_citable"]}:
            raise ValueError("fair_bootstrap_delivery_not_initial_frame")
        new_map = {sid: fingerprint for sid, fingerprint in visible_map.items() if sid not in committed}
        if (delivery["new_sentence_sha256"] != new_map
                or any(visible_map.get(sid) != fingerprint for sid, fingerprint in committed.items())):
            raise ValueError("fair_delivery_increment_not_actual_state")
        next_attempt = delivery["next_attempt"]
        if next_attempt is None:
            if event["tool"] != "retrieve" and not (
                    event["tool"] == "read" and row["route"] != "autonomous"):
                raise ValueError("fair_delivery_unbound_receipt_time")
        elif (type(next_attempt) is not int or not event["trigger_attempt"] < next_attempt <= len(row["generation_attempts"])):
            raise ValueError("fair_delivery_future_or_invalid_receipt_time")
        else:
            actual_receipts = [i for i, a in enumerate(row["generation_attempts"])
                if i > event["trigger_attempt"] and
                    {v["sentence_id"]: hashlib.sha256(v["text"].encode()).hexdigest()
                     for v in a["observation"]["current_citable"]} == visible_map]
            if next_attempt != (actual_receipts[0] if actual_receipts else len(row["generation_attempts"])):
                raise ValueError("fair_delivery_receipt_time_not_actual_prompt")
        committed.update(visible_map)
    def audit_plan(plan: dict[str, Any], executed: list[dict[str, Any]]) -> None:
        expected = (["read"] if plan["read_source_ids"] else []) + ["rewrite"] * len(plan["queries"])
        # Rerank is skipped ONLY if no pool is available after queries/reads.
        final_pool = executed[-1]["candidate_ids"] if executed and "candidate_ids" in executed[-1] else tools[0].get("candidate_ids", [])
        if executed and executed[-1]["tool"] == "rerank":
            final_pool = executed[-1]["before_candidate_ids"]
        if final_pool:
            expected.append("rerank")
        if [e["tool"] for e in executed] != expected:
            # An explicit fatal controller failure may truncate the fixed list;
            # it is not a completed plan or a successful task.
            if not row["outcome"].startswith("controller_failure:") or [e["tool"] for e in executed] != expected[:len(executed)]:
                raise ValueError("fair_fixed_execution_sequence")
        offset = 1 if plan["read_source_ids"] else 0
        if offset and executed and executed[0].get("requested_context") != plan["read_source_ids"] and executed[0]["status"] == "completed":
            raise ValueError("fair_fixed_read_arguments")
        for q, e in zip(plan["queries"], executed[offset:]):
            if e["tool"] != "rewrite" or e["query_sha256"] != hashlib.sha256(q["query"].encode()).hexdigest() or e["query_purpose"] != q["purpose"]:
                raise ValueError("fair_fixed_query_arguments")
    if row["route"] == "deterministic_workflow":
        audit_plan({"read_source_ids": [p["source_id"] for p in row["initial_frame"]["preview_only"]][:5],
                    "queries": targeted_query.deterministic_queries(row["initial_frame"]["immutable_claim"])}, tools[1:])
    phase = "plan" if row["route"] == "fixed_multiquery" else "verdict" if row["route"] == "deterministic_workflow" else "gate"
    for ordinal, a in enumerate(row["generation_attempts"]):
        obs = a["observation"]
        before = [e for e in tools if e["trigger_attempt"] < ordinal]
        actual_queries = [e for e in before if e["tool"] == "rewrite"]
        if (a["stage"] != phase or obs["phase"] != phase
                or obs["remaining_calls"] != 5 - ordinal
                or obs["remaining_tools"] != 5 - len(before)
                or len(obs["prior_queries"]) != len(actual_queries)
                or any(hashlib.sha256(q.encode()).hexdigest() != e["query_sha256"]
                       for q, e in zip(obs["prior_queries"], actual_queries))):
            raise ValueError("fair_phase_cost_or_queries")
        if ordinal == 0 and row["route"] != "deterministic_workflow":
            if {k: obs[k] for k in row["initial_frame"]} != row["initial_frame"]:
                raise ValueError("fair_actual_initial_information_drift")
        visible = {v["sentence_id"]: v["text"] for v in obs["current_citable"]}
        feedback_tools = [e for e in before if "candidate_ids" in e]
        history = [{k: e.get(k) for k in ("tool", "status", "error_type", "query_sha256", "query_purpose",
            "search_returned_ids", "search_empty", "candidate_ids", "requested_context")} for e in feedback_tools]
        if obs["tool_feedback_history"] != history:
            raise ValueError("fair_actual_feedback_history")
        if feedback_tools:
            latest = {k: feedback_tools[-1].get(k) for k in ("tool", "query_sha256", "query_purpose",
                "search_returned_ids", "search_empty", "new_source_ids", "before_candidate_ids", "candidate_ids", "requested_context", "status", "error_type")}
            if obs["tool_feedback"] != latest:
                raise ValueError("fair_actual_latest_feedback")
        if phase == "verdict":
            schema = action_schema(["abstain"] + (["answer"] if visible else []), [], list(visible), 5, targeted=True)
            for branch in schema["anyOf"]:
                if "reason" in branch["properties"]:
                    branch["properties"]["reason"]["enum"] = ["insufficient_evidence", "conflicting_evidence"]
        elif phase == "plan":
            schema = fair_acquisition.planning_schema(obs["readable_source_ids"])
        else:
            ranked = False
            for event in before:
                if event["tool"] == "rewrite" and event["status"] == "completed":
                    ranked = False
                elif event["tool"] == "rerank" and "candidate_ids" in event:
                    ranked = True
            pool = feedback_tools[-1]["candidate_ids"] if feedback_tools else []
            readable = [p["source_id"] for p in obs["preview_only"]]
            fully = {alias for alias, real in aliases.items() if all(f"{alias}:{i}" in visible for i in range(len(sources[real].sentences)))}
            acquire, query, rank = ordinal <= 2 and len(before) < 5, len(actual_queries) < 2, bool(pool and not ranked)
            if (obs["acquisition_available"] != acquire or obs["query_available"] != query
                    or obs["rerank_available"] != rank or obs["readable_source_ids"] != readable
                    or set(readable) & fully or not set(readable) <= set(pool)
                    or obs["allowed_actions"] != ["stop"] + (["acquire"] if acquire and (query or readable or rank) else [])):
                raise ValueError("fair_availability_reconstruction")
            gate = coverage_acquisition.schema if row["protocol"] == coverage_acquisition.PROTOCOL else fair_acquisition.gate_schema
            schema = gate(obs["readable_source_ids"],
                can_acquire=obs["acquisition_available"], can_query=obs["query_available"],
                can_rerank=obs["rerank_available"])
        if a["schema"] != schema:
            raise ValueError("fair_schema_drift")
        if obs["initial_frame_sha256"] != row["initial_frame_sha256"]:
            raise ValueError("fair_initial_frame_binding")
        received = a.get("received_tool_events", [])
        expected_received = [i for i, e in enumerate(events) if "delivery" in e and e["trigger_attempt"] < ordinal
                             and set(e["delivery"]["visible_sentence_sha256"]) <= set(visible)]
        if received != expected_received:
            raise ValueError("fair_delivery_receipt_omitted_or_fabricated")
        for event_id in received:
            event = events[event_id]
            if event["trigger_attempt"] >= ordinal:
                raise ValueError("future_feedback_received")
            for sid, fingerprint in event["delivery"]["visible_sentence_sha256"].items():
                if sid not in visible or hashlib.sha256(visible[sid].encode()).hexdigest() != fingerprint:
                    raise ValueError("fair_delivery_not_actual_prompt")
        if a["status"] == "valid_decision":
            d = a["decision"]
            if phase == "gate" and row["protocol"] == coverage_acquisition.PROTOCOL:
                full = [alias for alias, real in aliases.items() if all(f"{alias}:{i}" in visible for i in range(len(sources[real].sentences)))]
                if coverage_acquisition.parse(a["proposed_decision"], obs, fully_shown=full) != d:
                    raise ValueError("coverage_validated_decision_drift")
            triggered = [e for e in tools if e["trigger_attempt"] == ordinal]
            if phase == "verdict" and ordinal != len(row["generation_attempts"]) - 1:
                raise ValueError("fair_verdict_must_terminate")
            if phase == "gate" and d["action"] == "stop":
                if triggered:
                    raise ValueError("acquisition_after_stop")
                phase = "verdict"
            elif phase == "gate" and d["action"] == "acquire":
                expected_tool = "rewrite" if d["tool"] == "query" else d["tool"]
                not_started_failure = (not triggered and a.get("execution_status") == "not_started"
                    and ordinal == len(row["generation_attempts"]) - 1
                    and row["outcome"].startswith("controller_failure:"))
                if not not_started_failure and (len(triggered) != 1 or triggered[0]["tool"] != expected_tool):
                    raise ValueError("proposal_without_matching_execution")
                if triggered:
                    e = triggered[0]
                    if a.get("execution_status") != e["status"]:
                        raise ValueError("fair_execution_status")
                    if d["tool"] == "query" and (e["query_sha256"] != hashlib.sha256(d["query"].encode()).hexdigest() or e["query_purpose"] != d["purpose"]):
                        raise ValueError("fair_gate_query_arguments")
                    if d["tool"] == "read" and e["status"] == "completed" and e["requested_context"] != d["source_ids"]:
                        raise ValueError("fair_gate_read_arguments")
            elif phase == "plan":
                audit_plan(d, triggered)
                phase = "verdict"
            elif triggered:
                raise ValueError("tool_during_verdict")
    if row["answer"] is not None and phase != "verdict":
        raise ValueError("gate_not_terminal_verdict")
    final = row["generation_attempts"][-1] if row["generation_attempts"] else {}
    if row["outcome"].startswith("model_abstention:"):
        if (final.get("stage") != "verdict" or final.get("status") != "valid_decision"
                or final.get("decision", {}).get("action") != "abstain"
                or row["outcome"] != "model_abstention:" + final["decision"]["reason"]):
            raise ValueError("failure_not_verifier_abstention")
    if row["outcome"] == "ids_validated_semantics_unmeasured" and row["answer"] is None:
        raise ValueError("answer_outcome_without_answer")
    for event in tools:
        if event.get("delivery"):
            for sid, fingerprint in event["delivery"]["visible_sentence_sha256"].items():
                alias, index = sid.split(":")
                if hashlib.sha256(sources[aliases[alias]].sentences[int(index)].encode()).hexdigest() != fingerprint:
                    raise ValueError("delivery_source_mutation")
        if event["status"] == "failed" and "candidate_ids" in event and (event["before_candidate_ids"] != event["candidate_ids"]
                                             or event["before_context"] != event["requested_context"]):
            raise ValueError("failed_tool_polluted_state")


def score(run: dict[str, Any], gold: dict[str, Any]) -> dict[str, Any]:
    """Post-audit only. All N retained; semantic support remains unmeasured."""
    binding = run["fair_binding"]
    validate_binding(binding)
    if binding["protocol"] != run["protocol"]:
        raise ValueError("fair_binding_execution_protocol_mismatch")
    task_ids = [r["task_id"] for r in run["runs"] if r["route"] == fair_acquisition.ROUTES[0]]
    if (gold.get("tasks_sha256") != binding["tasks_sha256"] or set(gold["claims"]) != set(task_ids)
            or len(run["runs"]) != 3 * binding["task_count"]):
        raise ValueError("fair_gold_or_matrix_identity")
    result: dict[str, Any] = {"protocol": binding["protocol"], "study_kind": binding["study_kind"],
        "semantic_citation_support": "unmeasured; no human blind labels or model judge",
        "citation_proxy": "official decisive evidence-ID concordance, NOT semantic support",
        "feedback_causality": "whole_policy_comparison_only", "routes": {}, "paired_bootstrap": {}}
    vectors: dict[str, dict[str, list[float]]] = {}
    retrieval_vectors: dict[str, dict[str, list[float]]] = {}
    for route in fair_acquisition.ROUTES:
        rows = [r for r in run["runs"] if r["route"] == route]
        if [r["task_id"] for r in rows] != task_ids:
            raise ValueError("fair_score_incomplete_order")
        counts: Counter[str] = Counter()
        correct, joint, proxy = [], [], []
        retrieval: dict[str, list[float]] = {key: [] for key in ("recall@5", "mrr@10", "ndcg@10", "evidence_f1")}
        for row in rows:
            g = gold["claims"][row["task_id"]]
            if g["claim_sha256"] != row["immutable_claim_sha256"]:
                raise ValueError("fair_gold_claim_sha")
            answer, outcome = row["answer"], row["outcome"]
            expected = g["label"]
            if expected not in {"SUPPORTS", "REFUTES", "NOT_ENOUGH_INFO", "DISPUTED"}:
                raise ValueError("unsupported_official_label")
            intentional = outcome in {"model_abstention:insufficient_evidence", "model_abstention:conflicting_evidence"}
            ok = bool(answer and answer["label"] == expected) or (
                expected == "NOT_ENOUGH_INFO" and outcome == "model_abstention:insufficient_evidence") or (
                expected == "DISPUTED" and outcome == "model_abstention:conflicting_evidence")
            citations = answer["citations"] if answer else []
            if g["evidence_ids"]:
                metrics = per_claim_retrieval_metrics(
                    Claim(row["task_id"], "scorer metadata only", evidence_ids=tuple(g["evidence_ids"])),
                    Prediction(row["task_id"], tuple(row["delivered_evidence_ids"])),
                    ks=(5, 10), evidence_k=5)
                for key in retrieval:
                    retrieval[key].append(metrics[key])
            matched = bool(citations) and all(c["source_id"] in g["evidence_ids"] for c in citations)
            correct.append(float(ok))
            joint.append(float(ok and matched))
            proxy.append(float(matched))
            counts["answered" if answer else "intentional_abstention" if intentional else "execution_failure"] += 1
            for a in row["generation_attempts"]:
                if a["stage"] == "gate":
                    counts["gate_attempts"] += 1
                    if a.get("decision", {}).get("action") == "stop":
                        counts["validated_stop"] += 1
                    if a.get("decision", {}).get("action") == "acquire":
                        counts["validated_acquisition"] += 1
                    if any(row["events"][i]["delivery"]["new_sentence_sha256"]
                           for i in a.get("received_tool_events", []) if row["events"][i]["trigger_attempt"] >= 0):
                        counts["gate_received_new_full_text"] += 1
            for e in row["events"]:
                if "tool" in e:
                    counts["tool_" + e["tool"]] += 1
                    counts["tool_failure"] += e["status"] == "failed"
                    counts["empty_query"] += e.get("search_empty") is True
        n = len(rows)
        vectors[route] = {"official_task_correct": correct, "task_and_official_id_proxy_joint": joint}
        retrieval_vectors[route] = retrieval
        result["routes"][route] = {"all_task_denominator": n,
            "official_task_correct": sum(correct) / n, "task_and_official_id_proxy_joint": sum(joint) / n,
            "official_id_concordance_all_tasks": sum(proxy) / n, "behavior": dict(counts),
            "retrieval": {"denominator": len(retrieval["recall@5"]),
                "scope": "final ranked Top20 candidate retrieval proxy; legacy delivered_evidence_ids includes preview-only/not-read sources, NOT complete-text delivery; official evidence-bearing tasks incl failures; Top5 F1, NOT citation or entailment F1",
                "metrics": {key: sum(values) / len(values) if values else None for key, values in retrieval.items()}},
            "physical_cost_by_task": [{"task_id": r["task_id"], "generation": r["physical_cost"],
                "reranker": r["reranker_cost"], "tool_calls": r["tool_calls"],
                "cumulative_model_prompt_tokens": r["cumulative_model_prompt_tokens"], "elapsed_ms": r["elapsed_ms"]} for r in rows]}
    for comparator in fair_acquisition.ROUTES[:-1]:
        result["paired_bootstrap"]["autonomous-minus-" + comparator] = {
            key: paired_bootstrap(vectors[comparator][key], vectors["autonomous"][key], samples=5000, seed=20261003)
            for key in vectors[comparator]}
        if retrieval_vectors[comparator]["recall@5"]:
            result["paired_bootstrap"]["autonomous-minus-" + comparator]["retrieval_secondary"] = {
                key: paired_bootstrap(retrieval_vectors[comparator][key], retrieval_vectors["autonomous"][key], samples=5000, seed=20261003)
                for key in retrieval_vectors[comparator]}
    return result
