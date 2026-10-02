"""Five-route paired scorer; gold is supplied only AFTER provenance/cost audit.

Evidence F1 is existing Top5 retrieval F1, not citation F1. Decisive label
accuracy uses only official SUPPORTS/REFUTES. Nondecisive tasks remain in work,
failure and latency denominators, never relabelled as automatic NEI successes.
"""

from __future__ import annotations

from collections import Counter
import json
import math
from pathlib import Path
from typing import Any

import numpy as np

from .agent_v3 import Source, parse_action, action_schema
from . import stop_acquire
from .metrics import paired_bootstrap, per_claim_retrieval_metrics
from .models import Claim, Prediction
from .scifact_utility_runtime import ledger_cost, reranker_cost
from .targeted_query import PROTOCOL, ROUTES, decode, validate_plan
from .scifact_utility_contract import identity
from .targeted_replay import BUDGET, STUDY, sha, route_protocol
from .verification import normalise_claim

METRICS = ("recall@5", "mrr@10", "ndcg@10", "evidence_f1")


def audit_acquisition(row: dict[str, Any], sources: dict[str, Source]) -> None:
    """Replay only the new gate state; physical responses are checked below."""
    aliases = row["visible_source_ids"]
    tools = [e for e in row["events"] if "tool" in e]
    if (len(tools) != row["tool_calls"] or any(e["status"] == "running" for e in tools)
            or any(type(e.get("trigger_attempt")) is not int for e in tools)):
        raise ValueError("gate_tool_accounting")
    if not tools or tools[0]["tool"] != "retrieve" or tools[0]["trigger_attempt"] != -1:
        raise ValueError("gate_initial_retrieval")
    phase = "gate"
    queries: list[str] = []
    for ordinal, attempt in enumerate(row["generation_attempts"]):
        obs = attempt["observation"]
        before = [e for e in tools if e["trigger_attempt"] < ordinal]
        feedback_tools = [e for e in before if "candidate_ids" in e]
        if (attempt["stage"] != phase or obs.get("phase") != phase
                or obs.get("protocol") != stop_acquire.PROTOCOL
                or obs["remaining_calls"] != 5 - ordinal
                or obs["remaining_tools"] != 5 - len(before)
                or obs["prior_queries"] != queries):
            raise ValueError("gate_phase_budget_or_query_state")
        if feedback_tools:
            expected = {k: feedback_tools[-1].get(k) for k in (
                "tool", "query_sha256", "query_purpose", "search_returned_ids", "search_empty",
                "new_source_ids", "before_candidate_ids", "candidate_ids", "requested_context",
                "status", "error_type")}
            if obs["tool_feedback"] != expected:
                raise ValueError("gate_latest_feedback")
        elif obs["tool_feedback"] is not None:
            raise ValueError("gate_fabricated_feedback")
        shown = {x["sentence_id"] for x in obs["current_citable"]}
        fully = [alias for alias, real in aliases.items()
                 if all(f"{alias}:{i}" in shown for i in range(len(sources[real].sentences)))]
        if phase == "gate":
            readable = [p["source_id"] for p in obs["preview_only"]]
            candidates = feedback_tools[-1]["candidate_ids"] if feedback_tools else []
            acquire, query = 5 - ordinal >= 3 and len(before) < 5, len(queries) < 2
            if (len(set(readable)) != len(readable) or set(readable) & set(fully)
                    or not set(readable) <= set(candidates)
                    or obs.get("readable_source_ids") != readable or obs.get("read_limit") != 5
                    or obs.get("acquisition_available") != acquire or obs.get("query_available") != query
                    or obs["allowed_actions"] != ["stop"] + (["acquire"] if acquire and (query or readable) else [])):
                raise ValueError("gate_read_mask_or_reservation")
            expected_schema = stop_acquire.gate_schema(readable, can_acquire=acquire, can_query=query)
        else:
            # Sentence enum ordering is part of the physical prompt.
            expected_schema = action_schema(["abstain"] + (["answer"] if shown else []),
                [], [s["sentence_id"] for s in obs["current_citable"]], 5, targeted=True)
            for branch in expected_schema["anyOf"]:
                if "reason" in branch["properties"]:
                    branch["properties"]["reason"]["enum"] = ["insufficient_evidence", "conflicting_evidence"]
        if attempt["schema"] != expected_schema:
            raise ValueError("gate_schema_state_drift")
        triggered = [e for e in tools if e["trigger_attempt"] == ordinal]
        if attempt["status"] != "valid_decision":
            # Fatal tool exceptions may follow a validated proposal; still charged.
            if triggered and not row["outcome"].startswith("controller_failure"):
                raise ValueError("gate_tool_without_valid_decision")
            continue
        decision = attempt["decision"]
        if attempt.get("decision_status") != "validated":
            raise ValueError("gate_unvalidated_decision")
        if phase == "gate":
            if decision["action"] == "stop":
                if triggered or attempt.get("phase_transition") != "verdict":
                    raise ValueError("gate_stop_transition")
                phase = "verdict"
            else:
                if (not triggered and ordinal == len(row["generation_attempts"]) - 1
                        and attempt.get("execution_status") == "not_started"
                        and row["outcome"] == "controller_failure:RuntimeError"
                        and row["events"][-1].get("failure_code") == "tool_budget_or_deadline"
                        and row["elapsed_ms"] >= BUDGET["timeout_seconds"] * 1000
                        and row["answer"] is None):
                    continue  # Physical gate charged; deadline prevented the tool from starting.
                if len(triggered) != 1:
                    raise ValueError("gate_missing_or_repeated_execution")
                event = triggered[0]
                kind = "read" if decision["tool"] == "read" else "rewrite"
                if (event["tool"] != kind or event.get("selection_origin") != "stop_acquire_gate"
                        or not event.get("model_selected")
                        or attempt.get("execution_status") != event["status"]):
                    raise ValueError("gate_execution_binding")
                if kind == "read" and event.get("requested_context") != decision["source_ids"]:
                    raise ValueError("gate_read_execution")
                if kind == "rewrite":
                    if (event["query_sha256"] != sha(decision["query"].encode())
                            or event["query_purpose"] != decision["purpose"]):
                        raise ValueError("gate_query_execution")
                    queries.append(decision["query"])
        elif triggered or decision["action"] not in {"answer", "abstain"} or decision.get("reason") == "budget":
            raise ValueError("verdict_cannot_execute_or_hide_budget_failure")
        elif ordinal != len(row["generation_attempts"]) - 1:
            raise ValueError("verdict_must_terminate")
    if any(e["trigger_attempt"] >= len(row["generation_attempts"]) for e in tools):
        raise ValueError("gate_orphan_execution")
    if row["answer"] is not None and (phase != "verdict" or row["generation_attempts"][-1]["stage"] != "verdict"):
        raise ValueError("gate_not_a_verdict")
    if row["outcome"] == "ids_validated_semantics_unmeasured" and row["answer"] is None:
        raise ValueError("gate_answered_without_answer")
    if row["outcome"].startswith("model_abstention:"):
        final = row["generation_attempts"][-1] if row["generation_attempts"] else {}
        if (final.get("stage") != "verdict" or final.get("status") != "valid_decision"
                or final.get("decision", {}).get("action") != "abstain"
                or row["outcome"] != "model_abstention:" + final["decision"]["reason"]):
            raise ValueError("gate_failure_not_abstention")


def terminal_category(outcome: str) -> str:
    if outcome == "ids_validated_semantics_unmeasured":
        return "answered"
    if outcome in {
        "model_abstention:insufficient_evidence",
        "model_abstention:conflicting_evidence",
        "model_abstention:budget",
    }:
        return "intentional_model_abstention"
    if outcome in {"validation_repair_exhausted", "generation_budget_exhausted"}:
        return "execution_exhausted"
    if outcome.startswith("controller_failure:") or outcome in {
        "deadline",
        "deadline_during_prompt_assembly",
    }:
        return "execution_failure"
    raise ValueError("unclassified_terminal_outcome")


def audit(
    run: dict[str, Any],
    tasks: list[dict[str, str]],
    sources: dict[str, Source],
    ledger: Path,
    *,
    synthetic: bool = False,
    tokenizer: Any = None,
    generation_contract: Any = None,
    reranker_directory: Path | None = None,
) -> None:
    expected = [(t["id"], route) for t in tasks for route in ROUTES]
    if (
        run["protocol"] not in {PROTOCOL, stop_acquire.PROTOCOL}
        or run["study_kind"] != STUDY
        or run["budget"] != BUDGET
        or [(r["task_id"], r["route"]) for r in run["runs"]] != expected
    ):
        raise ValueError("incomplete_or_changed_matrix")
    task_hashes = {
        t["id"]: sha(normalise_claim(t["claim_text"]).encode()) for t in tasks
    }
    physical = {
        p.name.removesuffix(".reserved.json"): json.loads(p.read_bytes())
        for p in ledger.glob("g*.reserved.json")
    }
    seen: set[str] = set()
    rank_requests = (
        {
            p.name.removesuffix(".reserved.json"): json.loads(p.read_bytes())
            for p in reranker_directory.glob("r*.reserved.json")
        }
        if reranker_directory
        else {}
    )
    rank_seen: set[str] = set()
    for row in run["runs"]:
        if (
            row["protocol"] != route_protocol(run["protocol"], row["route"])
            or row["budget"] != BUDGET
            or row["immutable_claim_sha256"] != task_hashes[row["task_id"]]
            or (not synthetic and row["provider_kind"] != "local_model")
        ):
            raise ValueError("row_identity_or_unconfigured_provider")
        gated = row["protocol"] == stop_acquire.PROTOCOL
        if gated:
            audit_acquisition(row, sources)
        if not math.isfinite(row["elapsed_ms"]) or row["elapsed_ms"] < 0:
            raise ValueError("invalid_elapsed")
        ids = sorted(
            (k for k, v in physical.items() if v["slot"] == row["slot"]),
            key=lambda k: int(k[1:]),
        )
        if (
            row["physical_cost"] != ledger_cost(ledger, ids)
            or len(ids) > 5
            or not len(ids)
            == row["model_calls"]
            == len(row["generation_attempts"])
            <= 5
            or not 0 <= row["tool_calls"] <= 5
        ):
            raise ValueError("physical_cost_or_budget_drift")
        seen.update(ids)
        if reranker_directory is not None:
            rank_ids = [
                key
                for key, request in rank_requests.items()
                if request["slot"] == row["slot"]
            ]
            rank_seen.update(rank_ids)
            if (
                row["reranker_cost"] != reranker_cost(reranker_directory, rank_ids)
                or len(rank_ids) > 1
                or sum(rank_requests[k]["requested_pairs"] for k in rank_ids)
                != row["rerank_pairs"]
            ):
                raise ValueError("rerank_slot_accounting_drift")
        elif not synthetic:
            raise ValueError("missing_reranker_ledger")
        aliases = row["visible_source_ids"]
        delivered = row["delivered_evidence_ids"]
        if (
            len(delivered) > 20
            or len(set(delivered)) != len(delivered)
            or not set(delivered) <= set(aliases.values())
        ):
            raise ValueError("invalid_delivered_candidates")
        completed_tools = [
            e
            for e in row["events"]
            if e.get("status") == "completed" and "candidate_ids" in e
        ]
        expected_delivered = (
            []
            if row["outcome"].startswith("controller_failure")
            or "deadline" in row["outcome"]
            else [aliases[a] for a in completed_tools[-1]["candidate_ids"]]
            if completed_tools
            else []
        )
        if delivered != expected_delivered:
            raise ValueError("delivered_ranking_not_tool_result")
        if delivered:
            final_observation = row["generation_attempts"][-1]["observation"]
            if delivered != [
                aliases[s] for s in final_observation["tool_feedback"]["candidate_ids"]
            ]:
                raise ValueError("delivered_ranking_not_physical_feedback")
        presented: dict[str, Any] = {}
        attempts = row["generation_attempts"]
        for ordinal, a in enumerate(attempts):
            observation = a["observation"]
            if (
                sha(observation["immutable_claim"].encode())
                != row["immutable_claim_sha256"]
            ):
                raise ValueError("final_claim_mutated")
            now = {v["sentence_id"]: v["text"] for v in observation["current_citable"]}
            if not presented.items() <= now.items():
                raise ValueError("previously_read_text_hidden")
            for sid, text in now.items():
                alias, index = sid.split(":")
                original = sources[aliases[alias]]
                if original.sentences[int(index)] != text:
                    raise ValueError("displayed_source_text_drift")
            presented = now
            key = a.get("diagnostics", {}).get("physical_attempt_id")
            # Generic exceptions can lack diagnostics; only failed terminal calls
            # may be matched by physical order, never successful decisions.
            if key is None and a["status"] == "terminal_failure":
                key = ids[ordinal]
            if (
                key != ids[ordinal]
                or physical[key]["observation"] != observation
                or physical[key]["schema"] != a["schema"]
            ):
                raise ValueError("physical_prompt_or_unique_id_drift")
            request = physical[key]
            finished = json.loads((ledger / (key + ".finished.json")).read_bytes())
            if (
                finished["slot"] != row["slot"]
                or finished["physical_attempt_id"] != key
            ):
                raise ValueError("physical_finish_slot_drift")
            response = finished.get("response")
            if response is not None:
                if finished["usage"] != a["usage"] or response["usage"] != a["usage"]:
                    raise ValueError("physical_usage_drift")
                if a["status"] == "valid_decision":
                    payload = decode(response["raw"])
                    actual = (
                        stop_acquire.parse_gate(payload, observation, fully_shown=[
                            alias for alias, real in aliases.items()
                            if all(f"{alias}:{i}" in now for i in range(len(sources[real].sentences)))])
                        if gated and a["stage"] == "gate" else validate_plan(
                            payload,
                            [
                                observation["immutable_claim"],
                                *observation["prior_queries"],
                            ],
                        )
                        if a["stage"] == "plan"
                        else parse_action(
                            payload,
                            observation["allowed_actions"],
                            now,
                            list(aliases),
                            5,
                            targeted=True,
                            seen_queries=[
                                observation["immutable_claim"],
                                *observation["prior_queries"],
                            ],
                        )
                    )
                    if actual != a["decision"]:
                        raise ValueError("decision_not_physical_output")
                    if gated and payload != a.get("proposed_decision"):
                        raise ValueError("gate_proposal_not_physical_output")
            elif a["status"] == "valid_decision":
                raise ValueError("decision_without_physical_return")
            if not synthetic:
                from .local_agent_v3 import render_v3_prompt
                from .scifact_generation import audit_generation, expected_parser_config
                from .scifact_natural_contract import complete_wire
                from .targeted_replay import MODEL_SHA

                if tokenizer is None or generation_contract is None:
                    raise ValueError("physical_audit_inputs_required")
                prompt = render_v3_prompt(tokenizer, observation, a["schema"])
                tokens = tokenizer.encode(prompt, add_special_tokens=False)
                if (
                    request["prompt_sha256"] != identity(prompt)
                    or request["token_ids_sha256"] != identity(tokens)
                    or len(tokens) != a["input_prompt_tokens"]
                    or request["actual_base_state"]["base_model_sha256"] != MODEL_SHA
                    or request["actual_base_state"]["adapter_loaded"] is not False
                ):
                    raise ValueError("physical_model_prompt_identity")
                if response is not None and a["status"] == "valid_decision":
                    complete_wire(
                        response, len(tokens), ledger.parent / row["slot"] / "private"
                    )
                    generation = json.loads(
                        (ledger / (key + ".generation.json")).read_bytes()
                    )
                    if generation["input_tokens"] != len(tokens):
                        raise ValueError("generation_receipt_input_length")
                    audit_generation(
                        generation,
                        request,
                        response["diagnostics"],
                        expected_parser_config(tokenizer),
                        trusted_contract=generation_contract,
                    )
        answer = row["answer"]
        if answer is not None:
            if (
                not attempts
                or attempts[-1]["status"] != "valid_decision"
                or attempts[-1]["decision"]["action"] != "answer"
            ):
                raise ValueError("answer_without_model_decision")
            if answer["label"] != attempts[-1]["decision"]["label"]:
                raise ValueError("answer_label_drift")
            requested = attempts[-1]["decision"]["sentence_ids"]
            expected_citations = [
                (aliases[s.split(":")[0]], int(s.split(":")[1])) for s in requested
            ]
            if [
                (c["source_id"], c["sentence_index"]) for c in answer["citations"]
            ] != expected_citations:
                raise ValueError("answer_citation_drift")
            for sid, citation in zip(requested, answer["citations"]):
                text = presented[sid]
                original = sources[citation["source_id"]]
                if (
                    citation["text"] != text
                    or citation["text_sha256"] != sha(text.encode())
                    or citation["source_text_sha256"] != original.text_sha256
                ):
                    raise ValueError("citation_not_read_original_text")
    if seen != set(physical):
        raise ValueError("orphan_physical_calls")
    if rank_seen != set(rank_requests) or len(rank_requests) > len(tasks) * 4:
        raise ValueError("orphan_or_excess_rerank_requests")


def score(
    run: dict[str, Any], gold: dict[str, Any], *, synthetic: bool = False
) -> dict[str, Any]:
    rows: dict[str, list[dict[str, Any]]] = {r: [] for r in ROUTES}
    for r in run["runs"]:
        g = gold["claims"][r["task_id"]]
        if g["claim_sha256"] != r["immutable_claim_sha256"]:
            raise ValueError("gold_claim_identity")
        answer = r["answer"]
        metrics: dict[str, Any] = {}
        if g["evidence_ids"]:
            metrics.update(
                per_claim_retrieval_metrics(
                    Claim(
                        r["task_id"],
                        "metadata-only; scoring uses IDs",
                        evidence_ids=tuple(g["evidence_ids"]),
                    ),
                    Prediction(r["task_id"], tuple(r["delivered_evidence_ids"])),
                    ks=(5, 10),
                    evidence_k=5,
                )
            )
            cited = {c["source_id"] for c in answer["citations"]} if answer else set()
            gold_ids = set(g["evidence_ids"])
            overlap = len(cited & gold_ids)
            metrics["citation_id_precision"] = overlap / len(cited) if cited else 0.0
            metrics["citation_id_recall"] = overlap / len(gold_ids)
            metrics["citation_id_f1"] = 2 * overlap / (len(cited) + len(gold_ids))
        if g.get("label") in {"SUPPORTS", "REFUTES"}:
            if not synthetic and r["provider_kind"] != "local_model":
                raise ValueError("cannot_score_unconfigured_verdict")
            metrics["official_decisive_label_accuracy"] = float(
                answer is not None and answer["label"] == g["label"]
            )
        rows[r["route"]].append(
            {
                "task_id": r["task_id"],
                **metrics,
                "cost": r["physical_cost"],
                "reranker_cost": r.get("reranker_cost"),
                "elapsed_ms": r["elapsed_ms"],
                "tool_calls": r["tool_calls"],
                "rerank_pairs": r["rerank_pairs"],
                "outcome": r["outcome"],
                "answered": answer is not None,
                "terminal_category": terminal_category(r["outcome"]),
                "operational_failure": terminal_category(r["outcome"])
                in {"execution_exhausted", "execution_failure"},
                "nondecisive_answer": not g["evidence_ids"] and answer is not None,
            }
        )
    aggregate: dict[str, Any] = {}
    for route, values in rows.items():
        if not values:
            raise ValueError("missing_route")
        evidence = [r for r in values if "evidence_f1" in r]
        labelled = [r for r in values if "official_decisive_label_accuracy" in r]
        known = all(r["cost"]["total_tokens"] is not None for r in values)
        rank_known = all(
            r["reranker_cost"] is not None
            and r["reranker_cost"]["unknown_requests"] == 0
            and r["reranker_cost"]["completed_pairs"]
            == r["reranker_cost"]["requested_pairs"]
            for r in values
        )
        aggregate[route] = {
            "slots": len(values),
            "retrieval_denominator": len(evidence),
            "label_denominator": len(labelled),
            "evidence_metrics": {
                m: float(np.mean([r[m] for r in evidence])) for m in METRICS
            }
            if evidence
            else None,
            "citation_id_metrics": {
                m: float(np.mean([r[m] for r in evidence]))
                for m in (
                    "citation_id_precision",
                    "citation_id_recall",
                    "citation_id_f1",
                )
            }
            if evidence
            else None,
            "official_decisive_label_accuracy": float(
                np.mean([r["official_decisive_label_accuracy"] for r in labelled])
            )
            if labelled
            else None,
            "outcomes": dict(Counter(r["outcome"] for r in values)),
            "terminal_categories": dict(
                Counter(r["terminal_category"] for r in values)
            ),
            "operational_failures": sum(r["operational_failure"] for r in values),
            "nondecisive_answers": sum(r["nondecisive_answer"] for r in values),
            "latency_ms_p50": float(
                np.percentile([r["elapsed_ms"] for r in values], 50)
            ),
            "latency_ms_p95": float(
                np.percentile([r["elapsed_ms"] for r in values], 95)
            ),
            "mean_elapsed_ms": float(np.mean([r["elapsed_ms"] for r in values])),
            "physical_calls": sum(r["cost"]["unique_physical_calls"] for r in values),
            "tool_calls": sum(r["tool_calls"] for r in values),
            "rerank_pairs": sum(r["rerank_pairs"] for r in values),
            "unknown_usage_attempts": sum(
                r["cost"]["unknown_usage_attempts"] for r in values
            ),
            "recorded_token_lower_bound": {
                k: sum(r["cost"]["known_token_lower_bound"][k] for r in values)
                for k in ("input_tokens", "output_tokens")
            },
            "mean_total_generator_tokens": sum(
                sum(r["cost"]["total_tokens"].values()) for r in values
            )
            / len(values)
            if known
            else None,
            "api_currency_cost": None,
            "reranker_accounting_complete": rank_known,
            "reranker_known_token_lower_bound": sum(
                r["reranker_cost"]["known_token_lower_bound"]
                for r in values
                if r["reranker_cost"] is not None
            ),
            "reranker_unknown_requests": sum(
                r["reranker_cost"]["unknown_requests"]
                for r in values
                if r["reranker_cost"] is not None
            ),
        }
    comparisons = {}
    for control in ("fixed_rerank", "fixed_multiquery"):
        intervals = {}
        for metric in (*METRICS, "official_decisive_label_accuracy"):
            left = [r for r in rows[control] if metric in r]
            right = [r for r in rows["adaptive"] if metric in r]
            if not left or [r["task_id"] for r in left] != [
                r["task_id"] for r in right
            ]:
                raise ValueError("empty_or_unpaired_metric")
            intervals[metric] = paired_bootstrap(
                [r[metric] for r in left],
                [r[metric] for r in right],
                samples=5000,
                seed=20260929,
            )
        a, b = aggregate["adaptive"], aggregate[control]
        quality = all(
            intervals[m]["ci_lower"] > 0
            for m in ("evidence_f1", "official_decisive_label_accuracy")
        ) and all(intervals[m]["mean_difference"] >= 0 for m in METRICS)
        cost = (
            a["reranker_accounting_complete"]
            and b["reranker_accounting_complete"]
            and a["reranker_known_token_lower_bound"]
            <= b["reranker_known_token_lower_bound"]
            and a["rerank_pairs"] <= b["rerank_pairs"]
            and a["mean_total_generator_tokens"] is not None
            and b["mean_total_generator_tokens"] is not None
            and a["mean_total_generator_tokens"] <= b["mean_total_generator_tokens"]
            and a["mean_elapsed_ms"] <= b["mean_elapsed_ms"]
            and a["operational_failures"] <= b["operational_failures"]
        )
        comparisons[control] = {
            "paired_bootstrap_5000": intervals,
            "quality_gate": quality,
            "cost_gate": cost,
        }
    return {
        "protocol": run.get("protocol", PROTOCOL),
        "study_kind": "synthetic_fixture_not_quality_evidence" if synthetic else STUDY,
        "independent_test": False,
        "routes": aggregate,
        "comparisons": comparisons,
        "quality_gate": all(c["quality_gate"] for c in comparisons.values()),
        "efficiency_gate": all(
            c["quality_gate"] and c["cost_gate"] for c in comparisons.values()
        ),
        "semantic_entailment": "unmeasured",
        "resume_promotion": False,
        "boundary": "Repeated validation; same item schema and caps, not identical decoder/control/call protocol. No pure feedback causal claim.",
    }
