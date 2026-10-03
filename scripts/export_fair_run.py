"""Export accepted aggregate measurements, never private rows, to portable JSON.

No inference, gold parsing, network or upload. The original private tree must be
backed up separately; this projection is not a replacement for raw evidence.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import re
from typing import Any

from climate_rag.fair_acquisition import COVERAGE_PROTOCOL, PROTOCOL, ROUTES
from climate_rag.semantic_proxy import matrix
from replay_fair_acquisition_result import digest

BEHAVIOR = {"answered", "intentional_abstention", "execution_failure", "gate_attempts",
            "validated_stop", "validated_acquisition", "gate_received_new_full_text",
            "tool_retrieve", "tool_read", "tool_rewrite", "tool_rerank", "tool_failure", "empty_query"}
METRICS = ("recall@5", "mrr@10", "ndcg@10", "evidence_f1")
QUALITY = ("official_task_correct", "task_and_official_id_proxy_joint", "official_id_concordance_all_tasks")
BOOTSTRAP = ("pair_count", "samples", "baseline_mean", "candidate_mean", "mean_difference",
             "ci_lower", "ci_upper", "two_sided_p")


def numeric(value: Any) -> Any:
    if value is not None and (type(value) not in {int, float} or not math.isfinite(value)):
        raise ValueError("aggregate_must_be_numeric")
    return value


def fields(value: dict[str, Any], keys: Any) -> dict[str, Any]:
    return {k: numeric(value[k]) for k in keys}


def route_cost(rows: list[dict[str, Any]]) -> dict[str, Any]:
    # Preserve unknown usage, not a zero-cost assertion or duplicated token sum.
    result: dict[str, Any] = {}
    for target, parent, key in (
        ("generation_calls", "generation", "unique_physical_calls"),
        ("unknown_generator_usage", "generation", "unknown_usage_attempts"),
        ("generation_ms", "generation", "elapsed_ms"),
        ("reranker_requests", "reranker", "physical_requests"),
        ("reranker_requested_pairs", "reranker", "requested_pairs"),
        ("reranker_completed_pairs", "reranker", "completed_pairs"),
        ("unknown_reranker_usage", "reranker", "unknown_requests"),
        ("reranker_ms_including_swaps", "reranker", "elapsed_ms_including_swaps"),
    ):
        values = [numeric(r[parent][key]) for r in rows]
        result[target] = sum(values) if all(v is not None for v in values) else None
    for key in ("input_tokens", "output_tokens"):
        result[key + "_known_lower_bound"] = sum(numeric(r["generation"]["known_token_lower_bound"][key]) for r in rows)
    result["reranker_tokens_known_lower_bound"] = sum(numeric(r["reranker"]["known_token_lower_bound"]) for r in rows)
    result["tool_calls"] = sum(numeric(r["tool_calls"]) for r in rows)
    elapsed = sorted(numeric(r["elapsed_ms"]) for r in rows)
    result.update(slot_elapsed_ms_sum=sum(elapsed), slot_elapsed_ms_mean=sum(elapsed) / len(elapsed),
                  slot_p50_nearest_rank_ms=elapsed[math.ceil(len(elapsed) * .50) - 1],
                  slot_p95_nearest_rank_ms=elapsed[math.ceil(len(elapsed) * .95) - 1])
    return result


def validate_cost_roster(compact: dict[str, Any], raw: dict[str, Any], cost: dict[str, Any]) -> None:
    all_rows = []
    for arm in ROUTES:
        saved = [r for r in raw["runs"] if r["route"] == arm]
        published = compact["routes"][arm]["physical_cost_by_task"]
        expected = [{"task_id": r["task_id"], "generation": r["physical_cost"],
                     "reranker": r["reranker_cost"], "tool_calls": r["tool_calls"],
                     "cumulative_model_prompt_tokens": r["cumulative_model_prompt_tokens"],
                     "elapsed_ms": r["elapsed_ms"]} for r in saved]
        if (len(set(r["task_id"] for r in published)) != len(published) or published != expected):
            raise ValueError("cost_roster_or_raw_measurement_mismatch")
        all_rows.extend(published)
    totals = route_cost(all_rows)
    g, r = cost["generation"], cost["reranker"]
    expected_global = {"generation_calls": g["unique_physical_calls"],
        "unknown_generator_usage": g["unknown_usage_attempts"], "generation_ms": g["elapsed_ms"],
        "reranker_requests": r["physical_requests"], "reranker_requested_pairs": r["requested_pairs"],
        "reranker_completed_pairs": r["completed_pairs"], "unknown_reranker_usage": r["unknown_requests"],
        "reranker_ms_including_swaps": r["elapsed_ms_including_swaps"],
        "input_tokens_known_lower_bound": g["known_token_lower_bound"]["input_tokens"],
        "output_tokens_known_lower_bound": g["known_token_lower_bound"]["output_tokens"],
        "reranker_tokens_known_lower_bound": r["known_token_lower_bound"]}
    for key, value in expected_global.items():
        if value is None or totals[key] is None:
            matches = value is totals[key]
        else:
            matches = math.isclose(numeric(value), totals[key], rel_tol=1e-12, abs_tol=1e-6)
        if not matches:
            raise ValueError("global_cost_mismatch:" + key)


def validate_public_identity(operator: dict[str, Any], reserved: dict[str, Any]) -> None:
    if (not re.fullmatch(r"[a-f0-9]{40}", str(operator["source_git"]))
            or not re.fullmatch(r"[a-f0-9]{64}", str(operator["release_sha256"]))
            or any(reserved[k] != operator[k] for k in ("source_git", "release_sha256", "planned_slots"))):
        raise ValueError("public_execution_identity_mismatch")


def project(compact: dict[str, Any]) -> dict[str, Any]:
    if compact["protocol"] not in {PROTOCOL, COVERAGE_PROTOCOL} or set(compact["routes"]) != set(ROUTES):
        raise ValueError("known_complete_fair_protocol_required")
    result: dict[str, Any] = {"schema": "fair-three-arm-redacted-result-v1",
        "protocol": compact["protocol"],
        "study_kind": ("consumed_fair32_coverage_regression_not_independent_quality_validation"
                       if compact["protocol"] == COVERAGE_PROTOCOL else
                       "public_v2_retrieval_exposed_development_policy_comparison_not_independent_test"),
        "semantic_citation_support": "not measured by official evidence-ID score; automatic NLI is separate",
        "citation_proxy": "official decisive evidence-ID concordance, NOT semantic support",
        "feedback_causality": "whole_policy_comparison_only", "routes": {}, "paired_bootstrap": {}}
    for arm in ROUTES:
        r = compact["routes"][arm]
        if r["all_task_denominator"] != 32 or len(r["physical_cost_by_task"]) != 32:
            raise ValueError("complete_frozen_denominator_required")
        result["routes"][arm] = {"all_task_denominator": 32, **fields(r, QUALITY),
            "behavior": fields(r["behavior"], sorted(set(r["behavior"]) & BEHAVIOR)),
            "retrieval": {"denominator": numeric(r["retrieval"]["denominator"]),
                "scope": "final ranked candidates on evidence-bearing tasks; not full-text delivery or entailment",
                "metrics": fields(r["retrieval"]["metrics"], METRICS)},
            "cost": route_cost(r["physical_cost_by_task"])}
    for arm in ROUTES[:2]:
        key = "autonomous-minus-" + arm
        src = compact["paired_bootstrap"][key]
        result["paired_bootstrap"][key] = {k: fields(src[k], BOOTSTRAP) for k in QUALITY[:2]}
        if "retrieval_secondary" in src:
            result["paired_bootstrap"][key]["retrieval_secondary"] = {
                k: fields(src["retrieval_secondary"][k], BOOTSTRAP) for k in METRICS}
    return result


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--run-dir", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    root = args.run_dir
    operator = json.loads((root / "operator-status.json").read_bytes())
    validate_public_identity(operator, json.loads((root / "reserved.json").read_bytes()))
    compact_path, cost_path, raw_path = root / "compact.json", root / "cost-before-quality.json", root / "inference/run.json"
    compact = json.loads(compact_path.read_bytes())
    cost = json.loads(cost_path.read_bytes())
    if (operator["status"] != "completed" or operator["error_type"] is not None
            or operator["compact_sha256"] != digest(compact_path)
            or operator["cost_sha256"] != digest(cost_path)
            or compact["source_git"] != operator["source_git"]
            or compact["raw_run_sha256"] != digest(raw_path)
            or compact["cost_sha256"] != digest(cost_path)
            or cost["completed_slots"] != 96 or cost["planned_slots"] != 96
            or cost["worker_proof"]["returncode"] != 0
            or cost["worker_proof"]["child_started"] is not True
            or cost["worker_proof"]["child_reaped"] is not True
            or cost["worker_proof"]["interrupted"] is not None):
        raise ValueError("completed_same_identity_run_required")
    raw = json.loads(raw_path.read_bytes())
    matrix(raw, expected_tasks=32, expected_protocol=compact["protocol"])
    validate_cost_roster(compact, raw, cost)
    result = project(compact)
    result.update(source_git=operator["source_git"], release_sha256=operator["release_sha256"],
                  raw_run_sha256=digest(raw_path), cost_sha256=digest(cost_path),
                  original_compact_sha256=digest(compact_path), completed_slots=96, planned_slots=96,
                  operator_elapsed_seconds=numeric(operator["elapsed_seconds"]),
                  publisher_script_sha256=digest(Path(__file__)))
    with args.output.open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(result, stream, indent=2)
        stream.write("\n")
    print(json.dumps({"output_sha256": digest(args.output), "slots": 96, "raw_rows_exported": False}))


if __name__ == "__main__":
    main()
