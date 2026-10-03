"""Offline compact audit; raw model reasons stay in private runs, gold outside inference."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import subprocess
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np

from climate_rag.io import write_json
from climate_rag.metrics import paired_bootstrap, per_claim_retrieval_metrics
from climate_rag.models import Claim, Prediction
from climate_rag.verification import normalise_claim

STRATEGIES = ("fixed_retrieval", "fixed_rerank", "adaptive")
WORK_COUNTERS = ("tool_calls", "model_calls", "generation_calls", "retrieval_calls",
                 "rerank_calls", "rerank_candidate_pairs")
QUALITY_METRICS = ("recall@5", "mrr@10", "ndcg@10", "evidence_f1")
SELECTION_PATH = Path(__file__).resolve().parents[1] / "docs/verified-runs/budget-agent-validation-selection-20260929.json"
SELECTION_SHA = "988b6682034a70966c8bbad5ff3c42202933fe85791b97a5f39e4a1b68071bc6"
CONTROLLER_REASONS = {
    "stage_failed", "deadline_exceeded", "budget_exhausted", "disallowed_action", "query_loop",
    "rewrite_dropped_numeric_constraint", "rewrite_dropped_entity", "rewrite_changed_qualifier",
    "rewrite_lexical_drift", "no_new_evidence",
}


def sha(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def parse_frozen(raw: bytes, expected: str, name: str) -> dict[str, Any]:
    if sha(raw) != expected:
        raise ValueError(f"{name} SHA mismatch")
    value = json.loads(raw)
    if not isinstance(value, dict):
        raise ValueError(f"{name} must be a JSON object")
    return value


def load_selection_manifest() -> dict[str, Any]:
    selection = parse_frozen(SELECTION_PATH.read_bytes(), SELECTION_SHA, "selection manifest")
    counts = (len(selection["selected_ids"]), selection["call_latency_denominator_per_strategy"],
              selection["retrieval_denominator"], selection["selected_decisive"], selection["selected_nondecisive"])
    if counts != (32, 32, 24, 24, 8):
        raise ValueError("frozen validation 32/24/8 denominator mismatch")
    return selection


def validate_matrix(run: dict[str, Any], protocol: dict[str, Any],
                    phase: str, protocol_sha: str) -> dict[str, str]:
    if phase not in {"pilot", "validation", "vnext"} or run["phase"] != phase or phase not in protocol:
        raise ValueError("phase mismatch")
    if run["protocol_sha256"] != protocol_sha:
        raise ValueError("run protocol mismatch")
    if run["corpus_sha256"] != protocol["corpus_sha256"] or run["study_kind"] != protocol["study_kind"]:
        raise ValueError("corpus/study identity mismatch")
    strategies = protocol["strategies"]
    if len(strategies) != 3 or set(strategies) != set(STRATEGIES):
        raise ValueError("protocol strategy set mismatch")
    tasks = protocol[phase]
    if not tasks or len({x["id"] for x in tasks}) != len(tasks):
        raise ValueError("empty/duplicate protocol tasks")
    claim_hashes = {x["id"]: sha(normalise_claim(x["claim_text"]).encode()) for x in tasks}
    expected = {(task_id, strategy) for task_id in claim_hashes for strategy in strategies}
    actual = Counter((x["task_id"], x["strategy"]) for x in run["runs"])
    if set(actual) != expected or any(count != 1 for count in actual.values()):
        raise ValueError("incomplete/duplicate/extra task-strategy matrix")
    for row in run["runs"]:
        if sha(row["claim_text"].encode()) != claim_hashes[row["task_id"]]:
            raise ValueError("run claim mismatch")
    return claim_hashes


def validate_gold(protocol: dict[str, Any], phase: str, protocol_sha: str,
                  claim_hashes: dict[str, str], gold_bytes: bytes | None,
                  selection: dict[str, Any] | None) -> dict[str, Any] | None:
    if phase != "validation":
        if gold_bytes is not None or selection is not None or protocol["official_gold_available"] is not False:
            raise ValueError("authored phase forbids official gold/selection")
        return None
    if protocol["official_gold_available"] is not True or gold_bytes is None or selection is None:
        raise ValueError("validation requires frozen gold and selection manifest")
    if selection["protocol_sha256"] != protocol_sha:
        raise ValueError("selection protocol mismatch")
    gold = parse_frozen(gold_bytes, selection["gold_sha256"], "gold")
    if gold["protocol_sha256"] != protocol_sha:
        raise ValueError("gold protocol mismatch")
    selected = selection["selected_ids"]
    if (len(selected) != len(set(selected)) or set(selected) != set(claim_hashes)
            or set(gold["claims"]) != set(claim_hashes)
            or selection["call_latency_denominator_per_strategy"] != len(claim_hashes)):
        raise ValueError("gold/selection task set mismatch")
    provenance = gold["provenance"]
    if (provenance["archive_sha256"] != selection["source_archive_sha256"]
            or provenance["member_sha256"] != selection["member_sha256"]
            or provenance["annotation_kind"] != "official_CLIMATE_FEVER_evidence_and_label"
            or provenance["split"] != "repeated_validation_replay"):
        raise ValueError("gold provenance mismatch")
    for task_id, expected_hash in claim_hashes.items():
        if gold["claims"][task_id]["claim_sha256"] != expected_hash:
            raise ValueError("gold claim mismatch")
    evidence_count = sum(bool(x["evidence_ids"]) for x in gold["claims"].values())
    if (evidence_count != selection["retrieval_denominator"]
            or evidence_count != selection["selected_decisive"]
            or len(claim_hashes) - evidence_count != selection["selected_nondecisive"]):
        raise ValueError("gold retrieval denominator mismatch")
    return gold


def outcome(row: dict[str, Any]) -> str:
    """A controller's abstained status alone is not model-chosen abstention."""
    if row["status"] not in {"answered", "abstained"} or not isinstance(row["reason"], str):
        raise ValueError("invalid status/reason")
    if (row["status"] == "answered") != (row["answer"] is not None):
        raise ValueError("answer/status mismatch")
    failures = [x for x in row["events"] if x["stage"] == "failure"]
    decisions = [x["decision"] for x in row["events"] if x["stage"] == "decision"]
    if failures:
        if row["answer"] is not None or row["reason"] not in {"stage_failed", "deadline_exceeded"}:
            raise ValueError("failure/status mismatch")
        return "controller_failure"
    if row["answer"] is not None:
        if row["reason"] != "source_and_numeric_checks_passed_semantics_unverified":
            raise ValueError("answer reason mismatch")
        return "mechanically_validated_answer_semantics_unverified"
    if decisions and decisions[-1]["action"] == "abstain" and row["reason"] == decisions[-1]["reason"]:
        return "model_requested_abstention" if row["provider_kind"] == "local_model" else "heuristic_control_abstention"
    if row["reason"].startswith("answer_validation:"):
        errors = row["reason"].removeprefix("answer_validation:").split(",")
        if (set(errors) <= {"unknown_citation", "quote_not_exact", "uncited_numeric_value"}
                and errors == sorted(set(errors))):
            return "controller_rejection"
        raise ValueError("unrecognized answer-validation reason")
    if row["reason"] in CONTROLLER_REASONS:
        return "controller_rejection"
    raise ValueError("unclassified abstention reason without matching decision")


def public_reason(row: dict[str, Any], category: str) -> str:
    # All other accepted reasons are fixed controller enums. Never expose model prose.
    if category in {"model_requested_abstention", "heuristic_control_abstention"}:
        return category
    return str(row["reason"])


def validate_accounting(row: dict[str, Any]) -> None:
    if row["provider_kind"] not in {"local_model", "heuristic_no_model"}:
        raise ValueError("unrecognized provider kind")
    counts = [row[key] for key in WORK_COUNTERS] + [row["usage"][key] for key in ("input_tokens", "output_tokens")]
    if any(type(value) is not int or value < 0 for value in counts):
        raise ValueError("invalid work/token counter")
    if row["generation_calls"] != row["model_calls"] or type(row["usage_known"]) is not bool:
        raise ValueError("inconsistent generation/usage accounting")
    if not math.isfinite(row["elapsed_ms"]) or row["elapsed_ms"] < 0:
        raise ValueError("invalid latency")


def score(run: dict[str, Any], protocol_bytes: bytes, *, phase: str,
          expected_protocol_sha256: str, gold_bytes: bytes | None = None,
          selection: dict[str, Any] | None = None) -> dict[str, Any]:
    protocol = parse_frozen(protocol_bytes, expected_protocol_sha256, "protocol")
    claim_hashes = validate_matrix(run, protocol, phase, expected_protocol_sha256)
    gold = validate_gold(protocol, phase, expected_protocol_sha256, claim_hashes, gold_bytes, selection)
    rows: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in run["runs"]:
        validate_accounting(row)
        category = outcome(row)
        metrics: dict[str, Any] = {}
        if gold is not None:
            item = gold["claims"][row["task_id"]]
            if item["evidence_ids"]:
                claim = Claim(row["task_id"], row["claim_text"], evidence_ids=tuple(item["evidence_ids"]))
                metrics = per_claim_retrieval_metrics(claim, Prediction(
                    row["task_id"], tuple(row["delivered_evidence_ids"])), ks=(5, 10), evidence_k=5)
            if row["provider_kind"] == "local_model" and item.get("label") in {"SUPPORTS", "REFUTES"}:
                metrics["official_decisive_label_correct"] = int(
                    row["answer"] is not None and row["answer"]["label"] == item["label"])
        rows[row["strategy"]].append({
            "task_id": row["task_id"], **metrics, **{key: row[key] for key in WORK_COUNTERS},
            "elapsed_ms": row["elapsed_ms"], "answered": row["answer"] is not None,
            "usage_known": row["usage_known"], "usage": row["usage"],
            "status": row["status"], "reason": public_reason(row, category), "outcome": category,
            "provider_kind": row["provider_kind"], "provider": row.get("provider"),
        })
    aggregate: dict[str, Any] = {}
    for strategy, values in rows.items():
        result: dict[str, Any] = {
            "count": len(values), "answered": sum(x["answered"] for x in values),
            "latency_ms_p50": float(np.percentile([x["elapsed_ms"] for x in values], 50)),
            "latency_ms_p95": float(np.percentile([x["elapsed_ms"] for x in values], 95)),
            "status_counts": dict(Counter(x["status"] for x in values)),
            "reason_counts": dict(Counter(x["reason"] for x in values)),
            "outcome_counts": dict(Counter(x["outcome"] for x in values)),
            "unknown_token_accounting_runs": sum(not x["usage_known"] for x in values),
            "generation_token_totals_exact": all(x["usage_known"] for x in values),
            "operational_failures": sum(x["outcome"] == "controller_failure" for x in values),
            "evidence_metrics": None, "retrieval_denominator": None,
            "official_decisive_label_accuracy": None, "label_denominator": None,
        }
        for key in WORK_COUNTERS:
            result[key + "_sum"] = sum(x[key] for x in values)
            result[key + "_mean"] = float(np.mean([x[key] for x in values]))
        for key in ("input_tokens", "output_tokens"):
            result["recorded_generation_" + key] = sum(x["usage"][key] for x in values)
            result["complete_accounting_generation_" + key] = sum(x["usage"][key] for x in values if x["usage_known"])
            result["partial_accounting_recorded_" + key] = sum(x["usage"][key] for x in values if not x["usage_known"])
        if gold is not None:
            evidence_rows = [x for x in values if "recall@5" in x]
            result["evidence_metrics"] = {
                key: float(np.mean([x[key] for x in evidence_rows])) if evidence_rows else None
                for key in QUALITY_METRICS}
            result["retrieval_denominator"] = len(evidence_rows)
            labels = [x["official_decisive_label_correct"] for x in values if "official_decisive_label_correct" in x]
            result["official_decisive_label_accuracy"] = float(np.mean(labels)) if labels else None
            result["label_denominator"] = len(labels)
        aggregate[strategy] = result
    bootstrap = {}
    if gold is not None:
        baseline = sorted((x for x in rows["fixed_retrieval"] if "recall@5" in x), key=lambda x: x["task_id"])
        for strategy in ("fixed_rerank", "adaptive"):
            candidate = sorted((x for x in rows[strategy] if "recall@5" in x), key=lambda x: x["task_id"])
            if [x["task_id"] for x in baseline] != [x["task_id"] for x in candidate]:
                raise ValueError("unpaired task sets")
            bootstrap[strategy] = {key: paired_bootstrap(
                [x[key] for x in baseline], [x[key] for x in candidate], samples=5000, seed=20260929)
                for key in QUALITY_METRICS} if baseline else None
    comparisons = {}
    for strategy in ("fixed_rerank", "adaptive"):
        paired = rows["fixed_retrieval"] + rows[strategy]
        complete = all(x["usage_known"] for x in paired)
        same_model = (all(x["provider_kind"] == "local_model" and x["provider"] for x in paired)
                      and len({x["provider"] for x in paired}) == 1)
        comparisons[strategy] = {
            "generation_token_comparison_available": bool(complete and same_model),
            "unavailable_reason": "unknown_token_accounting" if not complete else
                (None if same_model else "no_common_real_generation_provider"),
            "monetary_cost_comparison_available": False,
            "monetary_cost_reason": "No currency/GPU/API pricing model; tokens exclude reranker work",
        }
    return {
        "phase": phase, "protocol_sha256": expected_protocol_sha256,
        "gold_sha256": sha(gold_bytes) if gold_bytes is not None else None,
        "inference_source_git_sha": run.get("code_sha"),
        "matrix": {"expected_slots": len(claim_hashes) * 3, "actual_slots": len(run["runs"]),
                   "task_denominator_per_strategy": len(claim_hashes), "validated": True},
        "aggregates": aggregate, "paired_bootstrap": bootstrap or None, "cost_comparison": comparisons,
        "gold_source": gold["provenance"] if gold else None, "semantic_supportability": None,
        "boundary": "Validation replay only; authored tasks have null official quality. Quote/number checks are not entailment. Partial tokens are lower bounds, not savings. Model prose is excluded; reason_counts contains controller enums or event-grounded abstention categories.",
    }


def scorer_identity() -> dict[str, Any]:
    root = Path(__file__).resolve().parents[1]
    revision = (root / "SOURCE_REVISION").read_text().strip()
    dirty: bool | None = None
    if not re.fullmatch(r"[0-9a-f]{40}", revision):
        revision = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root, text=True).strip()
        dirty = bool(subprocess.check_output(["git", "status", "--porcelain"], cwd=root, text=True).strip())
    return {"git_sha": revision, "working_tree_dirty": dirty, "script_sha256": sha(Path(__file__).read_bytes())}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--expected-protocol-sha256", required=True)
    parser.add_argument("--phase", choices=["pilot", "validation", "vnext"], required=True)
    parser.add_argument("--gold", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    run_bytes = args.run.read_bytes()
    selection = load_selection_manifest() if args.phase == "validation" else None
    result = score(json.loads(run_bytes), args.protocol.read_bytes(), phase=args.phase,
                   expected_protocol_sha256=args.expected_protocol_sha256,
                   gold_bytes=args.gold.read_bytes() if args.gold else None, selection=selection)
    result.update(run_sha256=sha(run_bytes), scorer_identity=scorer_identity(),
                  selection_manifest_sha256=sha(SELECTION_PATH.read_bytes()) if selection else None)
    if args.output.exists():
        raise ValueError("refuse to overwrite score output")
    write_json(args.output, result)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
