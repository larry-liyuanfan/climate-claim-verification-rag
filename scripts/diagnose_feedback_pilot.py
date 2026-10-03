"""Content-free v2 failure shapes from private run/raw, never semantic labels."""

from __future__ import annotations

import argparse
import hashlib
import json
import statistics
from collections import Counter, defaultdict
from pathlib import Path


def diagnose(run, raw_by_hash):
    result = {}
    for route in ("fixed_retrieval", "fixed_rerank", "adaptive"):
        rows = [row for row in run["runs"] if row["strategy"] == route]
        if not rows:
            raise ValueError("missing route")
        counts, first, allowed, reasons = (Counter() for _ in range(4))
        lengths, errors = [], Counter()
        by_category = defaultdict(list)
        for row in rows:
            initial = row["generation_attempts"][0]
            allowed[",".join(initial["allowed_actions"])] += 1
            raw = raw_by_hash[initial["diagnostics"]["output_sha256"]]
            try:
                action = json.loads(raw).get("action")
                first[
                    action
                    if action in {"answer", "abstain", "rewrite", "rerank"}
                    else "other"
                ] += 1
            except (json.JSONDecodeError, AttributeError):
                first["unparsed_json"] += 1
            for attempt in row["generation_attempts"]:
                diag = attempt["diagnostics"]
                raw = raw_by_hash[diag["output_sha256"]]
                if hashlib.sha256(raw.encode()).hexdigest() != diag["output_sha256"]:
                    raise ValueError("private raw response SHA mismatch")
                lengths.append(attempt["usage"]["output_tokens"])
                by_category[diag["category"]].append(attempt["usage"]["output_tokens"])
                counts["eos_true"] += diag.get("eos_observed") is True
                counts["token_cap_reached"] += (
                    diag.get("reached_max_new_tokens") is True
                )
                counts["contains_think_tag"] += "<think>" in raw or "</think>" in raw
                for error in diag.get("errors", []):
                    # Shape only; never stringify an arbitrary model field or input.
                    if (
                        error.get("loc") == ["answer", "statements"]
                        and error.get("type") == "too_long"
                    ):
                        errors["too_many_answer_statements"] += 1
                    else:
                        errors["other_schema_error"] += 1
            for event in row["events"]:
                if event["stage"] != "decision":
                    continue
                decision = event["decision"]
                if decision["action"] == "abstain":
                    reasons[
                        "exact_prompt_example_reason"
                        if decision["reason"] == "Missing required evidence."
                        else "other_reason"
                    ] += 1
                if decision["action"] == "answer":
                    context = set(row["context_evidence_ids"])
                    candidates = set(row["candidate_evidence_ids"])
                    for statement in decision["statements"]:
                        sid = statement["evidence_id"]
                        counts["answer_citation_in_context"] += sid in context
                        counts["answer_citation_preview_only"] += (
                            sid in candidates and sid not in context
                        )
                        counts["answer_citation_outside_candidates"] += (
                            sid not in candidates
                        )
            counts["rows_with_nonempty_full_context"] += bool(
                row["context_evidence_ids"]
            )
            counts["rows_with_5_full_context_docs"] += (
                len(row["context_evidence_ids"]) == 5
            )
            counts["repair_then_abstain_rows"] += (
                row["validation_repairs"] > 0
                and row["events"][-1].get("decision", {}).get("action") == "abstain"
            )
        result[route] = {
            "first_response_action_before_validation": dict(first),
            "first_allowed_actions": dict(allowed),
            "counts": dict(counts),
            "schema_errors": dict(errors),
            "output_tokens": {
                "min": min(lengths),
                "median": statistics.median(lengths),
                "max": max(lengths),
            },
            "output_tokens_by_category": dict(by_category),
            "abstain_reason_shape": dict(reasons),
        }
    return {
        "schema_version": "feedback-v2-failure-shapes-v1",
        "routes": result,
        "claim_text_or_evidence_ids_exported": False,
        "semantic_support_assessed": False,
        "causal_ablation": False,
        "caution": "Available full context is not verified entailment; parsed actions need not pass final validation. Citation partition uses final snapshot, valid here because no model-selected tool ran.",
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--raw-directory", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError("diagnostic output exists")
    payload = args.run.read_bytes()
    run = json.loads(payload)
    if any(
        e.get("feedback", {}).get("model_selected")
        for r in run["runs"]
        for e in r["events"]
    ):
        raise ValueError("final-snapshot diagnosis only applies to no-model-tool pilot")
    files = list(args.raw_directory.rglob("*.txt"))
    raw = {
        hashlib.sha256(p.read_bytes()).hexdigest(): p.read_bytes().decode("utf-8")
        for p in files
    }
    result = diagnose(run, raw)
    result.update(
        {
            "private_run_sha256": hashlib.sha256(payload).hexdigest(),
            "private_raw_file_count": len(files),
            "diagnostic_script_sha256": hashlib.sha256(
                Path(__file__).read_bytes()
            ).hexdigest(),
        }
    )
    with args.output.open("x", encoding="utf-8") as stream:
        json.dump(result, stream, indent=2, sort_keys=True)
        stream.write("\n")


if __name__ == "__main__":
    main()
