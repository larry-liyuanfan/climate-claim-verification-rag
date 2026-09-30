"""Small, stdlib-only receipt audit; never runs inference or recomputes frozen scores.

Execute on Spartan against private r2 files. Only aggregate counters and bounded
anonymous case summaries leave the run directory, never model prose or row dumps.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import tarfile
import types
from collections import Counter, defaultdict
from pathlib import Path

OPERATOR_GIT = "947645b3feed914e30d02dc5dbd05859c4a19667"
OPERATOR_ARCHIVE_SHA = "f9aa535d33cd8068c9a747aa8aba431a036b56e55cf0cb7a959daa36132e4d42"
OPERATOR_SCRIPT_SHA = "2ec2dd9e9664305c4b870def9df3a6d8ba0fb17b0d8984149e0ac79cac170b75"
STATUS_SHA = "4f1e7a0e2e3d2fcf2dc64ddf5d4ac128e65dda59e41af2b880a16a65f9e12f34"
SCORER_LF_SHA = "5a3afc492dacb786e67051d9d6900afa671f742a0222149f9cd9a7677e37956b"
RELEASE = "climate-full-72eaa90-20260930-r2"
STRATEGIES = ("fixed_retrieval", "fixed_rerank", "adaptive")
ACTIONS = {"answer", "abstain", "rewrite", "rerank"}
OUTCOMES = {"controller_failure", "controller_rejection", "model_requested_abstention",
            "mechanically_validated_answer_semantics_unverified"}
WORK_KEYS = ("tool_calls", "model_calls", "generation_calls", "retrieval_calls",
             "rerank_calls", "rerank_candidate_pairs")
BUDGET = {"candidate_k": 20, "context_k": 5, "max_input_tokens_per_call": 8192,
          "max_model_calls": 3, "max_output_tokens_per_call": 512,
          "max_tool_calls": 3, "timeout_seconds": 120.0}


def sha(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def require(condition: bool) -> None:
    if not condition:
        raise ValueError("compact audit contract mismatch; inspect private sources in place")


def receipt(path: Path) -> dict:
    raw = path.read_bytes()
    return {"sha256": sha(raw), "bytes": len(raw)}


def category(row: dict) -> str:
    if any(e["stage"] == "failure" for e in row["events"]):
        require(row["answer"] is None)
        return "controller_failure"
    if row["answer"] is not None:
        return "mechanically_validated_answer_semantics_unverified"
    decisions = [e["decision"] for e in row["events"] if e["stage"] == "decision"]
    if decisions and decisions[-1]["action"] == "abstain" and row["reason"] == decisions[-1]["reason"]:
        return "model_requested_abstention"
    return "controller_rejection"


def quantile(values: list[float], q: float) -> float:
    ordered = sorted(values)
    position = (len(ordered) - 1) * q
    lo, hi = math.floor(position), math.ceil(position)
    return ordered[lo] + (ordered[hi] - ordered[lo]) * (position - lo)


def inspect_payload(raw: bytes) -> list[str]:
    """Structural diagnosis, NOT replay through the frozen Pydantic validator."""
    try:
        value = json.loads(raw)
    except (ValueError, UnicodeError):
        return ["not_json_object"]
    if not isinstance(value, dict):
        return ["not_json_object"]
    flags = []
    if value.get("action") != "rewrite" and value.get("query") is not None:
        flags.append("query_on_nonrewrite")
    if isinstance(value.get("reason"), str) and len(value["reason"]) > 500:
        flags.append("reason_over_500_characters")
    if value.get("action") != "answer" and (value.get("statements") or value.get("label") is not None):
        flags.append("answer_fields_on_nonanswer")
    return flags or ["no_inspected_field_violation"]


def row_summary(row: dict) -> dict:
    decisions = [e["decision"]["action"] for e in row["events"] if e["stage"] == "decision"]
    require(set(decisions) <= ACTIONS)
    failures = [e for e in row["events"] if e["stage"] == "failure"]
    return {
        "outcome": category(row), "parsed_model_actions": decisions,
        "quote_mismatch_rejected": row["reason"] == "answer_validation:quote_not_exact",
        "candidate_count": len(row["candidate_evidence_ids"]),
        "context_count": len(row["context_evidence_ids"]),
        "delivered_count": len(row["delivered_evidence_ids"]),
        "generated_response_failure": bool(failures) and all(e["error_type"] == "GeneratedResponseError" for e in failures),
        "elapsed_ms": row["elapsed_ms"], "input_tokens": row["usage"]["input_tokens"],
        "output_tokens": row["usage"]["output_tokens"], "usage_known": row["usage_known"],
        **{k: row[k] for k in WORK_KEYS},
    }


def aggregate_rows(rows: list[dict], gold: dict | None, payloads: dict[str, bytes]) -> dict:
    """Descriptive counters/context diagnostics; never rescore the frozen outputs."""
    actions, stages, diagnostics, schema_errors, attachments = (Counter() for _ in range(5))
    outcomes: dict[str, list[dict]] = defaultdict(list)
    generation_ms = retrieval_ms = rerank_ms = 0.0
    eos = max_tokens = matched_failures = failures_with_diagnostic = 0
    field_flags, error_types = Counter(), Counter()
    retained_recall5, candidate_gold_hits, suppressed_with_gold = [], 0, 0
    violations = Counter()
    for row in rows:
        require(row["provider_kind"] == "local_model")
        outcomes[category(row)].append(row)
        require(row["context_evidence_ids"] == row["candidate_evidence_ids"][:5])
        for key, ceiling in (("generation_calls", 3), ("tool_calls", 3)):
            violations[key] += row[key] > ceiling
        violations["row_elapsed_over_120s"] += row["elapsed_ms"] > 120000
        # Actual run has one generation per row: this checks exact per-call input.
        violations["single_call_input_over_8192"] += row["generation_calls"] == 1 and row["usage"]["input_tokens"] > 8192
        if gold is not None and gold[row["task_id"]]["evidence_ids"]:
            ids = set(gold[row["task_id"]]["evidence_ids"])
            hits = len(ids & set(row["context_evidence_ids"]))
            retained_recall5.append(hits / len(ids))
            candidate_gold_hits += bool(ids & set(row["candidate_evidence_ids"]))
            suppressed_with_gold += bool(hits and not row["delivered_evidence_ids"])
        for event in row["events"]:
            stages[event["stage"] if event["stage"] in {"retrieve", "rerank", "rewrite_retrieve", "decision", "failure"} else "other"] += 1
            if event["stage"] == "retrieve":
                retrieval_ms += event["elapsed_ms"]
            elif event["stage"] == "rerank":
                rerank_ms += event["elapsed_ms"]
            elif event["stage"] == "decision":
                action = event["decision"]["action"]
                require(action in ACTIONS)
                actions[action] += 1
            if event["stage"] == "failure":
                error_types[event["error_type"] if event["error_type"] in {"GeneratedResponseError", "TimeoutError", "ValueError"} else "other"] += 1
            diag = event.get("generation_diagnostics")
            if not diag:
                continue
            diag_kind = diag["category"]
            require(diag_kind in {"validated", "schema_validation", "json_decode", "decode_error"})
            diagnostics[diag_kind] += 1
            eos += diag.get("eos_observed") is True
            max_tokens += diag.get("reached_max_new_tokens") is True
            generation_ms += diag.get("generation_elapsed_ms", 0.0)
            violations["output_over_512"] += diag.get("output_tokens", 0) > 512
            flag = diag.get("private_attachment", "disabled")
            require(flag in {"disabled", "saved_owner_only", "skipped_per_response_limit", "skipped_total_limit", "write_failed"})
            attachments[flag] += 1
            for error in diag.get("errors", []):
                name = ("root_value_error" if error == {"loc": [], "type": "value_error"} else
                        "reason_string_too_long" if error == {"loc": ["reason"], "type": "string_too_long"} else "other_redacted")
                schema_errors[name] += 1
            if event["stage"] == "failure":
                failures_with_diagnostic += 1
                raw = payloads.get(diag.get("output_sha256"))
                if raw is not None:
                    matched_failures += 1
                    field_flags.update(inspect_payload(raw))
    cost = {}
    for name, group in outcomes.items():
        cost[name] = {
            "count": len(group), "generation_calls": sum(x["generation_calls"] for x in group),
            "recorded_input_tokens": sum(x["usage"]["input_tokens"] for x in group),
            "recorded_output_tokens": sum(x["usage"]["output_tokens"] for x in group),
            "unknown_usage_rows": sum(not x["usage_known"] for x in group),
            "elapsed_ms_sum": sum(x["elapsed_ms"] for x in group),
        }
    elapsed = [row["elapsed_ms"] for row in rows]
    return {
        "parsed_model_action_counts": dict(actions), "completed_event_counts": dict(stages),
        "generation_diagnostic_counts": dict(diagnostics), "schema_error_counts": dict(schema_errors),
        "failure_error_types": dict(error_types), "eos_observed_count": eos,
        "max_new_tokens_reached_count": max_tokens, "private_attachment_status_counts": dict(attachments),
        "cost_by_outcome": cost, "elapsed_ms_sum": sum(elapsed),
        "latency_ms_p50_check": quantile(elapsed, 0.5), "latency_ms_p95_check": quantile(elapsed, 0.95),
        "generation_elapsed_ms_sum": generation_ms, "retrieval_elapsed_ms_sum": retrieval_ms,
        "rerank_elapsed_ms_sum": rerank_ms, "budget_violation_counts": dict(violations),
        "private_failure_field_inspection": {
            "failure_diagnostic_rows": failures_with_diagnostic,
            "sha_matched_payload_rows": matched_failures,
            "unavailable_payload_rows": failures_with_diagnostic - matched_failures,
            "structural_flag_counts": dict(field_flags),
            "scope": "hash-matched retained private JSON only; duplicates can match retained bytes; not frozen-validator replay",
        },
        "ungated_retained_context_diagnostic": None if gold is None else {
            "denominator": len(retained_recall5),
            "mean_gold_recall_at_context5": sum(retained_recall5) / len(retained_recall5),
            "queries_with_gold_in_candidate20": candidate_gold_hits,
            "queries_with_context_gold_but_empty_delivered": suppressed_with_gold,
            "scope": "post-tool/pre-delivery retained snapshot; diagnostic only, never replaces frozen delivered-evidence scores",
        },
    }


def select_cases(phase_rows: dict[str, list[dict]]) -> list[dict]:
    """At most five outcome-stratified case summaries, not representative accuracy.

    Keep task/claim IDs and their enumerable hashes private. The fixed selection
    predicate and lexicographic tie-break reproduce the mapping in place.
    """
    groups = {}
    for phase, rows in phase_rows.items():
        for row in rows:
            groups.setdefault((phase, row["task_id"]), {})[row["strategy"]] = row
    criteria = [
        ("exact_quote_rejection", lambda p, g: any(x["reason"] == "answer_validation:quote_not_exact" for x in g.values())),
        ("validation_model_abstention", lambda p, g: p == "validation" and category(g["adaptive"]) == "model_requested_abstention"),
        ("validation_all_routes_schema_failure", lambda p, g: p == "validation" and all(category(x) == "controller_failure" for x in g.values())),
        ("authored_model_abstention", lambda p, g: p == "vnext" and category(g["adaptive"]) == "model_requested_abstention"),
        ("authored_all_routes_schema_failure", lambda p, g: p == "vnext" and all(category(x) == "controller_failure" for x in g.values())),
    ]
    used, selected = set(), []
    for name, predicate in criteria:
        matches = [(key, group) for key, group in sorted(groups.items()) if key not in used and predicate(key[0], group)]
        if not matches:
            continue
        key, group = matches[0]
        used.add(key)
        selected.append({"case_id": name, "phase": key[0], "eligible_remaining_groups": len(matches),
                         "selection": "first task ID in lexical order, unused group",
                         "routes": {route: row_summary(group[route]) for route in STRATEGIES}})
    return selected


def build(root: Path, archive: Path) -> dict:
    require(receipt(archive)["sha256"] == OPERATOR_ARCHIVE_SHA)
    with tarfile.open(archive) as bundle:
        source = bundle.extractfile("scripts/run_budget_agent_full_operator.py").read()
        selection_raw = bundle.extractfile("docs/verified-runs/budget-agent-validation-selection-20260929.json").read()
        authored_raw = bundle.extractfile("configs/budget_agent_vnext_20260929.json").read()
    require(sha(source) == OPERATOR_SCRIPT_SHA)
    operator = types.ModuleType("frozen_operator")
    exec(compile(source, "frozen_operator.py", "exec"), operator.__dict__)
    result = root / "runs" / RELEASE
    status_raw = (result / "operator-status.json").read_bytes()
    require(sha(status_raw) == STATUS_SHA)
    state = json.loads(status_raw)
    require(state["status"] == state["stage"] == "complete" and state["job_id"] == "31543304")
    require(state["operator_git_sha"] == OPERATOR_GIT and state["operator_archive_sha256"] == OPERATOR_ARCHIVE_SHA)
    require(state["operator_script_sha256"] == OPERATOR_SCRIPT_SHA and state["input_extractions"] == 1)
    require(state["inference_git_sha"] == operator.INFERENCE and state["scorer_git_sha"] == operator.SCORER)
    require(sha(selection_raw) == operator.SELECTION_SHA)
    require(sha(authored_raw) == operator.PROTOCOLS["vnext"][1])
    scorer_archive = root / "envs" / operator.ARCHIVES["scorer"][0]
    require(receipt(scorer_archive)["sha256"] == operator.ARCHIVES["scorer"][1])
    with tarfile.open(scorer_archive) as bundle:
        scorer_bytes = bundle.extractfile("scripts/score_budget_agent.py").read()
    require(sha(scorer_bytes.replace(b"\r\n", b"\n")) == SCORER_LF_SHA)
    scorer_line_endings = {
        "archive_sha256": operator.ARCHIVES["scorer"][1], "member_bytes": len(scorer_bytes),
        "actual_member_sha256": sha(scorer_bytes), "crlf_count": scorer_bytes.count(b"\r\n"),
        "canonical_git_blob_lf_sha256": SCORER_LF_SHA,
        "only_crlf_normalization_matches_canonical_blob": True,
    }
    selection, authored = json.loads(selection_raw), json.loads(authored_raw)
    gold_path = root / "envs" / f"budget-agent-validation-gold-{operator.GOLD_SHA}.json"
    require(receipt(gold_path)["sha256"] == operator.GOLD_SHA)
    gold = json.loads(gold_path.read_bytes())["claims"]
    source_receipts, phases, all_rows = {"operator-status.json": receipt(result / "operator-status.json")}, {}, {}
    for phase in operator.PHASES:
        directory = result / phase
        verified = operator.verify_phase(directory, phase)
        require(state["phases"][phase] == {"status": "complete", "outputs": verified})
        for name, expected in verified.items():
            require(state["file_receipts"][phase + "/" + name] == expected)
            source_receipts[phase + "/" + name] = expected
        run, score = (json.loads((directory / name).read_bytes()) for name in ("run.json", "score.json"))
        require(score["scorer_identity"]["script_sha256"] == sha(scorer_bytes))
        require(run["budget"] == BUDGET)
        expected_ids = set(selection["selected_ids"]) if phase == "validation" else {x["id"] for x in authored["vnext"]}
        require(Counter((x["task_id"], x["strategy"]) for x in run["runs"]) == Counter((i, s) for i in expected_ids for s in STRATEGIES))
        claims = gold if phase == "validation" else {x["id"]: {"claim_sha256": sha(x["claim_text"].encode())} for x in authored["vnext"]}
        require(all(sha(x["claim_text"].encode()) == claims[x["task_id"]]["claim_sha256"] for x in run["runs"]))
        payloads = {}
        for path in (directory / "private-responses").glob("*.txt"):
            require(path.resolve().is_relative_to(directory.resolve()) and path.stat().st_size <= 32768)
            raw = path.read_bytes()
            require(state["file_receipts"][path.relative_to(result).as_posix()] == {"bytes": len(raw), "sha256": sha(raw)})
            payloads[sha(raw)] = raw
        supplemental = {}
        for route in STRATEGIES:
            rows = [x for x in run["runs"] if x["strategy"] == route]
            extra = aggregate_rows(rows, gold if phase == "validation" else None, payloads)
            aggregate = score["aggregates"][route]
            require(aggregate["outcome_counts"] == dict(Counter(category(x) for x in rows)))
            require(aggregate["count"] == len(rows) and aggregate["answered"] == sum(x["answer"] is not None for x in rows))
            for key in WORK_KEYS:
                require(aggregate[key + "_sum"] == sum(x[key] for x in rows))
            for key in ("input_tokens", "output_tokens"):
                require(aggregate["recorded_generation_" + key] == sum(x["usage"][key] for x in rows))
            for q in ("50", "95"):
                require(math.isclose(extra["latency_ms_p" + q + "_check"], aggregate["latency_ms_p" + q]))
            require(aggregate["retrieval_denominator"] == (24 if phase == "validation" else None))
            require(aggregate["label_denominator"] == (23 if phase == "validation" else None))
            supplemental[route] = extra
        # Frozen scorer is already compact and excludes row/model prose. Its bytes
        # were checked against the immutable completed-operator status above.
        phases[phase] = {"frozen_score": score, "budget": run["budget"],
                         "execution_manifest": run["execution_manifest"], "prompt_identity": run["prompt_identity"],
                         "neural_work_accounting": run["neural_work_accounting"], "supplemental": supplemental,
                         "private_attachment_files": sum(1 for _ in (directory / "private-responses").glob("*.txt"))}
        all_rows[phase] = run["runs"]
    return {"schema_version": "1.0", "job_id": "31543304", "release_id": RELEASE,
            "audit_script_sha256": receipt(Path(__file__))["sha256"],
            "source_receipts": source_receipts,
            "scorer_archive_line_endings": scorer_line_endings,
            "identities": {k: state[k] for k in ("inference_git_sha", "scorer_git_sha", "operator_git_sha", "operator_archive_sha256", "operator_script_sha256", "wrapper_sha256", "gold_sha256", "selection_sha256", "archives", "model_manifest_files")},
            "operator_elapsed_seconds": state["elapsed_seconds"], "input_extractions": 1,
            "gpu_memory_peak": state["gpu_memory_peak"], "gold_label_counts": dict(Counter(x["label"] for x in gold.values())),
            "phases": phases, "case_summaries": select_cases(all_rows),
            "frozen_scoring_recomputed": False, "posthoc_diagnostic_statistics_computed": True,
            "inference_repeated": False, "raw_text_or_prediction_rows_exported": False,
            "boundary": "Repeated validation and authored tasks kept separate. Delivered evidence is failure-gated, not initial retrieval. Parsed actions are not executed tools. Mechanical checks are not semantic support. Case summaries are selected examples, not additional quality metrics."}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--operator-archive", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    require(args.output.resolve().parent == (args.root / "runs" / RELEASE).resolve())
    require(not args.output.exists())
    compact = build(args.root, args.operator_archive)
    with args.output.open("x", encoding="utf-8") as stream:
        json.dump(compact, stream, ensure_ascii=True, indent=2, sort_keys=True)
        stream.write("\n")
    print(json.dumps(receipt(args.output)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
