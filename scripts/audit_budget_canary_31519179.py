"""Summarize public diagnostic receipts and redacted in-place inspection, never raw text."""
from __future__ import annotations

import argparse
import hashlib
import json
import tarfile
from collections import Counter
from pathlib import Path


def sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifacts", type=Path, required=True)
    parser.add_argument("--inspection", type=Path, required=True)
    parser.add_argument("--frozen-validation", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    archive = args.artifacts / "budget-agent-diagnostic-canary-6a764a1-20260929.tar.gz"
    archive_sha = sha(archive.read_bytes())
    assert archive_sha == "12ad6f9db73b52a272afe7f17df995aab6e8ab43558cffaaa9229a2ac635e095"
    with tarfile.open(archive) as bundle:
        assert set(bundle.getnames()) == {"result", "result/run.json", "result/run.consumed.json"}
        stream = bundle.extractfile("result/run.json")
        assert stream is not None
        raw = stream.read()
    report = json.loads(raw)
    assert report["code_sha"] == "6a764a1b4f9137cf8e8c29abfbc14572c3a1039d"
    assert not report["working_tree_dirty"] and not report["dense_enabled"]
    old = json.loads(Path("docs/verified-runs/budget-agent-gpu-pilot-31510619.json").read_text())
    assert report["execution_manifest"] == old["execution_manifest"]
    assert report["corpus_sha256"] == old["corpus_sha256"]
    rows = report["runs"]
    assert len(rows) == 9 and len({(x["task_id"], x["strategy"]) for x in rows}) == 9
    assert report["official_evidence_metrics"] is None and report["human_semantic_evaluation"] is None
    diagnostics = [e["generation_diagnostics"] for row in rows for e in row["events"]
                   if "generation_diagnostics" in e]
    inspection = json.loads(args.inspection.read_text())
    assert inspection["job_id"] == "31519179" and inspection["file_count"] == 9
    assert Counter(x["output_sha256"] for x in diagnostics) == Counter(x["output_sha256"] for x in inspection["records"])
    assert all(x["category"] == "json_valid" and x["object"] for x in inspection["records"])
    assert all(x["cross_field_conflicts_by_unchanged_rules"] == ["query_only_allowed_for_rewrite"] for x in inspection["records"])
    frozen_validation = json.loads(args.frozen_validation.read_text())
    assert frozen_validation["job_id"] == "31519179"
    assert frozen_validation["pydantic_version"] == "2.13.5"
    assert frozen_validation["class_source_sha256"] == "d482b922ada5f3d080143d8060df2bbd1a97afb226281120d99763f4cfd194c2"
    assert Counter(x["output_sha256"] for x in diagnostics) == Counter(
        x["output_sha256"] for x in frozen_validation["records"])
    assert all(not x["valid"] and x["query_only_rule_confirmed"] and x["error_count"] == 1
               for x in frozen_validation["records"])
    operator_path = args.artifacts / "operator-status.json"
    operator = json.loads(operator_path.read_text())
    assert operator["job_id"] == "31519179" and operator["exit_status"] == 0
    slim_rows = []
    for row in rows:
        error = [e for e in row["events"] if e["stage"] == "failure"]
        assert len(error) == 1 and error[0]["error_type"] == "GeneratedResponseError"
        slim_rows.append({key: row[key] for key in ("task_id", "strategy", "status", "reason", "usage", "usage_known", "generation_calls", "retrieval_calls", "rerank_calls", "rerank_candidate_pairs", "elapsed_ms")}
                         | {"diagnostics": error[0]["generation_diagnostics"]})
    result = {
        "job_id": "31519179", "release_id": "climate-response-diagnostic-6a764a1-20260929",
        "kind": "actual protocol diagnosis, not a successful Agent or quality evaluation",
        "source_git_sha": report["code_sha"],
        "operator_sha256": "c257d8b49a1e2c2112cb209fbca0e86e8ae6ca1989c7c18f4d2f860c15e30dd6",
        "archive_sha256": archive_sha, "archive_bytes": archive.stat().st_size,
        "run_sha256": sha(raw), "run_bytes": len(raw),
        "log_sha256": sha((args.artifacts / "slurm-agent-diagnostic-31519179.log").read_bytes()),
        "operator_status_sha256": sha(operator_path.read_bytes()),
        "execution_manifest": report["execution_manifest"], "corpus_sha256": report["corpus_sha256"],
        "rows": 9, "usage_known": all(x["usage_known"] for x in rows),
        "generation_attempts": sum(x["generation_calls"] for x in rows),
        "usage": {key: sum(x["usage"][key] for x in rows) for key in ("input_tokens", "output_tokens")},
        "error_categories": dict(Counter(x["category"] for x in diagnostics)),
        "eos_observed_count": sum(x["eos_observed"] is True for x in diagnostics),
        "token_cap_reached_count": sum(x["reached_max_new_tokens"] for x in diagnostics),
        "private_raw_saved_count": sum(x["private_attachment"] == "saved_owner_only" for x in diagnostics),
        "whole_response_cpu_json_parse_successes": 9,
        "exact_frozen_pydantic_query_rule_failures": len(frozen_validation["records"]),
        "frozen_validation_receipt_sha256": sha(args.frozen_validation.read_bytes()),
        "rejected_raw_action_candidates": dict(Counter(x["action"] for x in inspection["records"])),
        "accepted_decisions": 0, "published_answers": 0,
        "observed_conflict": "All9 non-rewrite actions contain a non-null, nonempty query; unchanged payload validator rejects them. Candidate actions are not accepted model/tool actions.",
        "citation_supportability": None,
        "gpu_memory_peak": None,
        "slurm": {"state": "COMPLETED", "exit_code": "0:0", "elapsed_seconds": 142,
                  "total_cpu_seconds": 125.008, "batch_maxrss_kib": 18102124,
                  "batch_maxrss_gib": 18102124 / 1048576,
                  "cpus": 8, "ram_gib": 32, "gpu": "1 full A100"},
        "operator": operator,
        "query_elapsed_seconds_sum": sum(x["elapsed_ms"] for x in rows) / 1000,
        "generation_call_elapsed_seconds_sum": sum(x["generation_elapsed_ms"] for x in diagnostics) / 1000,
        "rows_compact": slim_rows,
        "private_originals_exported": False,
        "private_directory_actual_mode": "2700 (inherited setgid); access bits0700; files0600",
        "old_pilot_root_cause_retroactively_assigned": False,
        "behavior_fix_present_in_this_canary": False,
        "scope": "Terminal diagnosis receipt, not the state of subsequent CPU-only repairs or releases.",
        "recommended_fix": "Explicitly describe conditional fields in the prompt while preserving strict validation; new CPU contract tests and a separately released small canary, never post-hoc query deletion or tolerant parsing.",
    }
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({key: result[key] for key in ("rows", "usage", "error_categories", "rejected_raw_action_candidates", "slurm")}, indent=2))


if __name__ == "__main__":
    main()
