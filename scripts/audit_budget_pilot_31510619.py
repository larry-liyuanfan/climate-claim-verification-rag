"""Reproduce compact evidence from the immutable pilot; never run a model or read labels."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import tarfile
from collections import Counter
from pathlib import Path

ARCHIVE_SHA = "808c7ae89af7c135b601295da87a2acb7727345c6429ccd4e220b658abd89a48"
RUN_SHA = "1b147faa5e84440f2fadc4d4f50a946575460254f54197c3b50f9e97f164c6d8"
SOURCE_SHA = "ca9fa53f2d4b70091fd08a8a1512e263f329fc61"
TASKS = ("pilot-ice", "pilot-multi", "pilot-empty")
STRATEGIES = ("fixed_retrieval", "fixed_rerank", "adaptive")


def digest(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifacts", type=Path, required=True)
    parser.add_argument("--models", type=Path, required=True)
    parser.add_argument("--evidence", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    archive = args.artifacts / "budget-agent-pilot-ca9fa53-release-20260929.tar.gz"
    assert digest(archive.read_bytes()) == ARCHIVE_SHA
    with tarfile.open(archive) as bundle:
        assert set(bundle.getnames()) == {"result", "result/run.json", "result/run.consumed.json"}
        stream = bundle.extractfile("result/run.json")
        assert stream is not None
        payload = stream.read()
    assert digest(payload) == RUN_SHA
    report = json.loads(payload)
    diagnostic_dir = args.artifacts / "budget-agent-pilot-ca9fa53-release-20260929-diagnostics"
    assert (diagnostic_dir / "final-run.json").read_bytes() == payload
    operator_path = diagnostic_dir / "operator-status.json"
    operator = json.loads(operator_path.read_text())
    assert operator["job_id"] == "31510619" and operator["exit_status"] == 0
    assert report["code_sha"] == SOURCE_SHA and not report["working_tree_dirty"]
    assert not report["dense_enabled"] and report["phase"] == "pilot"
    assert report["official_evidence_metrics"] is None
    assert report["human_semantic_evaluation"] is None
    rows = report["runs"]
    assert len(rows) == 9
    assert {(x["task_id"], x["strategy"]) for x in rows} == {
        (task, strategy) for task in TASKS for strategy in STRATEGIES}
    for kind, key in (("generator", "model"), ("reranker", "reranker")):
        path = args.models / kind / "model_manifest.json"
        raw = path.read_bytes()
        assert digest(raw) == report["execution_manifest"][f"{key}_manifest_sha256"]
        assert digest(json.dumps(json.loads(raw), sort_keys=True).encode()) == report[f"{key}_sha256"]
    corpus_raw = args.evidence.read_bytes()
    assert digest(corpus_raw) == report["corpus_sha256"]
    corpus = {x["evidence_id"] for line in corpus_raw.splitlines() if line
              for x in [json.loads(line)]}
    assert len(corpus) == report["document_count"] == 5240
    for row in rows:
        assert set(row["candidate_evidence_ids"]) <= corpus
        assert row["usage_known"] and row["provider_kind"] == "local_model"
    sample_path = diagnostic_dir / "gpu-samples.csv"
    sample_lines = sample_path.read_text().splitlines()
    memory = []
    for line in csv.reader(sample_lines):
        try:
            if len(line) == 6:
                memory.append(float(line[3]))
        except ValueError:
            continue
    log_path = args.artifacts / "slurm-agent-pilot-31510619.log"
    log = log_path.read_text()
    assert "3/3" in log and "2/2" in log and "Completed 9 bounded local-qwen runs" in log
    cases = []
    keys = ("strategy", "status", "reason", "generation_calls", "usage", "usage_known",
            "retrieval_calls", "rerank_calls", "rerank_candidate_pairs", "elapsed_ms", "events")
    for task in TASKS:
        group = [row for row in rows if row["task_id"] == task]
        cases.append({"task_id": task, "authored_question": group[0]["claim_text"],
                      "source": "result/run.json runs filtered by task_id; no lost text reconstructed",
                      "runs": [{**{key: row[key] for key in keys},
                                "candidate_count": len(row["candidate_evidence_ids"]),
                                "context_count": len(row["context_evidence_ids"]),
                                "delivered_count": len(row["delivered_evidence_ids"]),
                                "answer": row["answer"]} for row in group]})
    compact = {
        "job_id": "31510619", "kind": "authored real-model integration pilot; failed structured decisions",
        "source_git_sha": SOURCE_SHA,
        "operator_git_sha": "ae5166b872c9e0f71be09df1dd557c1804c01bf7",
        "operator_wrapper_sha256": "569d824e10645c0fa0ce4604e33810a95085969bdf730adc182e5483227780a8",
        "archive_sha256": ARCHIVE_SHA, "archive_bytes": archive.stat().st_size,
        "run_sha256": RUN_SHA, "run_bytes": len(payload),
        "log_sha256": digest(log_path.read_bytes()),
        "operator_status_sha256": digest(operator_path.read_bytes()),
        "execution_manifest": report["execution_manifest"],
        "corpus_sha256": report["corpus_sha256"], "documents": 5240,
        "model_canonical_manifest_sha256": report["model_sha256"],
        "reranker_canonical_manifest_sha256": report["reranker_sha256"],
        "rows": 9, "unique_task_route_pairs": 9,
        "real_weights_loaded_basis": "Both checkpoint loads completed; GPU provider generated tokens and nonempty reranker stages completed.",
        "generation_attempts": sum(x["generation_calls"] for x in rows),
        "usage": {key: sum(x["usage"][key] for x in rows) for key in ("input_tokens", "output_tokens")},
        "all_usage_known": all(x["usage_known"] for x in rows),
        "failure_types": dict(Counter(e["error_type"] for x in rows for e in x["events"] if e["stage"] == "failure")),
        "validated_decisions": sum(e["stage"] == "decision" for x in rows for e in x["events"]),
        "answers_published": sum(x["answer"] is not None for x in rows),
        "model_chosen_actions": [],
        "retrieval_calls": sum(x["retrieval_calls"] for x in rows),
        "rerank_calls": sum(x["rerank_calls"] for x in rows),
        "reranker_pair_attempts": sum(x["rerank_candidate_pairs"] for x in rows),
        "completed_nonempty_rerank_stages": sum(e["stage"] == "rerank" and e["candidate_count"] > 0 for x in rows for e in x["events"]),
        "successful_forwards_direct_counter": None,
        "forward_inference": "Two completed 20-pair stages imply 40 successful batch-size-1 forwards by frozen code; this is not a separately instrumented counter.",
        "route_elapsed_seconds_sum": sum(x["elapsed_ms"] for x in rows) / 1000,
        "operator": operator,
        "slurm": {"state": "COMPLETED", "exit_code": "0:0", "elapsed_seconds": 153,
                  "total_cpu_seconds": 124.678, "batch_maxrss_kib": 18192068,
                  "batch_maxrss_gib": 18192068 / 1048576,
                  "cpus": 8, "memory_gib": 32, "gpu": "1 full A100",
                  "start_scheduler_time": "2026-09-29T15:39:16", "end_scheduler_time": "2026-09-29T15:41:49"},
        "gpu_sampling": {"sha256": digest(sample_path.read_bytes()), "lines": len(sample_lines),
                         "valid_samples": len(memory), "sampled_max_memory_mib": max(memory) if memory else None,
                         "no_device_lines": sample_lines.count("No devices were found"),
                         "stderr_bytes": (diagnostic_dir / "gpu-sampler-errors.log").stat().st_size},
        "citation_precision": None, "citation_completeness": None, "semantic_supportability": None,
        "negative_conclusion": "All nine outputs failed JSON/schema processing before any accepted action; controller abstention is not model-chosen abstention. Lost raw text prevents subtype/root-cause diagnosis.",
        "full_evaluation_ready": False, "full_evaluation_released": False,
        "cases": cases,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(compact, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({key: compact[key] for key in ("rows", "generation_attempts", "usage", "failure_types", "gpu_sampling")}, indent=2))


if __name__ == "__main__":
    main()
