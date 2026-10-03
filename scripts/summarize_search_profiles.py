from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

from climate_rag.io import iter_evidence, load_claims, read_json, read_jsonl, write_json
from climate_rag.public_v2 import file_sha256
from climate_rag.public_v2_runtime import decisive_claims, predictions_from_rows
from climate_rag.representation_eval import build_pareto_report, evaluate_representation_pair


def validate_profile_artifacts(root: Path, summary: dict[str, Any], claim_ids: set[str]) -> None:
    """Reject stale/tampered rankings or traces before combining their results."""
    for filename, key in (("predictions.jsonl", "prediction_sha256"), ("traces.jsonl", "trace_sha256")):
        path = root / filename
        if file_sha256(path) != summary[key]:
            raise ValueError(f"profile artifact hash mismatch: {filename}")
        ids = [row["claim_id"] for row in read_jsonl(path)]
        if len(ids) != len(claim_ids) or set(ids) != claim_ids:
            raise ValueError(f"profile query IDs mismatch: {filename}")
    expected = hashlib.sha256(json.dumps(sorted(claim_ids)).encode()).hexdigest()
    if summary["query_ids_sha256"] != expected or summary["query_count"] != len(claim_ids):
        raise ValueError("profile query identity mismatch")


def main() -> int:
    parser = argparse.ArgumentParser(description="Summarize exactly three frozen validation profiles")
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--prepared-dir", required=True)
    parser.add_argument("--selection-dir", required=True)
    args = parser.parse_args()
    root = Path(args.run_dir)
    names = ("ltr", "rerank20", "rerank100")
    summaries = {name: read_json(root / name / "summary.json") for name in names}
    traces = {name: list(read_jsonl(root / name / "traces.jsonl")) for name in names}
    anchor = summaries["ltr"]
    expected_candidates = {row["claim_id"]: row["candidate_ids_sha256"] for row in traces["ltr"]}
    for name in names:
        current = summaries[name]
        if current["profile"] != name or len(traces[name]) != current["query_count"]:
            raise ValueError(f"profile identity or trace count mismatch: {name}")
        if len({row["claim_id"] for row in traces[name]}) != len(traces[name]):
            raise ValueError(f"duplicate query traces: {name}")
        for key in ("source_hashes", "query_ids_sha256", "model_revisions", "git_commit", "device"):
            if current[key] != anchor[key]:
                raise ValueError(f"profile comparison mismatch: {name}: {key}")
        for key in ("torch", "cuda", "python_hash_seed", "gpu_name", "omp_threads", "encoder_dtype"):
            if current["runtime"][key] != anchor["runtime"][key]:
                raise ValueError(f"profile runtime mismatch: {name}: {key}")
        if {row["claim_id"]: row["candidate_ids_sha256"] for row in traces[name]} != expected_candidates:
            raise ValueError(f"profile candidate pools differ: {name}")
    claims_path = Path(args.selection_dir) / "validation-claims.json"
    evidence_path = Path(args.prepared_dir) / "evidence.jsonl"
    if file_sha256(claims_path) != anchor["source_hashes"]["validation"]:
        raise ValueError("validation claims changed")
    if file_sha256(evidence_path) != anchor["source_hashes"]["evidence"]:
        raise ValueError("evidence mapping changed")
    claims = decisive_claims(load_claims(claims_path))
    for name in names:
        validate_profile_artifacts(root / name, summaries[name], set(claims))
    documents = list(iter_evidence(evidence_path))
    predictions = {name: predictions_from_rows(list(read_jsonl(root / name / "predictions.jsonl")))
                   for name in names}
    paired = {}
    for name in names[1:]:
        comparison, _ = evaluate_representation_pair(
            claims, documents, predictions["ltr"], predictions[name], evidence_k=5,
            bootstrap_samples=5000, seed=20260927,
        )
        paired[name] = comparison
    profiles = []
    for name in names:
        row = summaries[name]
        total_seconds = row["timings"]["end_to_end_ms"]["sum"] / 1000
        profiles.append({
            "name": name, "evidence_f1": row["metrics"]["evidence_f1@5"],
            "p95_ms": row["timings"]["end_to_end_ms"]["p95"],
            "memory_bytes": row["peak_torch_allocated_bytes"],
            "comparability_group": "same-public-validation-same-candidates-single-process-" + anchor["device"],
            "serial_request_seconds_per_1000": total_seconds * 1000 / row["query_count"],
            "cost_boundary": "time proxy, excludes load/allocation idle; not billed cost or savings",
        })
    write_json(root / "comparison.json", {
        "schema_version": 1, "evidence_track": "public-validation-only",
        "query_count": len(claims), "test_claims_loaded": 0,
        "profiles": summaries, "paired_vs_ltr": paired,
        "pareto": build_pareto_report(profiles),
        "candidate_pools_identical": True, "adapter_promotion": "not evaluated",
        "boundary": "one frozen configuration comparison; not independent test or production SLA",
    })
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
