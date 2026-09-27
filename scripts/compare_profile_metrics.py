"""Compare two saved profile metric vectors; no model, corpus or test access."""
from __future__ import annotations

import argparse
import math
from pathlib import Path
from typing import Any

from climate_rag.io import read_json, read_jsonl, write_json
from climate_rag.metrics import paired_bootstrap
from public_v2_stage_manifest import verify_manifest

METRICS = ("recall@5", "recall@10", "recall@50", "mrr@10", "ndcg@10", "evidence_f1@5")


def compare_saved_profiles(root: Path, baseline: str, candidate: str) -> dict[str, Any]:
    manifest = verify_manifest(root)
    summaries = {}
    vectors = {}
    for name in (baseline, candidate):
        if name not in {"ltr", "rerank20", "rerank100"}:
            raise ValueError("unknown frozen profile")
        summary = read_json(root / name / "summary.json")
        rows = list(read_jsonl(root / name / "query-metrics.jsonl"))
        keyed = {str(row["claim_id"]): row for row in rows}
        if len(rows) != len(keyed) or len(rows) != summary["query_count"]:
            raise ValueError("duplicate or missing metric query")
        for metric in METRICS:
            values = [float(row[metric]) for row in rows]
            if not all(math.isfinite(value) and 0 <= value <= 1 for value in values):
                raise ValueError("metric values must be finite and in [0, 1]")
            if not math.isclose(sum(values) / len(values), summary["metrics"][metric], abs_tol=1e-12):
                raise ValueError("per-query mean disagrees with profile summary")
        summaries[name], vectors[name] = summary, keyed
    left, right = summaries[baseline], summaries[candidate]
    for key in ("source_hashes", "query_ids_sha256", "git_commit", "model_revisions", "device"):
        if left[key] != right[key]:
            raise ValueError(f"profile comparison mismatch: {key}")
    if set(vectors[baseline]) != set(vectors[candidate]):
        raise ValueError("profile query sets differ")
    ids = sorted(vectors[baseline])
    return {
        "schema_version": 1, "baseline": baseline, "candidate": candidate,
        "query_count": len(ids), "bootstrap_samples": 5000, "seed": 20260927,
        "source_payload_tree_sha256": manifest["payload_tree_sha256"],
        "paired_bootstrap": {
            metric: paired_bootstrap(
                [vectors[baseline][qid][metric] for qid in ids],
                [vectors[candidate][qid][metric] for qid in ids],
                samples=5000, seed=20260927,
            ) for metric in METRICS
        },
        "boundary": "validation-only paired query bootstrap; crossing zero is not equivalence; no new inference",
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--baseline", default="rerank20")
    parser.add_argument("--candidate", default="rerank100")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    root, output = Path(args.run_dir).resolve(), Path(args.output).resolve()
    if output.is_relative_to(root) or output.exists():
        raise ValueError("write to a new path outside the immutable source bundle")
    write_json(output, compare_saved_profiles(root, args.baseline, args.candidate))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
