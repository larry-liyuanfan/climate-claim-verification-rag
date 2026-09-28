"""Post-hoc scoring only: gold never enters retrieval or model execution."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np

from climate_rag.io import write_json
from climate_rag.metrics import paired_bootstrap, per_claim_retrieval_metrics
from climate_rag.models import Claim, Prediction


def score(run: dict[str, Any], gold: dict[str, Any] | None = None) -> dict[str, Any]:
    rows: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in run["runs"]:
        metrics: dict[str, Any] = {}
        if gold is not None:
            if gold["protocol_sha256"] != run["protocol_sha256"]:
                raise ValueError("gold protocol mismatch")
            item = gold["claims"][row["task_id"]]
            if item["claim_sha256"] != hashlib.sha256(row["claim_text"].encode()).hexdigest():
                raise ValueError("gold claim mismatch")
            if item["evidence_ids"]:
                claim = Claim(row["task_id"], row["claim_text"], evidence_ids=tuple(item["evidence_ids"]))
                metrics = per_claim_retrieval_metrics(claim, Prediction(
                    row["task_id"], tuple(row["delivered_evidence_ids"])), ks=(5, 10), evidence_k=5)
            if row.get("provider_kind") == "local_model" and item.get("label") in {"SUPPORTS", "REFUTES"}:
                metrics["official_decisive_label_correct"] = int(
                    row["answer"] is not None and row["answer"]["label"] == item["label"])
        rows[row["strategy"]].append({"task_id": row["task_id"], **metrics,
            "tool_calls": row["tool_calls"], "model_calls": row["model_calls"],
            "elapsed_ms": row["elapsed_ms"], "answered": row["answer"] is not None,
            "usage_known": row["usage_known"], "usage": row["usage"],
            "reason": row["reason"]})
    aggregate: dict[str, Any] = {}
    for strategy, values in rows.items():
        aggregate[strategy] = {"count": len(values), "answered": sum(x["answered"] for x in values),
            "tool_calls_mean": float(np.mean([x["tool_calls"] for x in values])),
            "generation_calls_mean": float(np.mean([x["model_calls"] for x in values])),
            "latency_ms_p50": float(np.percentile([x["elapsed_ms"] for x in values], 50)),
            "latency_ms_p95": float(np.percentile([x["elapsed_ms"] for x in values], 95)),
            "unknown_token_accounting_runs": sum(not x["usage_known"] for x in values),
            "known_generation_input_tokens": sum(x["usage"]["input_tokens"] for x in values if x["usage_known"]),
            "known_generation_output_tokens": sum(x["usage"]["output_tokens"] for x in values if x["usage_known"]),
            "operational_failures": sum(x["reason"] in {"stage_failed", "deadline_exceeded"}
                                        for x in values)}
        if gold is not None:
            decisive = [x for x in values if "recall@5" in x]
            aggregate[strategy]["evidence_metrics"] = {
                key: float(np.mean([x[key] for x in decisive])) if decisive else None
                for key in ("recall@5", "mrr@10", "ndcg@10", "evidence_f1")}
            aggregate[strategy]["retrieval_denominator"] = len(decisive)
            labels = [x["official_decisive_label_correct"] for x in values
                      if "official_decisive_label_correct" in x]
            aggregate[strategy]["official_decisive_label_accuracy"] = float(np.mean(labels)) if labels else None
            aggregate[strategy]["label_denominator"] = len(labels)
    bootstrap = {}
    if gold is not None:
        baseline = sorted((x for x in rows["fixed_retrieval"] if "recall@5" in x), key=lambda x: x["task_id"])
        for strategy in ("fixed_rerank", "adaptive"):
            candidate = sorted((x for x in rows[strategy] if "recall@5" in x), key=lambda x: x["task_id"])
            if [x["task_id"] for x in baseline] != [x["task_id"] for x in candidate]:
                raise ValueError("unpaired task sets")
            bootstrap[strategy] = {key: paired_bootstrap(
                [x[key] for x in baseline], [x[key] for x in candidate],
                samples=5000, seed=20260929)
                for key in ("recall@5", "mrr@10", "ndcg@10", "evidence_f1")} if baseline else None
    return {"aggregates": aggregate, "paired_bootstrap": bootstrap or None,
            "gold_source": gold["provenance"] if gold else None,
            "semantic_supportability": None,
            "boundary": "No gold means null quality, not zero. Citation/number checks are not entailment."}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--gold", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    run = json.loads(args.run.read_text(encoding="utf-8"))
    gold = json.loads(args.gold.read_text(encoding="utf-8")) if args.gold else None
    write_json(args.output, score(run, gold))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
