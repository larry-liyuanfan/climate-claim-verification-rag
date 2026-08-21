from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

from climate_rag.artifacts import write_run_artifacts
from climate_rag.metrics import paired_bootstrap
from climate_rag.routing import (
    ROUTER_FEATURE_NAMES,
    agreement_features,
    cross_fit_route,
    group_prediction_rows,
    hashed_text_features,
)


METRICS = ("recall@5", "mrr@10", "ndcg@10", "evidence_f1")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Cross-fit a cost-aware weak/strong rerank router")
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--source-metrics", type=Path, required=True)
    parser.add_argument("--claims", type=Path, help="optional claim text for inference-safe difficulty features")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--weak-system", default="rrf")
    parser.add_argument("--strong-system", default="rrf_reranker_fusion_balanced")
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--regularization", type=float, default=1.0)
    parser.add_argument("--gain-preservation", type=float, default=0.8)
    parser.add_argument("--weak-seconds", type=float, default=0.01288)
    parser.add_argument("--bootstrap-samples", type=int, default=5000)
    return parser.parse_args()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def main() -> None:
    args = parse_args()
    started = datetime.now(timezone.utc).isoformat()
    grouped = group_prediction_rows(read_jsonl(args.predictions))
    required = {"bm25", "dense", args.weak_system, args.strong_system}
    claim_ids = sorted(grouped)
    missing = {claim_id: sorted(required - set(grouped[claim_id])) for claim_id in claim_ids}
    missing = {key: value for key, value in missing.items() if value}
    if missing:
        raise ValueError(f"missing required system rows: {missing}")

    agreement_matrix = np.vstack(
        [
            agreement_features(
                grouped[claim_id]["bm25"]["predicted_evidence_ids"],
                grouped[claim_id]["dense"]["predicted_evidence_ids"],
                grouped[claim_id][args.weak_system]["predicted_evidence_ids"],
            )
            for claim_id in claim_ids
        ]
    )
    feature_names = list(ROUTER_FEATURE_NAMES)
    if args.claims:
        claims = json.loads(args.claims.read_text(encoding="utf-8"))
        missing_claims = sorted(set(claim_ids) - set(claims))
        if missing_claims:
            raise ValueError(f"claim text is missing for: {missing_claims}")
        text_matrix = np.vstack(
            [hashed_text_features(str(claims[claim_id]["claim_text"])) for claim_id in claim_ids]
        )
        features = np.column_stack((agreement_matrix, text_matrix))
        feature_names.extend(
            [
                "text_log_token_count",
                "text_digit_token_fraction",
                "text_negation_fraction",
                "text_bracket_count",
                "text_question_mark_count",
                *(f"text_hash_{index}" for index in range(text_matrix.shape[1] - 5)),
            ]
        )
    else:
        features = agreement_matrix
    weak_f1 = np.asarray(
        [grouped[claim_id][args.weak_system]["evidence_f1"] for claim_id in claim_ids],
        dtype=np.float64,
    )
    strong_f1 = np.asarray(
        [grouped[claim_id][args.strong_system]["evidence_f1"] for claim_id in claim_ids],
        dtype=np.float64,
    )
    selected, predicted_gains, fold_reports = cross_fit_route(
        claim_ids,
        features,
        strong_f1 - weak_f1,
        fold_count=args.folds,
        regularization=args.regularization,
        gain_preservation=args.gain_preservation,
    )

    source_metrics = json.loads(args.source_metrics.read_text(encoding="utf-8"))
    reranker_timing = source_metrics["reranker_timing"]
    strong_rate = float(selected.mean())
    metrics: dict[str, Any] = {
        "claim_count": len(claim_ids),
        "evaluation_protocol": "deterministic hash-based cross-fitting on the fixed dev split",
        "weak_system": args.weak_system,
        "strong_system": args.strong_system,
        "fold_count": args.folds,
        "gain_preservation_target": args.gain_preservation,
        "router_strong_call_rate": strong_rate,
        "router_calls_avoided_rate": 1.0 - strong_rate,
        "estimated_mean_seconds_per_query": args.weak_seconds
        + strong_rate * float(reranker_timing["mean_seconds_per_query"]),
        "weak_path_seconds_per_query": args.weak_seconds,
        "strong_increment_mean_seconds_per_query": float(reranker_timing["mean_seconds_per_query"]),
        "strong_increment_p95_seconds_per_query": float(reranker_timing["p95_seconds_per_query"]),
        "feature_names": feature_names,
        "folds": fold_reports,
    }
    output_rows: list[dict[str, Any]] = []
    for index, claim_id in enumerate(claim_ids):
        output_rows.append(
            {
                "claim_id": claim_id,
                "selected_system": args.strong_system if selected[index] else args.weak_system,
                "predicted_strong_gain": float(predicted_gains[index]),
            }
        )
    for metric in METRICS:
        weak = np.asarray(
            [grouped[claim_id][args.weak_system][metric] for claim_id in claim_ids], dtype=np.float64
        )
        strong = np.asarray(
            [grouped[claim_id][args.strong_system][metric] for claim_id in claim_ids], dtype=np.float64
        )
        routed = np.where(selected, strong, weak)
        full_gain = float(strong.mean() - weak.mean())
        routed_gain = float(routed.mean() - weak.mean())
        metrics[metric] = {
            "always_weak": float(weak.mean()),
            "always_strong": float(strong.mean()),
            "cross_fit_router": float(routed.mean()),
            "router_gain_vs_weak": routed_gain,
            "router_gain_preserved_fraction": routed_gain / full_gain if full_gain else None,
            "router_vs_weak_paired_bootstrap": paired_bootstrap(
                weak, routed, samples=args.bootstrap_samples
            ),
            "router_vs_strong_paired_bootstrap": paired_bootstrap(
                strong, routed, samples=args.bootstrap_samples
            ),
        }

    write_run_artifacts(
        args.output_dir,
        command="evaluate-rerank-router",
        arguments={
            key: str(value) if isinstance(value, Path) else value
            for key, value in vars(args).items()
        },
        metrics=metrics,
        started_at=started,
        inputs=tuple(
            path
            for path in (args.predictions, args.source_metrics, args.claims)
            if path is not None
        ),
        predictions=output_rows,
        notes=(
            "This is a dev-selection experiment, not an independent held-out generalisation claim.",
            "Router features use only pre-reranker candidate-list agreement; gold labels are used only inside training folds and evaluation.",
            "Latency is an analytical estimate from measured stage timings, not a new end-to-end load test.",
        ),
        repository=Path(__file__).resolve().parents[1],
    )


if __name__ == "__main__":
    main()
