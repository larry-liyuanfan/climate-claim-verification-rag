from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

from evaluate_dense_model_gate import _evaluate_model

from climate_rag.artifacts import write_run_artifacts
from climate_rag.embedding_adapter_gate import (
    heldout_query_texts,
    select_heldout_claims,
)
from climate_rag.encoder_gate import (
    compare_metric_rows,
    evidence_preserving_reservoir_sample,
    required_evidence_ids,
    screening_decision,
)
from climate_rag.io import iter_evidence, load_claims, read_jsonl


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare a Qwen3 embedding LoRA adapter with its exact base model."
    )
    parser.add_argument("--claims", required=True)
    parser.add_argument("--evidence", required=True)
    parser.add_argument("--eval-dataset", required=True)
    parser.add_argument("--adapter", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--model", default="Qwen/Qwen3-Embedding-0.6B")
    parser.add_argument("--truncate-dim", type=int, default=1024)
    parser.add_argument("--sample-size", type=int, default=5_000)
    parser.add_argument("--top-k", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--bootstrap-samples", type=int, default=2_000)
    parser.add_argument("--seed", type=int, default=17)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    started_at = datetime.now(timezone.utc).isoformat()
    all_claims = load_claims(args.claims)
    claims = select_heldout_claims(
        all_claims, heldout_query_texts(read_jsonl(args.eval_dataset))
    )
    sample = evidence_preserving_reservoir_sample(
        iter_evidence(args.evidence),
        required_evidence_ids(claims),
        sample_size=args.sample_size,
        seed=args.seed,
    )
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    results: dict[str, object] = {}
    rows: dict[str, list[dict[str, object]]] = {}
    predictions: list[dict[str, object]] = []
    for label, adapter in (("base", None), ("adapted", args.adapter)):
        metrics, metric_rows, output_rows = _evaluate_model(
            args.model,
            claims=claims,
            documents=sample.documents,
            device=args.device,
            truncate_dim=args.truncate_dim,
            batch_size=args.batch_size,
            top_k=args.top_k,
            adapter_path=adapter,
            run_label=label,
        )
        results[label] = metrics
        rows[label] = metric_rows
        predictions.extend(output_rows)
        print(
            json.dumps(
                {
                    "event": "encoder_completed",
                    "encoder": label,
                    "recall@5": metrics["recall@5"],
                    "mrr@10": metrics["mrr@10"],
                },
                sort_keys=True,
            ),
            flush=True,
        )
    comparisons = compare_metric_rows(
        rows["base"],
        rows["adapted"],
        samples=args.bootstrap_samples,
        seed=args.seed,
    )
    metrics = {
        "scope": "claim-grouped held-out, evidence-preserving sampled-corpus adapter gate",
        "heldout_claim_count": len(claims),
        "source_document_count": sample.source_document_count,
        "sample_document_count": len(sample.documents),
        "required_evidence_count": len(sample.required_ids),
        "encoders": results,
        "adapted_vs_base": comparisons,
        "screening_decision": screening_decision(comparisons),
    }
    write_run_artifacts(
        output_dir,
        command="embedding-adapter-gate",
        arguments=vars(args),
        metrics=metrics,
        started_at=started_at,
        inputs=[args.claims, args.evidence, args.eval_dataset, args.adapter],
        predictions=predictions,
        notes=[
            "Only claim-grouped held-out query texts from the training eval JSONL are evaluated.",
            "Every labelled positive is retained; remaining evidence rows are a seeded reservoir sample.",
            "A sampled pass only authorises a full-corpus fixed-dev gate; it is not production evidence.",
        ],
        repository=Path(__file__).resolve().parents[1],
    )
    print(json.dumps(metrics, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
