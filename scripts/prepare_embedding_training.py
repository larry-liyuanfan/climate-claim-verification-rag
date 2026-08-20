from __future__ import annotations

import argparse
from datetime import datetime, timezone
from pathlib import Path

from climate_rag.artifacts import write_run_artifacts
from climate_rag.embedding_training import build_swift_infonce_dataset
from climate_rag.io import iter_evidence, load_claims, read_jsonl, write_jsonl


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Prepare claim-grouped Qwen3-Embedding InfoNCE train/eval JSONL."
    )
    parser.add_argument("--claims", required=True)
    parser.add_argument("--evidence", required=True)
    parser.add_argument("--hard-negatives", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--negatives-per-positive", type=int, default=4)
    parser.add_argument("--eval-ratio", type=float, default=0.1)
    parser.add_argument("--split-seed", type=int, default=45)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    started_at = datetime.now(timezone.utc).isoformat()
    dataset = build_swift_infonce_dataset(
        load_claims(args.claims),
        iter_evidence(args.evidence),
        read_jsonl(args.hard_negatives),
        negatives_per_positive=args.negatives_per_positive,
        eval_ratio=args.eval_ratio,
        split_seed=args.split_seed,
    )
    if not dataset.train_rows or not dataset.eval_rows:
        raise ValueError("claim-grouped split must produce non-empty train and eval rows")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    write_jsonl(output_dir / "train.jsonl", dataset.train_rows)
    write_jsonl(output_dir / "eval.jsonl", dataset.eval_rows)
    write_run_artifacts(
        output_dir,
        command="prepare-embedding-training",
        arguments=vars(args),
        metrics=dataset.metrics,
        started_at=started_at,
        inputs=[args.claims, args.evidence, args.hard_negatives],
        notes=[
            "Rows use the current ms-swift messages/positive_messages/negative_messages format.",
            "Splitting is claim-grouped; all gold evidence IDs are excluded from each claim's negatives.",
            "This artifact prepares a training gate and does not establish a retrieval improvement.",
        ],
        repository=Path(__file__).resolve().parents[1],
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
