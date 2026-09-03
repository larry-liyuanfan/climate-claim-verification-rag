from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any

from jsonschema import Draft202012Validator

from climate_rag.io import read_json, write_json
from climate_rag.public_v2 import file_sha256

ABSOLUTE_PATH = re.compile(r"^(?:[A-Za-z]:[\\/]|/data/gpfs/)")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Publish compact redacted public-v2 evidence."
    )
    parser.add_argument("--protocol", required=True)
    parser.add_argument("--prepare-metrics", required=True)
    parser.add_argument("--base-metrics", required=True)
    parser.add_argument("--pilot-selection", required=True)
    parser.add_argument("--full-selection", required=True)
    parser.add_argument("--full-metrics", action="append", required=True)
    parser.add_argument("--downstream-metrics", required=True)
    parser.add_argument("--external-metrics", required=True)
    parser.add_argument("--external-ledger", required=True)
    parser.add_argument("--sacct-json", required=True)
    parser.add_argument("--schema", required=True)
    parser.add_argument("--output-dir", required=True)
    return parser.parse_args()


def _object(path: str | Path) -> dict[str, Any]:
    value = read_json(path)
    if not isinstance(value, dict):
        raise TypeError(f"expected JSON object: {path}")
    return value


def _metric_summary(metrics: dict[str, Any]) -> dict[str, Any]:
    return {
        name: metrics[name] for name in ("recall@5", "mrr@10", "ndcg@10", "evidence_f1")
    }


def _assert_redacted(value: Any, location: str = "root") -> None:
    if isinstance(value, dict):
        for key, child in value.items():
            if any(token in key.casefold() for token in ("path", "dir", "root")):
                raise ValueError(
                    f"path-like key is forbidden in compact artifact: {location}.{key}"
                )
            _assert_redacted(child, f"{location}.{key}")
    elif isinstance(value, list):
        for index, child in enumerate(value):
            _assert_redacted(child, f"{location}[{index}]")
    elif isinstance(value, str) and ABSOLUTE_PATH.search(value):
        raise ValueError(f"absolute path leaked into compact artifact at {location}")


def main() -> int:
    args = parse_args()
    protocol = _object(args.protocol)
    prepared = _object(args.prepare_metrics)
    base = _object(args.base_metrics)
    pilot = _object(args.pilot_selection)
    full = _object(args.full_selection)
    full_metrics = [_object(path) for path in args.full_metrics]
    downstream = _object(args.downstream_metrics)
    external = _object(args.external_metrics)
    external_ledger = _object(args.external_ledger)
    sacct = read_json(args.sacct_json)
    if not isinstance(sacct, list):
        raise TypeError("sacct JSON must be a list")
    pilot_by_id = {str(row["id"]): row for row in pilot["all_pilot_results"]}
    full_by_id = {str(row["adapter_config_id"]): row for row in full_metrics}
    matrix: list[dict[str, Any]] = []
    for config in protocol["adapter_matrix"]:
        identifier = str(config["id"])
        row: dict[str, Any] = {
            "id": identifier,
            "config": config,
            "pilot_decision": pilot_by_id[identifier]["pilot"],
            "pilot_metrics_sha256": pilot_by_id[identifier]["metrics_sha256"],
            "advanced_to_full": identifier in pilot["selected_for_full"],
        }
        if identifier in full_by_id:
            metrics = full_by_id[identifier]
            row["full"] = {
                "base": _metric_summary(metrics["base"]),
                "candidate": _metric_summary(metrics["candidate"]),
                "paired_bootstrap": metrics["paired_bootstrap"],
                "decision": metrics["decision"],
                "diagnostic_slices": metrics["diagnostic_slices"],
                "adapter_sha256": metrics["adapter_sha256"],
                "metrics_sha256": file_sha256(
                    next(
                        path
                        for path in args.full_metrics
                        if _object(path)["adapter_config_id"] == identifier
                    )
                ),
            }
        matrix.append(row)
    compact = {
        "schema_version": 1,
        "evidence_status": "verified-public-validation-and-one-shot-external-transfer",
        "protocol": {
            "id": protocol["protocol_id"],
            "git_commit": downstream["resource"]["git_commit"],
            "protocol_sha256": file_sha256(args.protocol),
            "seed": protocol["seed"],
            "bootstrap_samples": protocol["evaluation"]["bootstrap_samples"],
            "candidate_width": protocol["downstream_ranking"]["candidate_width"],
            "feature_order": protocol["downstream_ranking"]["feature_order"],
        },
        "data": {
            "climate_fever": prepared["climate_fever"],
            "scifact": external["dataset_manifest"],
        },
        "baselines": {
            "selection_boundary": base["selection_boundary"],
            "bm25": base["bm25"],
            "qwen3_embedding_0_6b_base": base["qwen3_embedding_0_6b_base"],
            "dense_vs_bm25": base["dense_vs_bm25"],
            "training_data": base["training"],
        },
        "adapter_matrix": matrix,
        "full_selection": full,
        "downstream": {
            "systems": downstream["systems"],
            "paired_bootstrap_vs_bm25": downstream["paired_bootstrap_vs_bm25"],
            "diagnostic_slices_vs_bm25": downstream["diagnostic_slices_vs_bm25"],
            "index": downstream["index"],
            "timing": downstream["timing"],
            "resource": downstream["resource"],
            "lambdamart": downstream["lambdamart"],
            "reranker": downstream["reranker"],
            "selected_adapter": downstream["selected_adapter"],
        },
        "external_transfer": {
            "dataset": "SciFact",
            "base": _metric_summary(external["base"]),
            "candidate": _metric_summary(external["candidate"]),
            "paired_bootstrap": external["paired_bootstrap"],
            "diagnostic_slices": external["diagnostic_slices"],
            "completed_evaluations": external_ledger["completed_evaluations"],
            "attempt_count": len(external_ledger["attempts"]),
            "metrics_sha256": file_sha256(args.external_metrics),
            "truth_boundary": external["truth_boundary"],
        },
        "slurm_jobs": sacct,
        "artifact_hashes": {
            "prepare_metrics_sha256": file_sha256(args.prepare_metrics),
            "base_metrics_sha256": file_sha256(args.base_metrics),
            "pilot_selection_sha256": file_sha256(args.pilot_selection),
            "full_selection_sha256": file_sha256(args.full_selection),
            "downstream_metrics_sha256": file_sha256(args.downstream_metrics),
            "external_metrics_sha256": file_sha256(args.external_metrics),
            "external_ledger_sha256": file_sha256(args.external_ledger),
            "sacct_sha256": file_sha256(args.sacct_json),
        },
        "truth_boundaries": [
            "All CLIMATE-FEVER candidate selection uses the 1,075-train/230-validation v2 partitions; the historical consumed test is permanently sealed.",
            "Only decisive-evidence validation claims enter retrieval metrics; the full 230-claim partition count remains reported.",
            "Pilot advancement is not promotion. Promotion requires Recall@5 paired CI lower bound above zero and no secondary mean regression.",
            "The SciFact result is one post-freeze external transfer event and cannot trigger tuning or a repeated quality run.",
            "ANN, LambdaMART and reranker timings are component measurements on allocated Spartan hardware, not an online SLA.",
            "Predictions, checkpoints, model caches and indexes remain on Spartan; Git contains only this compact redacted record.",
        ],
    }
    _assert_redacted(compact)
    schema = _object(args.schema)
    Draft202012Validator(schema).validate(compact)
    output = Path(args.output_dir)
    if output.exists():
        raise FileExistsError(f"output must be unique: {output}")
    output.mkdir(parents=True)
    write_json(output / "public-v2-compact.json", compact)
    print(json.dumps(compact, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
