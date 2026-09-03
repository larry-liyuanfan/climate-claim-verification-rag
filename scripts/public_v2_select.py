from __future__ import annotations

import argparse
import json
from pathlib import Path

from climate_rag.io import write_json
from climate_rag.public_v2 import (
    file_sha256,
    load_metrics_files,
    load_public_v2_protocol,
    select_full_candidate,
    select_pilot_candidates,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Apply frozen public-v2 selection rules."
    )
    parser.add_argument("--protocol", required=True)
    parser.add_argument("--phase", choices=("pilot", "full"), required=True)
    parser.add_argument("--metrics", action="append", required=True)
    parser.add_argument("--output-dir", required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    output = Path(args.output_dir)
    if output.exists():
        raise FileExistsError(f"output must be unique: {output}")
    output.mkdir(parents=True)
    protocol = load_public_v2_protocol(args.protocol)
    metrics = load_metrics_files(args.metrics)
    if args.phase == "pilot":
        result = select_pilot_candidates(
            protocol["adapter_matrix"],
            metrics,
            maximum=int(protocol["evaluation"]["max_full_candidates"]),
        )
        write_json(output / "pilot_selection.json", result)
    else:
        result = select_full_candidate(metrics)
        selected_id = result["selected_candidate_id"]
        if selected_id is not None:
            selected_metrics = metrics[str(selected_id)]
            frozen = {
                "schema_version": 1,
                "protocol_id": protocol["protocol_id"],
                "protocol_sha256": file_sha256(args.protocol),
                "selected_candidate_id": selected_id,
                "selected_candidate_promoted": result["selected_candidate_promoted"],
                "adapter_path": selected_metrics["adapter_path"],
                "adapter_sha256": selected_metrics["adapter_sha256"],
                "candidate_embeddings_path": selected_metrics[
                    "candidate_embeddings_path"
                ],
                "candidate_embeddings_sha256": file_sha256(
                    selected_metrics["candidate_embeddings_path"]
                ),
                "climate_validation_metrics_sha256": selected_metrics["metrics_sha256"],
                "dense_model": protocol["models"]["dense"],
                "downstream_ranking": protocol["downstream_ranking"],
                "external_transfer": protocol["external_transfer"],
                "test_access": "forbidden",
                "tuning_after_freeze": "forbidden",
            }
            write_json(output / "frozen_config.json", frozen)
            result["frozen_config_sha256"] = file_sha256(output / "frozen_config.json")
        write_json(output / "full_selection.json", result)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
