from __future__ import annotations

import argparse
import json
from pathlib import Path

from climate_rag.io import read_json, write_json
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
    parser.add_argument(
        "--phase", choices=("pilot", "full", "negative-closeout"), required=True
    )
    parser.add_argument("--metrics", action="append")
    parser.add_argument("--pilot-selection")
    parser.add_argument("--base-embeddings")
    parser.add_argument("--diagnostic-log")
    parser.add_argument("--diagnostic-job-id")
    parser.add_argument("--diagnostic-exit-code")
    parser.add_argument("--output-dir", required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    output = Path(args.output_dir)
    if output.exists():
        raise FileExistsError(f"output must be unique: {output}")
    output.mkdir(parents=True)
    protocol = load_public_v2_protocol(args.protocol)
    metrics = load_metrics_files(args.metrics or [])
    if args.phase == "pilot":
        if not args.metrics:
            raise ValueError("pilot selection requires metrics")
        result = select_pilot_candidates(
            protocol["adapter_matrix"],
            metrics,
            maximum=int(protocol["evaluation"]["max_full_candidates"]),
        )
        write_json(output / "pilot_selection.json", result)
    elif args.phase == "full":
        if not args.metrics:
            raise ValueError("full selection requires metrics")
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
                "adapter_artifact": "selected-pilot-checkpoint",
                "adapter_sha256": selected_metrics["adapter_sha256"],
                "candidate_embeddings_artifact": (
                    "selected-full-evaluation-candidate-embeddings"
                ),
                "candidate_embeddings_sha256": selected_metrics[
                    "candidate_embeddings_sha256"
                ],
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
    else:
        required = {
            "pilot_selection": args.pilot_selection,
            "base_embeddings": args.base_embeddings,
            "diagnostic_log": args.diagnostic_log,
            "diagnostic_job_id": args.diagnostic_job_id,
            "diagnostic_exit_code": args.diagnostic_exit_code,
        }
        missing = [name for name, value in required.items() if not value]
        if missing:
            raise ValueError(
                "negative closeout is missing required inputs: " + ", ".join(missing)
            )
        pilot = read_json(str(args.pilot_selection))
        if not isinstance(pilot, dict):
            raise TypeError("pilot selection must be an object")
        selected = pilot.get("selected_for_full")
        if not pilot.get("diagnostic_fallback") or not isinstance(selected, list):
            raise ValueError("negative closeout requires the pilot diagnostic fallback")
        if len(selected) != 1:
            raise ValueError("negative closeout requires exactly one diagnostic candidate")
        pilot_rows = pilot.get("all_pilot_results")
        if not isinstance(pilot_rows, list) or any(
            not isinstance(row, dict)
            or bool(row.get("pilot", {}).get("advance_eligible"))
            for row in pilot_rows
        ):
            raise ValueError("negative closeout requires zero pilot-eligible adapters")
        diagnostic_log = Path(str(args.diagnostic_log))
        log_text = diagnostic_log.read_text(encoding="utf-8", errors="replace")
        missing_keys = "Found missing adapter keys while loading the checkpoint" in log_text
        ci_exception = "KeyError: 'lower'" in log_text
        if not missing_keys or not ci_exception:
            raise ValueError("diagnostic log does not contain the expected failure evidence")
        diagnostic_id = str(selected[0])
        diagnostic = {
            "candidate_id": diagnostic_id,
            "slurm_job_id": str(args.diagnostic_job_id),
            "state": "FAILED",
            "exit_code": str(args.diagnostic_exit_code),
            "log_sha256": file_sha256(diagnostic_log),
            "missing_adapter_keys_detected": True,
            "promotion_statistics_exception": "KeyError:lower",
            "adapter_integrity": "failed",
            "promotion_gate_status": "not-evaluable",
            "quality_retry_authorized": False,
        }
        result = {
            "schema_version": 1,
            "full_result_count": 0,
            "promoted_candidate_ids": [],
            "selected_candidate_id": None,
            "selected_candidate_promoted": False,
            "all_full_results": [],
            "diagnostic_attempt": diagnostic,
            "downstream_mode": "base-only-negative-closeout",
            "external_transfer_status": "not-authorized",
            "boundary": (
                "All six fixed pilots tied the base and no adapter was eligible. The "
                "single pre-registered full diagnostic failed adapter-integrity checks; "
                "it is not a valid effectiveness result, no adapter is promoted, and no "
                "further adapter or external-transfer run is authorized."
            ),
        }
        frozen = {
            "schema_version": 1,
            "protocol_id": protocol["protocol_id"],
            "protocol_sha256": file_sha256(args.protocol),
            "selected_candidate_id": None,
            "selected_candidate_promoted": False,
            "diagnostic_candidate_id": diagnostic_id,
            "diagnostic_status": "invalid-adapter-integrity",
            "downstream_mode": "base-only-negative-closeout",
            "base_embeddings_artifact": "frozen-base-embeddings",
            "base_embeddings_sha256": file_sha256(str(args.base_embeddings)),
            "dense_model": protocol["models"]["dense"],
            "downstream_ranking": protocol["downstream_ranking"],
            "external_transfer": {
                "dataset": protocol["external_transfer"]["dataset"],
                "authorized": False,
                "status": "not-run",
                "reason": "no-adapter-passed-promotion-gate",
            },
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
