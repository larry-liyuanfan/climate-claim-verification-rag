from __future__ import annotations

import argparse
import gc
import json
import os
from pathlib import Path

from climate_rag.dense import SentenceTransformerEncoder
from climate_rag.io import (
    iter_evidence,
    load_claims,
    read_json,
    write_json,
    write_jsonl,
)
from climate_rag.public_v2 import (
    complete_external_transfer,
    file_sha256,
    load_public_v2_protocol,
    prepare_scifact_transfer,
    reserve_external_transfer,
    tree_sha256,
)
from climate_rag.public_v2_runtime import (
    build_dense_vectors,
    build_faiss_index,
    predictions_from_rows,
    score_dense_index,
)
from climate_rag.representation_eval import evaluate_representation_pair


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the sole post-freeze SciFact representation transfer evaluation."
    )
    parser.add_argument("--protocol", required=True)
    parser.add_argument("--frozen-config", required=True)
    parser.add_argument("--adapter-dir", required=True)
    parser.add_argument("--archive", required=True)
    parser.add_argument("--ledger", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch-size", type=int, default=32)
    return parser.parse_args()


def _release_gpu() -> None:
    gc.collect()
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except ImportError:
        pass


def main() -> int:
    args = parse_args()
    protocol = load_public_v2_protocol(args.protocol)
    frozen = read_json(args.frozen_config)
    if not isinstance(frozen, dict):
        raise TypeError("frozen config must be an object")
    external = frozen.get("external_transfer")
    if (
        not frozen.get("selected_candidate_promoted")
        or not isinstance(external, dict)
        or external.get("authorized") is False
    ):
        raise PermissionError(
            "SciFact access is forbidden because no climate adapter was promoted"
        )
    frozen_sha = file_sha256(args.frozen_config)
    attempt_id = os.environ.get("SLURM_JOB_ID", "local-attempt")
    reserve_external_transfer(
        args.ledger,
        frozen_config_sha256=frozen_sha,
        attempt_id=attempt_id,
        maximum_infrastructure_retries=int(
            protocol["infrastructure"]["maximum_same_configuration_retries"]
        ),
    )
    output = Path(args.output_dir)
    if output.exists():
        raise FileExistsError(f"output must be unique: {output}")
    output.mkdir(parents=True)
    prepared = output / "prepared-scifact"
    scifact_manifest = prepare_scifact_transfer(args.archive, prepared)
    documents = list(iter_evidence(prepared / "evidence.jsonl"))
    claims = load_claims(prepared / "test-claims.json")
    model = protocol["models"]["dense"]

    base_encoder = SentenceTransformerEncoder(
        str(model["name"]),
        device=args.device,
        truncate_dim=int(model["dimension"]),
        revision=str(model["revision"]),
    )
    base_vectors, base_vector_metrics = build_dense_vectors(
        base_encoder, documents, batch_size=args.batch_size
    )
    base_index, base_index_metrics = build_faiss_index(base_vectors, kind="flat")
    base_metrics, _, base_rows = score_dense_index(
        base_encoder,
        base_index,
        documents,
        claims,
        batch_size=args.batch_size,
        search_width=5,
        final_k=5,
    )
    del base_encoder, base_vectors, base_index
    _release_gpu()

    adapter_path = Path(args.adapter_dir)
    if tree_sha256(adapter_path) != str(frozen["adapter_sha256"]):
        raise ValueError("selected adapter changed after configuration freeze")
    candidate_encoder = SentenceTransformerEncoder(
        str(model["name"]),
        device=args.device,
        truncate_dim=int(model["dimension"]),
        revision=str(model["revision"]),
        adapter_path=str(adapter_path),
    )
    candidate_vectors, candidate_vector_metrics = build_dense_vectors(
        candidate_encoder, documents, batch_size=args.batch_size
    )
    candidate_index, candidate_index_metrics = build_faiss_index(
        candidate_vectors, kind="flat"
    )
    candidate_metrics, _, candidate_rows = score_dense_index(
        candidate_encoder,
        candidate_index,
        documents,
        claims,
        batch_size=args.batch_size,
        search_width=5,
        final_k=5,
    )
    paired, tagged = evaluate_representation_pair(
        claims,
        documents,
        predictions_from_rows(base_rows),
        predictions_from_rows(candidate_rows),
        bootstrap_samples=int(protocol["evaluation"]["bootstrap_samples"]),
        seed=int(protocol["seed"]),
    )
    metrics = {
        "schema_version": 1,
        "evidence_status": "one-shot-external-transfer",
        "dataset": "SciFact",
        "dataset_manifest": scifact_manifest,
        "frozen_config_sha256": frozen_sha,
        "selected_candidate_id": frozen["selected_candidate_id"],
        "selected_candidate_promoted_on_climate_validation": frozen[
            "selected_candidate_promoted"
        ],
        "base": base_metrics,
        "candidate": candidate_metrics,
        "paired_bootstrap": paired["paired_bootstrap"],
        "diagnostic_slices": paired["taxonomy"],
        "base_vector_build": base_vector_metrics,
        "candidate_vector_build": candidate_vector_metrics,
        "base_flat_index": base_index_metrics,
        "candidate_flat_index": candidate_index_metrics,
        "model": str(model["name"]),
        "model_revision": str(model["revision"]),
        "adapter_sha256": frozen["adapter_sha256"],
        "completed_external_evaluations": 1,
        "tuning_after_result": "forbidden",
        "git_commit": os.environ.get("CLIMATE_GIT_COMMIT", "unknown"),
        "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
        "truth_boundary": (
            "This is the single post-freeze zero-shot SciFact transfer event. Its test "
            "qrels were not used for CLIMATE-FEVER selection, and no result-triggered "
            "configuration change or repeat is permitted."
        ),
    }
    write_jsonl(output / "base_predictions.jsonl", base_rows)
    write_jsonl(output / "candidate_predictions.jsonl", candidate_rows)
    write_jsonl(output / "slices.jsonl", tagged)
    write_json(output / "metrics.json", metrics)
    metrics_sha = file_sha256(output / "metrics.json")
    ledger = complete_external_transfer(
        args.ledger, attempt_id=attempt_id, metrics_sha256=metrics_sha
    )
    write_json(
        output / "artifact_manifest.json",
        {
            "schema_version": 1,
            "metrics_sha256": metrics_sha,
            "payload_tree_sha256": tree_sha256(output),
            "external_ledger_sha256": file_sha256(args.ledger),
            "completed_evaluations": ledger["completed_evaluations"],
            "predictions": "spartan-only",
        },
    )
    print(json.dumps(metrics, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
