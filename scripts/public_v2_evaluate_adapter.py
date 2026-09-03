from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

import numpy as np

from climate_rag.dense import SentenceTransformerEncoder
from climate_rag.evaluation_protocol import enforce_frozen_test_policy
from climate_rag.io import (
    iter_evidence,
    load_claims,
    read_json,
    write_json,
    write_jsonl,
)
from climate_rag.public_v2 import (
    file_sha256,
    load_public_v2_protocol,
    paired_promotion_decision,
    pilot_advancement_decision,
    select_pilot_claim_ids,
    tree_sha256,
)
from climate_rag.public_v2_runtime import (
    build_dense_vectors,
    build_faiss_index,
    decisive_claims,
    predictions_from_rows,
    score_dense_index,
)
from climate_rag.representation_eval import evaluate_representation_pair


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate one pre-registered public-v2 dense adapter."
    )
    parser.add_argument("--protocol", required=True)
    parser.add_argument("--policy", required=True)
    parser.add_argument("--prepared-dir", required=True)
    parser.add_argument("--selection-dir", required=True)
    parser.add_argument("--base-dir", required=True)
    parser.add_argument("--adapter-dir", required=True)
    parser.add_argument("--training-record", required=True)
    parser.add_argument("--adapter-config-id", required=True)
    parser.add_argument("--mode", choices=("pilot", "full"), required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch-size", type=int, default=32)
    return parser.parse_args()


def _registered_adapter(protocol: dict[str, Any], identifier: str) -> dict[str, Any]:
    for row in protocol["adapter_matrix"]:
        if str(row["id"]) == identifier:
            return dict(row)
    raise ValueError(f"adapter is not pre-registered: {identifier}")


def main() -> int:
    args = parse_args()
    protocol = load_public_v2_protocol(args.protocol)
    registered = _registered_adapter(protocol, args.adapter_config_id)
    training_record = read_json(args.training_record)
    if not isinstance(training_record, dict):
        raise TypeError("training record must be an object")
    if training_record.get("adapter_config") != registered:
        raise ValueError("trained adapter configuration differs from preregistration")
    policy = enforce_frozen_test_policy(
        args.policy,
        split="validation",
        system_id=args.adapter_config_id,
    )
    output = Path(args.output_dir)
    if output.exists():
        raise FileExistsError(f"output must be unique: {output}")
    output.mkdir(parents=True)
    documents = list(iter_evidence(Path(args.prepared_dir) / "evidence.jsonl"))
    validation_all = load_claims(Path(args.selection_dir) / "validation-claims.json")
    validation = decisive_claims(validation_all)
    if args.mode == "pilot":
        raw_claims = read_json(Path(args.selection_dir) / "validation-claims.json")
        if not isinstance(raw_claims, dict):
            raise TypeError("validation claims must be an object")
        pilot_ids = select_pilot_claim_ids(
            raw_claims,
            size=int(protocol["evaluation"]["pilot_query_count"]),
            seed=int(protocol["evaluation"]["pilot_seed"]),
        )
        claims = {claim_id: validation[claim_id] for claim_id in pilot_ids}
    else:
        claims = validation

    model = protocol["models"]["dense"]
    base_encoder = SentenceTransformerEncoder(
        str(model["name"]),
        device=args.device,
        truncate_dim=int(model["dimension"]),
        revision=str(model["revision"]),
    )
    base_vectors = np.load(Path(args.base_dir) / "base_embeddings.npy", mmap_mode="r")
    base_flat, base_index_metrics = build_faiss_index(base_vectors, kind="flat")
    base_metrics, _, base_rows = score_dense_index(
        base_encoder,
        base_flat,
        documents,
        claims,
        batch_size=args.batch_size,
        search_width=5,
        final_k=5,
    )
    del base_encoder, base_flat

    adapter_path = Path(args.adapter_dir)
    candidate_encoder = SentenceTransformerEncoder(
        str(model["name"]),
        device=args.device,
        truncate_dim=int(model["dimension"]),
        revision=str(model["revision"]),
        adapter_path=str(adapter_path),
    )
    candidate_vectors, vector_metrics = build_dense_vectors(
        candidate_encoder, documents, batch_size=args.batch_size
    )
    embedding_path = output / "candidate_embeddings.npy"
    np.save(embedding_path, candidate_vectors)
    flat_path = output / "candidate_flat.faiss"
    candidate_flat, candidate_index_metrics = build_faiss_index(
        candidate_vectors, kind="flat", output_path=flat_path
    )
    candidate_metrics, _, candidate_rows = score_dense_index(
        candidate_encoder,
        candidate_flat,
        documents,
        claims,
        batch_size=args.batch_size,
        search_width=5,
        final_k=5,
    )
    base_predictions = predictions_from_rows(base_rows)
    candidate_predictions = predictions_from_rows(candidate_rows)
    paired, tagged_rows = evaluate_representation_pair(
        claims,
        documents,
        base_predictions,
        candidate_predictions,
        bootstrap_samples=int(protocol["evaluation"]["bootstrap_samples"]),
        seed=int(protocol["seed"]),
    )
    decision = (
        pilot_advancement_decision(paired)
        if args.mode == "pilot"
        else paired_promotion_decision(paired)
    )
    metrics = {
        "schema_version": 1,
        "adapter_config_id": args.adapter_config_id,
        "adapter_config": registered,
        "mode": args.mode,
        "source_validation_claim_count": len(validation_all),
        "evaluated_decisive_claim_count": len(claims),
        "test_claims_loaded": 0,
        "base": base_metrics,
        "candidate": candidate_metrics,
        "paired_bootstrap": paired["paired_bootstrap"],
        "diagnostic_slices": paired["taxonomy"],
        "decision": decision,
        "base_index_rebuild": base_index_metrics,
        "candidate_vector_build": vector_metrics,
        "candidate_flat_index": candidate_index_metrics,
        "candidate_embedding_bytes": embedding_path.stat().st_size,
        "candidate_embeddings_path": str(embedding_path),
        "candidate_embeddings_sha256": file_sha256(embedding_path),
        "candidate_flat_index_path": str(flat_path),
        "adapter_parameter_count": candidate_encoder.adapter_parameter_count,
        "adapter_path": str(adapter_path),
        "adapter_sha256": tree_sha256(adapter_path),
        "training_record_sha256": file_sha256(args.training_record),
        "evaluation_policy": policy,
        "model": str(model["name"]),
        "model_revision": str(model["revision"]),
        "git_commit": os.environ.get("CLIMATE_GIT_COMMIT", "unknown"),
        "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
        "slurm_array_task_id": os.environ.get("SLURM_ARRAY_TASK_ID"),
        "boundary": (
            "Selection uses CLIMATE-FEVER v2 validation only. Pilot results are screens; "
            "full promotion requires the frozen paired rule. No test claim is loaded."
        ),
    }
    write_jsonl(output / "base_predictions.jsonl", base_rows)
    write_jsonl(output / "candidate_predictions.jsonl", candidate_rows)
    write_jsonl(output / "slices.jsonl", tagged_rows)
    write_json(output / "metrics.json", metrics)
    write_json(
        output / "artifact_manifest.json",
        {
            "schema_version": 1,
            "metrics_sha256": file_sha256(output / "metrics.json"),
            "artifact_tree_sha256": tree_sha256(output),
            "large_artifacts_publication": "spartan-only",
        },
    )
    print(json.dumps(metrics, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
