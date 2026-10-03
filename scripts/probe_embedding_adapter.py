"""Checkpoint integrity only: no qrels, corpus build, model selection or test use."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import time

from climate_rag.dense import SentenceTransformerEncoder
from climate_rag.io import write_json
from climate_rag.public_v2 import tree_sha256


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--adapter-dir", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    queries = ["How has global mean temperature changed?", "What affects Antarctic sea ice?"]
    documents = ["Carbon dioxide absorbs infrared radiation.",
                 "Ocean heat content varies over time."]
    revision = "97b0c614be4d77ee51c0cef4e5f07c00f9eb65b3"
    started = time.perf_counter()
    encoder = SentenceTransformerEncoder(
        "Qwen/Qwen3-Embedding-0.6B", device=args.device, truncate_dim=1024,
        revision=revision, adapter_path=args.adapter_dir,
    )
    probe = encoder.probe_adapter_effect(queries, documents)
    report = {
        "schema_version": 1,
        "git_commit": os.environ.get("CLIMATE_GIT_COMMIT", "unknown"),
        "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
        "model_revision": revision,
        "adapter_tree_sha256": tree_sha256(args.adapter_dir),
        "probe_text_sha256": hashlib.sha256(
            json.dumps([queries, documents], ensure_ascii=True).encode()
        ).hexdigest(),
        "weights": encoder.adapter_integrity,
        "output_probe": probe,
        "elapsed_seconds": time.perf_counter() - started,
        "quality_evaluation_performed": False,
        "test_claims_loaded": 0,
        "boundary": "original single diagnostic checkpoint, fixed label-free probes only",
    }
    write_json(args.output, report)
    print(json.dumps(report, indent=2))
    return 0 if probe["effect_verified"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
