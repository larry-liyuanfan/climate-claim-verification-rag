"""Fresh semantic-policy inference arm. Separate release and GPU allocation required."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import time
from pathlib import Path
from typing import Any

from climate_rag.agent_v3 import V3Budget
from climate_rag.local_agent_model import verify_model_files
from climate_rag.rerank import Qwen3CausalLMReranker
from climate_rag.scifact_grounding import parse_abstract
from climate_rag.scifact_retrieval import SciFactBM25, rerank_sources
from climate_rag.scifact_semantic_contract import (
    BUDGET, COMPACT_SHA, MODEL_SHA, PAIR, POLICIES, PREPARATION_GIT, PROMPTS,
    RELEASE, RERANKER_SHA, checked, load_inference, sha, write_once,
)
from climate_rag.scifact_semantic_execution import execute_slots, preflight
from climate_rag.scifact_semantic_policy import LocalQwenSemanticGapProvider


def verify_manifest(path: Path, directory: Path, expected: str) -> dict[str, str]:
    value: dict[str, str] = json.loads(path.read_bytes())
    if hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest() != expected:
        raise ValueError("semantic_model_manifest_identity")
    verify_model_files(directory, value)
    return value


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    for name in ("inference-dir", "model-dir", "model-manifest", "reranker-dir", "reranker-manifest", "output-dir"):
        p.add_argument("--" + name, type=Path, required=True)
    p.add_argument("--protocol-sha", required=True)
    p.add_argument("--policy", choices=POLICIES, required=True)
    args = p.parse_args()
    git = os.environ.get("CLIMATE_SOURCE_GIT", "")
    if (not os.environ.get("SLURM_JOB_ID") or not os.environ.get("CUDA_VISIBLE_DEVICES")
            or os.environ.get("CLIMATE_SEMANTIC_RELEASE") != RELEASE or os.environ.get("PYTHONHASHSEED") != "0"
            or any(os.environ.get(k) != "1" for k in ("HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE"))
            or V3Budget().model_dump() != BUDGET):
        raise ValueError("separately_released_offline_gpu_required")
    protocol, claims, corpus_bytes = load_inference(args.inference_dir, args.protocol_sha, git)
    source_sha, inference_sha = (os.environ[n] for n in ("CLIMATE_SOURCE_SHA256", "CLIMATE_INFERENCE_SHA256"))
    if any(len(s) != 64 or any(c not in "0123456789abcdef" for c in s) for s in (source_sha, inference_sha)):
        raise ValueError("archive_sha_required")
    abstracts = [parse_abstract(json.loads(line)) for line in corpus_bytes.splitlines()]
    corpus = {d.doc_id: d for d in abstracts}
    if len(corpus) != len(abstracts) or len(corpus) != 5183:
        raise ValueError("semantic_corpus_shape")
    manifest = verify_manifest(args.model_manifest, args.model_dir, MODEL_SHA)
    verify_manifest(args.reranker_manifest, args.reranker_dir, RERANKER_SHA)
    for name, expected in protocol["tokenizer_sha256"].items():
        checked(args.model_dir / name, expected)
    out = args.output_dir
    out.mkdir(mode=0o700)
    identity: dict[str, Any] = {"release_id": RELEASE, "comparison_protocol": PAIR, "arm": args.policy,
        "source_git": git, "source_archive_sha256": source_sha, "inference_archive_sha256": inference_sha,
        "preparation_source_git": PREPARATION_GIT, "preparation_compact_sha256": COMPACT_SHA,
        "protocol_sha256": args.protocol_sha, "prompt_sha256": PROMPTS[args.policy],
        "inference_file_sha256": protocol["inference_file_sha256"], "model_sha256": MODEL_SHA,
        "reranker_sha256": RERANKER_SHA, "gold_loaded": False}
    write_once(out / "consumed.json", identity | {"meaning": "attempt_reserved_not_all_slots_consumed"})
    (out / "private-responses").mkdir(mode=0o700)
    began = time.monotonic()
    backend = LocalQwenSemanticGapProvider(args.model_dir, manifest,
        private_dir=out / "private-responses", policy=args.policy)
    load_ms = (time.monotonic() - began) * 1000
    backend.start_slot(out / "private-responses" / "preflight")
    preflight(backend, args.policy, out, git, args.protocol_sha)
    began = time.monotonic()
    reranker = Qwen3CausalLMReranker(str(args.reranker_dir), device="cuda", dtype="bfloat16", max_length=2048, batch_size=1)
    reranker_ms = (time.monotonic() - began) * 1000
    began = time.monotonic()
    retrieve = SciFactBM25(corpus)
    index_ms = (time.monotonic() - began) * 1000
    rows = execute_slots(protocol, claims, backend, retrieve, lambda q, c: rerank_sources(reranker, q, c), corpus, out)
    write_once(out / "run.json", identity | {
        "runs": rows, "model_load_ms": load_ms, "reranker_load_ms": reranker_ms, "bm25_build_ms": index_ms,
        "preflight_sha256": sha((out / "runtime-preflight.json").read_bytes()),
        "private_storage_plan": {"directories": 1 + len(rows), "max_files_per_directory": 10,
            "max_bytes_per_directory": 5 * (backend.wire_byte_limit + 16384)},
    })


if __name__ == "__main__":
    main()
