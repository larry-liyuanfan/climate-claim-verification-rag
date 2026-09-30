"""Prepared paired-arm inference. Requires separate explicit GPU release; no gold loader."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import time
from pathlib import Path

from climate_rag.local_agent_model import verify_model_files
from climate_rag.local_bounded_scifact_provider import LocalQwenBoundedSciFactProvider
from climate_rag.public_v2 import file_sha256
from climate_rag.rerank import Qwen3CausalLMReranker
from climate_rag.scifact_diagnostic_runtime import validate_protocol
from climate_rag.scifact_bounded_runtime import ARMS, run_bounded_slot
from climate_rag.scifact_bounded_smoke import bounded_runtime_smoke
from climate_rag.scifact_grounding import parse_abstract
from climate_rag.scifact_retrieval import SciFactBM25, rerank_sources

RELEASE = "climate-scifact-bounded-gap-pair-20260930-v1"
PAIR_SHA = "4cc5033928a0833ee99ad64ecd5a7b10c7a1fde7ac4f219d6334d2349d9daaf6"

MODEL_SHA = "d1dd9783afdf4e0fbd21eee824834d71b86982f5a5d5f6f371fe07f2f76f3cf6"
RERANKER_SHA = "de1d4ac39101816774439e68881e2308c5e5f1bd94d0b0dc4c492a56c2681052"


def write_once(path, value):
    with path.open("x", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2, sort_keys=True)
        stream.write("\n")


def verify_manifest(path, model_dir, expected):
    value = json.loads(path.read_text())
    actual = hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()
    if actual != expected:
        raise ValueError("unreleased model manifest")
    verify_model_files(model_dir, value)
    return value


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("inference-dir", "model-dir", "model-manifest", "reranker-dir",
                 "reranker-manifest", "output-dir", "pair-protocol"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--expected-protocol-sha256", required=True)
    parser.add_argument("--arm", required=True, choices=ARMS)
    args = parser.parse_args()
    if (not os.environ.get("SLURM_JOB_ID") or not os.environ.get("CUDA_VISIBLE_DEVICES")
            or os.environ.get("CLIMATE_SCIFACT_PAIR_RELEASE") != RELEASE
            or os.environ.get("PYTHONHASHSEED") != "0"
            or any(os.environ.get(k) != "1" for k in ("HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE"))):
        raise ValueError("separately released offline GPU allocation required")
    if {p.name for p in args.inference_dir.iterdir()} != {"protocol.json", "corpus.jsonl", "claims.jsonl"}:
        raise ValueError("inference bundle has missing/extra files")
    if file_sha256(args.pair_protocol) != PAIR_SHA:
        raise ValueError("paired preregistration hash mismatch")
    pair = json.loads(args.pair_protocol.read_text())
    protocol_path = args.inference_dir / "protocol.json"
    if file_sha256(protocol_path) != args.expected_protocol_sha256:
        raise ValueError("protocol SHA mismatch")
    protocol = json.loads(protocol_path.read_text())
    for name in ("corpus.jsonl", "claims.jsonl"):
        if file_sha256(args.inference_dir / name) != protocol["inference_file_sha256"][name]:
            raise ValueError("inference file hash mismatch")
    claims = [json.loads(line) for line in (args.inference_dir / "claims.jsonl").read_text().splitlines()]
    validate_protocol(protocol, claims)
    if ([r["id"] for r in claims] != pair["ordered_claim_ids"]
            or args.expected_protocol_sha256 != pair["parent_protocol_sha256"]
            or protocol["inference_file_sha256"] != pair["inference_file_sha256"]):
        raise ValueError("paired claims/order/data contract mismatch")
    abstracts = [parse_abstract(json.loads(line)) for line in
                 (args.inference_dir / "corpus.jsonl").read_text().splitlines()]
    corpus = {d.doc_id: d for d in abstracts}
    if len(corpus) != 5183 or len(abstracts) != 5183:
        raise ValueError("original corpus shape mismatch")
    source_git = os.environ.get("CLIMATE_SOURCE_GIT", "")
    if len(source_git) != 40 or any(c not in "0123456789abcdef" for c in source_git):
        raise ValueError("frozen source SHA required")
    model_manifest = verify_manifest(args.model_manifest, args.model_dir, MODEL_SHA)
    verify_manifest(args.reranker_manifest, args.reranker_dir, RERANKER_SHA)
    out = args.output_dir
    out.mkdir(mode=0o700)  # no overwrite/resume or silent retry
    write_once(out / "consumed.json", {"source_git": source_git, "release_id": RELEASE, "arm": args.arm, "pair_sha256": PAIR_SHA})
    (out / "private-responses").mkdir(mode=0o700)
    began = time.monotonic()
    provider = LocalQwenBoundedSciFactProvider(args.model_dir, model_manifest,
        private_dir=out / "private-responses", gap=args.arm == ARMS[1])
    provider.start_slot(out / "private-responses" / "preflight")
    load_ms = (time.monotonic() - began) * 1000
    began = time.monotonic()
    preflight = bounded_runtime_smoke(provider)
    preflight["elapsed_ms"] = (time.monotonic() - began) * 1000
    write_once(out / "runtime-preflight.json", preflight)
    if preflight["status"] != "passed":
        raise ValueError("synthetic HF preflight failed; no real train inference")
    began = time.monotonic()
    model = Qwen3CausalLMReranker(str(args.reranker_dir), device="cuda", dtype="bfloat16",
                                max_length=2048, batch_size=1)
    reranker_load_ms = (time.monotonic() - began) * 1000
    began = time.monotonic()
    retrieve = SciFactBM25(corpus)
    index_ms = (time.monotonic() - began) * 1000
    lookup = {r["id"]: r["claim"] for r in claims}
    runs = []
    for slot, spec in enumerate(protocol["slots"]):
        raw_path = out / f"slot-{slot + 1:02d}-raw.json"
        provider.start_slot(out / "private-responses" / f"slot-{slot + 1:02d}")
        try:
            row = run_bounded_slot(spec["claim_id"], lookup[spec["claim_id"]], spec["route"], args.arm, provider,
                           retrieve, lambda q, c: rerank_sources(model, q, c), corpus,
                           persist_raw=lambda value: write_once(raw_path, value))
        except Exception as exc:
            # If durable I/O itself fails, stop: cannot safely pretend a ledger exists.
            if not raw_path.exists():
                raise
            row = {"claim_id": spec["claim_id"], "route": spec["route"], "arm": args.arm,
                   "result": json.loads(raw_path.read_text()), "prediction": None,
                   "export_error": type(exc).__name__}
        write_once(out / f"slot-{slot + 1:02d}.json", row)
        runs.append(row)
        for attempt in row["result"]["generation_attempts"]:
            diag = attempt.get("diagnostics", {})
            receipt = diag.get("private_attachment", {})
            if (not attempt["usage_known"] or receipt.get("truncated") is not False
                    or receipt.get("io_failed") is not False
                    or receipt.get("stored_bytes") != receipt.get("attempted_bytes")):
                raise ValueError("private wire/usage audit incomplete; durable cost retained, stop batch")
    write_once(out / "run.json", {
        "release_id": RELEASE, "source_git": source_git, "arm": args.arm,
        "pair_protocol_sha256": PAIR_SHA,
        "protocol_sha256": args.expected_protocol_sha256,
        "inference_file_sha256": protocol["inference_file_sha256"],
        "model_sha256": MODEL_SHA, "reranker_sha256": RERANKER_SHA,
        "model_load_ms": load_ms, "reranker_load_ms": reranker_load_ms,
        "bm25_build_ms": index_ms, "preflight_sha256": file_sha256(out / "runtime-preflight.json"),
        "scope": protocol["scope"], "gold_loaded": False, "runs": runs,
        "private_storage_plan": {"directories": 49, "max_files_per_directory": 10,
            "max_bytes_per_directory": 5 * (provider.wire_byte_limit + 16384),
            "wire_bytes_per_call": provider.wire_byte_limit},
    })


if __name__ == "__main__":
    main()
