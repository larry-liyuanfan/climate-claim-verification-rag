"""Run one frozen public-validation profile per fresh process (no training)."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import resource
import time
from pathlib import Path
from typing import Any

from climate_rag.bm25 import BM25Index
from climate_rag.dense import DenseRetriever, FaissANNIndex, SentenceTransformerEncoder
from climate_rag.evaluation_protocol import enforce_frozen_test_policy
from climate_rag.fusion import LightGBMLambdaMART
from climate_rag.io import iter_evidence, load_claims, write_json, write_jsonl
from climate_rag.metrics import evaluate_predictions
from climate_rag.models import Prediction
from climate_rag.public_v2 import file_sha256, load_public_v2_protocol
from climate_rag.public_v2_runtime import decisive_claims
from climate_rag.rerank import Qwen3CausalLMReranker
from climate_rag.search_profile import ProfiledSearch, summarize_request_timings


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", required=True)
    parser.add_argument("--policy", required=True)
    parser.add_argument("--prepared-dir", required=True)
    parser.add_argument("--selection-dir", required=True)
    parser.add_argument("--base-dir", required=True)
    parser.add_argument("--downstream-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--profile", choices=("ltr", "rerank20", "rerank100"), required=True)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    protocol = load_public_v2_protocol(args.protocol)
    policy = enforce_frozen_test_policy(args.policy, split="validation", system_id=args.profile)
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=False)
    evidence_path = Path(args.prepared_dir) / "evidence.jsonl"
    validation_path = Path(args.selection_dir) / "validation-claims.json"
    documents = list(iter_evidence(evidence_path))
    claims = decisive_claims(load_claims(validation_path))
    if len(documents) != 5240 or len(claims) != 126:
        raise ValueError("this profile is frozen to 5,240 documents / 126 validation queries")
    dense_model = protocol["models"]["dense"]
    ranking = protocol["downstream_ranking"]
    load_started = time.perf_counter()
    encoder = SentenceTransformerEncoder(
        dense_model["name"], revision=dense_model["revision"],
        truncate_dim=dense_model["dimension"], device=args.device,
    )
    index_path = Path(args.downstream_dir) / "base_dense_hnsw.faiss"
    backend = FaissANNIndex.load(index_path, {"dimension": 1024, "kind": "hnsw"})
    if backend.index.ntotal != len(documents):
        raise ValueError("saved index count and corpus mapping differ")
    dense = DenseRetriever(encoder, backend)
    dense.doc_ids = [row.evidence_id for row in documents]
    dense.texts = [row.text for row in documents]
    bm25_path = Path(args.base_dir) / "bm25.pkl.gz"
    bm25 = BM25Index.load(bm25_path)
    ranker = None
    reranker = None
    rerank_width = 0
    ltr_path = Path(args.downstream_dir) / "lambdamart_top100.txt"
    if args.profile == "ltr":
        ranker = LightGBMLambdaMART.load(ltr_path)
    else:
        rerank_width = 20 if args.profile == "rerank20" else 100
        model = protocol["models"]["reranker"]
        reranker = Qwen3CausalLMReranker(
            model["name"], device=args.device, max_length=2048, batch_size=4,
            dtype=model["dtype"], instruction=model["instruction"], revision=model["revision"],
        )
    import torch

    gpu = args.device.startswith("cuda")
    def synchronize() -> None:
        if gpu:
            torch.cuda.synchronize()

    search = ProfiledSearch(
        bm25, dense, ranker=ranker, reranker=reranker, rerank_width=rerank_width,
        recall_width=ranking["recall_width"], candidate_width=ranking["candidate_width"],
        rrf_k=ranking["rrf_k"], synchronize=synchronize,
        base_weight=ranking["reranker_fusion"]["base_weight"],
        reranker_weight=ranking["reranker_fusion"]["reranker_weight"],
    )
    load_seconds = time.perf_counter() - load_started
    # Identical unlabelled warmup, excluded from query metrics and latency quantiles.
    search.search("Carbon dioxide and the Earth's energy balance")
    if gpu:
        torch.cuda.reset_peak_memory_stats()
    predictions: dict[str, Prediction] = {}
    traces: list[dict[str, Any]] = []
    for claim_id, claim in sorted(claims.items()):
        ranked, trace = search.search(claim.text)
        predictions[claim_id] = Prediction(claim_id, tuple(row.evidence_id for row in ranked))
        traces.append({"claim_id": claim_id, **trace})
        print(json.dumps({"profile": args.profile, "completed": len(traces),
                          "end_to_end_ms": trace["end_to_end_ms"]}), flush=True)
    metrics, rows, _ = evaluate_predictions(
        claims, predictions, evidence_k=5, evaluate_labels=False,
    )
    write_jsonl(output / "predictions.jsonl", (
        {"claim_id": key, "evidence_ids": list(value.evidence_ids)}
        for key, value in sorted(predictions.items())
    ))
    write_jsonl(output / "query-metrics.jsonl", rows)
    write_jsonl(output / "traces.jsonl", traces)
    summary = {
        "schema_version": 1, "profile": args.profile, "split": "validation",
        "test_claims_loaded": 0, "adapter_used": False, "training_performed": False,
        "query_count": len(claims), "document_count": len(documents),
        "metrics": metrics, "timings": summarize_request_timings(traces),
        "load_seconds": load_seconds, "warmup_requests_excluded": 1,
        "device": args.device, "peak_rss_bytes": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024,
        "peak_torch_allocated_bytes": int(torch.cuda.max_memory_allocated()) if gpu else 0,
        "peak_torch_reserved_bytes": int(torch.cuda.max_memory_reserved()) if gpu else 0,
        "model_revisions": {name: data["revision"] for name, data in protocol["models"].items()},
        "runtime": {"torch": torch.__version__, "cuda": torch.version.cuda,
                    "python_hash_seed": os.environ.get("PYTHONHASHSEED"),
                    "gpu_name": torch.cuda.get_device_name() if gpu else None,
                    "omp_threads": os.environ.get("OMP_NUM_THREADS"),
                    "encoder_dtype": str(next(encoder._model.parameters()).dtype),
                    "reranker_dtype": str(next(reranker._model.parameters()).dtype) if reranker else None},
        "source_hashes": {"protocol": file_sha256(args.protocol),
                          "validation": file_sha256(validation_path), "evidence": file_sha256(evidence_path),
                          "index": file_sha256(index_path), "bm25": file_sha256(bm25_path),
                          "ltr": file_sha256(ltr_path)},
        "query_ids_sha256": hashlib.sha256(json.dumps(sorted(claims)).encode()).hexdigest(),
        "prediction_sha256": file_sha256(output / "predictions.jsonl"),
        "trace_sha256": file_sha256(output / "traces.jsonl"),
        "policy": policy, "git_commit": os.environ.get("CLIMATE_GIT_COMMIT", "unknown"),
        "job_id": os.environ.get("SLURM_JOB_ID"),
        "api_cost": {"requests": 0, "tokens": 0, "currency_charge": None},
        "boundary": "offline warmed serial in-process profile; not HTTP SLA or monetary savings",
    }
    write_json(output / "summary.json", summary)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
