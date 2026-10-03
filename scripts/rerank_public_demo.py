"""One live public RRF result -> exact public 4B checkpoint. No gold/test/Agent."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from time import perf_counter
from typing import Any

from climate_rag.io import iter_evidence
from climate_rag.models import RankedDocument
from climate_rag.local_agent_model import verify_model_files
from climate_rag.targeted_replay import MANIFEST_BYTES, RERANKER_SHA
from score_saved_citation_nli import digest
from build_public_learned_demo import CORPUS_SHA

MODEL = "Qwen/Qwen3-Reranker-4B"
REVISION = "22e683669bc0f0bd69640a1354a6d0aebcfeede5"


def run(search: Path, search_sha: str, evidence: Path, role_dir: Path) -> dict[str, Any]:
    if digest(search) != search_sha or digest(evidence) != CORPUS_SHA:
        raise ValueError("live_search_input_identity")
    value = json.loads(search.read_bytes())
    if (value["mode"] != "live_new_public_base_embedding_bm25_hnsw_rrf_not_historical_ltr_4b"
            or value["corpus_sha256"] != CORPUS_SHA):
        raise ValueError("live_public_search_required")
    manifest_file = role_dir / "model_manifest.json"
    source = json.loads((role_dir / "source.json").read_bytes())
    if (source["id"] != MODEL or source["sha"] != REVISION
            or digest(manifest_file) != MANIFEST_BYTES["reranker"]
            or verify_model_files(role_dir / "model", json.loads(manifest_file.read_bytes())) != RERANKER_SHA):
        raise ValueError("exact_historical_public_reranker_identity_required")
    lookup = {d.evidence_id: d for d in iter_evidence(evidence)}
    records = value["candidate_ranks"]["rrf"]
    if (len(records) != 20 or len({r["id"] for r in records}) != 20
            or any(r["id"] not in lookup for r in records)):
        raise ValueError("rrf_candidate_contract")
    rows = [RankedDocument(r["id"], float(r["score"]), int(r["rank"]), lookup[r["id"]].text, "rrf") for r in records]
    import torch
    from climate_rag.rerank import Qwen3CausalLMReranker
    torch.set_num_threads(4)
    started = perf_counter()
    ranker = Qwen3CausalLMReranker(str(role_dir / "model"), device="cpu", dtype="bfloat16", batch_size=1, max_length=2048)
    load_ms = (perf_counter() - started) * 1000
    claim = value["claim"]
    lengths = [len(ranker._tokenizer.encode(f"<Instruct>: {ranker.instruction}\n<Query>: {claim}\n<Document>: {r.text}", add_special_tokens=True))
               + len(ranker._prefix_tokens) + len(ranker._suffix_tokens) for r in rows]
    if max(lengths) > ranker.max_length:
        raise ValueError("reranker_pair_would_truncate")
    started = perf_counter()
    ranked = ranker.rerank(claim, rows, 20)
    rerank_ms = (perf_counter() - started) * 1000
    return {"mode": "live_public_bm25_learned_hnsw_rrf_4b_not_ltr_or_agent",
        "claim": claim, "parent_search_sha256": search_sha, "corpus_sha256": CORPUS_SHA,
        "model": MODEL, "revision": REVISION, "model_identity_sha256": RERANKER_SHA,
        "manifest_sha256": MANIFEST_BYTES["reranker"], "candidate_count": 20,
        "cpu_threads": 4, "dtype": "bfloat16", "batch_size": 1, "max_pair_tokens": max(lengths),
        "model_load_ms": load_ms, "rerank_ms": rerank_ms,
        "candidate_ranks": [{"id": r.evidence_id, "rank": r.rank, "score": r.score} for r in ranked],
        "evidence": [{"id": r.evidence_id, "text": r.text, "metadata": dict(lookup[r.evidence_id].metadata)} for r in ranked[:3]],
        "verdict": None, "task_accuracy": None, "ltr_restored": False,
        "timing_scope": "one public manual query; no P50/P95/SLA/quality uplift; setup/hash excluded"}


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--search", type=Path, required=True)
    p.add_argument("--search-sha", required=True)
    p.add_argument("--evidence", type=Path, required=True)
    p.add_argument("--role-dir", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    output = args.output.resolve()
    if (not args.output.is_absolute() or output.is_relative_to(Path(__file__).resolve().parents[1])
            or any(v.casefold() in {"onedrive", "求职"} for v in output.parts) or output.exists()):
        raise ValueError("new_private_local_output_required")
    result = run(args.search, args.search_sha, args.evidence, args.role_dir)
    with output.open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(result, stream, ensure_ascii=False, indent=2)
        stream.write("\n")
    print(json.dumps({"output_sha256": digest(output), "rerank_ms": result["rerank_ms"], "model_load_ms": result["model_load_ms"],
                      "candidate_count": result["candidate_count"], "mode": result["mode"]}))


if __name__ == "__main__":
    main()
