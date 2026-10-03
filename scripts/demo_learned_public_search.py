"""Actual public BM25 + pinned learned dense/HNSW + RRF, no LTR/4B/verdict."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys
from time import perf_counter
from typing import Any

from climate_rag.bm25 import BM25Index
from climate_rag.fusion import reciprocal_rank_fusion
from climate_rag.io import iter_evidence
from climate_rag.models import RankedDocument
from climate_rag.verification import normalise_claim
from build_public_learned_demo import CORPUS_SHA, MODEL, REVISION
from score_saved_citation_nli import digest


def run(evidence: Path, dense_dir: Path, claim: str) -> dict[str, Any]:
    text = normalise_claim(claim)
    if not 1 <= len(text) <= 2000:
        raise ValueError("claim_length")
    proof = json.loads((dense_dir / "reconstruction.json").read_bytes())
    if (digest(evidence) != CORPUS_SHA or proof["corpus_sha256"] != CORPUS_SHA
            or proof["model"] != MODEL or proof["revision"] != REVISION
            or digest(dense_dir / "index.faiss") != proof["index_sha256"]
            or digest(dense_dir / "dense_index.json") != proof["metadata_sha256"]):
        raise ValueError("reconstruction_identity_mismatch")
    spec = json.loads((dense_dir / "dense_index.json").read_bytes())
    model_dir = Path(spec["encoder"]["name"])
    if any(digest(model_dir / name) != sha for name, sha in proof["model_files"].items()):
        raise ValueError("embedding_model_bytes_changed")
    import numpy as np
    import faiss
    from climate_rag.dense import FaissANNIndex
    faiss.omp_set_num_threads(4)
    started = perf_counter()
    docs = list(iter_evidence(evidence))
    sparse = BM25Index().fit(docs)
    dense = FaissANNIndex.load(dense_dir / "index.faiss", spec["backend"])
    if spec["doc_ids"] != [d.evidence_id for d in docs] or spec["texts"] != [d.text for d in docs]:
        raise ValueError("dense_document_mapping_mismatch")
    dense.index.hnsw.efSearch = proof["ef_search"]
    setup_ms = (perf_counter() - started) * 1000
    timings = {}
    started = perf_counter()
    bm25 = sparse.search(text, 20)
    timings["bm25_ms"] = (perf_counter() - started) * 1000
    started = perf_counter()
    child = subprocess.run([sys.executable, "-X", "utf8", str(Path(__file__).with_name("encode_public_query.py"))],
        input=json.dumps({"claim": text, "model_dir": str(model_dir)}), text=True, capture_output=True, check=True)
    encoded = json.loads(child.stdout)
    timings["encoder_process_including_load_ms"] = (perf_counter() - started) * 1000
    timings["query_encode_ms"] = encoded["query_encode_ms"]
    vector = np.asarray(encoded["vector"], dtype=np.float32)
    if vector.shape != (1, proof["dimension"]) or not np.isfinite(vector).all():
        raise ValueError("query_vector_contract")
    started = perf_counter()
    scores, offsets = dense.search(vector, 20)
    ranked = sorted([(docs[int(i)].evidence_id, float(s), int(i)) for s, i in zip(scores[0], offsets[0], strict=True) if i >= 0], key=lambda r: (-r[1], r[0]))
    learned = [RankedDocument(evidence_id=eid, text=docs[i].text, score=score, rank=rank, source="dense")
               for rank, (eid, score, i) in enumerate(ranked, 1)]
    timings["hnsw_ms"] = (perf_counter() - started) * 1000
    started = perf_counter()
    fused = reciprocal_rank_fusion({"bm25": bm25, "dense": learned}, k=60, top_k=20)
    timings["rrf_ms"] = (perf_counter() - started) * 1000
    lookup = {d.evidence_id: d for d in docs}
    return {"mode": "live_new_public_base_embedding_bm25_hnsw_rrf_not_historical_ltr_4b",
        "claim": text, "corpus_sha256": CORPUS_SHA, "model": MODEL, "revision": REVISION,
        "documents": len(docs), "setup_ms": setup_ms, "query_model_load_ms": encoded["query_model_load_ms"], "request_stage_ms": timings,
        "candidate_width_per_retriever": 20, "fusion_k": 60, "ef_search": proof["ef_search"],
        "candidate_ranks": {name: [{"id": r.evidence_id, "rank": r.rank, "score": r.score} for r in rows]
                            for name, rows in (("bm25", bm25), ("dense", learned), ("rrf", fused))},
        "evidence": [{"id": r.evidence_id, "text": lookup[r.evidence_id].text,
                      "metadata": dict(lookup[r.evidence_id].metadata), "text_sha256": hashlib.sha256(lookup[r.evidence_id].text.encode()).hexdigest()}
                     for r in fused[:3]], "verdict": None, "quality_metrics": None,
        "timing_scope": "one local CPU request; stages exclude hash checks, evidence packing and serialization; encoder process/load reported separately, not summed twice; not E2E/SLA or A/B"}


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--evidence", type=Path, required=True)
    p.add_argument("--dense-dir", type=Path, required=True)
    p.add_argument("--claim", required=True)
    p.add_argument("--output", type=Path, help="Private local output, never Git/OneDrive")
    args = p.parse_args()
    if args.output is not None:
        output = args.output.resolve()
        if (not args.output.is_absolute() or output.is_relative_to(Path(__file__).resolve().parents[1])
                or any(v.casefold() in {"onedrive", "求职"} for v in output.parts) or output.exists()):
            raise ValueError("private_local_output_required")
    result = run(args.evidence, args.dense_dir, args.claim)
    if args.output is not None:
        with output.open("x", encoding="utf-8", newline="\n") as stream:
            json.dump(result, stream, ensure_ascii=False, indent=2)
            stream.write("\n")
        print(json.dumps({"output_sha256": digest(output), "request_stage_ms": result["request_stage_ms"], "mode": result["mode"]}))
    else:
        print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
