"""CPU-only contract demo. Uses repository-authored fixtures, never frozen test."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

from fastapi.testclient import TestClient

from climate_rag.bm25 import BM25Index
from climate_rag.evidence_packet import build_evidence_packet
from climate_rag.io import iter_evidence, write_json
from climate_rag.pipeline import HybridRetriever
from climate_rag.service import create_app

CASES = (
    ("lexical_evidence", "human carbon dioxide emissions warm the climate system"),
    ("year_constraint", "Arctic sea-ice extent declined from 1979 to 2020"),
    ("absent_evidence", "Martian volcanism xenolith zirconium"),
)


def run_demo(corpus: Path) -> dict[str, Any]:
    documents = {row.evidence_id: row for row in iter_evidence(corpus)}
    corpus_sha = hashlib.sha256(corpus.read_bytes()).hexdigest()
    app = create_app(HybridRetriever(bm25=BM25Index().fit(documents.values())))
    cases = []
    with TestClient(app) as client:
        for case_id, claim in CASES:
            request = {"claim_text": claim, "top_k": 2}
            response = client.post("/api/search", json=request)
            response.raise_for_status()
            search = response.json()
            packet = build_evidence_packet(
                search,
                documents,
                corpus_sha256=corpus_sha,
                corpus_reference="fixtures/evidence.json",
                evidence_track="repository_authored_contract_fixture",
            )
            trace = client.get(f"/api/traces/{search['trace_id']}")
            trace.raise_for_status()
            verification = client.post("/api/verify", json=request)
            verification.raise_for_status()
            cases.append(
                {
                    "case_id": case_id,
                    "request": request,
                    "packet": packet,
                    "separate_verify_response": verification.json()["verification"],
                    "trace_retrievable": trace.json()["trace_id"] == search["trace_id"],
                }
            )
    return {
        "scope": "in_process_API_contract_demo_not_quality_or_latency_evaluation",
        "corpus_sha256": corpus_sha,
        "backend": "BM25 fixture; no dense model, LTR, reranker or LLM loaded",
        "cases": cases,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    result = run_demo(Path(__file__).resolve().parents[1] / "fixtures/evidence.json")
    if args.output:
        write_json(args.output, result)
    else:
        print(json.dumps(result, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
