"""Run real LangChain retrieval on an explicitly supplied public evidence corpus."""

from __future__ import annotations

import argparse
import hashlib
from importlib.metadata import version
from pathlib import Path

from langsmith import tracing_context

from climate_rag.bm25 import BM25Index
from climate_rag.io import iter_evidence, write_json
from climate_rag.langchain_evidence import (
    ClimateEvidenceRetriever,
    create_evidence_chain,
    create_evidence_tool,
)
from climate_rag.pipeline import HybridRetriever


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--evidence", required=True, type=Path)
    parser.add_argument("--expected-sha256", required=True)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    digest = hashlib.sha256(args.evidence.read_bytes()).hexdigest()
    if digest != args.expected_sha256:
        raise ValueError("evidence SHA mismatch")
    documents = {row.evidence_id: row for row in iter_evidence(args.evidence)}
    retriever = ClimateEvidenceRetriever(
        pipeline=HybridRetriever(bm25=BM25Index().fit(documents.values())),
        documents=documents,
        corpus_sha256=digest,
        corpus_reference="CLIMATE-FEVER exported evidence.jsonl",
        evidence_track="public_corpus_manual_demo_not_split_evaluation",
        final_k=3,
    )
    chain = create_evidence_chain(retriever)
    tool = create_evidence_tool(retriever)
    # Authored demo queries, not loaded from any claim split or benchmark labels.
    queries = [
        "Arctic sea ice decline",
        "carbon dioxide greenhouse effect",
        "zxqv_nonexistent_term_20260927",
    ]
    with tracing_context(enabled=False):
        chain_packet = chain.invoke({"claim_text": queries[0]})
        cases = [tool.invoke({"claim_text": query}) for query in queries]
    assert [item["citation_id"] for item in chain_packet["items"]] == [
        item["citation_id"] for item in cases[0]["items"]
    ]
    if not cases[0]["items"] or not cases[1]["items"] or cases[2]["items"]:
        raise AssertionError("expected two populated packets and an empty OOV packet")
    write_json(
        args.output,
        {
            "langchain_core_version": version("langchain-core"),
            "corpus_sha256": digest,
            "document_count": len(documents),
            "mode": "BM25_only_small_public_corpus_CPU",
            "scope": "manual_application_demo_not_retrieval_quality_or_verdict_evaluation",
            "telemetry": "LangSmith tracing disabled; no LLM or external model call",
            "tool_schema": tool.get_input_schema().model_json_schema(),
            "chain_tool_first_case_agree": True,
            "cases": cases,
        },
    )
    print(
        "3 tool queries + 1 chain consistency call passed; "
        f"docs={len(documents)}; corpus_sha256={digest}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
