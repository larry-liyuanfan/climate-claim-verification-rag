"""Real LangChain retriever/LCEL/tool integration, with no implicit LLM call."""

from __future__ import annotations

import uuid
from typing import Any

from langchain_core.callbacks import CallbackManagerForRetrieverRun
from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever
from langchain_core.runnables import (
    RunnableLambda,
    RunnablePassthrough,
    RunnableSerializable,
)
from langchain_core.tools import StructuredTool
from pydantic import BaseModel, ConfigDict, Field

from .evidence_packet import build_evidence_packet
from .models import EvidenceDocument
from .pipeline import HybridRetriever
from .verification import extract_constraints, normalise_claim


class EvidenceQuery(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)
    claim_text: str = Field(min_length=1, max_length=10_000, pattern=r"\S")


class ClimateEvidenceRetriever(BaseRetriever):
    """Adapt configured BM25/dense/reranker routes to source-bearing Documents."""

    pipeline: HybridRetriever = Field(exclude=True)
    documents: dict[str, EvidenceDocument] = Field(exclude=True)
    corpus_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    corpus_reference: str
    evidence_track: str
    recall_k: int = Field(default=100, ge=1)
    rerank_k: int = Field(default=50, ge=1)
    final_k: int = Field(default=5, ge=1)

    def _get_relevant_documents(
        self, query: str, *, run_manager: CallbackManagerForRetrieverRun
    ) -> list[Document]:
        del run_manager
        claim = normalise_claim(EvidenceQuery(claim_text=query).claim_text)
        if not self.final_k <= self.rerank_k <= self.recall_k:
            raise ValueError("require final_k <= rerank_k <= recall_k")
        rows = self.pipeline.retrieve(
            claim,
            recall_k=self.recall_k,
            rerank_k=self.rerank_k,
            final_k=self.final_k,
        )
        packet = build_evidence_packet(
            {
                "evidence": [row.to_dict() for row in rows],
                "trace_id": str(uuid.uuid4()),
                "claim_text": claim,
                "queries": [claim],
                "constraints": extract_constraints(claim),
                "query_budget": 1,
                "re_retrieval_triggered": False,
            },
            self.documents,
            corpus_sha256=self.corpus_sha256,
            corpus_reference=self.corpus_reference,
            evidence_track=self.evidence_track,
        )
        return [
            Document(id=item["citation_id"], page_content=item["text"], metadata=item)
            for item in packet["items"]
        ]


def create_evidence_chain(
    retriever: ClimateEvidenceRetriever,
) -> RunnableSerializable[dict[str, Any], dict[str, Any]]:
    """Validate → normalize → invoke retriever → package, composed with LCEL."""

    def prepare(value: dict[str, Any]) -> dict[str, Any]:
        request = EvidenceQuery.model_validate(value)
        claim = normalise_claim(request.claim_text)
        return {"claim_text": claim, "constraints": extract_constraints(claim)}

    def package(value: dict[str, Any]) -> dict[str, Any]:
        return {
            "schema_version": "1.0",
            "trace_id": str(uuid.uuid4()),
            "claim_text": value["claim_text"],
            "status": "candidates_only" if value["documents"] else "no_evidence",
            "evidence_track": retriever.evidence_track,
            "items": [doc.metadata for doc in value["documents"]],
            "answer": None,
            "verdict": {"status": "not_evaluated", "label": None},
            "query_processing": {
                "method": "deterministic_whitespace_and_regex",
                "queries": [value["claim_text"]],
                "constraints": value["constraints"],
                "query_budget": 1,
                "re_retrieval_triggered": False,
            },
            "warnings": [
                "Retrieval scores and exact source spans do not prove entailment.",
                "No verdict model or answer generator is invoked by this tool.",
                "No evidence means the caller must abstain or request more evidence.",
            ],
        }

    return (
        RunnableLambda(prepare)
        | RunnablePassthrough.assign(
            documents=RunnableLambda(lambda value: value["claim_text"]) | retriever
        )
        | RunnableLambda(package)
    )


def create_evidence_tool(retriever: ClimateEvidenceRetriever) -> StructuredTool:
    chain = create_evidence_chain(retriever)

    def retrieve_climate_evidence(claim_text: str) -> dict[str, Any]:
        result: dict[str, Any] = chain.invoke({"claim_text": claim_text})
        return result

    return StructuredTool.from_function(
        func=retrieve_climate_evidence,
        name="retrieve_climate_evidence",
        description=(
            "Retrieve versioned climate evidence with citation IDs and exact spans. "
            "Returns candidates, NOT a factual verdict. No evidence requires abstention."
        ),
        args_schema=EvidenceQuery,
    )
