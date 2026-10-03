import hashlib
from pathlib import Path

import pytest

pytest.importorskip("langchain_core")

from langchain_core.documents import Document  # noqa: E402
from pydantic import ValidationError  # noqa: E402

from climate_rag.bm25 import BM25Index  # noqa: E402
from climate_rag.io import iter_evidence  # noqa: E402
from climate_rag.langchain_evidence import (  # noqa: E402
    ClimateEvidenceRetriever,
    create_evidence_chain,
    create_evidence_tool,
)
from climate_rag.pipeline import HybridRetriever  # noqa: E402


@pytest.fixture
def retriever():
    corpus = Path(__file__).resolve().parents[1] / "fixtures/evidence.json"
    documents = {row.evidence_id: row for row in iter_evidence(corpus)}
    return ClimateEvidenceRetriever(
        pipeline=HybridRetriever(bm25=BM25Index().fit(documents.values())),
        documents=documents,
        corpus_sha256=hashlib.sha256(corpus.read_bytes()).hexdigest(),
        corpus_reference="fixtures/evidence.json",
        evidence_track="fixture",
    )


def test_retriever_chain_and_structured_tool(retriever):
    query = "human carbon dioxide emissions"
    documents = retriever.invoke(query)
    assert isinstance(documents[0], Document)
    assert documents[0].metadata["evidence_id"] == "e1"
    assert documents[0].id == documents[0].metadata["citation_id"]
    chain = create_evidence_chain(retriever)
    tool = create_evidence_tool(retriever)
    from_chain = chain.invoke({"claim_text": query})
    from_tool = tool.invoke({"claim_text": query})
    assert from_chain["items"] == from_tool["items"]
    assert from_tool["query_processing"]["queries"] == [query]
    assert from_tool["query_processing"]["query_budget"] == 1
    assert from_tool["verdict"]["label"] is None and from_tool["answer"] is None
    assert tool.get_input_schema().model_json_schema()["additionalProperties"] is False


@pytest.mark.parametrize(
    "payload",
    [
        {"claim_text": " "},
        {"claim_text": "x", "sql": "SELECT *"},
        {"claim_text": 100},
        {"claim_text": "x" * 10_001},
    ],
)
def test_tool_rejects_invalid_input(retriever, payload):
    with pytest.raises(ValidationError):
        create_evidence_tool(retriever).invoke(payload)


def test_no_results_are_not_a_negative_verdict(retriever):
    packet = create_evidence_tool(retriever).invoke({"claim_text": "zxqv_unknown"})
    assert packet["status"] == "no_evidence"
    assert packet["items"] == [] and packet["verdict"]["label"] is None


def test_width_contract_and_source_identity(retriever):
    with pytest.raises(ValueError, match="final_k"):
        retriever.model_copy(update={"final_k": 200}).invoke("human carbon dioxide")
    retriever.documents.pop("e1")
    with pytest.raises(ValueError, match="unknown"):
        retriever.invoke("human carbon dioxide")
