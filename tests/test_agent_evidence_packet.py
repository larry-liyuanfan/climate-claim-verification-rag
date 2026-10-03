from __future__ import annotations

import copy
import hashlib
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from climate_rag.bm25 import BM25Index
from climate_rag.evidence_packet import build_evidence_packet
from climate_rag.io import iter_evidence
from climate_rag.models import EvidenceDocument
from climate_rag.pipeline import HybridRetriever
from climate_rag.service import create_app

CORPUS = Path(__file__).resolve().parents[1] / "fixtures/evidence.json"


@pytest.fixture
def packet_inputs():
    documents = {row.evidence_id: row for row in iter_evidence(CORPUS)}
    with TestClient(
        create_app(HybridRetriever(bm25=BM25Index().fit(documents.values())))
    ) as client:
        search = client.post(
            "/api/search",
            json={"claim_text": "human carbon dioxide emissions", "top_k": 2},
        ).json()
    kwargs = {
        "corpus_sha256": hashlib.sha256(CORPUS.read_bytes()).hexdigest(),
        "corpus_reference": "fixtures/evidence.json",
        "evidence_track": "fixture",
    }
    return search, documents, kwargs


def test_packet_binds_source_and_does_not_fabricate_verdict(packet_inputs):
    search, documents, kwargs = packet_inputs
    packet = build_evidence_packet(search, documents, **kwargs)
    item = packet["items"][0]
    assert item["evidence_id"] == "e1"
    assert item["source"]["url"] is None
    assert item["retrieval"]["route"] == "bm25"
    assert item["span"]["end"] == len(documents["e1"].text)
    assert (
        item["text_sha256"] == hashlib.sha256(documents["e1"].text.encode()).hexdigest()
    )
    assert kwargs["corpus_sha256"] in item["citation_id"]
    assert packet["answer"] is None and packet["verdict"]["label"] is None


@pytest.mark.parametrize("mutation", ["unknown", "duplicate", "text", "url", "digest"])
def test_packet_rejects_provenance_mismatch(packet_inputs, mutation):
    search, documents, kwargs = copy.deepcopy(packet_inputs)
    if mutation == "unknown":
        search["evidence"][0]["evidence_id"] = "unknown"
    elif mutation == "duplicate":
        search["evidence"].append(search["evidence"][0])
    elif mutation == "text":
        search["evidence"][0]["text"] = "invented text"
    elif mutation == "url":
        original = documents["e1"]
        documents["e1"] = EvidenceDocument(
            "e1", original.text, {"url": "javascript:bad"}
        )
    else:
        kwargs["corpus_sha256"] = "not-a-hash"
    with pytest.raises(ValueError):
        build_evidence_packet(search, documents, **kwargs)


def test_empty_packet_has_no_answer(packet_inputs):
    search, documents, kwargs = packet_inputs
    search["evidence"] = []
    packet = build_evidence_packet(search, documents, **kwargs)
    assert packet["status"] == "no_evidence"
    assert packet["items"] == [] and packet["answer"] is None


def test_three_case_cli_demo(tmp_path):
    import json
    import os
    import subprocess
    import sys

    root = CORPUS.parents[1]
    output = tmp_path / "demo.json"
    subprocess.run(
        [
            sys.executable,
            str(root / "scripts/demo_agent_evidence.py"),
            "--output",
            str(output),
        ],
        cwd=root,
        env={**os.environ, "PYTHONPATH": str(root / "src")},
        check=True,
    )
    report = json.loads(output.read_text(encoding="utf-8"))
    lexical, year, absent = report["cases"]
    assert lexical["packet"]["items"][0]["evidence_id"] == "e1"
    assert year["packet"]["items"][0]["evidence_id"] == "e2"
    assert year["packet"]["query_processing"]["constraints"]["years"] == [
        "1979",
        "2020",
    ]
    assert absent["packet"]["status"] == "no_evidence"
    assert all(case["trace_retrievable"] for case in report["cases"])
    assert (
        lexical["separate_verify_response"]["abstain_reason"]
        == "verifier_not_configured"
    )
    assert (
        absent["separate_verify_response"]["abstain_reason"]
        == "insufficient_evidence_coverage"
    )
