import json
from pathlib import Path

from fastapi.testclient import TestClient

from climate_rag.bm25 import BM25Index
from climate_rag.climate_fever import (
    ClimateFeverAnnotation,
    ClimateFeverRecord,
    audit_split_leakage,
    benchmark_public_bm25,
    grouped_split,
    prepare_public_benchmark,
    validate_split,
)
from climate_rag.models import EvidenceDocument, RankedDocument
from climate_rag.pipeline import HybridRetriever
from climate_rag.service import create_app
from climate_rag.verification import (
    Citation,
    FixedFixtureVerifier,
    VerificationResult,
    validate_verification_result,
)
from climate_rag.verification_metrics import citation_metrics, verification_metrics


def _record(claim_id: str, claim: str, evidence_id: str) -> ClimateFeverRecord:
    return ClimateFeverRecord(
        claim_id=claim_id,
        claim=claim,
        label="SUPPORTS",
        evidences=(
            ClimateFeverAnnotation(
                evidence_id=evidence_id,
                evidence_label="SUPPORTS",
                article="Article",
                text=f"Evidence for {claim}",
            ),
        ),
    )


def test_grouped_split_keeps_shared_and_near_duplicates_together() -> None:
    records = [
        _record("1", "Global warming raises sea levels", "e1"),
        _record("2", "Global warming raises sea level", "e2"),
        ClimateFeverRecord(
            claim_id="3",
            claim="Arctic ice is declining",
            label="SUPPORTS",
            evidences=(
                ClimateFeverAnnotation(
                    "e1",
                    "SUPPORTS",
                    "Article",
                    "Evidence for Global warming raises sea levels",
                ),
            ),
        ),
        _record("4", "Ocean heat content is increasing", "e4"),
        _record("5", "Methane traps heat in the atmosphere", "e5"),
        _record("6", "Carbon dioxide traps heat in the atmosphere", "e6"),
    ]
    split = grouped_split(records, seed=20260825, near_duplicate_threshold=0.70)
    validate_split(records, split)
    owner = {
        claim_id: name for name, claim_ids in split.items() for claim_id in claim_ids
    }
    assert owner["1"] == owner["2"] == owner["3"]


def test_grouped_split_keeps_evidence_document_variants_together() -> None:
    records = [
        ClimateFeverRecord(
            claim_id="1",
            claim="First unrelated claim",
            label="SUPPORTS",
            evidences=(
                ClimateFeverAnnotation(
                    "A:1", "SUPPORTS", "A", "One two three four five six."
                ),
            ),
        ),
        ClimateFeverRecord(
            claim_id="2",
            claim="Second unrelated claim",
            label="REFUTES",
            evidences=(
                ClimateFeverAnnotation(
                    "B:1", "REFUTES", "B", "One two three four five six seven."
                ),
            ),
        ),
        _record("3", "A third unrelated claim", "C:1"),
    ]
    split = grouped_split(records, seed=7, near_duplicate_threshold=0.8)
    owner = {
        claim_id: split_name
        for split_name, claim_ids in split.items()
        for claim_id in claim_ids
    }
    assert owner["1"] == owner["2"]
    assert audit_split_leakage(
        records,
        split,
        claim_similarity_threshold=0.8,
        evidence_similarity_threshold=0.8,
    )["status"] == "passed"


def test_split_audit_rejects_cross_partition_document_variants() -> None:
    records = [
        ClimateFeverRecord(
            claim_id="1",
            claim="First distinct claim",
            label="SUPPORTS",
            evidences=(
                ClimateFeverAnnotation(
                    "A:1", "SUPPORTS", "A", "Same evidence text."
                ),
            ),
        ),
        ClimateFeverRecord(
            claim_id="2",
            claim="Second distinct claim",
            label="REFUTES",
            evidences=(
                ClimateFeverAnnotation(
                    "A:2", "REFUTES", "A", "Same, evidence text!"
                ),
            ),
        ),
    ]
    audit = audit_split_leakage(
        records,
        {"train": ["1"], "validation": ["2"], "test": []},
    )
    assert audit["status"] == "failed"
    assert audit["normalised_evidence_text_cross_split"] == 1
    assert audit["supervised_relevance_status"] == "failed"


def test_split_audit_separates_non_decisive_document_variants() -> None:
    records = [
        ClimateFeverRecord(
            claim_id="1",
            claim="First distinct claim",
            label="NOT_ENOUGH_INFO",
            evidences=(
                ClimateFeverAnnotation(
                    "A:1", "NOT_ENOUGH_INFO", "A", "Nearly same evidence text here."
                ),
            ),
        ),
        ClimateFeverRecord(
            claim_id="2",
            claim="Second distinct claim",
            label="NOT_ENOUGH_INFO",
            evidences=(
                ClimateFeverAnnotation(
                    "A:2", "NOT_ENOUGH_INFO", "A", "Nearly same evidence text, here!"
                ),
            ),
        ),
    ]
    audit = audit_split_leakage(
        records,
        {"train": ["1"], "validation": [], "test": ["2"]},
        evidence_similarity_threshold=0.8,
    )
    assert audit["status"] == "failed"
    assert audit["near_duplicate_evidence_pairs_cross_split"] == 1
    assert audit["near_duplicate_decisive_evidence_pairs_cross_split"] == 0
    assert audit["supervised_relevance_status"] == "passed"


def test_prepare_and_benchmark_public_fixture(tmp_path: Path) -> None:
    source = tmp_path / "climate-fever.jsonl"
    rows = []
    for index in range(30):
        rows.append(
            {
                "claim_id": str(index),
                "claim": f"Climate signal number {index} is observed",
                "claim_label": "SUPPORTS",
                "evidences": [
                    {
                        "evidence_id": f"Article:{index}",
                        "evidence_label": "SUPPORTS",
                        "article": "Article",
                        "evidence": f"Climate signal number {index} is observed in records.",
                    },
                    {
                        "evidence_id": f"Background:{index}",
                        "evidence_label": "NOT_ENOUGH_INFO",
                        "article": "Background",
                        "evidence": f"Background context {index}.",
                    },
                ],
            }
        )
    source.write_text("\n".join(json.dumps(row) for row in rows), encoding="utf-8")
    prepared = tmp_path / "prepared"
    manifest = prepare_public_benchmark(source, prepared)
    assert manifest["claim_count"] == 30
    assert manifest["annotated_pair_count"] == 60
    assert sum(len(values) for values in manifest["split"].values()) == 30
    output = tmp_path / "benchmark"
    metrics = benchmark_public_bm25(prepared, output, split_name="test")
    assert metrics["claim_count"] > 0
    assert metrics["recall@5"] == 1.0
    assert metrics["fact_boundary"].startswith("Public external retrieval baseline")


def test_grounded_service_search_verify_trace_and_metrics() -> None:
    documents = [
        EvidenceDocument("e1", "Human carbon dioxide emissions warm the climate."),
        EvidenceDocument("e2", "Volcanic aerosols can cool the climate temporarily."),
    ]
    retriever = HybridRetriever(bm25=BM25Index().fit(documents))
    client = TestClient(create_app(retriever, verifier=FixedFixtureVerifier()))
    search = client.post(
        "/api/search",
        json={"claim_text": "human carbon dioxide emissions warm climate", "top_k": 2},
    )
    assert search.status_code == 200
    search_payload = search.json()
    assert search_payload["evidence"][0]["evidence_id"] == "e1"
    assert search_payload["query_budget"] == 2

    response = client.post(
        "/api/verify",
        json={"claim_text": "human carbon dioxide emissions warm climate", "top_k": 2},
    )
    assert response.status_code == 200
    payload = response.json()
    assert payload["verification"]["label"] == "SUPPORTS"
    assert payload["verification"]["citations"][0]["evidence_id"] == "e1"
    trace = client.get(f"/api/traces/{payload['trace_id']}")
    assert trace.status_code == 200
    assert trace.json()["operation"] == "verify"
    metrics = client.get("/metrics")
    assert metrics.status_code == 200
    assert "climate_rag_requests_total" in metrics.text


def test_invalid_citation_forces_abstention() -> None:
    result = VerificationResult(
        label="SUPPORTS",
        confidence=0.99,
        citations=(Citation("missing", "invented quote"),),
        rationale="unsupported",
        provider="test",
    )
    validated, warnings = validate_verification_result(
        result, [RankedDocument("e1", 1.0, 1, "Grounded evidence", "bm25")]
    )
    assert validated.label == "NOT_ENOUGH_INFO"
    assert "verdict_without_valid_citation" in warnings


def test_verification_and_citation_metrics_contracts() -> None:
    metrics = verification_metrics(
        ["SUPPORTS", "REFUTES", "NOT_ENOUGH_INFO"],
        ["SUPPORTS", "NOT_ENOUGH_INFO", "NOT_ENOUGH_INFO"],
        [0.9, 0.6, 0.8],
    )
    assert metrics["accuracy"] == 2 / 3
    citations = citation_metrics(
        {"c1": {"e1"}, "c2": {"e2"}}, {"c1": {"e1"}, "c2": {"e3"}}
    )
    assert citations["citation_precision"] == 0.5
    assert citations["citation_recall"] == 0.5
