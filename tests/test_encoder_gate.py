from climate_rag.encoder_gate import (
    compare_metric_rows,
    evidence_preserving_reservoir_sample,
    screening_decision,
)
from climate_rag.models import EvidenceDocument


def test_evidence_preserving_sample_is_deterministic_and_keeps_gold() -> None:
    documents = [EvidenceDocument(str(index), f"text {index}") for index in range(20)]
    first = evidence_preserving_reservoir_sample(
        documents, {"3", "17"}, sample_size=8, seed=19
    )
    second = evidence_preserving_reservoir_sample(
        documents, {"3", "17"}, sample_size=8, seed=19
    )
    assert first == second
    assert len(first.documents) == 8
    assert {"3", "17"} <= {row.evidence_id for row in first.documents}
    assert first.source_document_count == 20


def test_dense_screen_requires_positive_recall_interval() -> None:
    baseline = [
        {"claim_id": f"c{i}", "recall@5": 0.0, "mrr@10": 0.0, "ndcg@10": 0.0, "evidence_f1": 0.0}
        for i in range(8)
    ]
    candidate = [
        {"claim_id": f"c{i}", "recall@5": 1.0, "mrr@10": 1.0, "ndcg@10": 1.0, "evidence_f1": 1.0}
        for i in range(8)
    ]
    comparisons = compare_metric_rows(baseline, candidate, samples=200, seed=3)
    assert screening_decision(comparisons)["candidate_passes_sample_screen"] is True
