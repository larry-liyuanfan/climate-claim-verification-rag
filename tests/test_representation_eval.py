from __future__ import annotations

from climate_rag.models import Claim, EvidenceDocument, Prediction
from climate_rag.representation_eval import (
    build_pareto_report,
    evaluate_representation_pair,
    infer_query_taxonomy,
)


def test_representation_pair_reports_bootstrap_and_taxonomy() -> None:
    claims = {
        "q1": Claim("q1", "Australia warmed by 1 degree in 2020", evidence_ids=("e1",)),
        "q2": Claim("q2", "IPCC finds human influence", evidence_ids=("e2", "e3")),
        "q3": Claim("q3", "An unanswerable climate claim", evidence_ids=()),
    }
    evidence = [
        EvidenceDocument("e1", "In 2020 Australia was one degree warmer."),
        EvidenceDocument("e2", "Human influence has warmed the atmosphere."),
        EvidenceDocument("e3", "Human influence has warmed the ocean and land."),
        EvidenceDocument("noise", "Unrelated text."),
    ]
    baseline = {
        "q1": Prediction("q1", ("noise", "e1")),
        "q2": Prediction("q2", ("noise", "e2")),
        "q3": Prediction("q3", ("noise",)),
    }
    candidate = {
        "q1": Prediction("q1", ("e1", "noise")),
        "q2": Prediction("q2", ("e2", "e3")),
        "q3": Prediction("q3", ("noise",)),
    }
    metrics, rows = evaluate_representation_pair(
        claims, evidence, baseline, candidate, bootstrap_samples=5_000
    )
    assert metrics["bootstrap_samples"] == 5_000
    assert metrics["candidate"]["mrr@10"] > metrics["baseline"]["mrr@10"]
    assert metrics["taxonomy"]["multi_evidence"]["query_count"] == 1
    assert metrics["taxonomy"]["unanswerable"]["query_count"] == 1
    assert len(rows) == 3


def test_query_taxonomy_is_explicitly_multilabel() -> None:
    claim = Claim(
        "q",
        "Australia recorded 42 events in 2020",
        evidence_ids=("e1", "e2"),
    )
    labels = infer_query_taxonomy(
        claim,
        {
            "e1": EvidenceDocument("e1", "Completely disjoint terminology."),
            "e2": EvidenceDocument("e2", "Another unrelated sentence."),
        },
    )
    assert "year_or_numeric" in labels
    assert "geographic" in labels
    assert "multi_evidence" in labels
    assert "lexical_mismatch" in labels
    assert "semantic_paraphrase" in labels


def test_query_taxonomy_marks_spelling_variants_against_gold_text() -> None:
    labels = infer_query_taxonomy(
        Claim("q", "Global temprature is rising", evidence_ids=("e1",)),
        {"e1": EvidenceDocument("e1", "Global temperature has risen.")},
    )
    assert "spelling" in labels


def test_pareto_never_compares_different_measurement_scopes() -> None:
    report = build_pareto_report(
        [
            {
                "name": "fast",
                "comparability_group": "ann-only",
                "evidence_f1": 0.1,
                "p95_ms": 1,
                "memory_bytes": 10,
            },
            {
                "name": "quality",
                "comparability_group": "end-to-end",
                "evidence_f1": 0.9,
                "p95_ms": 100,
                "memory_bytes": 100,
            },
        ]
    )
    assert [profile["pareto_status"] for profile in report["profiles"]] == [
        "frontier",
        "frontier",
    ]
