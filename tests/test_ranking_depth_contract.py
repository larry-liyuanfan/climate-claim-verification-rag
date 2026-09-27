from __future__ import annotations

import math

import numpy as np
import pytest

from climate_rag.metrics import evaluate_predictions, per_claim_retrieval_metrics
from climate_rag.models import Claim, EvidenceDocument, Prediction
from climate_rag.public_v2_runtime import predictions_from_rows, score_dense_index
from climate_rag.representation_eval import evaluate_representation_pair


@pytest.mark.parametrize("rank", [6, 10, 11, 50, 51])
def test_ranking_depth_is_not_evidence_depth(rank: int) -> None:
    ranked = tuple(str(i) for i in range(1, 101))
    metric = per_claim_retrieval_metrics(
        Claim("q", "temperature", evidence_ids=(str(rank),)),
        Prediction("q", ranked), evidence_k=5,
    )
    assert metric["recall@5"] == 0
    assert metric["recall@10"] == int(rank <= 10)
    assert metric["recall@50"] == int(rank <= 50)
    assert metric["mrr@10"] == (1 / rank if rank <= 10 else 0)
    assert metric["ndcg@10"] == (1 / math.log2(rank + 1) if rank <= 10 else 0)
    assert metric["evidence_f1@5"] == metric["evidence_f1"] == 0


def test_f1_cutoff_preserves_legacy_and_does_not_depend_on_tail() -> None:
    claim = Claim("q", "temperature", evidence_ids=("1", "6"))
    for width in (50, 100):
        prediction = Prediction("q", tuple(str(i) for i in range(1, width + 1)))
        measured = per_claim_retrieval_metrics(claim, prediction, evidence_k=5)
        assert measured["evidence_f1@5"] == pytest.approx(2 / 7)
        assert measured["recall@10"] == 1
        assert per_claim_retrieval_metrics(claim, prediction)["evidence_f1"] == pytest.approx(
            4 / (width + 2)
        )
    for ids, expected in (((), 0), (("1",), 2 / 3)):
        measured = per_claim_retrieval_metrics(claim, Prediction("q", ids), evidence_k=5)
        assert measured["evidence_f1@5"] == pytest.approx(expected)


def test_public_full_ranking_survives_rows_pair_and_taxonomy() -> None:
    class Encoder:
        def encode_queries(self, texts, batch_size):
            return np.ones((len(texts), 2), dtype=np.float32)

    class Index:
        def search(self, vectors, width):
            return np.arange(width, 0, -1)[None, :], np.arange(width)[None, :]

    documents = [EvidenceDocument(str(i), "temperature evidence") for i in range(100)]
    claims = {"q": Claim("q", "temperature", label="SUPPORTS", evidence_ids=("5",))}
    metrics, _, rows = score_dense_index(
        Encoder(), Index(), documents, claims, batch_size=1, search_width=100
    )
    predictions = predictions_from_rows(rows)
    assert len(predictions["q"].evidence_ids) == 100
    assert metrics["mrr@10"] == pytest.approx(1 / 6)
    assert metrics["evidence_f1@5"] == 0
    assert "claim_accuracy" not in metrics
    report, tagged = evaluate_representation_pair(
        claims, documents, predictions, predictions, evidence_k=5
    )
    assert report["baseline"]["recall@10"] == 1
    assert report["paired_bootstrap"]["mrr@10"]["candidate_mean"] == pytest.approx(1 / 6)
    assert report["paired_bootstrap"]["evidence_f1@5"]["mean_difference"] == 0
    assert tagged[0]["candidate"]["evidence_f1@5"] == 0
    _, _, errors = evaluate_predictions(claims, predictions, evidence_k=5, evaluate_labels=False)
    assert errors[0]["categories"] == ["retrieval_miss"]
    with pytest.raises(ValueError, match="Recall@50"):
        score_dense_index(Encoder(), Index(), documents, claims, batch_size=1, search_width=5)
