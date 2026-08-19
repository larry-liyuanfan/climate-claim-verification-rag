from pathlib import Path

import numpy as np
import pytest

from climate_rag.dense import DenseRetriever, FaissANNIndex, HashDenseEncoder, NumpyFlatIndex
from climate_rag.fusion import LightGBMLambdaMART, LinearPairwiseLTR, reciprocal_rank_fusion
from climate_rag.io import iter_evidence
from climate_rag.models import RankedDocument
from climate_rag.negatives import mine_hard_negatives


FIXTURES = Path(__file__).parents[1] / "fixtures"


def test_hash_dense_retriever_round_trip(tmp_path: Path) -> None:
    documents = list(iter_evidence(FIXTURES / "evidence.json"))
    retriever = DenseRetriever(HashDenseEncoder(128), NumpyFlatIndex()).fit(documents)
    assert retriever.search("human carbon dioxide emissions", 1)[0].evidence_id == "e1"
    target = tmp_path / "dense"
    retriever.save(target)
    restored = DenseRetriever.load(target)
    assert restored.search("human carbon dioxide emissions", 2) == retriever.search(
        "human carbon dioxide emissions", 2
    )


def test_faiss_adapter_has_explicit_optional_dependency() -> None:
    try:
        index = FaissANNIndex(8, kind="flat")
    except RuntimeError as exc:
        assert "faiss" in str(exc).lower()
    else:
        index.build(np.eye(8, dtype=np.float32))
        scores, indices = index.search(np.eye(8, dtype=np.float32)[:1], 1)
        assert scores.shape == indices.shape == (1, 1)


def test_rrf_is_deterministic_and_deduplicates() -> None:
    rankings = {
        "bm25": [RankedDocument("b", 4, 1), RankedDocument("a", 3, 2)],
        "dense": [RankedDocument("a", 0.9, 1), RankedDocument("b", 0.8, 2)],
    }
    rows = reciprocal_rank_fusion(rankings, k=60)
    assert [row.evidence_id for row in rows] == ["a", "b"]
    assert len({row.evidence_id for row in rows}) == 2


def test_linear_pairwise_ltr_learns_order_and_persists(tmp_path: Path) -> None:
    features = np.asarray([[3.0, 1.0], [0.0, 0.0], [2.0, 1.0], [-1.0, 0.0]])
    labels = np.asarray([2, 0, 2, 0])
    groups = ["q1", "q1", "q2", "q2"]
    model = LinearPairwiseLTR(("quality", "overlap"), epochs=200)
    model.fit(features, labels, groups)
    scores = model.predict(features)
    assert scores[0] > scores[1]
    assert scores[2] > scores[3]
    path = tmp_path / "ltr.json"
    model.save(path)
    restored = LinearPairwiseLTR.load(path)
    np.testing.assert_allclose(restored.predict(features), scores)


def test_ltr_rejects_groups_without_preference_pairs() -> None:
    model = LinearPairwiseLTR(("x",))
    with pytest.raises(ValueError, match="unequal-label"):
        model.fit(np.asarray([[1.0], [2.0]]), np.asarray([0, 0]), ["q", "q"])


def test_lightgbm_ltr_round_trip_preserves_feature_count(tmp_path: Path) -> None:
    pytest.importorskip("lightgbm")
    features = np.asarray([[3.0, 1.0], [0.0, 0.0], [2.0, 1.0], [-1.0, 0.0]])
    labels = np.asarray([2, 0, 2, 0])
    groups = ["q1", "q1", "q2", "q2"]
    model = LightGBMLambdaMART(("quality", "overlap"), seed=17)
    model.fit(features, labels, groups)
    expected = model.predict(features)
    path = tmp_path / "ltr_model.txt"
    model.save(path)

    restored = LightGBMLambdaMART.load(path)
    np.testing.assert_allclose(restored.predict(features), expected)
    with pytest.raises(ValueError, match="feature matrix shape"):
        restored.predict(np.ones((1, 3)))


def test_hard_negative_mining_excludes_gold_and_tracks_sources() -> None:
    rankings = {
        "bm25": [RankedDocument("gold", 3, 1), RankedDocument("n1", 2, 2)],
        "dense": [RankedDocument("n1", 1, 1), RankedDocument("n2", 0.5, 2)],
    }
    rows = mine_hard_negatives(rankings, ["gold"], limit=2)
    assert [row["evidence_id"] for row in rows] == ["n1", "n2"]
    assert rows[0]["sources"] == {"bm25": 2, "dense": 1}

