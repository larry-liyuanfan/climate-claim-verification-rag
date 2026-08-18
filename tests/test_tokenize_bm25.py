from pathlib import Path

from climate_rag.bm25 import BM25Index
from climate_rag.io import iter_evidence
from climate_rag.tokenize import climate_tokenize


FIXTURES = Path(__file__).parents[1] / "fixtures"


def test_tokenizer_preserves_negation_and_normalizes_co2() -> None:
    tokens = climate_tokenize("CO2 does not warm the air without sunlight")
    assert "carbon_dioxide" in tokens
    assert "not" in tokens
    assert "without" in tokens
    assert "the" not in tokens


def test_bm25_ranks_exact_climate_evidence_first() -> None:
    index = BM25Index().fit(iter_evidence(FIXTURES / "evidence.json"))
    rows = index.search("Human carbon dioxide emissions warm climate", top_k=3)
    assert rows[0].evidence_id == "e1"
    assert [row.rank for row in rows] == [1, 2, 3]


def test_bm25_persistence_round_trip(tmp_path: Path) -> None:
    index = BM25Index().fit(iter_evidence(FIXTURES / "evidence.json"))
    path = tmp_path / "bm25.pkl.gz"
    index.save(path)
    restored = BM25Index.load(path)
    assert restored.search("Arctic sea ice 1979", 2) == index.search("Arctic sea ice 1979", 2)


def test_bm25_empty_and_unknown_queries_are_safe() -> None:
    index = BM25Index().fit(iter_evidence(FIXTURES / "evidence.json"))
    assert index.search("", 5) == []
    assert index.search("unseen_token_xyz", 5) == []
    assert index.search("climate", 0) == []

