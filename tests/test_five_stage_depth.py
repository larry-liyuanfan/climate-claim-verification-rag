"""Exercise the actual legacy entry point, not only the standalone scorer."""
import json
from pathlib import Path

import numpy as np
import pytest

from climate_rag import benchmark
from climate_rag.cli import main
from climate_rag.fusion import DEFAULT_FEATURES
from climate_rag.io import read_json, write_json
from climate_rag.models import RankedDocument


def test_five_stage_separates_rankings_from_served_evidence(tmp_path, monkeypatch):
    class FixedRetriever:
        def search(self, query, top_k):
            return [RankedDocument(f"d{i:02d}", 100.0 - i, i, "climate evidence")
                    for i in range(1, 61)][:top_k]

    class FixedRanker:
        feature_names = DEFAULT_FEATURES

        def predict(self, matrix):
            return np.zeros(len(matrix))

    class IdentityReranker:
        name = "identity-fixture-not-model"

        def rerank(self, query, candidates, top_k):
            return candidates[:top_k]

    monkeypatch.setattr(benchmark.BM25Index, "load", lambda *a, **kw: FixedRetriever())
    monkeypatch.setattr(benchmark.DenseRetriever, "load", lambda *a, **kw: FixedRetriever())
    monkeypatch.setattr(benchmark, "load_ranker", lambda *a, **kw: FixedRanker())
    monkeypatch.setattr(benchmark, "_make_reranker", lambda *a, **kw: IdentityReranker())
    claims = tmp_path / "claims.json"
    write_json(claims, {"q": {"claim_text": "climate", "claim_label": "SUPPORTS",
                             "evidences": ["d06", "d10", "d50"]}})
    config = tmp_path / "config.json"
    write_json(config, {
        "bm25_index": "fixture", "dense_index": "fixture", "ltr_model": "fixture",
        "recall_k": 60, "fusion_k": 60, "rerank_k": 60, "final_k": 5,
        "rerank_source": "rrf", "rerank_fusion": {"enabled": True},
        "ltr_fusion": {"enabled": True},
    })
    output = tmp_path / "benchmark"
    result, rows = benchmark.run_five_stage_benchmark(
        claims_path=claims, config_path=config, output_dir=output, bootstrap_samples=100,
    )
    assert result["ranking_k"] == 50
    assert result["final_k"] == result["metric_contract"]["evidence_k"] == 5
    for stage, metrics in result["systems"].items():
        assert metrics["recall@5"] == metrics["evidence_f1@5"] == 0
        assert metrics["recall@10"] == pytest.approx(2 / 3)
        assert metrics["recall@50"] == 1
        assert metrics["mrr@10"] == pytest.approx(1 / 6)
        assert "claim_accuracy" not in metrics and "macro_f1" not in metrics
        assert len(read_json(output / f"rankings_{stage}.json")["q"]["evidences"]) == 50
        assert len(read_json(output / f"predictions_{stage}.json")["q"]["evidences"]) == 5
    assert all(len(row["predicted_evidence_ids"]) == 50 for row in rows)
    # Reproduction CLI must use full ranks and the separately specified cutoff.
    reproduced = tmp_path / "reproduced"
    assert main([
        "evaluate", "--claims", str(claims), "--predictions", str(output / "rankings_rrf.json"),
        "--evidence-k", "5", "--retrieval-only", "--output-dir", str(reproduced),
    ]) == 0
    metrics = json.loads((reproduced / "metrics.json").read_text(encoding="utf-8"))
    assert metrics == result["systems"]["rrf"]


def test_retired_component_percentile_proxy_cannot_generate_new_pareto(tmp_path):
    retired = Path(__file__).parents[1] / "configs" / "search_profiles.verified.json"
    with pytest.raises(SystemExit) as error:
        main(["build-pareto", "--profiles", str(retired), "--output-dir", str(tmp_path / "report")])
    assert error.value.code == 2
    assert not (tmp_path / "report").exists()


def test_five_stage_rejects_silently_ignored_cli_evidence_cutoff(tmp_path, capsys):
    with pytest.raises(SystemExit) as error:
        main(["evaluate", "--claims", "unused.json", "--experiment-config", "unused.json",
              "--evidence-k", "10", "--output-dir", str(tmp_path / "report")])
    assert error.value.code == 2
    assert "five-stage configs use final_k" in capsys.readouterr().err
    assert not (tmp_path / "report").exists()
