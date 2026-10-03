from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pytest

from climate_rag.bm25 import BM25Index
from climate_rag.dense import DenseRetriever, HashDenseEncoder, NumpyFlatIndex
from climate_rag.fusion import DEFAULT_FEATURES
from climate_rag.models import EvidenceDocument
from climate_rag.rerank import DeterministicFeatureReranker
from climate_rag.search_profile import ProfiledSearch, summarize_request_timings


def test_profiles_keep_candidate_pool_and_measure_complete_request() -> None:
    documents = [EvidenceDocument(str(i), f"climate change evidence {i}") for i in range(120)]
    encoder = HashDenseEncoder(16)
    dense = DenseRetriever(encoder, NumpyFlatIndex()).fit(documents)
    lexical = BM25Index().fit(documents)

    class Ranker:
        feature_names = DEFAULT_FEATURES

        def predict(self, matrix):
            return np.zeros(len(matrix))

    counter = iter(float(i) for i in range(100))
    low = ProfiledSearch(lexical, dense, ranker=Ranker(), clock=lambda: next(counter))
    ranked, trace = low.search("climate evidence")
    high = ProfiledSearch(lexical, dense, reranker=DeterministicFeatureReranker(), rerank_width=20)
    reranked, high_trace = high.search("climate evidence")
    assert len(ranked) == len(reranked) == 100
    assert trace["candidate_ids_sha256"] == high_trace["candidate_ids_sha256"]
    assert set(row.evidence_id for row in ranked) == set(row.evidence_id for row in reranked)
    assert high_trace["reranker_scored_count"] == 20
    assert trace["encoding_ms"] == trace["ranking_ms"] == 1000
    assert trace["end_to_end_ms"] == 11000
    summary = summarize_request_timings([trace])
    assert summary["end_to_end_ms"]["p95"] == 11000
    with pytest.raises(ValueError, match="Top-100"):
        ProfiledSearch(lexical, dense, ranker=Ranker(), candidate_width=50)


def test_metric_publisher_requires_legacy_metrics_and_preserves_new_fields() -> None:
    path = Path(__file__).parents[1] / "scripts" / "public_v2_publish.py"
    spec = importlib.util.spec_from_file_location("public_v2_publish_test", path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    with pytest.raises(KeyError):
        module._metric_summary({})
    original = {name: 0.5 for name in ("recall@5", "mrr@10", "ndcg@10", "evidence_f1")}
    assert module._metric_summary(original) == original
    extended = {**original, "recall@50": 0.8, "evidence_f1@5": 0.5, "evidence_k": 5}
    assert module._metric_summary(extended) == extended


def test_ann_cutoff_is_explicit_in_downstream() -> None:
    # AST checks the actual downstream dict comprehensions, not a duplicate formula.
    import ast
    path = Path(__file__).parents[1] / "scripts" / "public_v2_downstream.py"
    tree = ast.parse(path.read_text(encoding="utf-8"))
    found = set()
    for node in ast.walk(tree):
        if not isinstance(node, ast.Assign) or not isinstance(node.value, ast.DictComp):
            continue
        names = [target.id for target in node.targets if isinstance(target, ast.Name)]
        if not any(name in {"flat_top5", "hnsw_top5"} for name in names):
            continue
        expression = ast.Expression(node.value)
        from climate_rag.models import Prediction
        scope = {
            "dense_prefix": "base_dense",
            "predictions": {
                "base_dense_flat": {"q": Prediction("q", tuple(str(i) for i in range(100)))},
                "base_dense_hnsw": {"q": Prediction("q", tuple(str(i) for i in range(5, 100)) + tuple(str(i) for i in range(5)))},
            },
        }
        values = eval(compile(expression, str(path), "eval"), scope)
        expected = set(map(str, range(5))) if "flat_top5" in names else set(map(str, range(5, 10)))
        assert values["q"] == expected
        found.update(names)
    assert found == {"flat_top5", "hnsw_top5"}


@pytest.mark.parametrize("filename", ["predictions.jsonl", "traces.jsonl"])
def test_profile_summary_rejects_stale_files_and_wrong_query_ids(tmp_path, filename) -> None:
    import hashlib
    import json
    from climate_rag.io import write_jsonl
    from climate_rag.public_v2 import file_sha256

    path = Path(__file__).parents[1] / "scripts" / "summarize_search_profiles.py"
    spec = importlib.util.spec_from_file_location("summarize_search_profiles_test", path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    write_jsonl(tmp_path / "predictions.jsonl", [{"claim_id": "q", "evidence_ids": ["d"]}])
    write_jsonl(tmp_path / "traces.jsonl", [{"claim_id": "q", "end_to_end_ms": 1.0}])
    keys = {"predictions.jsonl": "prediction_sha256", "traces.jsonl": "trace_sha256"}
    summary = {key: file_sha256(tmp_path / name) for name, key in keys.items()}
    summary.update(query_count=1, query_ids_sha256=hashlib.sha256(json.dumps(["q"]).encode()).hexdigest())
    module.validate_profile_artifacts(tmp_path, summary, {"q"})
    write_jsonl(tmp_path / filename, [{"claim_id": "wrong"}])
    with pytest.raises(ValueError, match="artifact hash mismatch"):
        module.validate_profile_artifacts(tmp_path, summary, {"q"})
    summary[keys[filename]] = file_sha256(tmp_path / filename)
    with pytest.raises(ValueError, match="query IDs mismatch"):
        module.validate_profile_artifacts(tmp_path, summary, {"q"})
