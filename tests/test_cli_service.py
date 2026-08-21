import json
from datetime import datetime
from pathlib import Path

from fastapi.testclient import TestClient

from climate_rag.bm25 import BM25Index
from climate_rag.cli import main
from climate_rag.io import iter_evidence
from climate_rag.pipeline import HybridRetriever
from climate_rag.service import create_app

FIXTURES = Path(__file__).parents[1] / "fixtures"


def test_cli_index_evaluate_negatives_and_ltr(tmp_path: Path) -> None:
    index_dir = tmp_path / "index"
    assert main(
        [
            "index",
            "--evidence",
            str(FIXTURES / "evidence.json"),
            "--output-dir",
            str(index_dir),
            "--backend",
            "both",
        ]
    ) == 0
    assert (index_dir / "bm25.pkl.gz").exists()
    assert (index_dir / "dense" / "dense_index.json").exists()
    assert (index_dir / "run_manifest.json").exists()
    manifest = json.loads((index_dir / "run_manifest.json").read_text(encoding="utf-8"))
    assert manifest["finished_at_utc"].endswith("+00:00")
    assert datetime.fromisoformat(manifest["started_at_utc"]) < datetime.fromisoformat(
        manifest["finished_at_utc"]
    )

    evaluation_dir = tmp_path / "evaluation"
    assert main(
        [
            "evaluate",
            "--claims",
            str(FIXTURES / "claims.json"),
            "--predictions",
            str(FIXTURES / "predictions_candidate.json"),
            "--baseline-predictions",
            str(FIXTURES / "predictions_baseline.json"),
            "--bootstrap-samples",
            "100",
            "--output-dir",
            str(evaluation_dir),
        ]
    ) == 0
    metrics = json.loads((evaluation_dir / "metrics.json").read_text(encoding="utf-8"))
    assert metrics["evidence_f1"] == 1.0
    assert "paired_bootstrap" in metrics

    negatives_dir = tmp_path / "negatives"
    assert main(
        [
            "mine-negatives",
            "--claims",
            str(FIXTURES / "claims.json"),
            "--bm25-index",
            str(index_dir / "bm25.pkl.gz"),
            "--dense-index",
            str(index_dir / "dense"),
            "--recall-k",
            "8",
            "--output-dir",
            str(negatives_dir),
        ]
    ) == 0
    assert (negatives_dir / "hard_negatives.jsonl").read_text(encoding="utf-8").strip()
    assert (negatives_dir / "ltr_features.jsonl").read_text(encoding="utf-8").strip()
    ltr_rows = [
        json.loads(line)
        for line in (negatives_dir / "ltr_features.jsonl").read_text(encoding="utf-8").splitlines()
        if line
    ]
    assert all(
        row["relevance"] == 0
        or row["features"]["bm25_reciprocal_rank"] > 0
        or row["features"]["dense_reciprocal_rank"] > 0
        for row in ltr_rows
    )
    negative_metrics = json.loads((negatives_dir / "metrics.json").read_text(encoding="utf-8"))
    assert negative_metrics["ltr_query_group_count"] > 0
    assert negative_metrics["ltr_skipped_query_count"] == 0

    ltr_dir = tmp_path / "ltr"
    assert main(
        [
            "train-fusion",
            "--features",
            str(FIXTURES / "ltr_features.jsonl"),
            "--algorithm",
            "linear",
            "--output-dir",
            str(ltr_dir),
        ]
    ) == 0
    assert (ltr_dir / "ltr_model.json").exists()


def test_five_stage_benchmark_entrypoint(tmp_path: Path, monkeypatch) -> None:
    index_dir = tmp_path / "index"
    main(
        [
            "index",
            "--evidence",
            str(FIXTURES / "evidence.json"),
            "--output-dir",
            str(index_dir),
            "--backend",
            "both",
        ]
    )
    ltr_dir = tmp_path / "ltr"
    main(
        [
            "train-fusion",
            "--features",
            str(FIXTURES / "ltr_features.jsonl"),
            "--algorithm",
            "linear",
            "--output-dir",
            str(ltr_dir),
        ]
    )
    config = tmp_path / "benchmark.json"
    monkeypatch.setenv("CLIMATE_TEST_ARTIFACT_DIR", str(tmp_path))
    config.write_text(
        json.dumps(
            {
                "bm25_index": "${CLIMATE_TEST_ARTIFACT_DIR}/index/bm25.pkl.gz",
                "dense_index": "${CLIMATE_TEST_ARTIFACT_DIR}/index/dense",
                "ltr_model": "${CLIMATE_TEST_ARTIFACT_DIR}/ltr/ltr_model.json",
                "claim_limit": 2,
                "recall_k": 8,
                "fusion_k": 8,
                "rerank_k": 8,
                "final_k": 5,
                "rerank_source": "rrf",
                "rerank_fusion": {
                    "enabled": True,
                    "k": 60,
                    "profiles": [
                        {
                            "name": "balanced",
                            "base_weight": 1.0,
                            "reranker_weight": 1.0,
                        }
                    ],
                },
                "reranker": {"kind": "deterministic"},
            }
        ),
        encoding="utf-8",
    )
    output_dir = tmp_path / "benchmark"
    assert main(
        [
            "evaluate",
            "--claims",
            str(FIXTURES / "claims.json"),
            "--experiment-config",
            str(config),
            "--bootstrap-samples",
            "100",
            "--output-dir",
            str(output_dir),
        ]
    ) == 0
    metrics = json.loads((output_dir / "metrics.json").read_text(encoding="utf-8"))
    assert set(metrics["systems"]) == {
        "bm25",
        "dense",
        "rrf",
        "ltr",
        "rrf_reranker",
        "rrf_reranker_fusion_balanced",
    }
    assert metrics["reranker"] == "deterministic-feature-fallback"
    assert metrics["reranker_base"] == "rrf"
    assert metrics["reranker_timing"]["query_count"] == metrics["claim_count"] == 2
    assert metrics["reranker_timing"]["candidate_pair_count"] > 0
    assert metrics["rerank_fusion"]["enabled"] is True
    assert (output_dir / "reranker_candidates.jsonl").exists()


def test_fastapi_service_returns_evidence_without_fake_label() -> None:
    index = BM25Index().fit(iter_evidence(FIXTURES / "evidence.json"))
    client = TestClient(create_app(HybridRetriever(bm25=index)))
    response = client.post(
        "/retrieve", json={"claim_text": "human carbon dioxide emissions", "top_k": 2}
    )
    assert response.status_code == 200
    payload = response.json()
    assert payload["evidence"][0]["evidence_id"] == "e1"
    assert payload["classification"] == {"status": "not_configured", "label": None}
