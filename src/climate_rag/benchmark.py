from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from .bm25 import BM25Index
from .dense import DenseRetriever
from .fusion import build_candidate_features, load_ranker, reciprocal_rank_fusion
from .io import load_claims, write_json
from .metrics import evaluate_predictions, paired_bootstrap
from .models import Prediction, RankedDocument
from .rerank import (
    DeterministicFeatureReranker,
    ModelStudioReranker,
    Qwen3CausalLMReranker,
    Reranker,
)


def _make_reranker(config: dict[str, Any]) -> Reranker:
    kind = str(config.get("kind", "deterministic"))
    if kind == "deterministic":
        return DeterministicFeatureReranker()
    if kind == "local":
        return Qwen3CausalLMReranker(
            str(config.get("model", "Qwen/Qwen3-Reranker-0.6B")),
            device=config.get("device"),
            max_length=int(config.get("max_length", 8192)),
            batch_size=int(config.get("batch_size", 8)),
            instruction=str(
                config.get(
                    "instruction",
                    "Given a climate claim, retrieve evidence that helps verify the claim",
                )
            ),
        )
    if kind == "model-studio":
        return ModelStudioReranker(
            str(config.get("model", "qwen3-rerank")),
            endpoint=str(
                config.get(
                    "endpoint",
                    "https://dashscope-intl.aliyuncs.com/api/v1/services/rerank/"
                    "text-rerank/text-rerank",
                )
            ),
            timeout_seconds=float(config.get("timeout_seconds", 30.0)),
        )
    raise ValueError(f"unsupported reranker kind: {kind}")


def _rank_with_ltr(
    query: str,
    bm25: list[RankedDocument],
    dense: list[RankedDocument],
    rrf: list[RankedDocument],
    ranker: Any,
) -> list[RankedDocument]:
    by_bm25 = {row.evidence_id: row for row in bm25}
    by_dense = {row.evidence_id: row for row in dense}
    candidates = {row.evidence_id: row for row in rrf}
    rows: list[dict[str, float]] = []
    ordered_ids = sorted(candidates)
    for evidence_id in ordered_ids:
        b_row = by_bm25.get(evidence_id)
        d_row = by_dense.get(evidence_id)
        candidate = candidates[evidence_id]
        rows.append(
            build_candidate_features(
                query,
                candidate.text,
                bm25_score=b_row.score if b_row else 0.0,
                bm25_rank=b_row.rank if b_row else None,
                dense_score=d_row.score if d_row else 0.0,
                dense_rank=d_row.rank if d_row else None,
            )
        )
    matrix = np.asarray(
        [[row.get(name, 0.0) for name in ranker.feature_names] for row in rows], dtype=np.float64
    )
    scores = ranker.predict(matrix)
    ranked = sorted(
        zip(ordered_ids, scores, rows, strict=True), key=lambda row: (-float(row[1]), row[0])
    )
    return [
        RankedDocument(
            evidence_id=evidence_id,
            score=float(score),
            rank=rank,
            text=candidates[evidence_id].text,
            source="ltr",
            features=features,
        )
        for rank, (evidence_id, score, features) in enumerate(ranked, start=1)
    ]


def run_five_stage_benchmark(
    *,
    claims_path: str | Path,
    config_path: str | Path,
    output_dir: str | Path,
    bootstrap_samples: int = 2000,
    seed: int = 17,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Run BM25, dense, RRF, LTR, and LTR+reranker on one fixed claim split."""
    config_source = Path(config_path)
    if config_source.suffix.lower() in {".yaml", ".yml"}:
        try:
            import yaml
        except ImportError as exc:
            raise RuntimeError("PyYAML is required for YAML benchmark configs") from exc
        config = yaml.safe_load(config_source.read_text(encoding="utf-8"))
    else:
        config = json.loads(config_source.read_text(encoding="utf-8"))
    claims = load_claims(claims_path)
    bm25 = BM25Index.load(config["bm25_index"])
    dense = DenseRetriever.load(config["dense_index"], device=config.get("device"))
    ranker = load_ranker(config["ltr_model"])
    reranker = _make_reranker(config.get("reranker", {}))
    recall_k = int(config.get("recall_k", 1000))
    fusion_k = int(config.get("fusion_k", 100))
    rerank_k = int(config.get("rerank_k", 50))
    final_k = int(config.get("final_k", 5))
    predictions: dict[str, dict[str, Prediction]] = {
        "bm25": {},
        "dense": {},
        "rrf": {},
        "ltr": {},
        "ltr_reranker": {},
    }
    for claim_id in sorted(claims):
        query = claims[claim_id].text
        bm25_rows = bm25.search(query, recall_k)
        dense_rows = dense.search(query, recall_k)
        rrf_rows = reciprocal_rank_fusion(
            {"bm25": bm25_rows, "dense": dense_rows}, k=int(config.get("rrf_k", 60)), top_k=fusion_k
        )
        ltr_rows = _rank_with_ltr(query, bm25_rows, dense_rows, rrf_rows, ranker)
        reranked = reranker.rerank(query, ltr_rows[:rerank_k], final_k)
        stage_rows = {
            "bm25": bm25_rows,
            "dense": dense_rows,
            "rrf": rrf_rows,
            "ltr": ltr_rows,
            "ltr_reranker": reranked,
        }
        for stage, rows in stage_rows.items():
            predictions[stage][claim_id] = Prediction(
                claim_id=claim_id,
                evidence_ids=tuple(row.evidence_id for row in rows[:final_k]),
            )
    target = Path(output_dir)
    target.mkdir(parents=True, exist_ok=True)
    system_metrics: dict[str, Any] = {}
    per_system_rows: dict[str, list[dict[str, Any]]] = {}
    long_rows: list[dict[str, Any]] = []
    for stage, stage_predictions in predictions.items():
        metrics, rows, _ = evaluate_predictions(claims, stage_predictions)
        system_metrics[stage] = metrics
        per_system_rows[stage] = rows
        write_json(
            target / f"predictions_{stage}.json",
            {
                claim_id: prediction.to_official_dict(claims[claim_id].text)
                for claim_id, prediction in stage_predictions.items()
            },
        )
        long_rows.extend({"system": stage, **row} for row in rows)
    baseline = {row["claim_id"]: row for row in per_system_rows["bm25"]}
    comparisons: dict[str, Any] = {}
    for stage in ("dense", "rrf", "ltr", "ltr_reranker"):
        candidate = {row["claim_id"]: row for row in per_system_rows[stage]}
        comparisons[stage] = {}
        for metric in ("recall@5", "evidence_f1", "mrr@10", "ndcg@10"):
            claim_ids = sorted(claims)
            comparisons[stage][metric] = paired_bootstrap(
                [baseline[claim_id][metric] for claim_id in claim_ids],
                [candidate[claim_id][metric] for claim_id in claim_ids],
                samples=bootstrap_samples,
                seed=seed,
            )
    return {
        "systems": system_metrics,
        "paired_bootstrap_vs_bm25": comparisons,
        "reranker": reranker.name,
        "claim_count": len(claims),
        "final_k": final_k,
    }, long_rows
