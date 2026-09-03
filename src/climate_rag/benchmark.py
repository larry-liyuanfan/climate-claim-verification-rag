from __future__ import annotations

import json
import os
import re
import time
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np

from .bm25 import BM25Index
from .dense import DenseRetriever
from .fusion import build_candidate_features, load_ranker, reciprocal_rank_fusion
from .io import load_claims, write_json, write_jsonl
from .metrics import evaluate_predictions, paired_bootstrap
from .models import Prediction, RankedDocument
from .rerank import (
    DeterministicFeatureReranker,
    ModelStudioReranker,
    Qwen3CausalLMReranker,
    Reranker,
    weighted_rank_fuse,
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
            dtype=str(config.get("dtype", "auto")),
            revision=config.get("revision"),
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


def _weighted_rank_fusion_profiles(
    config: dict[str, Any], *, label: str
) -> list[dict[str, Any]]:
    if not bool(config.get("enabled", False)):
        return []
    kind = str(config.get("kind", "weighted-rank-fusion"))
    if kind != "weighted-rank-fusion":
        raise ValueError(f"{label}.kind must be 'weighted-rank-fusion'")
    raw_profiles = config.get("profiles")
    if raw_profiles is None:
        raw_profiles = [
            {
                "name": "balanced",
                "base_weight": config.get("base_weight", 1.0),
                "reranker_weight": config.get("reranker_weight", 1.0),
            }
        ]
    if not isinstance(raw_profiles, list) or not raw_profiles:
        raise ValueError(f"{label}.profiles must be a non-empty list")
    profiles: list[dict[str, Any]] = []
    names: set[str] = set()
    for raw in raw_profiles:
        if not isinstance(raw, dict):
            raise TypeError(f"each {label} profile must be a mapping")
        name = str(raw.get("name", "")).strip()
        if not re.fullmatch(r"[a-z0-9_]+", name):
            raise ValueError(f"{label} profile names must match [a-z0-9_]+")
        if name in names:
            raise ValueError(f"duplicate {label} profile: {name}")
        names.add(name)
        profiles.append(
            {
                "name": name,
                "k": int(raw.get("k", config.get("k", 60))),
                "base_weight": float(raw.get("base_weight", 1.0)),
                "reranker_weight": float(raw.get("reranker_weight", 1.0)),
            }
        )
    return profiles


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
                rrf_score=candidate.score,
                rrf_rank=candidate.rank,
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
            import yaml  # type: ignore[import-untyped]
        except ImportError as exc:
            raise RuntimeError("PyYAML is required for YAML benchmark configs") from exc
        config = yaml.safe_load(config_source.read_text(encoding="utf-8"))
    else:
        config = json.loads(config_source.read_text(encoding="utf-8"))
    for key in ("bm25_index", "dense_index", "ltr_model"):
        if key in config:
            config[key] = os.path.expandvars(str(config[key]))
    claims = load_claims(claims_path)
    claim_limit = config.get("claim_limit")
    if claim_limit is not None:
        limit = int(claim_limit)
        if limit <= 0:
            raise ValueError("claim_limit must be positive when configured")
        claims = {claim_id: claims[claim_id] for claim_id in sorted(claims)[:limit]}
    bm25 = BM25Index.load(config["bm25_index"])
    dense = DenseRetriever.load(config["dense_index"], device=config.get("device"))
    ranker = load_ranker(config["ltr_model"])
    reranker_load_started = time.perf_counter()
    reranker = _make_reranker(config.get("reranker", {}))
    reranker_load_seconds = time.perf_counter() - reranker_load_started
    recall_k = int(config.get("recall_k", 1000))
    fusion_k = int(config.get("fusion_k", 100))
    rerank_k = int(config.get("rerank_k", 50))
    final_k = int(config.get("final_k", 5))
    rerank_source = str(config.get("rerank_source", "ltr"))
    if rerank_source not in {"rrf", "ltr"}:
        raise ValueError("rerank_source must be 'rrf' or 'ltr'")
    reranked_stage = f"{rerank_source}_reranker"
    fusion_config = config.get("rerank_fusion", {}) or {}
    fusion_profiles = _weighted_rank_fusion_profiles(fusion_config, label="rerank_fusion")
    fused_stages = {
        profile["name"]: f"{reranked_stage}_fusion_{profile['name']}"
        for profile in fusion_profiles
    }
    ltr_fusion_config = config.get("ltr_fusion", {}) or {}
    ltr_fusion_profiles = _weighted_rank_fusion_profiles(
        ltr_fusion_config, label="ltr_fusion"
    )
    ltr_fused_stages = {
        profile["name"]: f"rrf_ltr_fusion_{profile['name']}"
        for profile in ltr_fusion_profiles
    }
    predictions: dict[str, dict[str, Prediction]] = {
        "bm25": {},
        "dense": {},
        "rrf": {},
        "ltr": {},
        reranked_stage: {},
    }
    for fused_stage in fused_stages.values():
        predictions[fused_stage] = {}
    for fused_stage in ltr_fused_stages.values():
        predictions[fused_stage] = {}
    rerank_durations: list[float] = []
    reranked_pair_count = 0
    candidate_trace: list[dict[str, Any]] = []
    candidate_source_distribution: Counter[str] = Counter()
    serving_gold_total = 0
    serving_gold_reachable = 0
    for claim_id in sorted(claims):
        query = claims[claim_id].text
        bm25_rows = bm25.search(query, recall_k)
        dense_rows = dense.search(query, recall_k)
        rrf_rows = reciprocal_rank_fusion(
            {"bm25": bm25_rows, "dense": dense_rows}, k=int(config.get("rrf_k", 60)), top_k=fusion_k
        )
        bm25_ids = {row.evidence_id for row in bm25_rows}
        dense_ids = {row.evidence_id for row in dense_rows}
        rrf_ids = {row.evidence_id for row in rrf_rows}
        serving_gold_total += len(claims[claim_id].evidence_ids)
        serving_gold_reachable += sum(
            evidence_id in rrf_ids for evidence_id in claims[claim_id].evidence_ids
        )
        for row in rrf_rows:
            in_bm25 = row.evidence_id in bm25_ids
            in_dense = row.evidence_id in dense_ids
            source = (
                "both"
                if in_bm25 and in_dense
                else "bm25_only"
                if in_bm25
                else "dense_only"
            )
            candidate_source_distribution[source] += 1
        ltr_rows = _rank_with_ltr(query, bm25_rows, dense_rows, rrf_rows, ranker)
        ltr_fused_by_profile = {
            profile["name"]: weighted_rank_fuse(
                rrf_rows,
                ltr_rows,
                final_k,
                k=profile["k"],
                base_weight=profile["base_weight"],
                reranker_weight=profile["reranker_weight"],
            )
            for profile in ltr_fusion_profiles
        }
        rerank_candidates = rrf_rows if rerank_source == "rrf" else ltr_rows
        rerank_candidates = rerank_candidates[:rerank_k]
        rerank_started = time.perf_counter()
        reranked_all = reranker.rerank(query, rerank_candidates, len(rerank_candidates))
        rerank_durations.append(time.perf_counter() - rerank_started)
        reranked_pair_count += len(rerank_candidates)
        reranked = reranked_all[:final_k]
        fused_by_profile = {
            profile["name"]: weighted_rank_fuse(
                rerank_candidates,
                reranked_all,
                final_k,
                k=profile["k"],
                base_weight=profile["base_weight"],
                reranker_weight=profile["reranker_weight"],
            )
            for profile in fusion_profiles
        }
        base_by_id = {row.evidence_id: row for row in rerank_candidates}
        reranked_by_id = {row.evidence_id: row for row in reranked_all}
        for evidence_id in sorted(base_by_id.keys() | reranked_by_id.keys()):
            base_row = base_by_id.get(evidence_id)
            reranked_row = reranked_by_id.get(evidence_id)
            candidate_trace.append(
                {
                    "claim_id": claim_id,
                    "evidence_id": evidence_id,
                    "base_rank": base_row.rank if base_row is not None else None,
                    "base_score": base_row.score if base_row is not None else None,
                    "reranker_rank": reranked_row.rank if reranked_row is not None else None,
                    "reranker_score": reranked_row.score if reranked_row is not None else None,
                }
            )
        stage_rows: dict[str, list[RankedDocument]] = {
            "bm25": bm25_rows,
            "dense": dense_rows,
            "rrf": rrf_rows,
            "ltr": ltr_rows,
            reranked_stage: reranked,
        }
        for profile_name, fused in fused_by_profile.items():
            stage_rows[fused_stages[profile_name]] = fused
        for profile_name, fused in ltr_fused_by_profile.items():
            stage_rows[ltr_fused_stages[profile_name]] = fused
        for stage, ranked_rows in stage_rows.items():
            predictions[stage][claim_id] = Prediction(
                claim_id=claim_id,
                evidence_ids=tuple(
                    row.evidence_id for row in ranked_rows[:final_k]
                ),
            )
    target = Path(output_dir)
    target.mkdir(parents=True, exist_ok=True)
    write_jsonl(target / "reranker_candidates.jsonl", candidate_trace)
    system_metrics: dict[str, Any] = {}
    per_system_rows: dict[str, list[dict[str, Any]]] = {}
    long_rows: list[dict[str, Any]] = []
    for stage, stage_predictions in predictions.items():
        metrics, metric_rows, _ = evaluate_predictions(claims, stage_predictions)
        system_metrics[stage] = metrics
        per_system_rows[stage] = metric_rows
        write_json(
            target / f"predictions_{stage}.json",
            {
                claim_id: prediction.to_official_dict(claims[claim_id].text)
                for claim_id, prediction in stage_predictions.items()
            },
        )
        long_rows.extend({"system": stage, **row} for row in metric_rows)
    baseline = {row["claim_id"]: row for row in per_system_rows["bm25"]}
    comparisons: dict[str, Any] = {}
    for stage in predictions:
        if stage == "bm25":
            continue
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
    rerank_array = np.asarray(rerank_durations, dtype=np.float64)
    return {
        "systems": system_metrics,
        "paired_bootstrap_vs_bm25": comparisons,
        "reranker": reranker.name,
        "reranker_base": rerank_source,
        "rerank_fusion": {
            "enabled": bool(fusion_profiles),
            "kind": "weighted-rank-fusion" if fusion_profiles else None,
            "profiles": [
                {**profile, "stage": fused_stages[profile["name"]]}
                for profile in fusion_profiles
            ],
        },
        "ltr_fusion": {
            "enabled": bool(ltr_fusion_profiles),
            "kind": "weighted-rank-fusion" if ltr_fusion_profiles else None,
            "profiles": [
                {**profile, "stage": ltr_fused_stages[profile["name"]]}
                for profile in ltr_fusion_profiles
            ],
        },
        "reranker_timing": {
            "model_load_seconds": reranker_load_seconds,
            "query_count": len(rerank_durations),
            "candidate_pair_count": reranked_pair_count,
            "total_seconds": float(np.sum(rerank_array)),
            "mean_seconds_per_query": float(np.mean(rerank_array)),
            "p50_seconds_per_query": float(np.percentile(rerank_array, 50)),
            "p95_seconds_per_query": float(np.percentile(rerank_array, 95)),
        },
        "claim_count": len(claims),
        "final_k": final_k,
        "stage_contract": {
            "candidate_width": fusion_k,
            "feature_names": list(ranker.feature_names),
            "gold_positive_reachability": (
                serving_gold_reachable / serving_gold_total
                if serving_gold_total
                else 0.0
            ),
            "candidate_source_distribution": dict(
                sorted(candidate_source_distribution.items())
            ),
        },
    }, long_rows
