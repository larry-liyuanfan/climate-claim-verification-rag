from __future__ import annotations

import json
import math
from collections import defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

import numpy as np

from .io import write_json
from .models import RankedDocument
from .tokenize import climate_tokenize


DEFAULT_FEATURES = (
    "bm25_score",
    "bm25_reciprocal_rank",
    "dense_score",
    "dense_reciprocal_rank",
    "token_overlap",
    "number_overlap",
    "year_overlap",
    "query_length",
    "document_length",
)


def reciprocal_rank_fusion(
    rankings: Mapping[str, Sequence[RankedDocument]], *, k: int = 60, top_k: int | None = None
) -> list[RankedDocument]:
    if k <= 0:
        raise ValueError("RRF k must be positive")
    scores: dict[str, float] = defaultdict(float)
    texts: dict[str, str] = {}
    for source in sorted(rankings):
        seen: set[str] = set()
        for fallback_rank, row in enumerate(rankings[source], start=1):
            if row.evidence_id in seen:
                continue
            seen.add(row.evidence_id)
            rank = row.rank if row.rank > 0 else fallback_rank
            scores[row.evidence_id] += 1.0 / (k + rank)
            texts.setdefault(row.evidence_id, row.text)
    ordered = sorted(scores, key=lambda evidence_id: (-scores[evidence_id], evidence_id))
    if top_k is not None:
        ordered = ordered[: max(top_k, 0)]
    return [
        RankedDocument(
            evidence_id=evidence_id,
            score=scores[evidence_id],
            rank=rank,
            text=texts.get(evidence_id, ""),
            source="rrf",
        )
        for rank, evidence_id in enumerate(ordered, start=1)
    ]


def _numbers(text: str) -> set[str]:
    return {token for token in climate_tokenize(text) if any(character.isdigit() for character in token)}


def _years(text: str) -> set[str]:
    return {token for token in _numbers(text) if len(token) == 4 and token.isdigit()}


def overlap_ratio(left: set[str], right: set[str]) -> float:
    if not left:
        return 0.0
    return len(left & right) / len(left)


def build_candidate_features(
    query: str,
    document: str,
    *,
    bm25_score: float = 0.0,
    bm25_rank: int | None = None,
    dense_score: float = 0.0,
    dense_rank: int | None = None,
) -> dict[str, float]:
    query_tokens = set(climate_tokenize(query))
    document_tokens = set(climate_tokenize(document))
    return {
        "bm25_score": float(bm25_score),
        "bm25_reciprocal_rank": 0.0 if not bm25_rank else 1.0 / bm25_rank,
        "dense_score": float(dense_score),
        "dense_reciprocal_rank": 0.0 if not dense_rank else 1.0 / dense_rank,
        "token_overlap": overlap_ratio(query_tokens, document_tokens),
        "number_overlap": overlap_ratio(_numbers(query), _numbers(document)),
        "year_overlap": overlap_ratio(_years(query), _years(document)),
        "query_length": float(len(query_tokens)),
        "document_length": float(len(document_tokens)),
    }


class Ranker(Protocol):
    feature_names: tuple[str, ...]

    def fit(self, features: np.ndarray, labels: np.ndarray, groups: Sequence[str]) -> None: ...

    def predict(self, features: np.ndarray) -> np.ndarray: ...


@dataclass
class LinearPairwiseLTR:
    """Deterministic RankNet-style linear fallback when LightGBM is unavailable.

    For each query group it learns from pair differences where relevance differs.
    This is not LambdaMART; persisted metadata names the algorithm explicitly.
    """

    feature_names: tuple[str, ...] = DEFAULT_FEATURES
    learning_rate: float = 0.05
    epochs: int = 300
    l2: float = 1e-3
    seed: int = 17

    def __post_init__(self) -> None:
        self.weights = np.zeros(len(self.feature_names), dtype=np.float64)
        self.mean = np.zeros(len(self.feature_names), dtype=np.float64)
        self.scale = np.ones(len(self.feature_names), dtype=np.float64)

    def fit(self, features: np.ndarray, labels: np.ndarray, groups: Sequence[str]) -> None:
        matrix = np.asarray(features, dtype=np.float64)
        targets = np.asarray(labels, dtype=np.float64)
        if matrix.ndim != 2 or matrix.shape[1] != len(self.feature_names):
            raise ValueError("feature matrix shape does not match feature_names")
        if len(matrix) != len(targets) or len(matrix) != len(groups):
            raise ValueError("features, labels and groups must have equal length")
        self.mean = matrix.mean(axis=0)
        self.scale = matrix.std(axis=0)
        self.scale[self.scale < 1e-12] = 1.0
        normalized = (matrix - self.mean) / self.scale
        by_group: dict[str, list[int]] = defaultdict(list)
        for index, group in enumerate(groups):
            by_group[str(group)].append(index)
        pairs: list[np.ndarray] = []
        for group in sorted(by_group):
            indices = by_group[group]
            for left in indices:
                for right in indices:
                    if targets[left] > targets[right]:
                        pairs.append(normalized[left] - normalized[right])
        if not pairs:
            raise ValueError("LTR training requires at least one unequal-label pair within a group")
        differences = np.asarray(pairs)
        rng = np.random.default_rng(self.seed)
        weights = np.zeros(matrix.shape[1], dtype=np.float64)
        for epoch in range(self.epochs):
            order = rng.permutation(len(differences))
            shuffled = differences[order]
            margin = np.clip(shuffled @ weights, -40.0, 40.0)
            gradient = -(shuffled / (1.0 + np.exp(margin))[:, None]).mean(axis=0)
            gradient += self.l2 * weights
            step = self.learning_rate / math.sqrt(epoch + 1.0)
            weights -= step * gradient
        self.weights = weights

    def predict(self, features: np.ndarray) -> np.ndarray:
        matrix = np.asarray(features, dtype=np.float64)
        return ((matrix - self.mean) / self.scale) @ self.weights

    def save(self, path: str | Path) -> None:
        write_json(
            path,
            {
                "schema_version": 1,
                "algorithm": "linear_pairwise_ranknet_fallback",
                "feature_names": list(self.feature_names),
                "learning_rate": self.learning_rate,
                "epochs": self.epochs,
                "l2": self.l2,
                "seed": self.seed,
                "weights": self.weights.tolist(),
                "mean": self.mean.tolist(),
                "scale": self.scale.tolist(),
            },
        )

    @classmethod
    def load(cls, path: str | Path) -> "LinearPairwiseLTR":
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        if payload.get("algorithm") != "linear_pairwise_ranknet_fallback":
            raise ValueError("not a LinearPairwiseLTR model")
        model = cls(
            feature_names=tuple(payload["feature_names"]),
            learning_rate=float(payload["learning_rate"]),
            epochs=int(payload["epochs"]),
            l2=float(payload["l2"]),
            seed=int(payload["seed"]),
        )
        model.weights = np.asarray(payload["weights"], dtype=np.float64)
        model.mean = np.asarray(payload["mean"], dtype=np.float64)
        model.scale = np.asarray(payload["scale"], dtype=np.float64)
        return model


class LightGBMLambdaMART:
    def __init__(self, feature_names: Sequence[str] = DEFAULT_FEATURES, seed: int = 17) -> None:
        try:
            import lightgbm as lgb
        except ImportError as exc:
            raise RuntimeError("lightgbm is unavailable; install the 'ltr' extra") from exc
        self.feature_names = tuple(feature_names)
        self.seed = seed
        self._lgb = lgb
        self.model = lgb.LGBMRanker(
            objective="lambdarank",
            metric="ndcg",
            n_estimators=200,
            learning_rate=0.05,
            num_leaves=31,
            random_state=seed,
            deterministic=True,
            verbosity=-1,
        )

    def fit(self, features: np.ndarray, labels: np.ndarray, groups: Sequence[str]) -> None:
        order = sorted(range(len(groups)), key=lambda index: (str(groups[index]), index))
        sorted_features = np.asarray(features)[order]
        sorted_labels = np.asarray(labels)[order]
        counts: list[int] = []
        previous: str | None = None
        for index in order:
            group = str(groups[index])
            if group != previous:
                counts.append(1)
                previous = group
            else:
                counts[-1] += 1
        self.model.fit(sorted_features, sorted_labels, group=counts, feature_name=list(self.feature_names))

    def predict(self, features: np.ndarray) -> np.ndarray:
        return np.asarray(self.model.predict(features), dtype=np.float64)

    def save(self, path: str | Path) -> None:
        target = Path(path)
        self.model.booster_.save_model(str(target))
        write_json(
            target.with_suffix(target.suffix + ".json"),
            {
                "schema_version": 1,
                "algorithm": "lightgbm_lambdamart",
                "feature_names": list(self.feature_names),
                "seed": self.seed,
            },
        )

    @classmethod
    def load(cls, path: str | Path) -> "LightGBMLambdaMART":
        target = Path(path)
        metadata = json.loads(target.with_suffix(target.suffix + ".json").read_text(encoding="utf-8"))
        if metadata.get("algorithm") != "lightgbm_lambdamart":
            raise ValueError("not a LightGBM LambdaMART model")
        instance = cls(metadata["feature_names"], seed=int(metadata["seed"]))
        instance.model._Booster = instance._lgb.Booster(model_file=str(target))
        instance.model.fitted_ = True
        return instance


def train_ranker(
    features: np.ndarray,
    labels: np.ndarray,
    groups: Sequence[str],
    *,
    feature_names: Sequence[str] = DEFAULT_FEATURES,
    prefer_lightgbm: bool = True,
) -> Ranker:
    if prefer_lightgbm:
        try:
            model: Ranker = LightGBMLambdaMART(feature_names)
        except RuntimeError:
            model = LinearPairwiseLTR(tuple(feature_names))
    else:
        model = LinearPairwiseLTR(tuple(feature_names))
    model.fit(features, labels, groups)
    return model


def load_ranker(path: str | Path) -> Ranker:
    target = Path(path)
    if target.suffix.lower() == ".json":
        return LinearPairwiseLTR.load(target)
    return LightGBMLambdaMART.load(target)
