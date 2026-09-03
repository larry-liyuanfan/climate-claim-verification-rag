from __future__ import annotations

import hashlib
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .io import read_json

ALLOWED_TRACKS = frozenset(
    {"restricted_offline_dev", "public_validation", "public_frozen_test", "fixture"}
)


def stable_id_sha256(values: Sequence[str]) -> str:
    """Hash an ordered ID sequence without ambiguous concatenation."""

    digest = hashlib.sha256()
    for value in values:
        encoded = str(value).encode("utf-8")
        digest.update(len(encoded).to_bytes(8, "big"))
        digest.update(encoded)
    return digest.hexdigest()


@dataclass(frozen=True, slots=True)
class RetrievalRunContract:
    """Fields that must be identical in a paired representation comparison."""

    system_id: str
    track: str
    split: str
    query_id_sha256: str
    corpus_sha256: str
    candidate_universe_sha256: str
    candidate_width: int
    final_k: int
    model_sha256: str
    data_sha256: str

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> RetrievalRunContract:
        contract = cls(
            system_id=str(value["system_id"]),
            track=str(value["track"]),
            split=str(value["split"]),
            query_id_sha256=str(value["query_id_sha256"]),
            corpus_sha256=str(value["corpus_sha256"]),
            candidate_universe_sha256=str(value["candidate_universe_sha256"]),
            candidate_width=int(value["candidate_width"]),
            final_k=int(value["final_k"]),
            model_sha256=str(value["model_sha256"]),
            data_sha256=str(value["data_sha256"]),
        )
        contract.validate()
        return contract

    def validate(self) -> None:
        if self.track not in ALLOWED_TRACKS:
            raise ValueError(f"unknown evidence track: {self.track}")
        if self.candidate_width < 10:
            raise ValueError("candidate_width must be at least 10 for MRR/nDCG@10")
        if self.final_k < 5 or self.final_k > self.candidate_width:
            raise ValueError("final_k must be in [5, candidate_width]")
        for name in (
            "query_id_sha256",
            "corpus_sha256",
            "candidate_universe_sha256",
            "model_sha256",
            "data_sha256",
        ):
            value = getattr(self, name)
            if len(value) != 64 or any(
                character not in "0123456789abcdef" for character in value
            ):
                raise ValueError(f"{name} must be a lowercase SHA-256 digest")


def load_run_contract(path: str | Path) -> RetrievalRunContract:
    payload = read_json(path)
    if not isinstance(payload, dict):
        raise TypeError("retrieval run contract must be a JSON object")
    return RetrievalRunContract.from_mapping(payload)


def assert_paired_contracts(
    baseline: RetrievalRunContract, candidate: RetrievalRunContract
) -> dict[str, Any]:
    """Fail closed unless both runs differ only in the learned representation."""

    comparable_fields = (
        "track",
        "split",
        "query_id_sha256",
        "corpus_sha256",
        "candidate_universe_sha256",
        "candidate_width",
        "final_k",
        "data_sha256",
    )
    mismatches = {
        name: {
            "baseline": getattr(baseline, name),
            "candidate": getattr(candidate, name),
        }
        for name in comparable_fields
        if getattr(baseline, name) != getattr(candidate, name)
    }
    if mismatches:
        raise ValueError(f"paired comparison contract mismatch: {mismatches}")
    if baseline.system_id == candidate.system_id:
        raise ValueError("baseline and candidate system_id must differ")
    return {
        "status": "passed",
        "checked_fields": list(comparable_fields),
        "baseline_system_id": baseline.system_id,
        "candidate_system_id": candidate.system_id,
        "model_sha256_changed": baseline.model_sha256 != candidate.model_sha256,
    }


def enforce_frozen_test_policy(
    policy_path: str | Path,
    *,
    split: str,
    system_id: str,
    exact_baseline_reproduction: bool = False,
) -> dict[str, Any]:
    """Prevent a consumed public test from becoming another tuning surface."""

    policy = read_json(policy_path)
    if not isinstance(policy, dict):
        raise TypeError("public evaluation policy must be a JSON object")
    test = policy.get("frozen_test")
    if not isinstance(test, dict):
        raise TypeError("policy is missing a frozen_test object")
    if split != "test":
        return {
            "status": "allowed_non_test_split",
            "split": split,
            "frozen_test_status": str(test.get("status", "unknown")),
        }
    status = str(test.get("status"))
    if status == "unconsumed":
        return {"status": "allowed_unique_final_test", "split": split}
    consumed_system = str(test.get("consumed_system_id", ""))
    reproduction_allowed = bool(test.get("exact_baseline_reproduction_allowed", False))
    if (
        status == "consumed"
        and exact_baseline_reproduction
        and reproduction_allowed
        and system_id == consumed_system
    ):
        return {
            "status": "allowed_exact_consumed_baseline_reproduction",
            "split": split,
            "warning": "This is a reproduction, never a new independent-test result.",
        }
    raise ValueError(
        "public frozen test is already consumed; candidate evaluation is forbidden. "
        "Use public validation or the restricted offline-dev track."
    )


def audit_training_serving_contracts(
    training: Mapping[str, Any],
    serving: Mapping[str, Any],
    *,
    distribution_tv_limit: float = 0.15,
) -> dict[str, Any]:
    """Audit candidate reachability, feature schema and negative-source drift."""

    if not 0.0 <= distribution_tv_limit <= 1.0:
        raise ValueError("distribution_tv_limit must be in [0, 1]")
    training_features = tuple(str(value) for value in training.get("feature_names", ()))
    serving_features = tuple(str(value) for value in serving.get("feature_names", ()))
    training_width = int(training.get("candidate_width", 0))
    serving_width = int(serving.get("candidate_width", 0))
    reachable_rate = float(training.get("reachable_positive_rate", 0.0))
    train_distribution = _normalise_distribution(
        training.get("candidate_source_distribution", {})
    )
    serving_distribution = _normalise_distribution(
        serving.get("candidate_source_distribution", {})
    )
    sources = sorted(train_distribution.keys() | serving_distribution.keys())
    total_variation = 0.5 * sum(
        abs(train_distribution.get(name, 0.0) - serving_distribution.get(name, 0.0))
        for name in sources
    )
    checks = {
        "candidate_width_equal": training_width == serving_width and training_width > 0,
        "feature_names_equal": bool(training_features)
        and training_features == serving_features,
        "all_training_positives_reachable": reachable_rate == 1.0,
        "candidate_source_support_equal": set(train_distribution)
        == set(serving_distribution),
        "candidate_source_total_variation_within_limit": total_variation
        <= distribution_tv_limit,
    }
    return {
        "status": "passed" if all(checks.values()) else "failed",
        "checks": checks,
        "candidate_width": {"training": training_width, "serving": serving_width},
        "feature_names": {
            "training": list(training_features),
            "serving": list(serving_features),
        },
        "reachable_positive_rate": reachable_rate,
        "candidate_source_distribution": {
            "training": train_distribution,
            "serving": serving_distribution,
            "total_variation": total_variation,
            "limit": distribution_tv_limit,
        },
    }


def _normalise_distribution(value: Any) -> dict[str, float]:
    if not isinstance(value, Mapping) or not value:
        return {}
    counts = {str(name): float(count) for name, count in value.items()}
    if any(count < 0 for count in counts.values()):
        raise ValueError("candidate source counts cannot be negative")
    total = sum(counts.values())
    if total <= 0:
        return {}
    return {name: count / total for name, count in counts.items()}
