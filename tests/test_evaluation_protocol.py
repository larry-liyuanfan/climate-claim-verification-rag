from __future__ import annotations

import json
from pathlib import Path

import pytest

from climate_rag.evaluation_protocol import (
    RetrievalRunContract,
    assert_paired_contracts,
    audit_training_serving_contracts,
    enforce_frozen_test_policy,
    stable_id_sha256,
)

DIGEST_A = "a" * 64
DIGEST_B = "b" * 64


def _contract(system_id: str, *, candidate_width: int = 50) -> RetrievalRunContract:
    return RetrievalRunContract.from_mapping(
        {
            "system_id": system_id,
            "track": "restricted_offline_dev",
            "split": "official_dev",
            "query_id_sha256": DIGEST_A,
            "corpus_sha256": DIGEST_B,
            "candidate_universe_sha256": DIGEST_A,
            "candidate_width": candidate_width,
            "final_k": 5,
            "model_sha256": DIGEST_A if system_id == "base" else DIGEST_B,
            "data_sha256": DIGEST_B,
        }
    )


def test_paired_contract_rejects_candidate_width_mismatch() -> None:
    with pytest.raises(ValueError, match="candidate_width"):
        assert_paired_contracts(
            _contract("base"), _contract("adapted", candidate_width=100)
        )


def test_frozen_test_policy_blocks_new_candidate(tmp_path: Path) -> None:
    policy = tmp_path / "policy.json"
    policy.write_text(
        json.dumps(
            {
                "frozen_test": {
                    "status": "consumed",
                    "consumed_system_id": "bm25-v1",
                    "exact_baseline_reproduction_allowed": True,
                }
            }
        ),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="already consumed"):
        enforce_frozen_test_policy(policy, split="test", system_id="adapted-encoder")
    allowed = enforce_frozen_test_policy(
        policy,
        split="test",
        system_id="bm25-v1",
        exact_baseline_reproduction=True,
    )
    assert allowed["status"] == "allowed_exact_consumed_baseline_reproduction"


def test_training_serving_contract_audit_detects_feature_drift() -> None:
    audit = audit_training_serving_contracts(
        {
            "candidate_width": 100,
            "feature_names": ["bm25_score", "dense_score"],
            "reachable_positive_rate": 1.0,
            "candidate_source_distribution": {"bm25": 50, "dense": 50},
        },
        {
            "candidate_width": 100,
            "feature_names": ["bm25_score"],
            "candidate_source_distribution": {"bm25": 50, "dense": 50},
        },
    )
    assert audit["status"] == "failed"
    assert audit["checks"]["feature_names_equal"] is False


def test_stable_id_hash_is_order_sensitive_and_unambiguous() -> None:
    assert stable_id_sha256(["ab", "c"]) != stable_id_sha256(["a", "bc"])
    assert stable_id_sha256(["a", "b"]) != stable_id_sha256(["b", "a"])
