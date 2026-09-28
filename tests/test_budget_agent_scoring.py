import copy
import hashlib
import importlib.util
from pathlib import Path

import pytest


spec = importlib.util.spec_from_file_location(
    "agent_scoring", Path(__file__).resolve().parents[1] / "scripts/score_budget_agent.py"
)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def run_fixture():
    return {"protocol_sha256": "frozen", "runs": [
        {"strategy": strategy, "task_id": "a", "claim_text": "ice", "answer": None,
         "candidate_evidence_ids": ["gold"], "delivered_evidence_ids": [],
         "tool_calls": 1, "model_calls": 1, "elapsed_ms": 20.0,
         "usage_known": False, "usage": {"input_tokens": 0, "output_tokens": 0},
         "reason": "deadline_exceeded"}
        for strategy in ("fixed_retrieval", "fixed_rerank", "adaptive")
    ]}


def test_gold_is_posthoc_and_deadline_does_not_earn_recall():
    run = run_fixture()
    before = copy.deepcopy(run)
    gold = {"protocol_sha256": "frozen", "provenance": "synthetic contract fixture",
            "claims": {"a": {"claim_sha256": hashlib.sha256(b"ice").hexdigest(),
                              "evidence_ids": ["gold"]}}}
    scored = module.score(run, gold)
    assert run == before
    assert scored["aggregates"]["adaptive"]["evidence_metrics"]["recall@5"] == 0
    assert scored["paired_bootstrap"]["adaptive"]["recall@5"]["samples"] == 5000
    assert scored["aggregates"]["adaptive"]["unknown_token_accounting_runs"] == 1
    gold["claims"]["a"]["evidence_ids"] = ["different-gold"]
    module.score(run, gold)
    assert run == before  # no changed prompts/queries/predictions when gold changes


def test_missing_gold_not_zero_quality():
    scored = module.score(run_fixture())
    assert scored["gold_source"] is None
    assert "evidence_metrics" not in scored["aggregates"]["adaptive"]


def test_gold_hash_mismatch_rejected():
    with pytest.raises(ValueError, match="protocol"):
        module.score(run_fixture(), {"protocol_sha256": "different"})
