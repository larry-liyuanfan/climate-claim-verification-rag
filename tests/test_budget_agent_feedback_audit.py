"""Development-audit and pilot operator guards; synthetic CPU fixtures only."""
import copy
import importlib.util
import sys
from pathlib import Path

import pytest

from climate_rag.budget_agent import BudgetedEvidenceAgent
from climate_rag.rerank import DeterministicFeatureReranker
from test_budget_agent import answer, item
from test_budget_agent_feedback import SequenceModel, bad_json, budget

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("feedback_audit", ROOT / "scripts/score_budget_agent_feedback.py")
audit_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(audit_module)


def records():
    rows = []
    for route in audit_module.ROUTES:
        decisions = [bad_json(), answer()] if route != "adaptive" else [bad_json(), bad_json(), bad_json()]
        result = BudgetedEvidenceAgent(lambda _: {"items": [item()]}, SequenceModel(decisions),
                                      budget=budget(), reranker=DeterministicFeatureReranker()).run(
            {"claim_text": "Arctic ice"}, strategy=route)
        rows.append({"task_id": "fixture-one", **result})
    protocol = {"official_gold_available": False, "study_kind": "fixture", "corpus_sha256": "fixture-sha",
                "budget": budget().model_dump(), "strategies": list(audit_module.ROUTES),
                "pilot": [{"id": "fixture-one", "claim_text": "Arctic ice"}]}
    run = {"phase": "pilot", "budget": protocol["budget"], "dense_enabled": False,
           "study_kind": "fixture", "corpus_sha256": "fixture-sha", "protocol_sha256": "fixture-protocol",
           "code_sha": "fixture-source", "model_sha256": "fixture-model", "reranker_sha256": "fixture-reranker",
           "prompt_identity": {}, "runs": rows}
    return run, protocol


def test_audit_accounts_repaired_and_exhausted_errors_without_using_v1_scorer():
    run, protocol = records()
    report = audit_module.audit(run, protocol)
    assert report["complete_slots"] == 3
    fixed, adaptive = report["routes"]["fixed_retrieval"], report["routes"]["adaptive"]
    assert fixed["outcomes"] == {"mechanically_accepted_semantics_unmeasured": 1}
    assert adaptive["outcomes"] == {"validation_repair_exhausted": 1}
    assert fixed["input_tokens"] == 250 and adaptive["input_tokens"] == 450
    assert adaptive["generation_attempts"] == 3
    assert report["semantic_support_quality"] is report["retrieval_quality"] is None
    assert "Arctic" not in str(report) and "fixture-one" not in str(report)


@pytest.mark.parametrize("change", ["drop_row", "duplicate_row", "tokens", "claim", "gold", "budget"])
def test_audit_rejects_changed_contract_or_incomplete_costs(change):
    run, protocol = records()
    if change == "drop_row":
        run["runs"].pop()
    elif change == "duplicate_row":
        run["runs"].append(copy.deepcopy(run["runs"][0]))
    elif change == "tokens":
        run["runs"][0]["usage"]["output_tokens"] += 1
    elif change == "claim":
        run["runs"][0]["claim_text"] += " edited"
    elif change == "gold":
        protocol["official_gold_available"] = True
    else:
        run["budget"] = {**run["budget"], "max_tool_calls": 1}
    with pytest.raises(ValueError):
        audit_module.audit(run, protocol)


def test_operator_is_separate_release_with_allocated_preflight_before_pilot(monkeypatch):
    monkeypatch.syspath_prepend(str(ROOT / "scripts"))
    sys.modules.pop("run_budget_agent_feedback_operator", None)
    import run_budget_agent_feedback_operator as operator
    monkeypatch.delenv("SLURM_JOB_ID", raising=False)
    with pytest.raises(ValueError, match="released allocation"):
        operator.main()
    text = (ROOT / "scripts/run_budget_agent_feedback_operator.py").read_text()
    assert text.index('stage("preflight")') < text.index('stage("pilot")')
    assert "--gold" not in text and "score_budget_agent.py" not in text
    wrapper = (ROOT / "hpc/budget_agent_feedback_pilot.sbatch").read_text()
    assert "#SBATCH --no-requeue" in wrapper and "#SBATCH --time=00:45:00" in wrapper
    assert "#SBATCH --gres=gpu:A100:1" in wrapper
