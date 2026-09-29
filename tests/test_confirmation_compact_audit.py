"""Synthetic redaction checks only; no real result/raw response is opened."""
import importlib.util
import json
from pathlib import Path


def test_cpu_wrapper_preserves_module_dependency_path():
    wrapper = (Path(__file__).resolve().parents[1]
               / "hpc/budget_agent_confirmation_audit.sbatch").read_text()
    assert 'src${PYTHONPATH:+:${PYTHONPATH}}' in wrapper
    assert '#SBATCH --cpus-per-task=1' in wrapper
    assert '#SBATCH --gres' not in wrapper


def test_compact_row_excludes_prose_and_source_text():
    spec = importlib.util.spec_from_file_location("audit", Path(__file__).resolve().parents[1]
                                                / "scripts/audit_budget_confirmation_31520350.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    row = {key: 1 for key in ("tool_calls", "model_calls", "generation_calls", "retrieval_calls",
                             "rerank_calls", "rerank_candidate_pairs", "elapsed_ms")}
    row.update(task_id="fixture", strategy="adaptive", status="answered", usage_known=True,
               usage={"input_tokens": 1, "output_tokens": 1},
               answer={"label": "SUPPORTS", "reason": "private prose", "statements": [{"text": "private prose"}]},
               candidate_evidence_ids=["source"], context_evidence_ids=["source"],
               events=[{"stage": "decision", "decision": {"action": "answer", "reason": "private prose"},
                        "elapsed_ms": 1, "generation_diagnostics": {"category": "validated", "output_sha256": "hash"}},
                       {"stage": "rerank", "elapsed_ms": 1, "private_extra": "private prose"}])
    result = module.compact_row(row, "mechanically_validated_answer_semantics_unverified",
                                "source_and_numeric_checks_passed_semantics_unverified")
    assert "private prose" not in json.dumps(result)
    assert result["schema_valid_action_proposals"] == ["answer"]
    assert result["successful_model_tool_events"] == ["rerank"]
    assert result["final_statement_count"] == 1
