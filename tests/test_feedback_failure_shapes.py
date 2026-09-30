import hashlib
import importlib.util
from pathlib import Path

import pytest


def test_diagnostic_redacts_content_and_counts_preview_citation():
    spec = importlib.util.spec_from_file_location(
        "failure_shapes",
        Path(__file__).parents[1] / "scripts/diagnose_feedback_pilot.py",
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    raw = '{"action":"answer","private-marker":"secret-fixture"}'
    digest = hashlib.sha256(raw.encode()).hexdigest()
    row = {
        "generation_attempts": [
            {
                "allowed_actions": ["answer"],
                "usage": {"output_tokens": 30},
                "diagnostics": {
                    "output_sha256": digest,
                    "category": "validated",
                    "eos_observed": True,
                },
            }
        ],
        "events": [
            {
                "stage": "decision",
                "decision": {
                    "action": "answer",
                    "statements": [{"evidence_id": "preview-private-id"}],
                },
            }
        ],
        "context_evidence_ids": ["context-private-id"],
        "candidate_evidence_ids": ["context-private-id", "preview-private-id"],
        "validation_repairs": 0,
    }
    run = {
        "runs": [
            {**row, "strategy": s}
            for s in ("fixed_retrieval", "fixed_rerank", "adaptive")
        ]
    }
    result = module.diagnose(run, {digest: raw})
    assert "secret-fixture" not in str(result)
    assert "private-id" not in str(result)
    assert result["routes"]["adaptive"]["counts"]["answer_citation_preview_only"] == 1
    with pytest.raises(ValueError, match="SHA mismatch"):
        module.diagnose(run, {digest: raw + "tampered"})
