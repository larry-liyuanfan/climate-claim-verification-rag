"""Same official/strict scoring; new physical reference audit, three arms."""
from __future__ import annotations

from functools import partial
from pathlib import Path
from typing import Any

from climate_rag.scifact_evidence_commit import ARMS, PROTOCOL
from climate_rag.scifact_evidence_commit_runtime import audit_episode
from score_scifact_document_verifier import score_after_exit as score_original_entry
from run_scifact_grounding_train_operator import ROOT
import scifact_evidence_input as inputs


def score_after_exit(output: Path, load_tokenizer: Any, release: Any, root: Path = ROOT, *,
                     release_sha: str) -> dict[str, Any]:
    # Scope includes the exact release identity, not a client-supplied saved ref.
    from run_scifact_grounding_train_operator import sha
    import json
    reservation = json.loads((output / "reserved.json").read_bytes())
    proof = json.loads((output / "worker-exit.json").read_bytes())
    from climate_rag.scifact_natural_contract import require
    require(reservation["release_sha256"] == proof["release_sha256"] == release_sha,
            "trusted_release_scope_mismatch")
    return score_original_entry(output, load_tokenizer, release, root,
        protocol=PROTOCOL, arms=ARMS,
        input_adapter=inputs if inputs.prospective(release) else None,
        audit_fn=partial(audit_episode, run_identity=release_sha), baseline_arm="fixed_all",
        comparison_limits={
            "verify_is_prerequisite_not_spontaneous_demand": True,
            "fixed_top1_rule": "first frozen retrieval document, no gold selection",
            "call_caps": {"fixed_top1": 1, "fixed_all": 4, "adaptive": 5},
            "fixed_empty_positive_is_unresolved_not_NEI": True,
            "no_automatic_agent_gain": True,
            "exact_reservation_sha256": sha(output / "reserved.json"),
        })
