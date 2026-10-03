"""Protocol-bound cumulative exposure ledger, with physical call deduplication."""
from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

from .scifact_grounding_sft import VERSION, require
from .scifact_semantic_contract import encoded, sha


def hash_string(value: Any) -> bool:
    return isinstance(value, str) and len(value) == 64 and all(
        c in "0123456789abcdef" for c in value)


def cumulative_ledger(
    assignment: Mapping[str, Any], assignment_sha: str,
    historical: Mapping[str, Any], historical_sha: str,
    runs: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Each run binds protocol, planned IDs, terminal state and physical attempts.

    A missing/unknown real-claim receipt excludes its entire component. Failed
    jobs remain in the registry even if proven zero-call; unresolved zero-call
    jobs are never silently called unconsumed. Carried calls must reference an
    already registered physical identity and exactly match its receipt/hash.
    """
    require(hash_string(assignment_sha) and hash_string(historical_sha), "ledger_source_hash")
    eligible = set(assignment["eligible_train_ids"])
    consumed = set(historical["model_consumed_ids"])
    uncertain = set(historical["uncertain_ids"])
    require(consumed | uncertain <= eligible, "historical_outside_eligible")
    components = {i: str(assignment["claim_component"][str(i)]) for i in eligible}
    seen_runs: set[str] = set()
    physical: dict[str, dict[str, Any]] = {}
    carry_count = 0
    registry: list[dict[str, Any]] = []
    for run in runs:
        name = run["run_id"]
        require(isinstance(name, str) and name not in seen_runs, "duplicate_run")
        seen_runs.add(name)
        require(hash_string(run["protocol_sha256"]) and hash_string(run["receipt_sha256"]),
                "protocol_receipt_binding")
        planned = run["planned_claim_ids"]
        require(len(planned) == len(set(planned)) and set(planned) <= eligible, "planned_claim_matrix")
        require(run["state"] in {"completed", "failed", "unknown"}, "run_terminal_state")
        seen_claims: set[int] = set()
        for event in run["attempts"]:
            require({"physical_id", "claim_id", "receipt_sha256", "status", "carried"} <= set(event)
                    <= {"physical_id", "claim_id", "receipt_sha256", "status", "carried", "original_status", "usage_known"},
                    "attempt_fields")
            require(hash_string(event["receipt_sha256"]), "attempt_receipt_hash")
            require(event["status"] in {"completed", "failed", "unknown"}, "attempt_status")
            claim = event["claim_id"]
            require(claim is None or (type(claim) is int and claim in planned), "attempt_not_in_protocol")
            identity = event["physical_id"]
            normalized = {k: v for k, v in event.items() if k != "carried"}
            require(isinstance(identity, str) and bool(identity), "physical_call_identity")
            if event["carried"]:
                require(identity in physical and physical[identity] == normalized, "carry_identity_mismatch")
                carry_count += 1
            else:
                require(identity not in physical, "duplicate_physical_call")
                physical[identity] = normalized
            if claim is not None:
                consumed.add(claim)
                seen_claims.add(claim)
        # Explicit per-claim proven unattempted list must be receipt-bound.
        unattempted = run["proven_unattempted_claim_ids"]
        require(len(unattempted) == len(set(unattempted)) and set(unattempted) <= set(planned),
                "unattempted_outside_protocol")
        require(not seen_claims.intersection(unattempted), "attempted_and_unattempted")
        uncertain.update(set(planned) - seen_claims - set(unattempted))
        registry.append(dict(run))
    uncertain -= consumed
    excluded = {components[i] for i in consumed | uncertain}
    body = {
        "version": VERSION, "assignment_sha256": assignment_sha,
        "historical_ledger_sha256": historical_sha, "runs": registry,
        "model_consumed_ids": sorted(consumed), "uncertain_ids": sorted(uncertain),
        "component_excluded": sorted(excluded),
        "component_excluded_eligible_ids": sorted(i for i in eligible if components[i] in excluded),
        "new_physical_calls": len(physical), "carried_references_deduplicated": carry_count,
        "gold_preparation_seen_count": len(eligible),
        "scope": "TRAIN_only_gold_preparation_seen_not_independent_test",
    }
    return body | {"ledger_identity_sha256": sha(encoded(body))}
