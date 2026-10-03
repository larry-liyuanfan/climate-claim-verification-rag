"""Private TRAIN exposure ledger and selected-only CPU opportunity review."""
from __future__ import annotations

import hashlib
from collections import Counter
from collections.abc import Mapping, Sequence
from typing import Any

from .scifact_semantic_policy import POLICIES
from .scifact_semantic_runtime import semantic_packing_snapshot
from .scifact_train_diagnostic import classify_opportunity, select_component_distinct


def checked_ids(values: Sequence[Any]) -> set[int]:
    if any(type(i) is not int for i in values) or len(set(values)) != len(values):
        raise ValueError("duplicate_or_invalid_id")
    return set(values)


def verify_bytes(payload: bytes, expected_sha: str) -> None:
    if len(expected_sha) != 64 or hashlib.sha256(payload).hexdigest() != expected_sha:
        raise ValueError("frozen_input_sha_mismatch")


def consumption_ledger(assignments: Mapping[str, Any], audit: Sequence[Mapping[str, Any]],
                       planned: Sequence[Mapping[str, Any]], actual: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    eligible = checked_ids(assignments["eligible_train_ids"])
    if checked_ids([r["id"] for r in audit]) != eligible:
        raise ValueError("legacy_audit_must_cover_exact_eligible_train_ids")
    raw_components = assignments["claim_component"]
    if any(str(i) not in raw_components or raw_components[str(i)] is None for i in eligible):
        raise ValueError("missing_eligible_component")
    components = {i: str(raw_components[str(i)]) for i in eligible}
    evidence: dict[int, list[dict[str, Any]]] = {}
    consumed: set[int] = set()

    def add(record: Mapping[str, Any], attempted: bool) -> None:
        claim_id = record["claim_id"]
        if type(claim_id) is not int or claim_id not in eligible:
            raise ValueError("unknown_or_ineligible_consumption_id")
        sha = record["source_sha256"]
        if not isinstance(sha, str) or len(sha) != 64 or any(c not in "0123456789abcdef" for c in sha):
            raise ValueError("source_sha_required")
        evidence.setdefault(claim_id, []).append(dict(record))
        if attempted:
            consumed.add(claim_id)

    for r in planned:
        add(r, False)
    for r in actual:
        result = r["result"]
        if "claim_id" in result and result["claim_id"] != r["claim_id"]:
            raise ValueError("nested_consumption_identity_mismatch")
        attempts = result.get("generation_attempts")
        calls = result.get("model_calls")
        # Any attempt consumes the query, including failed/unknown-usage calls,
        # abstentions, truncated wire, and export errors. Missing trace is uncertain.
        attempted = (isinstance(attempts, list) and bool(attempts)) or (type(calls) is int and calls > 0)
        add({k: v for k, v in r.items() if k != "result"} | {
            "attempt_count": len(attempts) if isinstance(attempts, list) else None,
            "reported_model_calls": calls, "outcome": result.get("outcome"),
            "kind": "model_attempt" if attempted else "uncertain_trace"}, attempted)
    uncertain = set(evidence) - consumed
    excluded = {components[i] for i in consumed | uncertain}
    return {
        "schema_version": "scifact-private-consumption-v1",
        "model_consumed_ids": sorted(consumed), "uncertain_ids": sorted(uncertain),
        "component_excluded": sorted(excluded),
        "gold_preparation_seen_ids": sorted(eligible),
        "component_excluded_eligible_ids": sorted(i for i in eligible if components[i] in excluded),
        "records": [{"claim_id": i, "component": components[i],
                     "status": "consumed" if i in consumed else "uncertain", "evidence": evidence[i]}
                    for i in sorted(evidence)],
        "boundary": "Remaining train is gold-preparation-seen, not independent external test.",
    }


def select_unconsumed(audit: Sequence[Mapping[str, Any]], assignments: Mapping[str, Any],
                      ledger: Mapping[str, Any]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    components = {i: str(assignments["claim_component"][str(i)]) for i in assignments["eligible_train_ids"]}
    # Preserve only legacy ID/stratum; never mislabel old V1 packing as new packing.
    minimal = [{"id": r["id"], "stratum": r["stratum"]} for r in audit]
    selected, summary = select_component_distinct(minimal, components,
        excluded_components=ledger["component_excluded"])
    return [{"id": r["id"], "component": r["component"], "legacy_stratum": r["stratum"]} for r in selected], summary


def review_selected(selected: Sequence[Mapping[str, Any]], gold: Mapping[int, Any],
                     retrieve: Any, tokenizer: Any) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    ids = checked_ids([r["id"] for r in selected])
    if set(gold) != ids or len(ids) > 12:
        raise ValueError("selected_gold_only_required")
    reviews = []
    calls = 0
    for row in selected:
        g = gold[row["id"]]
        if g.claim_id != row["id"]:
            raise ValueError("selected_gold_identity")
        policies = {}
        for policy in POLICIES:
            def snapshot(read_ids: Sequence[str] = (), *, policy: str = policy) -> dict[str, Any]:
                nonlocal calls
                calls += 1
                return semantic_packing_snapshot(g.claim, retrieve, tokenizer, read_ids, policy)
            initial = snapshot()
            opportunity = classify_opportunity(g, initial, snapshot)
            if opportunity.get("excluded_reason"):
                opportunity["excluded_reason"] = "no_witness_under_declared_fixture_history"
            policies[policy] = opportunity
        left, right = [policies[p]["initial"] for p in POLICIES]
        if left["initial_identity"] != right["initial_identity"] or left["candidate_doc_ids"] != right["candidate_doc_ids"]:
            raise ValueError("semantic_initial_packing_mismatch")
        reviews.append(dict(row) | {"current_opportunity": policies})
    changes = Counter(f'{r["legacy_stratum"]}->{r["current_opportunity"][p]["stratum"]}:{p}'
                      for r in reviews for p in POLICIES)
    return reviews, {"selected_only_queries_reviewed": len(ids), "packing_probes": calls,
                     "model_calls": 0, "initial_contexts_comparable": True,
                     "legacy_to_current_opportunity": dict(changes), "reselection_after_probe": False,
                     "witness_scope": "declared uncertain/unknown/empty-gap read/abstain fixture history only"}
