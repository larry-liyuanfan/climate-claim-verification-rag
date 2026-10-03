"""Metadata-only sampling inside an already exposed FIT pool, never a test split.

The selector accepts ID/component projections only. No model result, label,
cohort, query text or readiness status may influence the frozen ordering.
"""
from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping, Sequence
from typing import Any

PROTOCOL = "scifact-natural-fit24-v1-20261001"
SALT = PROTOCOL
MAX_CLAIMS = 24
SCOPE = "metadata_sample_of_exposed_FIT144_not_independent_evaluation"


def digest(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, separators=(",", ":")).encode()).hexdigest()


def component_rows(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    result = []
    for row in rows:
        if (set(row) != {"id", "component"} or type(row["id"]) is not int
                or row["id"] <= 0 or not isinstance(row["component"], str) or not row["component"]):
            raise ValueError("id_component_projection_only")
        result.append({"id": row["id"], "component": row["component"]})
    if len({r["id"] for r in result}) != len(result):
        raise ValueError("duplicate_claim_id")
    return result


def select_metadata(pool: Sequence[Mapping[str, Any]], old12: Sequence[Mapping[str, Any]],
                    utility8: Sequence[Mapping[str, Any]], protected_components: Sequence[str]) -> dict[str, Any]:
    """Select once; an unavailable physical prefix later never causes replacement."""
    pool_rows, old_rows, utility_rows = map(component_rows, (pool, old12, utility8))
    protected = set(protected_components)
    if any(not isinstance(c, str) or not c for c in protected):
        raise ValueError("protected_component_type")
    known: dict[int, str] = {}
    for row in pool_rows + old_rows + utility_rows:
        if row["id"] in known and known[row["id"]] != row["component"]:
            raise ValueError("claim_component_drift")
        known[row["id"]] = row["component"]
    if protected & {r["component"] for r in pool_rows}:
        raise ValueError("FIT_pool_contains_protected_component")
    excluded = protected | {r["component"] for r in old_rows + utility_rows}
    available: dict[str, list[int]] = {}
    for row in pool_rows:
        if row["component"] not in excluded:
            available.setdefault(row["component"], []).append(row["id"])
    order = sorted(available, key=lambda c: (digest([SALT, "component", c]), c))
    selected = [{"id": min(available[c], key=lambda i: (digest([SALT, "claim", c, i]), i)),
                 "component": c} for c in order[:MAX_CLAIMS]]
    count = len(selected)
    return {"protocol": PROTOCOL, "scope": SCOPE, "salt": SALT,
            "ordering": "sha256(compact_utf8_json([salt,kind,component,(claim_id)])); deterministic lexical/id tie",
            "pool": sorted(pool_rows, key=lambda r: r["id"]),
            "excluded_components": sorted(excluded), "selected": selected,
            "counts": {"pool_claims": len(pool_rows), "pool_components": len({r["component"] for r in pool_rows}),
                "old_claims": len(old_rows), "utility_claims": len(utility_rows),
                "excluded_pool_claims": len(pool_rows) - sum(map(len, available.values())),
                "eligible_claims": sum(map(len, available.values())), "eligible_components": len(available),
                "selected_claims": count, "shortfall": MAX_CLAIMS - count},
            "ordered_ids_sha256": digest([r["id"] for r in selected]),
            "caps": {"natural_generations": 5 * count, "branch_generations": 2 * count,
                "physical_generations": 7 * count, "rerank_requests": 2 * count, "rerank_pairs": 40 * count,
                "natural_tool_calls_including_retrieval": 5 * count, "new_branch_tool_calls": 2 * count,
                "episode_seconds": 120, "episode_time_bound_seconds": 3 * count * 120},
            "replacement_allowed": False, "label_quotas": False,
            "gold_decoded": False, "protected_split_read": False, "gpu_authorized": False}
