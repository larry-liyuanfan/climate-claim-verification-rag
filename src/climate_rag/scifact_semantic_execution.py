"""Gold-free semantic execution bindings and durable exclusive slot helpers."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .evidence_gap_candidate import CANDIDATE_PROTOCOL
from .scifact_bounded_smoke import bounded_runtime_smoke
from .scifact_semantic_contract import (
    POLICIES, PROMPTS, write_once,
)
from .scifact_semantic_policy import policy_sha
from .scifact_semantic_runtime import run_semantic_slot


class PreflightPersistenceError(RuntimeError):
    """In-memory known costs survive; absence of durable totals is not zero."""

    def __init__(self, report: dict[str, Any]) -> None:
        super().__init__("preflight_summary_not_persisted: known costs remain in partial_report; missing costs are unknown, not zero")
        self.partial_report = report | {"summary_persisted": False}


def provider_binding(backend: Any, policy: str) -> dict[str, Any]:
    if (policy not in POLICIES or backend.policy != policy or backend.gap is not True
            or backend.wire_protocol != CANDIDATE_PROTOCOL or policy_sha(policy) != PROMPTS[policy]):
        raise ValueError("semantic_provider_binding")
    return {"policy": policy, "prompt_sha256": PROMPTS[policy], "gap": True}


def preflight(backend: Any, policy: str, out: Path, source_git: str, protocol_sha: str) -> dict[str, Any]:
    identity = provider_binding(backend, policy) | {"source_git": source_git,
        "protocol_sha256": protocol_sha, "execution_kind": backend.kind}

    def persist(index: int, phase: str, record: dict[str, Any]) -> None:
        write_once(out / f"preflight-{index + 1:02d}-{phase}.json", identity | {"record": record})

    report = bounded_runtime_smoke(backend, persist_case=persist) | identity
    try:
        write_once(out / "runtime-preflight.json", report)
    except Exception as exc:
        raise PreflightPersistenceError(report) from exc
    provider_binding(backend, policy)
    if report["status"] != "passed" or report["attempted_calls"] != 4:
        raise ValueError("semantic_synthetic_preflight_failed")
    return report


def execute_slots(protocol: dict[str, Any], claims: list[dict[str, Any]], backend: Any,
                  retrieve: Any, rerank: Any, corpus: Any, out: Path) -> list[dict[str, Any]]:
    policy, git = backend.policy, protocol["execution_source_git"]
    provider_binding(backend, policy)
    lookup = {r["id"]: r["claim"] for r in claims}
    rows = []
    for index, spec in enumerate(protocol["slots_per_arm"], 1):
        raw_path = out / f"slot-{index:02d}-raw.json"
        backend.start_slot(out / "private-responses" / f"slot-{index:02d}")
        try:
            row = run_semantic_slot(spec["claim_id"], lookup[spec["claim_id"]], spec["route"],
                backend, retrieve, rerank, corpus, git, persist_raw=lambda v: write_once(raw_path, v))
        except Exception as exc:
            if not raw_path.exists():
                raise  # no durable cost ledger: stop, never silently consume a new slot
            raw = json.loads(raw_path.read_bytes())
            row = {k: raw[k] for k in ("claim_id", "route", "arm", "source_git", "prompt_sha256", "comparison_protocol")}
            row.update(result=raw, prediction=None, export_error=type(exc).__name__)
        write_once(out / f"slot-{index:02d}.json", row)
        rows.append(row)
        # Persist unknown usage/errors before stopping; they cannot vanish as zero cost.
        for a in row["result"]["generation_attempts"]:
            r = a.get("diagnostics", {}).get("private_attachment", {})
            if (not a["usage_known"] or r.get("truncated") is not False or r.get("io_failed") is not False
                    or r.get("stored_bytes") != r.get("attempted_bytes")):
                raise ValueError("semantic_wire_or_usage_incomplete_durable_stop")
        if row.get("export_error"):
            raise ValueError("semantic_export_failed_durable_stop")
    return rows
