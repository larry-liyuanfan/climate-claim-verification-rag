"""Private, outcome-selected TRAIN diagnostic contracts; no data loading."""
from __future__ import annotations

import copy
import hashlib
import json
import math
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from .agent_v3 import Source, V3Budget, valid_search_query

PROTOCOL = "scifact-utility8-shared-prefix-v1"
ARMS = ("A", "B", "C")
MAX_GENERATIONS = 56


def identity(value: Any) -> str:
    # Physical insertion order is intentional: it affects the rendered prompt.
    return hashlib.sha256(json.dumps(value, ensure_ascii=False,
                                    separators=(",", ":")).encode()).hexdigest()


@dataclass
class UtilityDiagnostic:
    """Optional controller hook. Default production run does not instantiate it."""
    observe: Callable[[str, dict[str, Any]], None]
    prefix: dict[str, Any] | None = None
    intervention: dict[str, Any] | None = None

    def emit(self, kind: str, value: dict[str, Any]) -> None:
        try:
            self.observe(kind, copy.deepcopy(value))
        except Exception as exc:
            # Never turn a persistence/observer failure into a model repair.
            raise RuntimeError("diagnostic_observer_failed") from exc


def seal_prefix(payload: dict[str, Any]) -> dict[str, Any]:
    return {"payload": copy.deepcopy(payload), "sha256": identity(payload)}


def validate_prefix(prefix: dict[str, Any], claim: str, budget: V3Budget) -> dict[str, Any]:
    p: dict[str, Any] = copy.deepcopy(prefix["payload"])
    if (prefix["sha256"] != identity(p) or p["protocol"] != PROTOCOL
            or p["claim"] != claim or p["budget"] != budget.model_dump()
            or budget != V3Budget() or p["route"] != "adaptive"
            or not math.isfinite(p["elapsed_seconds"]) or p["elapsed_seconds"] < 0
            or p["tool_calls"] != 1 or len(p["events"]) != 1
            or p["events"][0].get("tool") != "retrieve"
            or p["events"][0].get("status") != "completed"
            or not p["attempt"].get("diagnostics", {}).get("physical_attempt_id")):
        raise ValueError("invalid_shared_prefix")
    rows = p["sources"]
    sources = {r["source_id"]: Source(r["source_id"], r["title"], tuple(r["sentences"])) for r in rows}
    aliases = p["aliases"]
    if (len(sources) != len(rows) or set(sources) != set(aliases)
            or list(aliases.values()) != [f"c{i}" for i in range(len(aliases))]
            or len(set(p["candidates"])) != len(p["candidates"])
            or set(p["candidates"]) != set(aliases.values())
            or p["selected"] != p["candidates"][:budget.context_k]
            or any(sources[r["source_id"]].text_sha256 != r["sha256"] for r in rows)):
        raise ValueError("shared_source_or_alias_changed")
    return p


def read_intervention(prefix: dict[str, Any]) -> tuple[dict[str, Any] | None, str | None]:
    """A attempt ZERO only. Never fall through an invalid reference to a new one."""
    p = prefix["payload"]
    if p["attempt"]["status"] != "valid_decision" or p["decision"] is None:
        return None, "attempt0_invalid"
    d = p["decision"]
    if d["action"] == "read" and d["source_ids"] == p["selected"]:
        return None, "attempt0_read_loop"
    if d["action"] == "rewrite" and not valid_search_query(p["claim"], d["query"]):
        return None, "attempt0_rewrite_invalid"
    references = (d.get("source_ids", []) if d["action"] == "read" else
                  [r["source_id"] for r in d.get("documents", [])])
    if references:
        # Validate the entire reference list before choosing its first entry.
        if len(set(references)) != len(references) or not set(references) <= set(p["candidates"]):
            return None, "attempt0_reference_invalid"
        chosen = references[0]
    else:
        previews = p["frame"]["observation"]["preview_only"]
        if not previews:
            return None, "initial_preview_unavailable"
        chosen = previews[0]["source_id"]
    if chosen not in p["candidates"] or [chosen] == p["selected"]:
        return None, "initial_read_unavailable_or_loop"
    return {"action": "read", "source_ids": [chosen]}, None
