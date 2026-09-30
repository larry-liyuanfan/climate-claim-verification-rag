"""Versioned LMFE compatibility adapter, NOT the authoritative output contract.

LMFE 0.11.3 interprets integer enum as unquoted StringParsingState: auto-pop
and ListParsingState both count the same item and reject legal orders/whitespace.
NumberParsingState avoids this bug. Only the decoder copy loses integer enum;
the frozen prompt/schema and strict post-generation parse remain unchanged.
LMFE does not reliably implement uniqueItems/maxItems:0. Such outputs are paid
schema failures, never repaired/retried. This adapter is not an ID-validity claim.
"""
from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from .scifact_semantic_contract import encoded, sha

DECODER_VERSION = "component-lmfe0113-number-state-v1"


def decoder_schema(schema: Mapping[str, Any]) -> dict[str, Any]:
    def visit(value: Any) -> Any:
        if isinstance(value, dict):
            return {k: visit(v) for k, v in value.items()
                    if not (k == "enum" and value.get("type") == "integer")}
        if isinstance(value, list):
            return [visit(v) for v in value]
        return value
    return dict(visit(dict(schema)))


def decoder_identity(schema: Mapping[str, Any]) -> dict[str, str]:
    return {"version": DECODER_VERSION, "canonical_schema_sha256": sha(encoded(schema)),
            "decoder_schema_sha256": sha(encoded(decoder_schema(schema)))}
