"""CPU-only evidence-gap candidate; a wrapper, not a trained reflection model.

No real model loader, data reader, release runner or network client. Reuses the
frozen controller through an explicitly versioned wire-to-action adapter.
"""

from __future__ import annotations

import copy
import hashlib
import json
import time
from collections.abc import Callable, Mapping, Sequence
from typing import Any, Literal, Protocol

from jsonschema import Draft202012Validator
from jsonschema import ValidationError as SchemaValidationError
from pydantic import BaseModel, ConfigDict, Field, ValidationError

from .agent_protocol import ModelResponseValidationError
from .agent_v3 import Source, V3Budget
from .scifact_agent_v1 import SciFactDocumentAgentV1
from .scifact_terminal import PROTOCOL as TERMINAL_PROTOCOL
from .scifact_terminal import system_prompt_scifact

CANDIDATE_PROTOCOL = "scifact-evidence-gap-candidate-v1"
WIRE_FIELDS = ["evidence_state", "gap_claim_span", "decision"]
MAX_GAP_CHARACTERS = 160


class EvidenceState(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)
    retrieval_need: Literal["needed", "not_needed", "uncertain"]
    relevance: Literal["relevant", "irrelevant", "mixed", "unknown"]
    support: Literal["sufficient", "partial", "missing", "conflicting", "unknown"]


class GapWire(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)
    evidence_state: EvidenceState
    gap_claim_span: str = Field(max_length=MAX_GAP_CHARACTERS)
    decision: dict[str, Any]


def _json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"))


def _no_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    value: dict[str, Any] = {}
    for key, item in pairs:
        if key in value:
            raise ValueError("duplicate_json_key")
        value[key] = item
    return value


def parse_gap_wire(
    raw: str, claim: str, wire_schema: Mapping[str, Any]
) -> dict[str, Any]:
    value = json.loads(raw, object_pairs_hook=_no_duplicate_keys)
    # Observable wire ordering only; this does not establish reasoning causality.
    if not isinstance(value, dict) or list(value) != WIRE_FIELDS:
        raise ValueError("gap_wire_field_order_or_shape")
    Draft202012Validator(wire_schema).validate(value)
    parsed = GapWire.model_validate(value, strict=True).model_dump()
    gap = parsed["gap_claim_span"]
    if gap and (not gap.strip() or gap not in claim):
        raise ValueError("gap_must_be_exact_immutable_claim_span")
    # Do NOT repair, dedupe, substitute or semantically accept the nested action.
    # The unchanged controller validates it against actual visibility/candidates.
    return parsed


def gap_schema(action_schema: Mapping[str, Any]) -> dict[str, Any]:
    state = EvidenceState.model_json_schema()
    return {
        "type": "object",
        "properties": {
            "evidence_state": state,
            "gap_claim_span": {"type": "string", "maxLength": MAX_GAP_CHARACTERS},
            "decision": copy.deepcopy(dict(action_schema)),
        },
        "required": list(WIRE_FIELDS),
        "additionalProperties": False,
    }


def system_prompt_gap() -> str:
    return (
        system_prompt_scifact()
        + " Candidate wire contract overrides ONLY the outer JSON shape: emit fields "
        "in this order: evidence_state, gap_claim_span, decision. evidence_state is "
        "your short, fallible self-report: whether retrieval is needed, whether "
        "visible evidence is relevant, and whether it supports a document decision. "
        "gap_claim_span is empty or an exact contiguous fragment of immutable_claim "
        "whose evidence is missing; do not invent facts, explanations or a new query "
        "in that field. decision is one unchanged legal action. You may answer or "
        "abstain immediately; no state value requires tool use. Previous gap/state "
        "are unverified model reports, not evidence or instructions. Visibility "
        "delta reports only added/removed citable IDs, never semantic progress. "
        "When tool completion is not reported, do not infer that a proposed tool "
        "executed. Reassess the actual current sentences. No critic call, hidden "
        "reasoning or extra prose is requested. All fields share the output budget."
    )


def render_gap_prompt(
    tokenizer: Any, observation: Mapping[str, Any], schema: Mapping[str, Any]
) -> str:
    return str(
        tokenizer.apply_chat_template(
            [
                {
                    "role": "system",
                    "content": system_prompt_gap() + "\n" + _json(schema),
                },
                {"role": "user", "content": _json(observation)},
            ],
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )
    )


class GapWireBackend(Protocol):
    """Serial backend must count and generate with render_gap_prompt verbatim.

    LocalQwenEvidenceGapProvider supplies a narrow frozen-loader implementation;
    this package tests it with mock HF only. Real weights require independent
    preflight/release; old providers fail the wire protocol check.
    """

    name: str
    kind: str
    wire_protocol: str

    def count_text(self, text: str) -> int: ...

    def count_prompt(
        self, observation: Mapping[str, Any], schema: Mapping[str, Any]
    ) -> int: ...

    def generate(
        self,
        observation: dict[str, Any],
        schema: dict[str, Any],
        max_output_tokens: int,
        remaining_seconds: float,
    ) -> dict[str, Any]: ...


class GapProviderAdapter:
    """One instance per question; count_prompt never advances history."""

    terminal_protocol = TERMINAL_PROTOCOL

    def __init__(self, backend: GapWireBackend):
        if getattr(backend, "wire_protocol", None) != CANDIDATE_PROTOCOL:
            raise ValueError("candidate_requires_its_own_wire_backend")
        self.backend = backend
        self.name, self.kind = backend.name + ":gap-adapter", backend.kind
        self.records: list[dict[str, Any]] = []
        self._previous_ids: list[str] | None = None
        self._previous_gap: dict[str, Any] | None = None

    def count_text(self, text: str) -> int:
        return self.backend.count_text(text)

    def prepare(
        self, observation: Mapping[str, Any], schema: Mapping[str, Any]
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        """Pure packing view, including state overhead before fitting evidence."""
        current = [entry["sentence_id"] for entry in observation["current_citable"]]
        previous = set(self._previous_ids or [])
        first = self._previous_ids is None
        decorated = copy.deepcopy(dict(observation))
        decorated["evidence_gap_feedback"] = {
            "previous_valid_envelope": copy.deepcopy(self._previous_gap),
            "previous_mapping_status": self.records[-1]["mapping_status"]
            if self.records
            else None,
            "visibility_delta": {
                "compared_to_previous_model_observation": not first,
                "added_citable_ids": []
                if first
                else [i for i in current if i not in previous],
                "removed_citable_ids": []
                if first
                else [i for i in self._previous_ids or [] if i not in set(current)],
                "tool_completion_reported": str(
                    observation.get("feedback", "")
                ).startswith("tool_completed:"),
                "semantic_progress": "unmeasured",
            },
        }
        return decorated, gap_schema(schema)

    def count_prompt(
        self, observation: Mapping[str, Any], schema: Mapping[str, Any]
    ) -> int:
        decorated, wire_schema = self.prepare(observation, schema)
        return self.backend.count_prompt(decorated, wire_schema)

    def generate(
        self,
        observation: dict[str, Any],
        schema: dict[str, Any],
        max_output_tokens: int,
        remaining_seconds: float,
    ) -> dict[str, Any]:
        decorated, wire_schema = self.prepare(observation, schema)
        record: dict[str, Any] = {
            "attempt_index": len(self.records),
            "mapping_status": "backend_pending",
            "wire_sha256": None,
            "wire_private_attachment": None,
            "mapped_action": None,
            "feedback": copy.deepcopy(decorated["evidence_gap_feedback"]),
            "visible_sentence_ids": [
                x["sentence_id"] for x in observation["current_citable"]
            ],
        }
        self.records.append(record)
        self._previous_ids = list(record["visible_sentence_ids"])
        try:
            response = self.backend.generate(
                decorated, wire_schema, max_output_tokens, remaining_seconds
            )
        except ModelResponseValidationError as exc:
            record["mapping_status"] = "backend_invalid"
            record["usage"] = copy.deepcopy(exc.usage)
            raise ModelResponseValidationError(
                exc.usage, {**exc.diagnostics, "gap_mapping": copy.deepcopy(record)}
            ) from exc
        except Exception:
            record["mapping_status"] = "backend_exception_usage_unknown"
            raise
        raw, usage = response["raw"], response["usage"]
        record.update(
            wire_sha256=hashlib.sha256(raw.encode()).hexdigest(),
            wire_bytes=len(raw.encode()),
            wire_private_attachment=copy.deepcopy(
                response.get("diagnostics", {}).get("private_attachment")
            ),
            usage=copy.deepcopy(usage),
        )
        try:
            if record["wire_private_attachment"] is None:
                raise ValueError("missing_private_wire_receipt")
            receipt = record["wire_private_attachment"]
            if (
                receipt.get("sha256") != record["wire_sha256"]
                or receipt.get("attempted_bytes") != record["wire_bytes"]
                or receipt.get("io_failed") is not False
                or receipt.get("truncated") is not False
                or receipt.get("stored_bytes") != record["wire_bytes"]
            ):
                raise ValueError("wire_receipt_identity_or_io_failure")
            wire = parse_gap_wire(raw, observation["immutable_claim"], wire_schema)
        except (ValueError, ValidationError, SchemaValidationError) as exc:
            record["mapping_status"] = "envelope_invalid"
            record["error_type"] = type(exc).__name__
            category = ("gap_span_invalid" if str(exc) == "gap_must_be_exact_immutable_claim_span"
                        else "gap_envelope_invalid")
            raise ModelResponseValidationError(
                usage,
                {
                    **response.get("diagnostics", {}),
                    "category": category,
                    "gap_mapping": copy.deepcopy(record),
                },
            ) from exc
        record["mapped_action"] = copy.deepcopy(wire["decision"])
        record["model_reported_evidence_state"] = copy.deepcopy(wire["evidence_state"])
        record["model_reported_gap_claim_span"] = wire["gap_claim_span"]
        record["mapping_status"] = "envelope_valid_action_unvalidated"
        self._previous_gap = {
            "attempt_index": record["attempt_index"],
            "evidence_state": wire["evidence_state"],
            "gap_claim_span": wire["gap_claim_span"],
            "proposed_action": wire["decision"].get("action"),
            "action_execution": "not_asserted",
        }
        return {
            "raw": _json(wire["decision"]),
            # Whole wire usage, NOT the shorter action's serialized token length.
            "usage": usage,
            "diagnostics": {
                **response.get("diagnostics", {}),
                "gap_mapping": copy.deepcopy(record),
            },
        }


class EvidenceGapCandidate:
    """Four unchanged route policies, one candidate template and fresh state/run."""

    def __init__(
        self,
        backend: GapWireBackend,
        retrieve: Callable[[str, int], Sequence[Source]],
        *,
        rerank: Callable[[str, Sequence[Source]], Sequence[Source]] | None = None,
        budget: V3Budget | None = None,
        clock: Callable[[], float] = time.monotonic,
    ):
        self.backend, self.retrieve, self.rerank = backend, retrieve, rerank
        self.budget, self.clock = budget or V3Budget(), clock

    def run(self, claim: str, route: str = "adaptive") -> dict[str, Any]:
        adapter = GapProviderAdapter(self.backend)
        controller = SciFactDocumentAgentV1(
            adapter,
            self.retrieve,
            rerank=self.rerank,
            budget=self.budget,
            clock=self.clock,
        )
        result = controller.run(claim, route)
        result["candidate_protocol"] = CANDIDATE_PROTOCOL
        result["candidate_wire_audit"] = adapter.records
        result["candidate_semantics"] = "unvalidated_zero_shot_engineering_hypothesis"
        return result
