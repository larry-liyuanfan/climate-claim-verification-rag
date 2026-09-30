"""Versioned action-discriminated wire contract; not grammar-constrained decoding."""
from __future__ import annotations

from typing import Annotated, Any, Literal, Union

from pydantic import BaseModel, ConfigDict, Field, TypeAdapter


class ModelResponseValidationError(ValueError):
    def __init__(self, usage: dict[str, int], diagnostics: dict[str, Any] | None = None,
                 repair_context: dict[str, Any] | None = None) -> None:
        super().__init__("local model generated an invalid structured response")
        self.usage = usage
        self.diagnostics = diagnostics or {}
        self.repair_context = repair_context or {}


class ActionBase(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)
    reason: str = Field(min_length=1, max_length=500, pattern=r"\S")


class WireStatement(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)
    text: str = Field(min_length=1, max_length=800, pattern=r"\S")
    evidence_id: str = Field(min_length=1, max_length=300)
    quote: str = Field(min_length=1, max_length=1500, pattern=r"\S")


class RewriteAction(ActionBase):
    action: Literal["rewrite"]
    query: str = Field(min_length=1, max_length=2000, pattern=r"\S")


class RerankAction(ActionBase):
    action: Literal["rerank"]


class AnswerAction(ActionBase):
    action: Literal["answer"]
    evidence_assessment: Literal["sufficient"]
    label: Literal["SUPPORTS", "REFUTES"]
    statements: list[WireStatement] = Field(min_length=1, max_length=3)


class AbstainAction(ActionBase):
    action: Literal["abstain"]
    evidence_assessment: Literal["insufficient", "conflicting", "unknown"]


WireDecision = Annotated[Union[RewriteAction, RerankAction, AnswerAction, AbstainAction],
                         Field(discriminator="action")]
WIRE_ADAPTER: TypeAdapter[WireDecision] = TypeAdapter(WireDecision)


def wire_schema() -> dict[str, Any]:
    return WIRE_ADAPTER.json_schema()


def parse_wire(value: Any) -> dict[str, Any]:
    # No coercion, dropping extra keys, JSON repair or action substitution.
    return WIRE_ADAPTER.validate_python(value, strict=True).model_dump()
