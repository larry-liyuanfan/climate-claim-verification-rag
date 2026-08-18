from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Mapping, Sequence


VALID_LABELS = frozenset({"SUPPORTS", "REFUTES", "NOT_ENOUGH_INFO", "DISPUTED"})


@dataclass(frozen=True, slots=True)
class EvidenceDocument:
    evidence_id: str
    text: str
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.evidence_id:
            raise ValueError("evidence_id must not be empty")
        if not isinstance(self.text, str):
            raise TypeError("evidence text must be a string")


@dataclass(frozen=True, slots=True)
class Claim:
    claim_id: str
    text: str
    label: str | None = None
    evidence_ids: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if not self.claim_id:
            raise ValueError("claim_id must not be empty")
        if not isinstance(self.text, str) or not self.text.strip():
            raise ValueError("claim text must not be empty")
        if self.label is not None and self.label not in VALID_LABELS:
            raise ValueError(f"unsupported claim label: {self.label}")


@dataclass(frozen=True, slots=True)
class RankedDocument:
    evidence_id: str
    score: float
    rank: int
    text: str = ""
    source: str = ""
    features: Mapping[str, float] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class Prediction:
    claim_id: str
    evidence_ids: tuple[str, ...]
    label: str | None = None
    label_scores: Mapping[str, float] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.label is not None and self.label not in VALID_LABELS:
            raise ValueError(f"unsupported prediction label: {self.label}")

    def to_official_dict(self, claim_text: str | None = None) -> dict[str, Any]:
        result: dict[str, Any] = {
            "claim_label": self.label,
            "evidences": list(self.evidence_ids),
        }
        if claim_text is not None:
            result["claim_text"] = claim_text
        if self.label_scores:
            result["label_scores"] = dict(self.label_scores)
        return result


def claim_from_mapping(claim_id: str, value: Mapping[str, Any]) -> Claim:
    raw_evidence = value.get("evidences", value.get("evidence_ids", ())) or ()
    label = value.get("claim_label", value.get("label"))
    return Claim(
        claim_id=str(claim_id),
        text=str(value.get("claim_text", value.get("text", ""))),
        label=str(label).upper() if label is not None else None,
        evidence_ids=tuple(str(item) for item in raw_evidence),
    )


def prediction_from_mapping(claim_id: str, value: Mapping[str, Any]) -> Prediction:
    raw_evidence: Sequence[Any] = value.get("evidences", value.get("evidence_ids", ())) or ()
    label = value.get("claim_label", value.get("label"))
    scores = value.get("label_scores", {}) or {}
    return Prediction(
        claim_id=str(claim_id),
        evidence_ids=tuple(str(item) for item in raw_evidence),
        label=str(label).upper() if label is not None else None,
        label_scores={str(key).upper(): float(score) for key, score in scores.items()},
    )

