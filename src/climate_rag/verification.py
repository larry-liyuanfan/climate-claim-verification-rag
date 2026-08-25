from __future__ import annotations

import json
import os
import re
import time
import urllib.error
import urllib.request
from collections.abc import Sequence
from dataclasses import asdict, dataclass, field
from typing import Any, Protocol

from .models import RankedDocument
from .tokenize import climate_tokenize

VERDICT_LABELS = frozenset({"SUPPORTS", "REFUTES", "NOT_ENOUGH_INFO"})


@dataclass(frozen=True, slots=True)
class Citation:
    evidence_id: str
    quote: str


@dataclass(frozen=True, slots=True)
class VerificationResult:
    label: str
    confidence: float
    citations: tuple[Citation, ...] = ()
    rationale: str = ""
    abstain_reason: str | None = None
    provider: str = ""
    usage: dict[str, int] = field(default_factory=dict)
    latency_ms: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class VerificationProvider(Protocol):
    name: str

    def verify(
        self, claim: str, evidence: Sequence[RankedDocument]
    ) -> VerificationResult: ...


class AbstainingVerifier:
    name = "not-configured"

    def verify(
        self, claim: str, evidence: Sequence[RankedDocument]
    ) -> VerificationResult:
        del claim, evidence
        return VerificationResult(
            label="NOT_ENOUGH_INFO",
            confidence=0.0,
            abstain_reason="verifier_not_configured",
            provider=self.name,
        )


def normalise_claim(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def extract_constraints(text: str) -> dict[str, list[str]]:
    years = sorted(set(re.findall(r"\b(?:18|19|20|21)\d{2}\b", text)))
    numbers = sorted(set(re.findall(r"(?<!\w)[+-]?\d+(?:\.\d+)?%?", text)))
    proper = sorted(
        set(re.findall(r"\b[A-Z][A-Za-z-]{2,}(?:\s+[A-Z][A-Za-z-]{2,})*", text))
    )
    return {"years": years, "numbers": numbers, "entities": proper[:12]}


def decompose_claim(text: str, *, max_queries: int = 2) -> list[str]:
    normalised = normalise_claim(text)
    parts = [
        part.strip(" ,;:-")
        for part in re.split(
            r"\b(?:and|while|whereas|because|but)\b|[;]",
            normalised,
            flags=re.IGNORECASE,
        )
        if part.strip(" ,;:-")
    ]
    candidates = [
        part
        for part in parts
        if part != normalised and len(climate_tokenize(part)) >= 4
    ]
    return candidates[:max_queries]


def evidence_coverage(claim: str, evidence: Sequence[RankedDocument]) -> float:
    claim_tokens = set(climate_tokenize(claim))
    if not claim_tokens:
        return 0.0
    evidence_tokens: set[str] = set()
    for row in evidence:
        evidence_tokens.update(climate_tokenize(row.text))
    return len(claim_tokens & evidence_tokens) / len(claim_tokens)


def _normalise_quote(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip().casefold()


def validate_verification_result(
    result: VerificationResult, evidence: Sequence[RankedDocument]
) -> tuple[VerificationResult, list[str]]:
    warnings: list[str] = []
    if result.label not in VERDICT_LABELS:
        warnings.append("unsupported_label")
    if not 0.0 <= result.confidence <= 1.0:
        warnings.append("confidence_out_of_range")
    evidence_by_id = {row.evidence_id: row for row in evidence}
    valid_citations: list[Citation] = []
    for citation in result.citations:
        row = evidence_by_id.get(citation.evidence_id)
        if row is None:
            warnings.append(f"unknown_citation:{citation.evidence_id}")
            continue
        quote = _normalise_quote(citation.quote)
        if not quote or quote not in _normalise_quote(row.text):
            warnings.append(f"invalid_quote:{citation.evidence_id}")
            continue
        valid_citations.append(citation)
    if result.label in {"SUPPORTS", "REFUTES"} and not valid_citations:
        warnings.append("verdict_without_valid_citation")
    if warnings:
        return (
            VerificationResult(
                label="NOT_ENOUGH_INFO",
                confidence=0.0,
                citations=tuple(valid_citations),
                rationale="Evidence or citation validation failed; no grounded verdict was emitted.",
                abstain_reason=";".join(sorted(set(warnings))),
                provider=result.provider,
                usage=result.usage,
                latency_ms=result.latency_ms,
            ),
            warnings,
        )
    return result, warnings


class ModelStudioStructuredVerifier:
    """Strict JSON verifier for Model Studio's OpenAI-compatible endpoint.

    Credentials are environment-only. A caller must configure both
    ``DASHSCOPE_API_KEY`` and ``MODEL_STUDIO_BASE_URL``; neither is accepted as a
    persisted CLI argument or written to traces.
    """

    def __init__(
        self,
        *,
        model: str = "qwen3.7-plus",
        timeout_seconds: float = 45.0,
        max_output_tokens: int = 800,
    ) -> None:
        self.model = model
        self.name = f"model-studio:{model}"
        self.timeout_seconds = timeout_seconds
        self.max_output_tokens = max_output_tokens

    def verify(
        self, claim: str, evidence: Sequence[RankedDocument]
    ) -> VerificationResult:
        api_key = os.getenv("DASHSCOPE_API_KEY")
        base_url = os.getenv("MODEL_STUDIO_BASE_URL")
        if not api_key or not base_url:
            raise RuntimeError(
                "DASHSCOPE_API_KEY and MODEL_STUDIO_BASE_URL are required for Model Studio"
            )
        evidence_payload = [
            {"evidence_id": row.evidence_id, "text": row.text} for row in evidence
        ]
        schema_instruction = (
            "Return JSON only with keys label, confidence, citations, rationale, "
            "abstain_reason. label must be SUPPORTS, REFUTES, or NOT_ENOUGH_INFO. "
            "Each citation must contain an evidence_id from the supplied list and an exact "
            "verbatim quote copied from that evidence. If evidence is insufficient, "
            "conflicting, or cannot be cited exactly, return NOT_ENOUGH_INFO."
        )
        body = json.dumps(
            {
                "model": self.model,
                "messages": [
                    {"role": "system", "content": schema_instruction},
                    {
                        "role": "user",
                        "content": json.dumps(
                            {"claim": claim, "evidence": evidence_payload},
                            ensure_ascii=False,
                        ),
                    },
                ],
                "response_format": {"type": "json_object"},
                "temperature": 0,
                "max_tokens": self.max_output_tokens,
            },
            ensure_ascii=False,
        ).encode("utf-8")
        endpoint = base_url.rstrip("/") + "/chat/completions"
        request = urllib.request.Request(
            endpoint,
            data=body,
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
            },
            method="POST",
        )
        started = time.perf_counter()
        try:
            with urllib.request.urlopen(
                request, timeout=self.timeout_seconds
            ) as response:
                payload = json.load(response)
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")
            raise RuntimeError(
                f"Model Studio verification failed ({exc.code}): {detail[:500]}"
            ) from exc
        content = payload["choices"][0]["message"]["content"]
        parsed = json.loads(content)
        usage = payload.get("usage", {}) or {}
        result = VerificationResult(
            label=str(parsed.get("label", "NOT_ENOUGH_INFO")).upper(),
            confidence=float(parsed.get("confidence", 0.0)),
            citations=tuple(
                Citation(str(item.get("evidence_id", "")), str(item.get("quote", "")))
                for item in parsed.get("citations", [])
            ),
            rationale=str(parsed.get("rationale", "")),
            abstain_reason=(
                str(parsed["abstain_reason"]) if parsed.get("abstain_reason") else None
            ),
            provider=self.name,
            usage={
                "input_tokens": int(usage.get("prompt_tokens", 0)),
                "output_tokens": int(usage.get("completion_tokens", 0)),
            },
            latency_ms=(time.perf_counter() - started) * 1000.0,
        )
        return validate_verification_result(result, evidence)[0]


class FixedFixtureVerifier:
    """Deterministic contract fixture for tests; never a semantic production model."""

    name = "fixed-fixture"

    def __init__(self, label: str = "SUPPORTS") -> None:
        self.label = label

    def verify(
        self, claim: str, evidence: Sequence[RankedDocument]
    ) -> VerificationResult:
        del claim
        if not evidence:
            return VerificationResult(
                label="NOT_ENOUGH_INFO",
                confidence=0.0,
                abstain_reason="empty_evidence",
                provider=self.name,
            )
        row = evidence[0]
        result = VerificationResult(
            label=self.label,
            confidence=0.9,
            citations=(Citation(row.evidence_id, row.text),),
            rationale="Deterministic contract fixture.",
            provider=self.name,
        )
        return validate_verification_result(result, evidence)[0]
