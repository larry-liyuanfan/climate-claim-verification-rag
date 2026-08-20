from __future__ import annotations

import hashlib
from collections import defaultdict
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from typing import Any

from .models import Claim, EvidenceDocument

DEFAULT_QUERY_INSTRUCTION = (
    "Given a climate claim, retrieve evidence that supports, refutes, or materially "
    "qualifies the claim."
)


@dataclass(frozen=True, slots=True)
class EmbeddingTrainingDataset:
    train_rows: tuple[dict[str, Any], ...]
    eval_rows: tuple[dict[str, Any], ...]
    metrics: Mapping[str, int | float]


def _normalise_text(value: str) -> str:
    return " ".join(value.split()).casefold()


def _is_eval_claim(claim_id: str, eval_ratio: float, split_seed: int) -> bool:
    if not 0.0 <= eval_ratio < 1.0:
        raise ValueError("eval_ratio must be in [0, 1)")
    digest = hashlib.sha256(f"{split_seed}:{claim_id}".encode()).digest()
    bucket = int.from_bytes(digest[:8], "big") / float(2**64)
    return bucket < eval_ratio


def _messages(text: str, *, instruction: str | None = None) -> list[dict[str, str]]:
    result: list[dict[str, str]] = []
    if instruction:
        result.append({"role": "system", "content": instruction})
    result.append({"role": "user", "content": text})
    return result


def build_swift_infonce_dataset(
    claims: Mapping[str, Claim],
    evidence: Iterable[EvidenceDocument],
    hard_negatives: Iterable[Mapping[str, Any]],
    *,
    negatives_per_positive: int = 4,
    eval_ratio: float = 0.1,
    split_seed: int = 45,
    query_instruction: str = DEFAULT_QUERY_INSTRUCTION,
) -> EmbeddingTrainingDataset:
    """Build claim-grouped Qwen3-Embedding InfoNCE rows for ms-swift.

    The claim ID, rather than the individual positive pair, determines the split.
    This prevents another gold evidence row for the same claim from leaking into
    evaluation. Gold IDs and normalised positive text are removed from the hard
    negative pool before serialisation.
    """

    if negatives_per_positive < 1:
        raise ValueError("negatives_per_positive must be positive")

    negatives_by_claim: dict[str, list[tuple[str, str]]] = defaultdict(list)
    required_ids: set[str] = set()
    for claim in claims.values():
        required_ids.update(claim.evidence_ids)
    for row in hard_negatives:
        claim_id = str(row.get("claim_id", ""))
        evidence_id = str(row.get("evidence_id", ""))
        if claim_id not in claims or not evidence_id:
            continue
        text = str(row.get("text", ""))
        negatives_by_claim[claim_id].append((evidence_id, text))
        required_ids.add(evidence_id)

    evidence_by_id: dict[str, str] = {}
    for document in evidence:
        if document.evidence_id in required_ids:
            evidence_by_id[document.evidence_id] = document.text

    train_rows: list[dict[str, Any]] = []
    eval_rows: list[dict[str, Any]] = []
    skipped_missing_positive = 0
    skipped_insufficient_negatives = 0
    removed_false_negatives = 0

    for claim_id in sorted(claims):
        claim = claims[claim_id]
        gold_ids = set(claim.evidence_ids)
        candidate_negatives: list[tuple[str, str]] = []
        seen_negative_ids: set[str] = set()
        for evidence_id, embedded_text in negatives_by_claim.get(claim_id, ()):
            if evidence_id in gold_ids or evidence_id in seen_negative_ids:
                removed_false_negatives += int(evidence_id in gold_ids)
                continue
            text = embedded_text.strip() or evidence_by_id.get(evidence_id, "").strip()
            if not text:
                continue
            seen_negative_ids.add(evidence_id)
            candidate_negatives.append((evidence_id, text))

        for positive_id in claim.evidence_ids:
            positive_text = evidence_by_id.get(positive_id, "").strip()
            if not positive_text:
                skipped_missing_positive += 1
                continue
            positive_normalised = _normalise_text(positive_text)
            chosen: list[str] = []
            seen_texts: set[str] = {positive_normalised}
            for _, negative_text in candidate_negatives:
                normalised = _normalise_text(negative_text)
                if not normalised or normalised in seen_texts:
                    removed_false_negatives += 1
                    continue
                seen_texts.add(normalised)
                chosen.append(negative_text)
                if len(chosen) == negatives_per_positive:
                    break
            if len(chosen) < negatives_per_positive:
                skipped_insufficient_negatives += 1
                continue

            output_row = {
                "messages": _messages(claim.text, instruction=query_instruction),
                "positive_messages": [_messages(positive_text)],
                "negative_messages": [_messages(text) for text in chosen],
            }
            target = (
                eval_rows
                if _is_eval_claim(claim_id, eval_ratio=eval_ratio, split_seed=split_seed)
                else train_rows
            )
            target.append(output_row)

    metrics: dict[str, int | float] = {
        "claim_count": len(claims),
        "required_evidence_count": len(required_ids),
        "resolved_evidence_count": len(evidence_by_id),
        "train_row_count": len(train_rows),
        "eval_row_count": len(eval_rows),
        "negatives_per_positive": negatives_per_positive,
        "eval_ratio": eval_ratio,
        "split_seed": split_seed,
        "skipped_missing_positive": skipped_missing_positive,
        "skipped_insufficient_negatives": skipped_insufficient_negatives,
        "removed_false_negatives": removed_false_negatives,
    }
    return EmbeddingTrainingDataset(tuple(train_rows), tuple(eval_rows), metrics)
