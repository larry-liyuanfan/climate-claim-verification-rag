"""Versioned original-SciFact gold scoring, separate from inference.

Metric definitions match allenai/scifact at the pinned SHA below, on its
supported gold format. The wrapper additionally rejects incomplete prediction
matrices and unknown/out-of-range evidence instead of dropping denominators.
No claim-level any-hit metric is named official SciFact F1.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from .scifact_grounding import (
    LABEL_TO_PROJECT,
    OFFICIAL_CODE_SHA,
    Abstract,
    GoldClaim,
    _integer,
)


@dataclass(frozen=True)
class PredictedAbstract:
    label: str
    sentences: tuple[int, ...]


@dataclass(frozen=True)
class ClaimPrediction:
    claim_id: int
    evidence: Mapping[int, PredictedAbstract]


def parse_prediction(
    row: Mapping[str, Any], corpus: Mapping[int, Abstract]
) -> ClaimPrediction:
    if set(row) != {"id", "evidence"} or not isinstance(row["evidence"], dict):
        raise ValueError("prediction requires exactly id and evidence")
    result: dict[int, PredictedAbstract] = {}
    for key, value in row["evidence"].items():
        if not isinstance(key, str) or not key.isdecimal() or str(int(key)) != key:
            raise ValueError("noncanonical predicted document ID")
        doc_id = int(key)
        if doc_id not in corpus or set(value) != {"label", "sentences"}:
            raise ValueError("unknown document or unexpected prediction fields")
        if value["label"] not in LABEL_TO_PROJECT:
            raise ValueError("prediction label must be SUPPORT or CONTRADICT")
        if not isinstance(value["sentences"], list):
            raise ValueError("prediction sentences must be a list")
        indices = tuple(_integer(i) for i in value["sentences"])
        if any(i >= len(corpus[doc_id].sentences) for i in indices):
            raise ValueError("out-of-range predicted sentence index")
        # Intentionally keep order and duplicates: the reference scorer does.
        result[doc_id] = PredictedAbstract(value["label"], indices)
    return ClaimPrediction(_integer(row["id"]), result)


def _metrics(correct: int, predicted: int, relevant: int) -> dict[str, float | int]:
    precision = correct / predicted if predicted else 0.0
    recall = correct / relevant if relevant else 0.0
    return {
        "correct": correct,
        "predicted": predicted,
        "relevant": relevant,
        "precision": precision,
        "recall": recall,
        "f1": 2 * precision * recall / (precision + recall)
        if precision + recall
        else 0.0,
    }


def score_original(
    gold: Sequence[GoldClaim], predictions: Sequence[ClaimPrediction]
) -> dict[str, Any]:
    expected = {g.claim_id: g for g in gold}
    predicted = {p.claim_id: p for p in predictions}
    if not expected or len(expected) != len(gold):
        raise ValueError("empty/duplicate gold matrix")
    if len(predicted) != len(predictions) or set(expected) != set(predicted):
        raise ValueError("missing/extra/duplicate prediction claim IDs")
    counts = {
        key: 0
        for key in (
            "gold_docs",
            "pred_docs",
            "doc_label",
            "doc_rat",
            "gold_sents",
            "pred_sents",
            "sent_sel",
            "sent_label",
        )
    }
    nei = nei_false_answers = mixed = abstained = 0
    for claim_id, claim in expected.items():
        pred = predicted[claim_id]
        nei += not claim.evidence
        nei_false_answers += not claim.evidence and bool(pred.evidence)
        mixed += claim.label_scope == "MIXED"
        abstained += not pred.evidence
        counts["gold_docs"] += len(claim.evidence)
        for gold_rats in claim.evidence.values():
            if len({r.label for r in gold_rats}) != 1:
                raise ValueError(
                    "official metrics do not support mixed labels within document"
                )
            all_indices = [i for r in gold_rats for i in r.sentences]
            if len(set(all_indices)) != len(all_indices):
                raise ValueError(
                    "official metrics assume disjoint alternative rationale sets"
                )
            counts["gold_sents"] += len(all_indices)
        for doc_id, doc in pred.evidence.items():
            if doc.label not in LABEL_TO_PROJECT:
                raise ValueError("NEI must be empty evidence, not a document label")
            counts["pred_docs"] += 1
            counts["pred_sents"] += len(doc.sentences)
            rats = claim.evidence.get(doc_id)
            if rats is None:
                continue
            label_ok = doc.label == rats[0].label
            counts["doc_label"] += label_ok
            first_three = set(doc.sentences[:3])
            counts["doc_rat"] += label_ok and any(
                set(r.sentences) <= first_three for r in rats
            )
            selected = set(doc.sentences)
            credit = {
                i for r in rats if set(r.sentences) <= selected for i in r.sentences
            }
            counts["sent_sel"] += len(credit)
            counts["sent_label"] += len(credit) * label_ok
    return {
        "schema_version": "scifact-original-gold-score-v1",
        "official_reference_sha": OFFICIAL_CODE_SHA,
        "claim_count": len(gold),
        "metrics": {
            "abstract_label_only": _metrics(
                counts["doc_label"], counts["pred_docs"], counts["gold_docs"]
            ),
            "abstract_rationalized": _metrics(
                counts["doc_rat"], counts["pred_docs"], counts["gold_docs"]
            ),
            "sentence_selection": _metrics(
                counts["sent_sel"], counts["pred_sents"], counts["gold_sents"]
            ),
            "sentence_label": _metrics(
                counts["sent_label"], counts["pred_sents"], counts["gold_sents"]
            ),
        },
        "separate_diagnostics_not_official_f1": {
            "nei_claim_count": nei,
            "nei_claims_with_false_evidence": nei_false_answers,
            "nei_false_evidence_rate": nei_false_answers / nei if nei else None,
            "mixed_label_claim_count": mixed,
            "empty_evidence_predictions": abstained,
            "claim_verdict_accuracy": None,
            "free_text_entailment": None,
        },
    }
