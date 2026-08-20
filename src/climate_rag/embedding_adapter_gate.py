from __future__ import annotations

from collections.abc import Iterable, Mapping
from typing import Any

from .models import Claim


def heldout_query_texts(rows: Iterable[Mapping[str, Any]]) -> set[str]:
    """Recover the exact query texts present in the claim-grouped eval JSONL."""

    texts: set[str] = set()
    for row in rows:
        messages = row.get("messages")
        if not isinstance(messages, list):
            raise TypeError("each eval row must contain a messages list")
        user_messages = [
            message.get("content")
            for message in messages
            if isinstance(message, dict) and message.get("role") == "user"
        ]
        if len(user_messages) != 1 or not isinstance(user_messages[0], str):
            raise ValueError("each eval row must contain exactly one textual user query")
        texts.add(user_messages[0])
    if not texts:
        raise ValueError("eval dataset contains no query texts")
    return texts


def select_heldout_claims(
    claims: Mapping[str, Claim], query_texts: set[str]
) -> dict[str, Claim]:
    selected = {claim_id: claim for claim_id, claim in claims.items() if claim.text in query_texts}
    selected_texts = {claim.text for claim in selected.values()}
    missing = query_texts - selected_texts
    if missing:
        raise ValueError(f"{len(missing)} eval query texts do not resolve to source claims")
    return selected


def full_corpus_promotion_decision(
    comparisons: Mapping[str, Mapping[str, float | int]],
) -> dict[str, object]:
    """Promote only on significant Recall@5 lift and mean non-regression elsewhere."""

    recall = comparisons["recall@5"]
    secondary = ("mrr@10", "ndcg@10", "evidence_f1")
    primary_pass = (
        float(recall["mean_difference"]) > 0
        and float(recall["ci_lower"]) > 0
    )
    non_regression = all(
        float(comparisons[metric]["mean_difference"]) >= 0 for metric in secondary
    )
    passed = primary_pass and non_regression
    return {
        "candidate_passes_full_corpus_gate": passed,
        "primary_recall_ci_positive": primary_pass,
        "secondary_mean_non_regression": non_regression,
        "decision": "promote_adapter" if passed else "retain_base_encoder",
        "boundary": (
            "The gate uses the untouched official dev claims and full evidence corpus; "
            "it is still an offline project evaluation, not an online production A/B test."
        ),
    }
