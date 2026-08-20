from climate_rag.embedding_training import build_swift_infonce_dataset
from climate_rag.models import Claim, EvidenceDocument


def test_build_swift_infonce_dataset_is_claim_grouped_and_filters_false_negatives() -> None:
    claims = {
        "claim-a": Claim("claim-a", "Claim A", evidence_ids=("gold-a1", "gold-a2")),
        "claim-b": Claim("claim-b", "Claim B", evidence_ids=("gold-b",)),
    }
    evidence = [
        EvidenceDocument("gold-a1", "Positive A1"),
        EvidenceDocument("gold-a2", "Positive A2"),
        EvidenceDocument("gold-b", "Positive B"),
        EvidenceDocument("n1", "Negative one"),
        EvidenceDocument("n2", "Negative two"),
        EvidenceDocument("n3", "Negative three"),
    ]
    negatives = [
        {"claim_id": "claim-a", "evidence_id": "gold-a1", "text": "Positive A1"},
        {"claim_id": "claim-a", "evidence_id": "n1", "text": "Negative one"},
        {"claim_id": "claim-a", "evidence_id": "n2", "text": "Negative two"},
        {"claim_id": "claim-b", "evidence_id": "n2", "text": "Negative two"},
        {"claim_id": "claim-b", "evidence_id": "n3", "text": "Negative three"},
    ]

    dataset = build_swift_infonce_dataset(
        claims,
        evidence,
        negatives,
        negatives_per_positive=2,
        eval_ratio=0.5,
        split_seed=45,
    )

    all_rows = [*dataset.train_rows, *dataset.eval_rows]
    assert len(all_rows) == 3
    assert all(len(row["negative_messages"]) == 2 for row in all_rows)
    assert all(len(row["positive_messages"]) == 1 for row in all_rows)
    assert dataset.metrics["removed_false_negatives"] >= 1

    # Both positives for claim-a must remain in one split, never across train/eval.
    claim_a_train = sum(row["messages"][-1]["content"] == "Claim A" for row in dataset.train_rows)
    claim_a_eval = sum(row["messages"][-1]["content"] == "Claim A" for row in dataset.eval_rows)
    assert (claim_a_train, claim_a_eval) in {(2, 0), (0, 2)}


def test_build_swift_infonce_dataset_skips_rows_without_enough_negatives() -> None:
    dataset = build_swift_infonce_dataset(
        {"claim": Claim("claim", "Claim", evidence_ids=("gold",))},
        [EvidenceDocument("gold", "Positive"), EvidenceDocument("n1", "Negative")],
        [{"claim_id": "claim", "evidence_id": "n1", "text": "Negative"}],
        negatives_per_positive=2,
        eval_ratio=0.0,
    )

    assert dataset.train_rows == ()
    assert dataset.metrics["skipped_insufficient_negatives"] == 1
