from __future__ import annotations

import hashlib
import io
import json
import tarfile

import pytest

from climate_rag.scifact_grounding import (
    Abstract,
    GoldClaim,
    Rationale,
    assemble_context,
    audit_groups,
    parse_abstract,
    parse_gold,
    read_original_archive,
)
from climate_rag.scifact_scoring import (
    ClaimPrediction,
    PredictedAbstract,
    parse_prediction,
    score_original,
)


def claim(identifier, evidence, *, text="fixture", cited=()):
    return GoldClaim(
        identifier,
        text,
        {
            doc: tuple(Rationale(label, tuple(indices)) for label, indices in rats)
            for doc, rats in evidence.items()
        },
        tuple(cited),
    )


def prediction(identifier, evidence):
    return ClaimPrediction(
        identifier,
        {
            doc: PredictedAbstract(label, tuple(indices))
            for doc, (label, indices) in evidence.items()
        },
    )


def test_official_full_example():
    gold = claim(
        52, {11: [("SUPPORT", [0, 1]), ("SUPPORT", [11])], 15: [("SUPPORT", [4])]}
    )
    pred = prediction(52, {11: ("SUPPORT", [1, 11, 13]), 16: ("CONTRADICT", [18, 20])})
    # Official documentation typo REFUTES is replaced with parser's CONTRADICT.
    metrics = score_original([gold], [pred])["metrics"]
    for name in ("abstract_label_only", "abstract_rationalized"):
        assert metrics[name]["precision"] == metrics[name]["recall"] == 0.5
        assert metrics[name]["f1"] == 0.5
    for name in ("sentence_selection", "sentence_label"):
        assert metrics[name]["precision"] == 0.2
        assert metrics[name]["recall"] == 0.25
        assert metrics[name]["f1"] == pytest.approx(2 / 9)


@pytest.mark.parametrize(
    "indices,abstract_f1",
    [
        ([4, 5, 6, 0, 1], 0),
        ([0, 1, 4, 5, 6], 1),
    ],
)
def test_first_three_only_applies_to_abstract(indices, abstract_f1):
    gold = claim(1, {11: [("SUPPORT", [0, 1])]})
    m = score_original([gold], [prediction(1, {11: ("SUPPORT", indices)})])["metrics"]
    assert m["abstract_label_only"]["f1"] == 1
    assert m["abstract_rationalized"]["f1"] == abstract_f1
    assert m["sentence_label"]["f1"] == pytest.approx(4 / 7)


def test_duplicate_predictions_are_not_silently_deduplicated():
    gold = claim(1, {11: [("SUPPORT", [0, 1])]})
    m = score_original([gold], [prediction(1, {11: ("SUPPORT", [0, 0, 1])})])["metrics"]
    assert m["sentence_label"]["precision"] == pytest.approx(2 / 3)
    assert m["sentence_label"]["f1"] == 0.8


def test_sentence_selection_and_label_are_different():
    gold = claim(1, {11: [("SUPPORT", [0])], 15: [("CONTRADICT", [1])]})
    m = score_original(
        [gold], [prediction(1, {11: ("SUPPORT", [0]), 15: ("SUPPORT", [1])})]
    )
    assert gold.label_scope == "MIXED"
    assert m["metrics"]["sentence_selection"]["f1"] == 1
    for name in ("abstract_label_only", "abstract_rationalized", "sentence_label"):
        assert m["metrics"][name]["f1"] == 0.5
    assert m["separate_diagnostics_not_official_f1"]["mixed_label_claim_count"] == 1


@pytest.mark.parametrize(
    "rats",
    [
        [("SUPPORT", [0]), ("CONTRADICT", [1])],
        [("SUPPORT", [0]), ("SUPPORT", [0, 1])],
    ],
)
def test_unsupported_gold_shapes_rejected_not_collapsed(rats):
    with pytest.raises(ValueError, match="official metrics"):
        score_original([claim(1, {11: rats})], [prediction(1, {})])


def test_nei_has_no_true_negative_f1_reward_and_false_answers_count():
    gold = claim(1, {}, cited=[11])
    empty = score_original([gold], [prediction(1, {})])
    assert all(m["f1"] == 0 for m in empty["metrics"].values())
    false = score_original([gold], [prediction(1, {11: ("SUPPORT", [0])})])
    assert false["metrics"]["abstract_label_only"]["predicted"] == 1
    assert false["separate_diagnostics_not_official_f1"]["nei_false_evidence_rate"] == 1
    with pytest.raises(ValueError, match="NEI must"):
        score_original([gold], [prediction(1, {11: ("NOT_ENOUGH_INFO", [])})])


def test_micro_metrics_require_complete_matrix():
    gold = [
        claim(1, {11: [("SUPPORT", [0])]}),
        claim(
            2, {12: [("SUPPORT", [0])], 13: [("SUPPORT", [0])], 14: [("SUPPORT", [0])]}
        ),
    ]
    preds = [prediction(1, {11: ("SUPPORT", [0])}), prediction(2, {})]
    assert all(m["f1"] == 0.4 for m in score_original(gold, preds)["metrics"].values())
    for invalid in (preds[:1], preds + preds[:1], preds + [prediction(3, {})]):
        with pytest.raises(ValueError, match="prediction claim IDs"):
            score_original(gold, invalid)


def test_original_parser_preserves_alternatives_and_no_gold_in_model_input():
    corpus = {11: Abstract(11, "title", ("zero", "one", "two"), False)}
    raw = {
        "id": 3,
        "claim": "A claim",
        "cited_doc_ids": [11],
        "evidence": {
            "11": [
                {"label": "SUPPORT", "sentences": [0, 1]},
                {"label": "SUPPORT", "sentences": [2]},
            ]
        },
    }
    parsed = parse_gold(raw, corpus)
    assert parsed.gold_row() == raw
    assert parsed.inference_row() == {"id": 3, "claim": "A claim"}
    assert parsed.label_scope == "SUPPORTS"
    assert parse_gold({**raw, "evidence": {}}, corpus).label_scope == "NOT_ENOUGH_INFO"


@pytest.mark.parametrize("value", [True, "0", -1, 3])
def test_bad_gold_sentence_index_rejected(value):
    corpus = {11: Abstract(11, "title", ("s",), False)}
    with pytest.raises(ValueError):
        parse_gold(
            {
                "id": 1,
                "claim": "a",
                "cited_doc_ids": [],
                "evidence": {"11": [{"label": "SUPPORT", "sentences": [value]}]},
            },
            corpus,
        )


def test_prediction_parser_preserves_order_and_checks_unknown_fields():
    corpus = {11: Abstract(11, "title", ("a", "b"), False)}
    raw = {"id": 1, "evidence": {"11": {"label": "SUPPORT", "sentences": [1, 0, 1]}}}
    assert parse_prediction(raw, corpus).evidence[11].sentences == (1, 0, 1)
    with pytest.raises(ValueError):
        parse_prediction({**raw, "gold_label": "SUPPORT"}, corpus)


def test_group_overlap_via_cited_source_variant_and_claim_paraphrase():
    corpus = {
        1: Abstract(1, "Version", ("Alpha beta gamma",), False),
        2: Abstract(2, "VERSION", ("Alpha beta gamma!",), False),
        3: Abstract(3, "Distinct source", ("Other biology",), False),
    }
    train = [
        claim(1, {}, text="Unrelated text", cited=[1]),
        claim(2, {}, text="A big blue red green atom", cited=[3]),
        claim(3, {}, text="Eligible unique different work"),
    ]
    dev = [
        claim(4, {}, text="Another text", cited=[2]),
        claim(5, {}, text="A big blue red green atom changes"),
    ]
    summary, private = audit_groups(corpus, {"train": train, "dev": dev})
    assert summary["train_quarantined_for_dev_overlap"] == 2
    assert private["eligible_train_ids"] == [3]
    assert private["dev_ids"] == [4, 5]


def test_archive_reads_only_allowlisted_members(tmp_path):
    path = tmp_path / "fixture.tar.gz"
    with tarfile.open(path, "w:gz") as archive:
        rows = {
            "data/corpus.jsonl": {
                "doc_id": 1,
                "title": "t",
                "abstract": ["s"],
                "structured": False,
            },
            "data/claims_train.jsonl": {
                "id": 1,
                "claim": "c",
                "evidence": {},
                "cited_doc_ids": [1],
            },
            "data/claims_dev.jsonl": {
                "id": 2,
                "claim": "d",
                "evidence": {},
                "cited_doc_ids": [1],
            },
        }
        for name, row in rows.items():
            payload = json.dumps(row).encode()
            info = tarfile.TarInfo(name)
            info.size = len(payload)
            archive.addfile(info, io.BytesIO(payload))
        sentinel = tarfile.TarInfo("data/claims_test.jsonl")
        sentinel.size = 5
        archive.addfile(sentinel, io.BytesIO(b"BOOM!"))
    sha = hashlib.sha256(path.read_bytes()).hexdigest()
    corpus, splits, hashes = read_original_archive(path, expected_sha256=sha)
    assert len(corpus) == len(splits["dev"]) == 1
    assert set(hashes) == set(rows)
    with pytest.raises(ValueError, match="SHA mismatch"):
        read_original_archive(path)


def test_context_uses_original_prefix_not_gold_and_counts_format_tokens():
    doc = Abstract(1, "title", ("alpha", "verylongsentence"), False)
    context = assemble_context([doc], count_tokens=len, max_tokens=35, max_docs=1)
    assert context["included"] == [{"doc_id": 1, "sentences": [0]}]
    assert context["omitted_sentences"] == 1
    assert context["overflow"] is True
    assert context["tokens"] == len(context["text"]) <= 35
    empty = assemble_context([doc], count_tokens=len, max_tokens=1, max_docs=1)
    assert empty["included"] == [] and empty["overflow"]
    with pytest.raises(ValueError, match="duplicate"):
        assemble_context([doc, doc], count_tokens=len, max_tokens=1000, max_docs=5)


def test_corpus_sentence_order_is_preserved():
    doc = parse_abstract(
        {"doc_id": 1, "title": "t", "abstract": ["z", "a"], "structured": False}
    )
    assert doc.sentences == ("z", "a")


def test_duplicate_cited_metadata_preserved_but_not_used_as_gold():
    corpus = {11: Abstract(11, "title", ("sentence",), False)}
    parsed = parse_gold(
        {"id": 1, "claim": "a", "evidence": {}, "cited_doc_ids": [11, 11]}, corpus
    )
    assert parsed.cited_doc_ids == (11, 11)
    assert parsed.evidence == {}
    summary, _ = audit_groups(corpus, {"train": [], "dev": [parsed]})
    assert summary["duplicate_cited_id_rows"]["dev"] == 1
