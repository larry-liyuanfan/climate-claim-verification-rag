"""Synthetic contract/scorer interoperability, never real dev model results."""

import copy
import inspect
import json

import pytest

from climate_rag.agent_v3 import SentenceAgentV3, V3Budget
from climate_rag.local_agent_v3 import fixed_grammar_config
from climate_rag.scifact_agent_v1 import SciFactDocumentAgentV1
from climate_rag.scifact_grounding import Abstract, GoldClaim, Rationale
from climate_rag.scifact_scoring import parse_prediction, score_original
from climate_rag.scifact_terminal import (
    PROTOCOL,
    action_schema,
    parse_action,
    render_scifact_prompt,
    source_from_abstract,
    system_prompt_scifact,
    to_original_prediction,
)


class FixtureProvider:
    name, kind, terminal_protocol = "synthetic-scifact", "fixture", PROTOCOL

    def __init__(self, actions):
        self.actions, self.observations = iter(actions), []

    def count_prompt(self, observation, schema):
        # Fixture units only, not a real tokenizer benchmark.
        return len(json.dumps([system_prompt_scifact(), observation, schema])) // 5

    def count_text(self, text):
        return len(text) // 5

    def generate(self, observation, schema, max_output_tokens, remaining_seconds):
        self.observations.append((observation, schema))
        action = next(self.actions)
        if isinstance(action, Exception):
            raise action
        return {
            "raw": json.dumps(action),
            "usage": {
                "input_tokens": self.count_prompt(observation, schema),
                "output_tokens": 50,
            },
        }


def corpus_fixture(count=6):
    return {
        i: Abstract(
            i,
            "Synthetic title",
            tuple(f"Document {i} sentence {j}." for j in range(10)),
            False,
        )
        for i in range(10, 10 + count)
    }


def doc(alias="c0", indices=(0,), label="SUPPORTS"):
    return {
        "source_id": alias,
        "label": label,
        "sentence_ids": [f"{alias}:{i}" for i in indices],
    }


def answer(*documents):
    return {"action": "answer", "documents": list(documents)}


ABSTAIN = {"action": "abstain", "reason": "insufficient_evidence"}


def run(actions, *, route="adaptive", corpus=None, context_k=2):
    corpus = corpus_fixture() if corpus is None else corpus
    provider = FixtureProvider(actions)
    engine = SciFactDocumentAgentV1(
        provider,
        lambda q, k: [source_from_abstract(d) for d in corpus.values()],
        rerank=lambda q, rows: rows,
        budget=V3Budget(context_k=context_k),
    )
    result = engine.run("A fixture claim", route)
    return result, provider, corpus


def converted(result, corpus):
    return to_original_prediction(42, result, corpus)


@pytest.mark.parametrize("labels", [("SUPPORTS", "SUPPORTS"), ("SUPPORTS", "REFUTES")])
def test_two_documents_two_sentences_and_mixed_labels_survive(labels):
    result, _, corpus = run(
        [answer(doc("c1", (2, 0), labels[1]), doc("c0", (1, 3), labels[0]))]
    )
    output = converted(result, corpus)
    assert list(output["prediction"]["evidence"]) == ["11", "10"]
    assert output["prediction"]["evidence"]["11"] == {
        "label": {"SUPPORTS": "SUPPORT", "REFUTES": "CONTRADICT"}[labels[1]],
        "sentences": [2, 0],
    }
    assert output["claim_verdict_accuracy"] is None
    assert output["termination_reason"] == "ids_validated_semantics_unmeasured"


def test_four_sentences_and_first_three_order_are_not_sorted_or_truncated():
    corpus = corpus_fixture(1)
    gold = [GoldClaim(42, "synthetic", {10: (Rationale("SUPPORT", (3,)),)}, ())]
    scores = []
    for indices in [(4, 0, 1, 3), (3, 4, 0, 1)]:
        result, _, _ = run([answer(doc(indices=indices))], corpus=corpus)
        row = converted(result, corpus)["prediction"]
        assert row["evidence"]["10"]["sentences"] == list(indices)
        scores.append(score_original(gold, [parse_prediction(row, corpus)])["metrics"])
    assert [s["abstract_rationalized"]["f1"] for s in scores] == [0, 1]
    assert [s["sentence_label"]["f1"] for s in scores] == [0.4, 0.4]


def test_alternative_rationale_and_noncontiguous_original_indices():
    result, _, corpus = run([answer(doc(indices=(7, 2)))])
    row = converted(result, corpus)["prediction"]
    gold = [
        GoldClaim(
            42,
            "synthetic",
            {10: (Rationale("SUPPORT", (0, 1)), Rationale("SUPPORT", (7,)))},
            (),
        )
    ]
    score = score_original(gold, [parse_prediction(row, corpus)])
    assert row["evidence"]["10"]["sentences"] == [7, 2]
    assert score["metrics"]["abstract_rationalized"]["f1"] == 1
    assert score["metrics"]["sentence_label"]["correct"] == 1


@pytest.mark.parametrize("action", [ABSTAIN, RuntimeError("fixture failure")])
def test_empty_evidence_preserves_abstain_versus_failure(action):
    result, _, corpus = run([action])
    output = converted(result, corpus)
    assert output["prediction"] == {"id": 42, "evidence": {}}
    expected = (
        "controller_failure:RuntimeError"
        if isinstance(action, Exception)
        else "model_abstention:insufficient_evidence"
    )
    assert output["termination_reason"] == expected


def test_empty_retrieval_allows_abstain_but_not_answer():
    result, provider, corpus = run([ABSTAIN], corpus={})
    assert "answer" not in provider.observations[0][0]["allowed_actions"]
    assert converted(result, corpus)["prediction"]["evidence"] == {}


@pytest.mark.parametrize(
    "bad",
    [
        answer(doc(), doc()),
        answer(doc(indices=(0, 0))),
        answer({**doc(), "sentence_ids": ["c1:0"]}),
        answer(doc("c5")),
        answer(doc(indices=(99,))),
        answer(doc(indices=tuple(range(9)))),
        {**answer(doc()), "label": "SUPPORTS"},
    ],
)
def test_invalid_document_or_sentence_references_are_charged_not_repaired(bad):
    result, _, corpus = run([bad, ABSTAIN])
    assert result["generation_attempts"][0]["status"] == "validation_failed"
    assert result["validation_repairs"] == 1
    assert result["usage"]["output_tokens"] == 100
    assert converted(result, corpus)["prediction"]["evidence"] == {}


def test_read_invalidates_old_context_then_valid_new_document_is_accepted():
    result, provider, corpus = run(
        [
            {"action": "read", "source_ids": ["c5"]},
            answer(doc()),
            answer(doc("c5", (7, 1))),
        ]
    )
    assert (
        result["generation_attempts"][1]["error_code"]
        == "sentence_not_currently_visible"
    )
    assert result["events"][1]["model_selected"]
    assert all(
        s["sentence_id"].startswith("c5:")
        for s in provider.observations[1][0]["current_citable"]
    )
    assert converted(result, corpus)["prediction"]["evidence"]["15"]["sentences"] == [
        7,
        1,
    ]


@pytest.mark.parametrize(
    "route,tools",
    [
        ("fixed_retrieval", 1),
        ("fixed_rerank", 2),
        ("deterministic_extra", 3),
        ("adaptive", 1),
    ],
)
def test_four_routes_preserve_shared_caps_and_actual_cost_ledger(route, tools):
    result, _, _ = run([answer(doc())], route=route)
    assert result["tool_calls"] == tools
    assert result["model_calls"] == 1 and result["usage"]["output_tokens"] == 50
    assert result["budget"] == V3Budget(context_k=2).model_dump()
    assert result["equal_caps_not_equal_actual_cost"]
    assert result["unknown_usage_attempts"] == 0


def test_total_citation_cap_independent_of_per_document_cap():
    payload = answer(doc("c0", range(8)), doc("c1", range(8)), doc("c2", range(5)))
    visible = {f"c{i}:{j}": {} for i in range(3) for j in range(8)}
    with pytest.raises(ValueError, match="total_sentence_budget"):
        parse_action(payload, ["answer"], visible, ["c0", "c1", "c2"], 3)


def test_converter_rejects_mutated_text_and_never_silently_deduplicates():
    result, _, corpus = run([answer(doc(indices=(2, 7)))])
    for mutation in ("text", "duplicate", "protocol", "outcome"):
        changed = copy.deepcopy(result)
        citations = changed["answer"]["documents"][0]["citations"]
        if mutation == "text":
            citations[0]["text"] = "invented"
        elif mutation == "duplicate":
            citations.append(copy.deepcopy(citations[0]))
        elif mutation == "protocol":
            changed["protocol"] = "sentence-id-v3"
        else:
            changed["outcome"] = "model_abstention:budget"
        with pytest.raises(ValueError):
            converted(changed, corpus)


def test_provider_contract_and_separate_prompt_renderer():
    provider = FixtureProvider([])
    provider.terminal_protocol = "sentence-id-v3"
    with pytest.raises(ValueError, match="own provider"):
        SciFactDocumentAgentV1(provider, lambda q, k: [])

    class Tokenizer:
        def apply_chat_template(self, messages, **kwargs):
            assert kwargs == {
                "tokenize": False,
                "add_generation_prompt": True,
                "enable_thinking": False,
            }
            return json.dumps(messages)

    rendered = json.loads(render_scifact_prompt(Tokenizer(), {"fixture": "data"}, {}))
    assert "each relevant scientific document" in rendered[0]["content"]
    assert "opposite labels" in rendered[0]["content"]
    assert rendered[1]["content"] == '{"fixture":"data"}'


def test_lmfe_nested_schema_accepts_multi_document_and_rejects_cross_document():
    from lmformatenforcer import JsonSchemaParser

    schema = action_schema(
        ["answer", "abstain"],
        ["c0", "c1"],
        [f"c{i}:{j}" for i in range(2) for j in range(4)],
        2,
    )

    def accepted(payload):
        parser = JsonSchemaParser(schema, config=fixed_grammar_config())
        for char in json.dumps(payload, separators=(",", ":")):
            if char not in parser.get_allowed_characters():
                return False
            parser = parser.add_character(char)
        return parser.can_end()

    assert accepted(answer(doc("c0", (3, 1, 2, 0)), doc("c1", (0, 1), "REFUTES")))
    assert accepted(ABSTAIN)
    assert not accepted(answer({**doc("c0"), "sentence_ids": ["c1:0"]}))
    assert "$ref" not in json.dumps(schema)


def test_versioned_controller_policy_is_unchanged_except_terminal_handling():
    old = inspect.getsource(SentenceAgentV3.run)
    old_block = """                        answer = {
                            "label": decision["label"],
                            "citations": [visible[s] for s in decision["sentence_ids"]],
                            "rationale": None,
                            "semantic_support": "unmeasured",
                        }"""
    assert old_block in old
    expected = old.replace(
        old_block, "                        answer = render_answer(decision, visible)"
    )
    expected = expected.replace('"protocol": "sentence-id-v3"', '"protocol": PROTOCOL')
    assert inspect.getsource(SciFactDocumentAgentV1.run) == expected
