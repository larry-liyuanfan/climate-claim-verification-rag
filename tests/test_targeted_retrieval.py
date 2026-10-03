from __future__ import annotations

import copy

import pytest

from climate_rag.agent_v3 import (
    SentenceAgentV3,
    Source,
    V3Budget,
    action_schema,
    valid_search_query,
)
from climate_rag.targeted_query import (
    PROTOCOL,
    ROUTES,
    deterministic_queries,
    planning_schema,
    query_fields,
    validate_plan,
    validate_query,
)
from test_agent_v3 import ABSTAIN, Provider

CLAIM = "CO2 was never above 800 ppm in 2010 and sea level did not change"
Q1 = {"purpose": "counter_evidence", "query": "historical carbon dioxide observations"}
Q2 = {"purpose": "subquestion", "query": "sea level observations"}
OLD = Source("z", "Initial source", ("Initial complete text.",))
NEW = Source("a", "New source", ("New complete counter evidence.",))


def engine(provider, retrieve=None, rerank=None, budget=None):
    return SentenceAgentV3(
        provider,
        retrieve or (lambda q, k: [OLD]),
        rerank=rerank or (lambda q, rows: rows),
        budget=budget or V3Budget(context_k=2),
        protocol=PROTOCOL,
    )


def test_targeted_queries_do_not_change_legacy_claim_guard():
    assert not valid_search_query(CLAIM, Q1["query"])
    assert validate_query(Q1, [CLAIM]) == Q1
    assert validate_query(Q2, [CLAIM]) == Q2
    for q in deterministic_queries(CLAIM):
        assert validate_query(q, [CLAIM]) == q
    rewrite = action_schema(["rewrite"], [], [], 2, targeted=True)["anyOf"][0]
    plan = planning_schema()["properties"]["queries"]["items"]
    assert {k: rewrite["properties"][k] for k in query_fields()} == plan["properties"]


@pytest.mark.parametrize(
    "query",
    [
        dict(Q1, query=" "),
        dict(Q1, query=CLAIM),
        dict(Q1, query="x" * 301),
        dict(Q1, purpose="execute"),
        dict(Q1, claim="changed"),
    ],
)
def test_query_operational_guard(query):
    with pytest.raises(ValueError):
        validate_query(query, [CLAIM])


def test_fixed_plan_empty_and_duplicate_queries():
    assert (
        validate_plan({"action": "plan_queries", "queries": []}, [CLAIM])["queries"]
        == []
    )
    with pytest.raises(ValueError):
        validate_plan({"action": "plan_queries", "queries": [Q1, Q1]}, [CLAIM])


def test_new_evidence_feedback_and_read_memory_are_real():
    observed_queries = []

    def retrieve(q, k):
        observed_queries.append(q)
        return [OLD] if q == CLAIM else [NEW]

    provider = Provider(
        [
            dict(action="rewrite", **Q1),
            {"action": "answer", "label": "REFUTES", "sentence_ids": ["c1:0", "c0:0"]},
        ]
    )
    result = engine(provider, retrieve).run(CLAIM)
    assert observed_queries == [CLAIM, Q1["query"]]
    after = provider.observations[1][0]
    assert after["immutable_claim"] == CLAIM
    assert after["tool_feedback"]["new_source_ids"] == ["c1"]
    assert {x["text"] for x in after["current_citable"]} == {
        OLD.sentences[0],
        NEW.sentences[0],
    }
    assert result["answer"]["label"] == "REFUTES"
    assert len(result["answer"]["citations"]) == 2
    assert result["events"][1]["selection_origin"] == "model_feedback"


class FeedbackProvider(Provider):
    def __init__(self):
        super().__init__([])

    def generate(self, obs, schema, cap, seconds):
        if not obs["prior_queries"]:
            action = dict(action="rewrite", **Q1)
        elif obs["tool_feedback"]["search_empty"]:
            action = (
                dict(action="rewrite", **Q2)
                if len(obs["prior_queries"]) == 1
                else ABSTAIN
            )
        else:
            action = ABSTAIN
        self.actions = iter([action])
        return super().generate(obs, schema, cap, seconds)


def test_feedback_changes_next_action_but_upfront_plan_does_not():
    adaptive, fixed = [], []
    for empty in (False, True):
        queries = []

        def retrieve(q, k):
            queries.append(q)
            return [OLD] if q == CLAIM else ([] if empty else [NEW])

        p = FeedbackProvider()
        r = engine(p, retrieve).run(CLAIM)
        adaptive.append([x["action"] for x in r["generation_attempts"]])
        queries.clear()
        p = Provider([{"action": "plan_queries", "queries": [Q1, Q2]}, ABSTAIN])
        r = engine(p, retrieve).run(CLAIM, "fixed_multiquery")
        assert p.observations[0][0]["current_citable"] == []
        assert p.observations[0][0]["tool_feedback"] is None
        assert r["generation_attempts"][0]["stage"] == "plan"
        fixed.append(copy.deepcopy(queries))
        assert r["model_calls"] == 2 and r["tool_calls"] == 4
        assert all(
            e["selection_origin"] == "model_upfront_fixed"
            for e in r["events"]
            if e["tool"] == "rewrite"
        )
    assert adaptive == [["rewrite", "abstain"], ["rewrite", "rewrite", "abstain"]]
    assert fixed == [[CLAIM, Q1["query"], Q2["query"]]] * 2


def test_rrf_is_recomputed_from_original_rankings_order_independent():
    docs = {str(i): Source(str(i), "fixture", (str(i),)) for i in range(8)}
    orders = {
        CLAIM: list("01234567"),
        Q1["query"]: list("67123450"),
        Q2["query"]: list("76543210"),
    }
    answers = []
    for qs in ([Q1, Q2], [Q2, Q1]):
        p = Provider([{"action": "plan_queries", "queries": qs}, ABSTAIN])
        result = engine(
            p, lambda q, k: [docs[i] for i in orders[q]], budget=V3Budget(candidate_k=5)
        ).run(CLAIM, "fixed_multiquery")
        answers.append(result["delivered_evidence_ids"])
    assert answers[0] == answers[1]


@pytest.mark.parametrize("route", ROUTES)
def test_all_routes_share_caps_final_schema_and_original_claim(route):
    actions = (
        [{"action": "plan_queries", "queries": [Q1, Q2]}]
        if route == "fixed_multiquery"
        else []
    ) + [ABSTAIN]
    p = Provider(actions)
    r = engine(p).run(CLAIM, route)
    assert r["budget"] == V3Budget(context_k=2).model_dump()
    assert all(o["immutable_claim"] == CLAIM for o, _ in p.observations)
    assert r["model_calls"] <= 5 and r["tool_calls"] <= 5
    assert r["protocol"] == PROTOCOL


def test_repeat_query_charged_but_not_reexecuted_and_preview_not_citable():
    docs = [Source(str(i), "fixture", (f"Document {i}.",)) for i in range(5)]
    p = Provider(
        [
            dict(action="rewrite", **Q1),
            dict(action="rewrite", **Q1),
            {"action": "answer", "label": "SUPPORTS", "sentence_ids": ["c4:0"]},
            ABSTAIN,
        ]
    )
    r = engine(p, lambda q, k: docs, budget=V3Budget(context_k=1)).run(CLAIM)
    assert r["tool_calls"] == 2 and r["model_calls"] == 4
    assert r["generation_attempts"][1]["status"] == "validation_failed"
    assert r["generation_attempts"][2]["error_code"] == "known_candidate_not_read"
    assert r["answer"] is None


def test_failed_search_preserves_call_and_does_not_deliver_old_candidates():
    def retrieve(q, k):
        if q != CLAIM:
            raise OSError("fixture failure")
        return [OLD]

    p = Provider([dict(action="rewrite", **Q1)])
    r = engine(p, retrieve).run(CLAIM)
    assert r["answer"] is None and r["delivered_evidence_ids"] == []
    assert r["model_calls"] == 1 and r["usage"]["output_tokens"] == 25
    assert r["events"][1]["status"] == "failed"


def test_fixed_plan_before_search_including_repairs_and_generation_budget():
    queries = []
    p = Provider(
        [
            {"action": "plan_queries", "queries": [Q1, Q1]},
            {"action": "plan_queries", "queries": []},
            ABSTAIN,
        ]
    )

    def retrieve(q, k):
        assert len(p.observations) == 2
        queries.append(q)
        return [OLD]

    r = engine(p, retrieve).run(CLAIM, "fixed_multiquery")
    assert queries == [CLAIM] and r["model_calls"] == 3 and r["validation_repairs"] == 1
    assert all(o["current_citable"] == [] for o, _ in p.observations[:2])


def test_memory_overflow_is_failure_not_silently_hidden_old_text():
    class AfterReturn(Provider):
        def generate(self, *args):
            response = super().generate(*args)
            self.overhead = 100000
            return response

    p = AfterReturn([dict(action="rewrite", **Q1), ABSTAIN])
    r = engine(p, budget=V3Budget(max_input_tokens=1024)).run(CLAIM)
    assert r["answer"] is None and len(p.observations) == 1
    assert r["outcome"].startswith("controller_failure")
    assert r["generation_attempts"][0]["status"] == "valid_decision"
    assert [e["status"] for e in r["events"][:2]] == ["completed", "completed"]
    assert r["events"][-1]["failure_code"] == "retained_context_capacity"


def test_new_full_sentence_capacity_failure_not_successful_read_or_answer():
    huge = Source("new", "fixture", ("Z" * 12000,))
    p = Provider([dict(action="rewrite", **Q1), ABSTAIN])
    r = engine(
        p,
        lambda q, k: [OLD] if q == CLAIM else [huge],
        budget=V3Budget(max_input_tokens=1024),
    ).run(CLAIM)
    assert r["answer"] is None and len(p.observations) == 1
    assert r["generation_attempts"][0]["status"] == "valid_decision"
    assert r["events"][-1]["failure_code"] == "selected_context_capacity"
    assert r["delivered_evidence_ids"] == []
    assert any(e.get("stage") == "context_delivery" for e in r["events"])


def test_targeted_duplicate_json_key_is_rejected_not_silently_overwritten():
    p = Provider(
        [
            '{"action":"abstain","reason":"budget","reason":"insufficient_evidence"}',
            ABSTAIN,
        ]
    )
    r = engine(p).run(CLAIM)
    assert r["generation_attempts"][0]["error_code"] == "duplicate_json_key"
