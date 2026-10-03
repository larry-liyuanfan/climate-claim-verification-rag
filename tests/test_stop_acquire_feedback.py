"""Feedback-dependent SYNTHETIC contract; no model-quality/benchmark evidence."""

import json

import pytest

from climate_rag import stop_acquire
from climate_rag.targeted_replay import run_matrix
from climate_rag.targeted_score import audit
from test_stop_acquire import DOCS, SEARCH, STOP
from test_targeted_replay import FixtureProvider, FixtureRerank, QUERY, TASKS, sha

SECOND_QUERY = "independent glacier retreat field observations"


class FeedbackFixture(FixtureProvider):
    """Same observation -> same action; no call counter or labels in this policy."""

    @staticmethod
    def decide(observation):
        found = [x for x in observation["current_citable"] if "refute" in x["text"]]
        if observation["phase"] == "verdict":
            assert found
            return {"action": "answer", "label": "REFUTES",
                    "sentence_ids": [found[0]["sentence_id"]]}
        if found:
            return STOP
        feedback = observation["tool_feedback"]
        if feedback and (feedback["search_empty"] is True or feedback["status"] == "failed"):
            return {**SEARCH, "query": SECOND_QUERY}
        return SEARCH

    def generate(self, observation, schema, max_output_tokens, remaining_seconds):
        if observation.get("protocol") != stop_acquire.PROTOCOL:
            return super().generate(observation, schema, max_output_tokens, remaining_seconds)
        action = self.decide(observation)
        return {"raw": json.dumps(action), "usage": {
            "input_tokens": self.count_prompt(observation, schema), "output_tokens": 32}}


@pytest.mark.parametrize("first_result", ["relevant", "empty", "timeout"])
def test_same_initial_state_branches_on_actual_return_and_preserves_cost(tmp_path, first_result):
    def retrieve(query, width):
        if query == QUERY["query"]:
            if first_result == "empty":
                return []
            if first_result == "timeout":
                raise TimeoutError("synthetic recoverable tool failure")
            return [DOCS["new"]]
        if query == SECOND_QUERY:
            return [DOCS["new"]]
        return list(DOCS.values())[:6]

    run = run_matrix(TASKS, FeedbackFixture(), retrieve,
                     FixtureRerank(tmp_path / "reranker-ledger"), tmp_path / "inference",
                     protocol=stop_acquire.PROTOCOL)
    audit(run, TASKS, DOCS, tmp_path / "inference/ledger", synthetic=True,
          reranker_directory=tmp_path / "reranker-ledger")
    for row in (r for r in run["runs"] if r["route"] == "adaptive"):
        attempts = row["generation_attempts"]
        initial = attempts[0]["observation"]
        assert attempts[0]["proposed_decision"] == SEARCH
        assert [v["sentence_id"] for v in initial["current_citable"]] == [f"c{i}:0" for i in range(5)]
        assert initial["tool_feedback"]["candidate_ids"] == [f"c{i}" for i in range(6)]
        assert initial["prior_queries"] == []
        next_obs = attempts[1]["observation"]
        assert next_obs["prior_queries"] == [QUERY["query"]]
        expected = STOP if first_result == "relevant" else {**SEARCH, "query": SECOND_QUERY}
        assert attempts[1]["proposed_decision"] == expected
        assert next_obs["tool_feedback"]["status"] == ("failed" if first_result == "timeout" else "completed")
        if first_result == "empty":
            assert next_obs["tool_feedback"]["search_empty"] is True
        elif first_result == "timeout":
            assert next_obs["tool_feedback"]["error_type"] == "TimeoutError"
        gates = 2 if first_result == "relevant" else 3
        assert len(attempts) == gates + 1
        assert attempts[-2]["decision"] == STOP
        assert attempts[-1]["stage"] == "verdict"
        old = {x["sentence_id"]: x["text"] for x in initial["current_citable"]}
        for ordinal, attempt in enumerate(attempts):
            observation = attempt["observation"]
            assert observation["immutable_claim"] == initial["immutable_claim"]
            visible = {x["sentence_id"]: x["text"] for x in observation["current_citable"]}
            assert all(visible.get(k) == v for k, v in old.items())
            decision = attempt["decision"]
            if decision["action"] == "acquire":
                event, = [e for e in row["events"] if e.get("trigger_attempt") == ordinal]
                assert event["query_sha256"] == sha(decision["query"].encode())
                feedback = attempts[ordinal + 1]["observation"]["tool_feedback"]
                assert feedback["query_sha256"] == event["query_sha256"]
                assert feedback["status"] == event["status"] == attempt["execution_status"]
        sentence = attempts[-1]["decision"]["sentence_ids"][0]
        assert row["visible_source_ids"][sentence.split(":")[0]] == "new"
        assert any(x["sentence_id"] == sentence and "refute" in x["text"]
                   for x in attempts[-1]["observation"]["current_citable"])
        assert row["model_calls"] == row["physical_cost"]["unique_physical_calls"] == gates + 1 <= 5
        assert row["tool_calls"] == gates <= 5
        assert row["usage"]["output_tokens"] == 32 * (gates + 1)
        assert row["usage"]["input_tokens"] == sum(a["usage"]["input_tokens"] for a in attempts)


def test_fixture_decision_is_independent_of_call_ordinal_and_hidden_gold():
    observation = {"phase": "gate", "current_citable": [],
                   "tool_feedback": {"search_empty": True, "status": "completed"}}
    provider = FeedbackFixture()
    provider.calls = 999
    assert provider.decide(observation) == {**SEARCH, "query": SECOND_QUERY}
    observation["current_citable"] = [{"sentence_id": "c6:0", "text": "Synthetic refute result."}]
    assert provider.decide(observation) == STOP
