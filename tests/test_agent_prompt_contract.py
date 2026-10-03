"""CPU protocol fixtures; passing these is not evidence of real-model compliance."""
import hashlib
import json

import pytest
from pydantic import ValidationError

from climate_rag.budget_agent import AgentBudget, AgentDecision
from climate_rag.local_agent_model import (
    GeneratedResponseError, agent_prompt_identity, build_agent_system_prompt,
)
from test_model_response_diagnostics import stub_provider

STATEMENT = {"text": "Fixture assertion", "evidence_id": "fixture-1", "quote": "Fixture quote"}


@pytest.mark.parametrize("action", ["rewrite", "rerank", "answer", "abstain"])
@pytest.mark.parametrize("query", ["omitted", None, "", " ", "Fixture query"])
def test_query_contract_keeps_strict_preexisting_classification(action, query):
    payload = {"action": action, "reason": "CPU fixture"}
    if query != "omitted":
        payload["query"] = query
    if action == "answer":
        payload.update(label="SUPPORTS", evidence_assessment="sufficient", statements=[STATEMENT])
    valid = (query == "Fixture query") if action == "rewrite" else query in ("omitted", None)
    serialized = json.dumps(payload)
    provider = stub_provider(serialized)
    if valid:
        expected = AgentDecision.model_validate(payload).model_dump()
        assert provider.decide({"remaining_seconds": 10}, AgentBudget())["decision"] == expected
    else:
        with pytest.raises(ValidationError):
            AgentDecision.model_validate(payload)
        with pytest.raises(GeneratedResponseError) as captured:
            provider.decide({"remaining_seconds": 10}, AgentBudget())
        assert captured.value.diagnostics["category"] == "schema_validation"
    assert json.dumps(payload) == serialized  # no query deletion/repair


@pytest.mark.parametrize("action", ["rewrite", "rerank", "abstain"])
@pytest.mark.parametrize("forbidden", [{"label": "SUPPORTS"}, {"statements": [STATEMENT]}])
def test_non_answer_still_rejects_label_or_statements(action, forbidden):
    payload = {"action": action, "reason": "CPU fixture", **forbidden}
    if action == "rewrite":
        payload["query"] = "Fixture query"
    with pytest.raises(ValidationError):
        AgentDecision.model_validate(payload)


@pytest.mark.parametrize("missing", ["label", "statements", "evidence_assessment"])
def test_answer_still_requires_each_conditional_field(missing):
    payload = {"action": "answer", "reason": "CPU fixture", "label": "REFUTES",
               "evidence_assessment": "sufficient", "statements": [STATEMENT]}
    del payload[missing]
    with pytest.raises(ValidationError):
        AgentDecision.model_validate(payload)


def test_actual_model_message_contains_all_conditions_and_unchanged_schema():
    provider = stub_provider('{"action":"abstain","reason":"CPU fixture"}')
    observation = {"remaining_seconds": 10}
    provider.decide(observation, AgentBudget())
    system = provider.tokenizer.messages[0]["content"]
    assert system == build_agent_system_prompt()
    for condition in (
        "For rewrite, query must be a nonempty, non-whitespace string.",
        "For rerank, answer and abstain, query must be null or omitted",
        "For rewrite, rerank and abstain, label must be null or omitted",
        "statements must be an empty list or omitted.",
        "For answer, evidence_assessment must be sufficient",
        "SUPPORTS or REFUTES, and statements must contain 1 to 3 cited statements.",
        "Each cited statement requires text, evidence_id and an exact source quote.",
    ):
        assert condition in system
    assert system.endswith(json.dumps(AgentDecision.model_json_schema()))
    assert provider.tokenizer.messages[1] == {"role": "user", "content": json.dumps(observation)}
    assert "Fixture" not in system


def test_static_prompt_identity_covers_exact_model_visible_system_text():
    identity = agent_prompt_identity()
    assert identity["system_prompt_sha256"] == hashlib.sha256(build_agent_system_prompt().encode()).hexdigest()
    assert identity["schema_json_sha256"] == hashlib.sha256(
        json.dumps(AgentDecision.model_json_schema()).encode()).hexdigest()
    assert "excludes observation" in identity["scope"]
