"""CPU fixed-output fixtures, NOT reconstructions of lost pilot responses."""
import hashlib
import json
from contextlib import nullcontext
from types import SimpleNamespace

import pytest
from jsonschema import Draft202012Validator

from climate_rag.budget_agent import AgentBudget, AgentDecision, BudgetedEvidenceAgent
from climate_rag.local_agent_model import GeneratedResponseError, LocalQwenDecisionProvider
from climate_rag.model_diagnostics import PRIVATE_RESPONSE_LIMIT_BYTES, response_diagnostics


class FakeInputs(dict):
    input_ids = SimpleNamespace(shape=(1, 1))

    def to(self, device):
        return self


class FakeTokenizer:
    def __init__(self, text):
        self.text = text

    def apply_chat_template(self, messages, **kwargs):
        self.messages = messages
        return "fixed CPU fixture prompt"

    def __call__(self, prompt, **kwargs):
        return FakeInputs(input_ids=[0])

    def decode(self, tokens, **kwargs):
        return self.text


def stub_provider(text, private_dir=None, tokens=None):
    provider = object.__new__(LocalQwenDecisionProvider)
    provider._torch = SimpleNamespace(inference_mode=nullcontext)
    provider.tokenizer = FakeTokenizer(text)
    provider.private_response_dir = private_dir
    provider.name = "cpu-fixed-output-fixture"
    provider.model = SimpleNamespace(
        device="cpu", generation_config=SimpleNamespace(eos_token_id=[9]),
        generate=lambda **kwargs: [[0, *(tokens if tokens is not None else [2, 3, 9])]],
    )
    return provider


@pytest.mark.parametrize("text,category", [
    ('{"action":', "json_decode"),
    ('```json\n{"action":"abstain","reason":"none"}\n```', "json_decode"),
    ('{"action":"abstain","reason":"none","query":"leaked query"}', "schema_validation"),
    ('{"action":"answer","reason":"none"}', "schema_validation"),
    ('{"action":"rewrite","reason":"none"}', "schema_validation"),
])
def test_invalid_output_is_rejected_without_repair_and_keeps_usage(text, category):
    provider = stub_provider(text)
    with pytest.raises(GeneratedResponseError) as captured:
        provider.decide({"remaining_seconds": 10}, AgentBudget())
    error = captured.value
    assert error.usage == {"input_tokens": 1, "output_tokens": 3}
    assert error.diagnostics["category"] == category
    assert error.diagnostics["output_sha256"] == hashlib.sha256(text.encode()).hexdigest()
    assert error.diagnostics["eos_observed"] is True
    assert not error.diagnostics["reached_max_new_tokens"]
    if category == "json_decode":
        assert error.diagnostics["line"] >= 1 and error.diagnostics["column"] >= 1
    else:
        assert set(error.diagnostics["errors"][0]) == {"loc", "type"}
    assert "leaked query" not in json.dumps(error.diagnostics)


def test_legal_output_unchanged_and_private_raw_attachment(tmp_path):
    tmp_path.chmod(0o700)
    raw = '  {"action":"abstain","reason":"No evidence"}\n'
    response = stub_provider(raw, tmp_path).decide({"remaining_seconds": 10}, AgentBudget())
    assert response["decision"]["action"] == "abstain"
    assert response["diagnostics"]["category"] == "validated"
    assert response["diagnostics"]["output_characters"] == len(raw)
    assert [x.read_bytes() for x in tmp_path.iterdir()] == [raw.encode()]
    assert raw not in json.dumps(response["diagnostics"])


def test_failure_diagnostics_and_tokens_survive_controller():
    result = BudgetedEvidenceAgent(lambda query: {"items": []}, stub_provider("not JSON")).run(
        {"claim_text": "CPU fixture"}, strategy="fixed_retrieval",
    )
    assert result["reason"] == "stage_failed" and result["answer"] is None
    assert result["generation_calls"] == 1 and result["usage_known"]
    assert result["usage"] == {"input_tokens": 1, "output_tokens": 3}
    assert result["events"][-1]["generation_diagnostics"]["category"] == "json_decode"


def test_safe_schema_locations_redact_extra_keys():
    text = '{"action":"abstain","reason":"none","sensitive extra field":42}'
    with pytest.raises(GeneratedResponseError) as captured:
        stub_provider(text).decide({"remaining_seconds": 10}, AgentBudget())
    diagnostic = captured.value.diagnostics
    assert diagnostic["errors"] == [{"loc": ["<redacted>"], "type": "extra_forbidden"}]
    assert "sensitive" not in json.dumps(diagnostic)


def test_token_limit_and_eos_are_observations_not_stop_cause():
    response = stub_provider('{"action":"abstain","reason":"none"}', tokens=[2] * 128).decide(
        {"remaining_seconds": 10}, AgentBudget(max_output_tokens_per_call=128),
    )
    assert response["diagnostics"]["reached_max_new_tokens"]
    assert response["diagnostics"]["eos_observed"] is False
    assert "stop_reason" not in response["diagnostics"]


def test_oversize_private_attachment_is_skipped_not_truncated(tmp_path):
    tmp_path.chmod(0o700)
    report = response_diagnostics("x" * (PRIVATE_RESPONSE_LIMIT_BYTES + 1), output_tokens=5,
                                  max_new_tokens=512, eos_observed=None,
                                  generation_elapsed_ms=1, private_dir=tmp_path)
    assert report["private_attachment"] == "skipped_per_response_limit"
    assert not list(tmp_path.iterdir())


def test_private_file_count_limit_and_storage_failure_do_not_relax_decisions(tmp_path):
    tmp_path.chmod(0o700)
    for index in range(32):
        (tmp_path / f"existing-{index}.txt").touch()
    raw = '{"action":"abstain","reason":"none"}'
    response = stub_provider(raw, tmp_path).decide({"remaining_seconds": 10}, AgentBudget())
    assert response["diagnostics"]["private_attachment"] == "skipped_total_limit"
    assert len(list(tmp_path.iterdir())) == 32
    response = stub_provider(raw, tmp_path / "missing").decide({"remaining_seconds": 10}, AgentBudget())
    assert response["diagnostics"]["private_attachment"] == "write_failed"
    assert response["decision"]["action"] == "abstain"


def test_cross_field_constraint_is_not_fully_expressed_by_json_schema():
    value = {"action": "answer", "reason": "CPU counterexample"}
    Draft202012Validator(AgentDecision.model_json_schema()).validate(value)
    with pytest.raises(GeneratedResponseError) as captured:
        stub_provider(json.dumps(value)).decide({"remaining_seconds": 10}, AgentBudget())
    assert captured.value.diagnostics["category"] == "schema_validation"
    # This counterexample does not reveal what the actual pilot emitted.


def test_prepared_canary_is_separately_gated_and_excludes_private_result_packaging():
    from pathlib import Path

    script = (Path(__file__).resolve().parents[1] / "hpc/budget_agent_diagnostic_canary.sbatch").read_text()
    assert "${CLIMATE_CANARY_RELEASE_ID:?" in script
    assert "${CLIMATE_EXPECTED_SOURCE_GIT:?" in script
    assert "#SBATCH --no-requeue" in script
    assert 'mkdir -m 700 "${CLIMATE_PRIVATE_RESPONSE_DIR}"' in script
    assert '"${directory}/run.json" "${directory}/run.consumed.json"' in script
