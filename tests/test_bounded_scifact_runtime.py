"""Synthetic F/F+G contract/cost tests. No real model or scientific outcomes."""
import copy
import hashlib
import json
import types

import pytest

from climate_rag import local_bounded_scifact_provider as module
from climate_rag.agent_protocol import ModelResponseValidationError
from climate_rag.evidence_gap_candidate import CANDIDATE_PROTOCOL, GapProviderAdapter, render_gap_prompt
from climate_rag.local_bounded_scifact_provider import LocalQwenBoundedSciFactProvider, raw_action
from climate_rag.scifact_bounded_runtime import ARMS, run_bounded_slot
from climate_rag.scifact_grounding import Abstract
from climate_rag.scifact_terminal import PROTOCOL, render_scifact_prompt, source_from_abstract
from test_local_scifact_provider import mocked_provider
from test_scifact_train_diagnostic import TinyTokenizer


ABSTAIN = {"action": "abstain", "reason": "insufficient_evidence"}


def wire(decision, span=""):
    return {"evidence_state": {"retrieval_need": "uncertain", "relevance": "unknown", "support": "unknown"},
            "gap_claim_span": span, "decision": decision}


def provider_fixture(tmp_path, monkeypatch, actions, gap=False, **kwargs):
    old, calls, _ = mocked_provider(tmp_path, monkeypatch, actions, **kwargs)
    provider = LocalQwenBoundedSciFactProvider.__new__(LocalQwenBoundedSciFactProvider)
    provider.__dict__.update(old.__dict__)
    provider.gap, provider.plain_partition = gap, object()
    provider.wire_byte_limit = 32768
    provider.wire_protocol = CANDIDATE_PROTOCOL if gap else PROTOCOL
    provider.terminal_protocol = provider.wire_protocol
    parsers = []
    def prefix(data, action, envelope, **kwargs):
        config = types.SimpleNamespace(alphabet="synthetic", max_consecutive_whitespaces=12,
                                       force_json_field_order=gap, max_json_array_length=20)
        result = types.SimpleNamespace(token_enforcer=types.SimpleNamespace(root_parser=types.SimpleNamespace(config=config)))
        parsers.append((result, action, envelope))
        return result
    monkeypatch.setattr(module, "build_bounded_scifact_prefix", prefix)
    return provider, calls, parsers


@pytest.mark.parametrize("gap", [False, True])
def test_provider_actual_prompt_usage_and_whole_wire_receipt(tmp_path, monkeypatch, gap):
    from climate_rag.scifact_runtime_smoke import smoke_cases
    case = smoke_cases()[0]
    from climate_rag.scifact_terminal import action_schema
    case["schema"] = action_schema(["abstain"], [], [], 5)
    provider, calls, parsers = provider_fixture(tmp_path, monkeypatch, [wire(ABSTAIN) if gap else ABSTAIN], gap)
    active = GapProviderAdapter(provider) if gap else provider
    expected = active.count_prompt(case["observation"], case["schema"])
    output = active.generate(case["observation"], case["schema"], 512, 45)
    assert output["usage"]["input_tokens"] == expected == len(calls[0]["input_ids"].ids)
    assert output["diagnostics"]["raw_wire_action"] == "abstain"
    assert json.loads(output["raw"]) == ABSTAIN
    assert output["diagnostics"]["actual_observation"]["visible"]
    assert (parsers[0][2] is not None) == gap
    if gap:
        assert output["usage"]["output_tokens"] > len(output["raw"])
        assert output["diagnostics"]["gap_mapping"]["mapping_status"] == "envelope_valid_action_unvalidated"


@pytest.mark.parametrize("failure", ["model", "grammar"])
def test_provider_failure_is_charged_and_logging_restored(tmp_path, monkeypatch, failure):
    import logging
    from climate_rag.scifact_runtime_smoke import smoke_cases
    provider, _, _ = provider_fixture(tmp_path, monkeypatch, [ABSTAIN],
                                      raise_model=failure == "model", log=failure == "grammar")
    case = smoke_cases()[0]
    before = logging.getLogger().handlers[:]
    with pytest.raises(ModelResponseValidationError) as captured:
        provider.generate(case["observation"], case["schema"], 512, 45)
    assert captured.value.usage["input_tokens"] > 0
    assert logging.getLogger().handlers == before
    assert bool(captured.value.diagnostics.get("output_usage_unknown")) == (failure == "model")


class SyntheticBackend:
    kind, name = "fixture", "synthetic-counted-backend"

    def __init__(self, actions, gap):
        self.actions, self.gap = iter(actions), gap
        self.wire_protocol = CANDIDATE_PROTOCOL if gap else PROTOCOL
        self.terminal_protocol = self.wire_protocol
        self.base = types.SimpleNamespace(tokenizer=TinyTokenizer())
        self.observations = []

    def count_text(self, text):
        return len(self.base.tokenizer.encode(text))

    def count_prompt(self, observation, schema):
        fn = render_gap_prompt if self.gap else render_scifact_prompt
        return self.count_text(fn(self.base.tokenizer, observation, schema))

    def generate(self, observation, schema, max_output_tokens, remaining_seconds):
        self.observations.append(copy.deepcopy(observation))
        raw = json.dumps(next(self.actions), separators=(",", ":"))
        return {"raw": raw, "usage": {"input_tokens": self.count_prompt(observation, schema), "output_tokens": 50},
                "diagnostics": {"raw_wire_action": raw_action(raw, self.gap), "private_attachment": {
                    "sha256": hashlib.sha256(raw.encode()).hexdigest(), "attempted_bytes": len(raw.encode()),
                    "stored_bytes": len(raw.encode()), "truncated": False, "io_failed": False}}}


def run(actions, gap=False, large=False):
    corpus = {i: Abstract(i, "Fixture", tuple(f"Sentence {j} " + ('x' * 220 if large else '.')
                                            for j in range(30 if large else 3)), False) for i in range(100, 120)}
    sources = [source_from_abstract(a) for a in corpus.values()]
    backend = SyntheticBackend(actions, gap)
    row = run_bounded_slot(42, "Fixture claim", "adaptive", ARMS[int(gap)], backend,
                           lambda q, k: sources[:k], lambda q, c: c, corpus)
    return row["result"], backend


def test_common_initial_packing_identical_actual_cost_not_padded():
    a, fa = run([ABSTAIN], large=True)
    b, gb = run([wire(ABSTAIN)], gap=True, large=True)
    assert a["initial_context_identity"] == b["initial_context_identity"]
    assert a["outcome"].startswith("model_abstention") and b["outcome"].startswith("model_abstention")
    assert a["usage"]["input_tokens"] < b["usage"]["input_tokens"] <= 8192
    assert fa.observations[0]["current_citable"] == gb.observations[0]["current_citable"]


@pytest.mark.parametrize("gap", [False, True])
def test_proposal_validation_execution_and_next_feedback_stay_distinct(gap):
    decisions = [{"action": "read", "source_ids": [f"c{i}" for i in range(5)]},
                 {"action": "read", "source_ids": ["c5"]}, ABSTAIN]
    result, _ = run([wire(d) for d in decisions] if gap else decisions, gap)
    audit = result["decision_execution_audit"]
    assert audit[0]["raw_wire_action"] == "read" and audit[0]["strict_status"] == "validation_failed"
    assert audit[0]["actual_event_index"] is None
    assert audit[1]["actual_event_status"] == "completed"
    assert audit[1]["next_feedback"].startswith("tool_completed:")
    assert audit[1]["next_strict_action"] == "abstain"
    assert result["initial_context_identity"]["previews"][0]["source_id"] == "c5"
    assert not result["trace_unlinked_events"] and result["usage"]["output_tokens"] == 150


def test_invalid_gap_span_cost_and_concrete_repair_not_forced_tool():
    result, backend = run([wire(ABSTAIN, "invented missing claim"), wire(ABSTAIN)], gap=True)
    assert result["usage"]["output_tokens"] == 100 and result["model_calls"] == 2
    assert "exact immutable_claim substring" in backend.observations[1]["feedback"]
    assert result["outcome"].startswith("model_abstention")
    assert not any(e.get("model_selected") for e in result["events"])


def test_shared_directory_quota_truncation_never_becomes_complete_receipt(tmp_path, monkeypatch):
    from climate_rag.private_diagnostics_v3 import PrivateDiagnosticStore
    from climate_rag.scifact_terminal import action_schema
    observation = {"immutable_claim": "Synthetic", "current_citable": [], "preview_only": []}
    schema = action_schema(["abstain"], [], [], 5)
    a, _, _ = provider_fixture(tmp_path, monkeypatch, [ABSTAIN])
    b, _, _ = provider_fixture(tmp_path, monkeypatch, [ABSTAIN])
    a.private_store = PrivateDiagnosticStore(tmp_path, max_files=1, max_bytes=1000)
    a.generate(observation, schema, 512, 45)
    b.private_store = PrivateDiagnosticStore(tmp_path, max_files=1, max_bytes=1000)
    with pytest.raises(ModelResponseValidationError) as captured:
        b.generate(observation, schema, 512, 45)
    assert captured.value.diagnostics["category"] == "private_response_incomplete"
    assert captured.value.usage["output_tokens"] > 0


def test_slot_directories_exclusive_and_worst_case_budget(tmp_path, monkeypatch):
    provider, _, _ = provider_fixture(tmp_path, monkeypatch, [ABSTAIN])
    first = tmp_path / "slot-01"
    provider.start_slot(first)
    with pytest.raises(FileExistsError):
        provider.start_slot(first)
    assert first.is_dir()
    provider.start_slot(tmp_path / "slot-02")


@pytest.mark.parametrize("gap", [False, True])
def test_versioned_runtime_smoke_includes_charged_truncation(tmp_path, monkeypatch, gap):
    from climate_rag.scifact_bounded_smoke import bounded_runtime_smoke
    from test_local_scifact_provider import payload
    actions = [payload(), payload("c5", "c6"), payload(), payload("c5", "c6")]
    provider, calls, _ = provider_fixture(tmp_path, monkeypatch, [wire(a) for a in actions] if gap else actions, gap)
    report = bounded_runtime_smoke(provider)
    assert report["status"] == "passed" and len(calls) == 4
    assert report["records"][-1]["usage"]["output_tokens"] == 1
    if gap:
        assert report["records"][-1]["failure"] == "provider_response_invalid"


def test_duplicate_raw_action_is_not_claimed_as_an_observed_action():
    assert raw_action('{"action":"read","action":"abstain"}', False) is None


@pytest.mark.parametrize("fault", ["identity", "preview", "order", "missing"])
def test_pair_comparability_failures_retain_cost_and_do_not_score(fault):
    from climate_rag.scifact_bounded_comparison import compare_initial_contexts
    from climate_rag.scifact_diagnostic_runtime import ROUTES
    identity = {"visible": [{"sentence_id": "c0:0", "sha256": "a"}, {"sentence_id": "c0:1", "sha256": "b"}],
                "previews": [{"source_id": "c5", "sha256": "c", "citable": False}]}
    rows = [{"claim_id": 1, "route": route, "arm": arm,
             "result": {"initial_context_identity": copy.deepcopy(identity), "usage": {"input_tokens": 10, "output_tokens": 2}}}
            for route in ROUTES for arm in ARMS]
    bad = rows[1]["result"]
    if fault == "missing":
        bad["initial_context_identity"] = None
    elif fault == "order":
        bad["initial_context_identity"]["visible"].reverse()
    else:
        bad["initial_context_identity"]["visible" if fault == "identity" else "previews"][0]["sha256"] = "changed"
    gate = compare_initial_contexts(rows, [1])
    assert not gate["all_comparable"] and sum(not p["comparable"] for p in gate["pairs"]) == 1
    assert gate["pairs"][0]["costs"][ARMS[1]]["input_tokens"] == 10
    with pytest.raises(ValueError, match="matrix"):
        compare_initial_contexts(rows + rows[:1], [1])
