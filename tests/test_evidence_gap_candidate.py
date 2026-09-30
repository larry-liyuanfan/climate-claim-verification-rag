"""Synthetic CPU contracts only; no real query, model weights or quality claims."""

import copy
import hashlib
import inspect
import json
import os
import stat
import string
import types
from pathlib import Path

import pytest
from jsonschema import ValidationError as SchemaValidationError

from climate_rag.agent_protocol import ModelResponseValidationError
from climate_rag.agent_v3 import Source, V3Budget
from climate_rag.evidence_gap_candidate import (
    CANDIDATE_PROTOCOL,
    EvidenceGapCandidate,
    GapProviderAdapter,
    gap_schema,
    parse_gap_wire,
    render_gap_prompt,
)
from climate_rag.local_gap_provider import LocalQwenEvidenceGapProvider
from climate_rag.evidence_gap_grammar import build_ordered_gap_prefix
from climate_rag.local_scifact_provider import LocalQwenSciFactProvider
from climate_rag.private_diagnostics_v3 import PrivateDiagnosticStore
from climate_rag.scifact_agent_v1 import SciFactDocumentAgentV1
from climate_rag.scifact_terminal import action_schema

from test_local_scifact_provider import mocked_provider

CLAIM = "Synthetic valve measurement increased in 2020."
ABSTAIN = {"action": "abstain", "reason": "insufficient_evidence"}
ROUTES = ["fixed_retrieval", "fixed_rerank", "deterministic_extra", "adaptive"]


def wire(decision=None, gap="", **state):
    return {
        "evidence_state": {
            "retrieval_need": "uncertain",
            "relevance": "unknown",
            "support": "unknown",
            **state,
        },
        "gap_claim_span": gap,
        "decision": copy.deepcopy(decision if decision is not None else ABSTAIN),
    }


def answer(alias="c0", indices=(0,), label="SUPPORTS"):
    return {
        "action": "answer",
        "documents": [
            {
                "source_id": alias,
                "label": label,
                "sentence_ids": [f"{alias}:{i}" for i in indices],
            }
        ],
    }


class CharTokenizer:
    def apply_chat_template(self, messages, **kwargs):
        assert kwargs == {
            "tokenize": False,
            "add_generation_prompt": True,
            "enable_thinking": False,
        }
        return json.dumps(messages, ensure_ascii=False)

    def encode(self, text, *, add_special_tokens=False):
        assert not add_special_tokens
        return list(map(ord, text))


class FixtureBackend:
    name, kind, wire_protocol = "synthetic-gap", "fixture", CANDIDATE_PROTOCOL

    def __init__(self, actions, root, tokenizer=None, **quota):
        self.actions, self.requests = iter(actions), []
        self.tokenizer = tokenizer or CharTokenizer()
        self.store = PrivateDiagnosticStore(root, **quota)

    def count_text(self, text):
        return len(self.tokenizer.encode(text, add_special_tokens=False))

    def count_prompt(self, observation, schema):
        return self.count_text(render_gap_prompt(self.tokenizer, observation, schema))

    def generate(self, observation, schema, max_output_tokens, remaining_seconds):
        self.requests.append(copy.deepcopy((observation, schema)))
        value = next(self.actions)
        if isinstance(value, Exception):
            raise value
        raw = (
            value
            if isinstance(value, str)
            else json.dumps(value, separators=(",", ":"))
        )
        receipt = self.store.sink("response", 32768)
        receipt.write(raw)
        return {
            "raw": raw,
            "usage": {
                "input_tokens": self.count_prompt(observation, schema),
                "output_tokens": self.count_text(raw),
            },
            "diagnostics": {"private_attachment": receipt.receipt()},
        }


def sources():
    return [
        Source(
            str(i),
            "Synthetic document",
            tuple(f"Synthetic document {i}, sentence {j}." for j in range(3)),
        )
        for i in range(6)
    ]


def run(tmp_path, actions, route="adaptive", *, context_k=1, **quota):
    backend = FixtureBackend(actions, tmp_path, **quota)
    candidate = EvidenceGapCandidate(
        backend,
        lambda query, k: sources(),
        rerank=lambda query, rows: rows,
        budget=V3Budget(context_k=context_k),
    )
    return candidate.run(CLAIM, route), backend


def request():
    observation = {
        "immutable_claim": CLAIM,
        "allowed_actions": ["abstain", "read", "answer"],
        "current_citable": [{"sentence_id": "c0:0", "text": "Synthetic."}],
        "preview_only": [],
        "feedback": None,
        "remaining_calls": 5,
        "remaining_tools": 4,
    }
    return observation, action_schema(
        observation["allowed_actions"], ["c0", "c1"], ["c0:0"], 1
    )


@pytest.mark.parametrize("decision", [ABSTAIN, answer()])
@pytest.mark.parametrize("route", ROUTES)
def test_all_routes_may_terminate_immediately_with_same_template(
    tmp_path, decision, route
):
    result, backend = run(
        tmp_path, [wire(decision, "valve", retrieval_need="needed")], route
    )
    assert result["model_calls"] == 1
    assert not any(e.get("model_selected") for e in result["events"])
    assert result["candidate_protocol"] == CANDIDATE_PROTOCOL
    assert (
        result["answer"] is not None
        if decision["action"] == "answer"
        else result["answer"] is None
    )
    obs, schema = backend.requests[0]
    assert (
        list(schema["properties"])
        == schema["required"]
        == ["evidence_state", "gap_claim_span", "decision"]
    )
    assert obs["evidence_gap_feedback"]["previous_valid_envelope"] is None
    assert result["usage"]["output_tokens"] > backend.count_text(json.dumps(decision))


def test_repeated_packing_is_pure_and_count_generate_match(tmp_path):
    backend = FixtureBackend([wire()], tmp_path)
    adapter = GapProviderAdapter(backend)
    obs, schema = request()
    before = copy.deepcopy((obs, schema))
    expected = adapter.count_prompt(obs, schema)
    empty = {**obs, "current_citable": []}
    adapter.count_prompt(empty, action_schema(["abstain"], [], [], 1))
    assert adapter.count_prompt(obs, schema) == expected and adapter.records == []
    assert (obs, schema) == before
    result = adapter.generate(obs, schema, 512, 120)
    assert result["usage"]["input_tokens"] == expected
    assert (
        result["diagnostics"]["gap_mapping"]["wire_sha256"]
        == hashlib.sha256(
            next(tmp_path.glob("*-response.txt")).read_bytes()
        ).hexdigest()
    )
    assert "wire_raw" not in json.dumps(adapter.records)


def test_read_carries_gap_and_final_visibility_delta_without_semantic_claim(tmp_path):
    result, backend = run(
        tmp_path,
        [wire({"action": "read", "source_ids": ["c1"]}, "valve"), wire(answer("c1"))],
    )
    feedback = backend.requests[1][0]["evidence_gap_feedback"]
    assert feedback["previous_valid_envelope"]["gap_claim_span"] == "valve"
    assert feedback["previous_valid_envelope"]["action_execution"] == "not_asserted"
    delta = feedback["visibility_delta"]
    assert delta["tool_completion_reported"] is True
    assert delta["added_citable_ids"] == [f"c1:{i}" for i in range(3)]
    assert delta["removed_citable_ids"] == [f"c0:{i}" for i in range(3)]
    assert delta["semantic_progress"] == "unmeasured"
    assert result["answer"]["documents"][0]["source_id"] == "1"


def test_rejected_read_does_not_claim_execution_or_drop_cost(tmp_path):
    result, backend = run(
        tmp_path, [wire({"action": "read", "source_ids": ["c0"]}, "valve"), wire()]
    )
    delta = backend.requests[1][0]["evidence_gap_feedback"]["visibility_delta"]
    assert not delta["tool_completion_reported"]
    assert delta["added_citable_ids"] == delta["removed_citable_ids"] == []
    assert result["model_calls"] == 2 and result["validation_repairs"] == 1
    assert not any(e.get("model_selected") for e in result["events"])


@pytest.mark.parametrize("bad", ["not in original", " ", "Synthetic " * 20])
def test_invalid_gap_charged_and_never_silently_repaired(tmp_path, bad):
    result, backend = run(tmp_path, [wire(gap=bad), wire()])
    assert result["generation_attempts"][0]["status"] == "provider_response_invalid"
    assert result["usage"] == {
        key: sum(a["usage"][key] for a in result["generation_attempts"])
        for key in ("input_tokens", "output_tokens")
    }
    assert (
        backend.requests[1][0]["evidence_gap_feedback"]["previous_valid_envelope"]
        is None
    )
    assert result["candidate_wire_audit"][0]["mapped_action"] is None


def test_duplicate_json_keys_wrong_order_extra_state_and_disallowed_action_rejected():
    obs, schema = request()
    schema = gap_schema(schema)
    good = wire()
    assert parse_gap_wire(json.dumps(good), CLAIM, schema) == good
    bad = [
        json.dumps(good).replace(
            '"gap_claim_span": ""', '"gap_claim_span": "", "gap_claim_span": "valve"'
        ),
        json.dumps(
            {
                "decision": ABSTAIN,
                "evidence_state": good["evidence_state"],
                "gap_claim_span": "",
            }
        ),
        json.dumps(wire({"action": "rerank"})),
        json.dumps(wire(extra="injected")),
    ]
    for raw in bad:
        with pytest.raises((ValueError, SchemaValidationError)):
            parse_gap_wire(raw, CLAIM, schema)


def test_stale_ids_and_duplicate_citations_not_substituted(tmp_path):
    result, _ = run(
        tmp_path,
        [
            wire({"action": "read", "source_ids": ["c1"]}),
            wire(answer("c0")),
            wire(answer("c1", (0, 0))),
            wire(),
        ],
    )
    assert result["validation_repairs"] == 2 and result["answer"] is None
    assert result["generation_attempts"][1]["status"] == "provider_response_invalid"
    assert result["generation_attempts"][2]["status"] == "validation_failed"
    assert result["candidate_wire_audit"][2]["mapped_action"] == answer("c1", (0, 0))


def test_same_claim_repeated_runs_do_not_share_gap_history(tmp_path):
    backend = FixtureBackend([wire(gap="valve"), wire()], tmp_path)
    candidate = EvidenceGapCandidate(backend, lambda query, k: sources())
    candidate.run(CLAIM, "fixed_retrieval")
    candidate.run(CLAIM, "adaptive")
    assert all(
        o["evidence_gap_feedback"]["previous_valid_envelope"] is None
        for o, _ in backend.requests
    )


def test_private_quota_truncation_preserves_action_cost_and_reports_missing_raw(
    tmp_path,
):
    result, _ = run(tmp_path, [wire(answer())], max_files=1, max_bytes=10)
    mapping = result["candidate_wire_audit"][0]
    receipt = mapping["wire_private_attachment"]
    assert receipt["truncated"] and receipt["stored_bytes"] == 10
    assert result["answer"] is not None and result["usage"]["output_tokens"] > 10
    assert "wire_raw" not in mapping


def test_unknown_backend_usage_is_not_free_success(tmp_path):
    failure = ModelResponseValidationError(
        {"input_tokens": 123, "output_tokens": 0}, {"output_usage_unknown": True}
    )
    result, _ = run(tmp_path, [failure, wire()])
    assert result["unknown_usage_attempts"] == 1
    assert result["usage"]["input_tokens"] >= 123
    assert result["candidate_wire_audit"][0]["mapping_status"] == "backend_invalid"


def test_frozen_provider_is_not_implicitly_reused_as_wire_backend():
    with pytest.raises(ValueError, match="own_wire_backend"):
        GapProviderAdapter(types.SimpleNamespace(wire_protocol="old"))


def test_bare_candidate_backend_cannot_bypass_mapping():
    provider = LocalQwenEvidenceGapProvider.__new__(LocalQwenEvidenceGapProvider)
    with pytest.raises(ValueError, match="own provider prompt"):
        SciFactDocumentAgentV1(provider, lambda query, k: sources())


def test_wrong_attachment_identity_is_charged_and_rejected(tmp_path):
    backend = FixtureBackend([wire()], tmp_path)
    original = backend.generate

    def corrupt(*args):
        response = original(*args)
        response["diagnostics"]["private_attachment"]["sha256"] = "0" * 64
        return response

    backend.generate = corrupt
    obs, schema = request()
    with pytest.raises(ModelResponseValidationError) as failure:
        GapProviderAdapter(backend).generate(obs, schema, 512, 120)
    assert failure.value.usage["output_tokens"] > 0
    assert failure.value.diagnostics["gap_mapping"]["mapped_action"] is None


def candidate_mock(tmp_path, monkeypatch, actions, **kwargs):
    prior, calls, _ = mocked_provider(tmp_path, monkeypatch, actions, **kwargs)
    provider = LocalQwenEvidenceGapProvider.__new__(LocalQwenEvidenceGapProvider)
    provider.__dict__.update(prior.__dict__)
    callbacks = []

    def build(data, schema):
        config = types.SimpleNamespace(
            alphabet="fixture",
            max_consecutive_whitespaces=12,
            force_json_field_order=True,
            max_json_array_length=20,
        )
        callback = types.SimpleNamespace(
            token_enforcer=types.SimpleNamespace(
                root_parser=types.SimpleNamespace(config=config)
            )
        )
        callbacks.append(callback)
        return callback

    monkeypatch.setattr(
        "climate_rag.local_gap_provider.build_ordered_gap_prefix", build
    )
    return provider, calls, callbacks


def test_mock_hf_real_candidate_method_counts_generates_and_maps_same_template(
    tmp_path, monkeypatch
):
    provider, calls, callbacks = candidate_mock(tmp_path, monkeypatch, [wire(answer())])
    adapter = GapProviderAdapter(provider)
    obs, schema = request()
    expected = adapter.count_prompt(obs, schema)
    decorated, full_schema = adapter.prepare(obs, schema)
    expected_tokens = provider.base.tokenizer.encode(
        render_gap_prompt(provider.base.tokenizer, decorated, full_schema)
    )
    response = adapter.generate(obs, schema, 512, 120)
    # Recorded mapping contains the same visibility; count/generate sees one full wire.
    assert (
        response["usage"]["input_tokens"] == expected == len(calls[0]["input_ids"].ids)
    )
    assert json.loads(response["raw"]) == answer()
    assert response["diagnostics"]["effective_parser_config"]["force_json_field_order"]
    assert calls[0]["prefix_allowed_tokens_fn"] is callbacks[0]
    assert calls[0]["input_ids"].ids == expected_tokens
    assert calls[0]["max_new_tokens"] == 512 and calls[0]["do_sample"] is False


@pytest.mark.parametrize(
    "kwargs,category",
    [
        ({"log": True}, "grammar_backend_error"),
        ({"raise_model": True}, "model_or_grammar_failure"),
    ],
)
def test_mock_hf_candidate_failure_retains_usage(
    tmp_path, monkeypatch, kwargs, category
):
    provider, _, _ = candidate_mock(tmp_path, monkeypatch, [wire()], **kwargs)
    obs, schema = request()
    with pytest.raises(ModelResponseValidationError) as failure:
        GapProviderAdapter(provider).generate(obs, schema, 512, 120)
    assert failure.value.usage["input_tokens"] > 0
    assert failure.value.diagnostics["category"] == category


def test_frozen_model_generate_call_is_unchanged_in_candidate_source():
    def block(cls):
        source = inspect.getsource(cls.generate)
        return source.split("output = self.base.model.generate(", 1)[1].split(
            "except Exception as exc:", 1
        )[0]

    assert block(LocalQwenEvidenceGapProvider) == block(LocalQwenSciFactProvider)


def test_actual_core_callback_enforces_field_order_without_touching_old_config():
    from lmformatenforcer import TokenEnforcerTokenizerData
    from climate_rag.local_agent_v3 import fixed_grammar_config

    data = TokenEnforcerTokenizerData(
        [(ord(c), c, False) for c in string.printable],
        lambda ids: "".join(chr(i) for i in ids),
        9999,
        use_bitmask=False,
        vocab_size=10000,
    )
    _, schema = request()
    schema = gap_schema(schema)

    def accepts(value):
        callback = build_ordered_gap_prefix(data, schema)
        assert (
            callback.token_enforcer.root_parser.config.alphabet
            == data.tokenizer_alphabet
        )
        ids = [ord("X")]
        for token in map(ord, json.dumps(value, separators=(",", ":"))):
            if token not in callback(
                0, types.SimpleNamespace(tolist=lambda: list(ids))
            ):
                return False
            ids.append(token)
        return 9999 in callback(0, types.SimpleNamespace(tolist=lambda: list(ids)))

    good = wire()
    assert accepts(good)
    assert not accepts(
        {
            "decision": ABSTAIN,
            "evidence_state": good["evidence_state"],
            "gap_claim_span": "",
        }
    )
    assert not accepts({"evidence_state": good["evidence_state"], "decision": ABSTAIN})
    assert accepts(good)
    assert fixed_grammar_config().force_json_field_order is False


def test_candidate_grammar_rejects_environment_override(monkeypatch):
    monkeypatch.setenv("LMFE_FORCE_JSON_FIELD_ORDER", "1")
    with pytest.raises(ValueError, match="environment overrides"):
        build_ordered_gap_prefix(object(), {})


def test_smoke_creates_exclusive_absolute_private_fixture_directory(
    tmp_path, monkeypatch
):
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1] / "scripts"))
    from smoke_evidence_gap_candidate import ScriptedBackend

    target = tmp_path / "new-parent" / "private"
    backend = ScriptedBackend(CharTokenizer(), [], target)
    assert backend.store.root.is_absolute() and target.is_dir()
    if os.name == "posix":
        assert stat.S_IMODE(target.stat().st_mode) & 0o077 == 0
    with pytest.raises(FileExistsError):
        ScriptedBackend(CharTokenizer(), [], target)
