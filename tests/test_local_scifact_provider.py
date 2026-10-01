import contextlib
import copy
import inspect
import importlib.util
import json
import logging
import sys
import types
from pathlib import Path

import pytest

from climate_rag import local_scifact_provider
from climate_rag.agent_protocol import ModelResponseValidationError
from climate_rag.agent_v3 import V3Budget
from climate_rag.local_agent_v3 import LocalQwenV3Provider
from climate_rag.local_scifact_provider import LocalQwenSciFactProvider
from climate_rag.private_diagnostics_v3 import PrivateDiagnosticStore
from climate_rag.scifact_agent_v1 import SciFactDocumentAgentV1
from climate_rag.scifact_grounding import Abstract
from climate_rag.scifact_runtime_smoke import smoke_cases, synthetic_runtime_smoke
from climate_rag.scifact_terminal import (
    PROTOCOL,
    render_scifact_prompt,
    source_from_abstract,
    to_original_prediction,
)


def payload(alias="c0", second="c1"):
    return {
        "action": "answer",
        "documents": [
            {
                "source_id": alias,
                "label": "SUPPORTS",
                "sentence_ids": [
                    f"{alias}:3",
                    f"{alias}:1",
                    f"{alias}:2",
                    f"{alias}:0",
                ],
            },
            {
                "source_id": second,
                "label": "REFUTES",
                "sentence_ids": [f"{second}:2", f"{second}:0"],
            },
        ],
    }


def mocked_provider(tmp_path, monkeypatch, actions, *, log=False, raise_model=False):
    calls, parsers = [], []
    integration = types.ModuleType("lmformatenforcer.integrations.transformers")

    def prefix_builder(data, parser):
        callback = types.SimpleNamespace(parser=parser)
        parsers.append(callback)
        return callback

    integration.build_transformers_prefix_allowed_tokens_fn = prefix_builder
    integration.build_token_enforcer_tokenizer_data = (
        lambda tokenizer, use_bitmask: object()
    )
    monkeypatch.setitem(sys.modules, integration.__name__, integration)

    class Tensor:
        def __init__(self, ids):
            self.ids, self.shape = ids, (1, len(ids))

    class Inputs(dict):
        def __init__(self, ids):
            self.input_ids = Tensor(ids)
            super().__init__(input_ids=self.input_ids)

        def to(self, device):
            return self

    class Tokenizer:
        eos_token_id = 300000

        def apply_chat_template(self, messages, **kwargs):
            assert kwargs == {
                "tokenize": False,
                "add_generation_prompt": True,
                "enable_thinking": False,
            }
            return json.dumps(messages)

        def encode(self, text, *, add_special_tokens=False):
            assert add_special_tokens is False
            return list(map(ord, text))

        def __call__(self, text, *, return_tensors, add_special_tokens):
            assert return_tensors == "pt" and add_special_tokens is False
            return Inputs(self.encode(text, add_special_tokens=False))

        def decode(self, ids, *, skip_special_tokens):
            assert skip_special_tokens
            return "".join(chr(i) for i in ids if i != self.eos_token_id)

    tokenizer, responses = Tokenizer(), iter(actions)

    def generate(**kwargs):
        calls.append(kwargs)
        assert kwargs["do_sample"] is False and kwargs["num_beams"] == 1
        if log:
            logging.error("synthetic-private-grammar-detail")
        if raise_model:
            raise RuntimeError("synthetic generation failure")
        value = next(responses)
        raw = (
            value
            if isinstance(value, str)
            else json.dumps(value, separators=(",", ":"))
        )
        generated = tokenizer.encode(raw) + [tokenizer.eos_token_id]
        generated = generated[: kwargs["max_new_tokens"]]
        return [kwargs["input_ids"].ids + generated]

    provider = LocalQwenSciFactProvider.__new__(LocalQwenSciFactProvider)
    provider.name = "synthetic-mocked-hf"
    provider.private_dir, provider.private_store = (
        tmp_path,
        PrivateDiagnosticStore(tmp_path),
    )
    provider.tokenizer_data = object()
    provider.base = types.SimpleNamespace(
        tokenizer=tokenizer,
        name="synthetic-base",
        model=types.SimpleNamespace(
            device="cpu",
            generate=generate,
            generation_config=types.SimpleNamespace(
                eos_token_id=tokenizer.eos_token_id
            ),
        ),
        _torch=types.SimpleNamespace(inference_mode=contextlib.nullcontext),
    )
    return provider, calls, parsers


def test_template_count_and_generation_tokenization_are_identical(
    tmp_path, monkeypatch
):
    provider, calls, parsers = mocked_provider(tmp_path, monkeypatch, [payload()])
    case = smoke_cases()[0]
    expected = provider.count_prompt(case["observation"], case["schema"])
    result = provider.generate(case["observation"], case["schema"], 512, 45)
    assert result["usage"]["input_tokens"] == expected == len(calls[0]["input_ids"].ids)
    assert calls[0]["input_ids"].ids == provider.base.tokenizer.encode(
        render_scifact_prompt(
            provider.base.tokenizer, case["observation"], case["schema"]
        )
    )
    assert (
        result["diagnostics"]["eos_observed"]
        and not result["diagnostics"]["reached_max_new_tokens"]
    )
    assert provider.terminal_protocol == PROTOCOL and len(parsers) == 1


def test_dynamic_prefix_state_fresh_a_b_a_and_truncation(tmp_path, monkeypatch):
    provider, calls, parsers = mocked_provider(
        tmp_path,
        monkeypatch,
        [payload(), payload("c5", "c6"), payload(), payload("c5", "c6")],
    )
    result = synthetic_runtime_smoke(provider, execution_kind="mock_hf")
    assert result["status"] == "passed"
    assert len({id(p.parser) for p in parsers}) == len({id(p) for p in parsers}) == 4
    assert parsers[0].parser is not parsers[2].parser
    assert calls[-1]["max_new_tokens"] == 1
    assert result["records"][-1]["usage"]["output_tokens"] == 1
    assert result["records"][-1]["diagnostics"]["eos_observed"] is False


def test_runtime_smoke_retains_usage_after_invalid_complete_output(
    tmp_path, monkeypatch
):
    provider, _, _ = mocked_provider(
        tmp_path,
        monkeypatch,
        ["{", payload("c5", "c6"), payload(), payload("c5", "c6")],
    )
    result = synthetic_runtime_smoke(provider, execution_kind="mock_hf")
    assert result["status"] == "failed"
    assert result["records"][0]["usage"]["output_tokens"] == 2
    assert result["records"][0]["failure"] == "JSONDecodeError"
    assert all(r["passed"] for r in result["records"][1:])


def test_runtime_smoke_second_count_failure_keeps_first_generation_cost(
    tmp_path, monkeypatch
):
    provider, calls, _ = mocked_provider(
        tmp_path, monkeypatch, [payload(), payload(), payload("c5", "c6")]
    )
    count_prompt = provider.count_prompt
    counts = [0]

    def fail_second(observation, schema):
        counts[0] += 1
        if counts[0] == 2:
            raise ValueError("synthetic tokenizer failure")
        return count_prompt(observation, schema)

    monkeypatch.setattr(provider, "count_prompt", fail_second)
    result = synthetic_runtime_smoke(provider, execution_kind="mock_hf")
    assert result["status"] == "failed" and len(result["records"]) == 4
    assert result["records"][0]["usage"]["output_tokens"] > 0
    assert result["records"][1]["failure_stage"] == "before_generation"
    assert result["records"][1]["provider_generate_attempted"] is False
    assert result["attempted_calls"] == len(calls) == 3
    assert all(r["passed"] for r in result["records"][2:])


def load_smoke_script(monkeypatch):
    scripts = Path(__file__).resolve().parents[1] / "scripts"
    monkeypatch.syspath_prepend(str(scripts))
    spec = importlib.util.spec_from_file_location(
        "synthetic_smoke_fixture", scripts / "smoke_scifact_provider.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_runtime_model_args_require_explicit_optin_before_loading(
    tmp_path, monkeypatch
):
    script = load_smoke_script(monkeypatch)
    monkeypatch.setattr(
        sys,
        "argv",
        ["smoke", "--model-dir", str(tmp_path), "--output", str(tmp_path / "new.json")],
    )
    with pytest.raises(ValueError, match="default mode"):
        script.main()
    assert not (tmp_path / "new.json").exists()


def test_core_lmfe_token_prefix_without_hf_torch_bridge(monkeypatch):
    script = load_smoke_script(monkeypatch)

    class AsciiTokenizer:
        eos_token_id, all_special_ids = 128, [128]

        def __len__(self):
            return 129

        def encode(self, text, **kwargs):
            return list(map(ord, text))

        def decode(self, ids):
            return "".join(chr(i) for i in ids if i != 128)

    tokenizer = AsciiTokenizer()
    data = script.cpu_tokenizer_data(tokenizer)
    case = smoke_cases()[0]
    assert script.prefix_accepts(
        data, tokenizer, "synthetic", case["schema"], payload()
    )[0]
    cross_doc = copy.deepcopy(payload())
    cross_doc["documents"][0]["sentence_ids"][0] = "c1:3"
    assert len(cross_doc["documents"]) == 2
    assert not script.prefix_accepts(
        data, tokenizer, "synthetic", case["schema"], cross_doc
    )[0]
    assert not script.prefix_accepts(
        data, tokenizer, "synthetic", smoke_cases()[1]["schema"], payload()
    )[0]


def test_mocked_provider_controller_maps_labels_and_original_sentence_order(
    tmp_path, monkeypatch
):
    provider, _, _ = mocked_provider(tmp_path, monkeypatch, [payload()])
    corpus = {
        i: Abstract(i, "Fixture", ("zero", "one", "two", "three"), False)
        for i in [100, 101]
    }
    result = SciFactDocumentAgentV1(
        provider, lambda q, k: [source_from_abstract(d) for d in corpus.values()]
    ).run("Fixture claim")
    converted = to_original_prediction(1, result, corpus)
    assert converted["prediction"]["evidence"] == {
        "100": {"label": "SUPPORT", "sentences": [3, 1, 2, 0]},
        "101": {"label": "CONTRADICT", "sentences": [2, 0]},
    }


@pytest.mark.parametrize("bad_kind", ["old_context", "cross_document"])
def test_stale_or_cross_doc_output_from_provider_is_rejected(
    tmp_path, monkeypatch, bad_kind
):
    bad = (
        payload()
        if bad_kind == "old_context"
        else {
            "action": "answer",
            "documents": [
                {"source_id": "c2", "label": "SUPPORTS", "sentence_ids": ["c3:0"]}
            ],
        }
    )
    provider, _, _ = mocked_provider(
        tmp_path,
        monkeypatch,
        [
            {"action": "read", "source_ids": ["c2", "c3"]},
            bad,
            {"action": "abstain", "reason": "insufficient_evidence"},
        ],
    )
    docs = [
        source_from_abstract(
            Abstract(i, "Fixture", ("zero", "one", "two", "three"), False)
        )
        for i in range(4)
    ]
    result = SciFactDocumentAgentV1(
        provider, lambda q, k: docs, budget=V3Budget(context_k=2)
    ).run("Fixture claim")
    assert result["answer"] is None and result["validation_repairs"] == 1
    assert result["generation_attempts"][1]["error_code"] in {
        "sentence_not_currently_visible",
        "cross_document_reference",
    }
    assert result["usage"]["output_tokens"] > 0


@pytest.mark.parametrize("failure", ["model", "grammar_log", "diagnostic", "disk"])
def test_failures_keep_known_or_unknown_usage_and_private_logs(
    tmp_path, monkeypatch, capsys, failure
):
    provider, _, _ = mocked_provider(
        tmp_path,
        monkeypatch,
        [payload()],
        log=failure in {"model", "grammar_log"},
        raise_model=failure == "model",
    )
    if failure == "diagnostic":
        monkeypatch.setattr(
            local_scifact_provider,
            "response_diagnostics",
            lambda *a, **k: (_ for _ in ()).throw(OSError("private-detail")),
        )
    if failure == "disk":
        monkeypatch.setattr(
            provider.private_store,
            "append",
            lambda *a, **k: (_ for _ in ()).throw(OSError("private-detail")),
        )
    original = logging.getLogger().handlers[:]
    case = smoke_cases()[0]
    with pytest.raises(ModelResponseValidationError) as captured:
        provider.generate(case["observation"], case["schema"], 512, 45)
    assert captured.value.usage["input_tokens"] == provider.count_prompt(
        case["observation"], case["schema"]
    )
    if failure == "model":
        assert captured.value.diagnostics["output_usage_unknown"] is True
    else:
        assert captured.value.usage["output_tokens"] > 0
    assert logging.getLogger().handlers == original
    assert "private-detail" not in str(captured.value.diagnostics)
    assert "synthetic-private-grammar-detail" not in capsys.readouterr().err


def test_private_quota_persists_between_generations(tmp_path, monkeypatch):
    provider, _, _ = mocked_provider(tmp_path, monkeypatch, [payload()] * 4)
    provider.private_store = PrivateDiagnosticStore(tmp_path, max_files=2, max_bytes=64)
    case = smoke_cases()[0]
    for _ in range(4):
        response = provider.generate(case["observation"], case["schema"], 512, 45)
        assert json.loads(response["raw"]) == payload()
    assert sum(p.stat().st_size for p in tmp_path.iterdir()) <= 64
    assert len(list(tmp_path.iterdir())) <= 2
    assert response["diagnostics"]["private_attachment"]["truncated"]


def test_inherits_loader_initialization_and_rejects_environment_override(
    tmp_path, monkeypatch
):
    seen = []

    def base_init(self, model_dir, manifest, *, private_dir, device):
        seen.append((model_dir, manifest, private_dir, device))
        self.base = types.SimpleNamespace(name="frozen-qwen")

    monkeypatch.setattr(LocalQwenV3Provider, "__init__", base_init)
    provider = LocalQwenSciFactProvider(
        tmp_path, {"model": "hash"}, private_dir=tmp_path, device="cpu"
    )
    assert seen == [(tmp_path, {"model": "hash"}, tmp_path, "cpu")]
    assert PROTOCOL in provider.name
    monkeypatch.setenv("LMFE_MAX_JSON_ARRAY_LENGTH", "1")
    with pytest.raises(ValueError, match="environment"):
        LocalQwenSciFactProvider(tmp_path, {}, private_dir=tmp_path)
    with pytest.raises(ValueError, match="environment"):
        provider.generate({}, {}, 512, 1)


def test_generation_parity_except_renderer_env_guard_and_effective_config_receipt():
    old = inspect.getsource(LocalQwenV3Provider.generate)
    expected = old.replace("render_v3_prompt(self.base.tokenizer, observation, schema)", "self.render(observation, schema)")
    expected = expected.replace("JsonSchemaParser(schema,", "JsonSchemaParser(self.decoder_schema(schema),")
    expected = expected.replace(
        "        from lmformatenforcer import JsonSchemaParser",
        "        require_clean_grammar_environment()\n        from lmformatenforcer import JsonSchemaParser",
    )
    extra = """        diagnostics["effective_parser_config"] = {
            "alphabet_sha256": hashlib.sha256(
                parser.config.alphabet.encode()
            ).hexdigest(),
            "max_consecutive_whitespaces": parser.config.max_consecutive_whitespaces,
            "force_json_field_order": parser.config.force_json_field_order,
            "max_json_array_length": parser.config.max_json_array_length,
        }
"""
    actual = inspect.getsource(LocalQwenSciFactProvider.generate)
    assert extra in actual
    actual = actual.replace(extra, "")
    actual = actual.replace('        diagnostics.update(decoder.metadata)\n', '')
    actual = actual.replace('decoder.grammar', '"lm-format-enforcer0.11.3/fresh-inline-anyOf"')
    actual = actual.replace('decoder.config_identity', 'grammar_config_identity()')
    actual = actual.replace(
        '        decoder = self.build_decoder(observation, schema)\n        parser, prefix_fn = decoder.parser, decoder.prefix\n',
        '        parser = JsonSchemaParser(self.decoder_schema(schema), config=fixed_grammar_config())\n'
        '        prefix_fn = build_transformers_prefix_allowed_tokens_fn(\n            self.tokenizer_data, parser\n        )\n')
    imports = ('        from lmformatenforcer import JsonSchemaParser\n'
        '        from lmformatenforcer.integrations.transformers import (\n'
        '            build_transformers_prefix_allowed_tokens_fn,\n        )\n\n')
    actual = actual.replace('        begin = time.perf_counter()\n', imports + '        begin = time.perf_counter()\n')
    assert actual == expected  # only the reviewed hook extraction differs
    hook = inspect.getsource(LocalQwenSciFactProvider.build_decoder)
    assert 'JsonSchemaParser(self.decoder_schema(schema), config=fixed_grammar_config())' in hook
    assert 'build_transformers_prefix_allowed_tokens_fn(self.tokenizer_data, parser)' in hook
