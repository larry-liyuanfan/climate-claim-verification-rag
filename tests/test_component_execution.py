from __future__ import annotations

import copy
import json
import types

import pytest

from climate_rag.agent_protocol import ModelResponseValidationError
from climate_rag.component_audit import aggregate, reconcile
from climate_rag.component_decoder import decoder_identity, decoder_schema
from climate_rag.component_execution import durable, execute_matrix, execute_slot
from climate_rag.component_preflight import preflight_cases
from climate_rag.local_agent_v3 import fixed_grammar_config
from climate_rag.local_component_provider import LocalQwenComponentProvider
from climate_rag.local_scifact_provider import LocalQwenSciFactProvider
from climate_rag.scifact_component_contract import ContractError, messages, packing, parse, schema_for
from climate_rag.scifact_component_preparation import prepare_claim
from climate_rag.scifact_component_runtime import ComponentPersistenceError
from climate_rag.scifact_semantic_contract import encoded, sha
from test_scifact_components import TokenizerFixture, example


def slot(spec=None, i=0):
    spec = spec or preflight_cases()[0][0]
    return {"slot": i, "input": spec, "packing": packing(TokenizerFixture(), spec)}


class Fake:
    tokenizer = TokenizerFixture()
    def __init__(self, fail_at=-1, wire=None, error=None):
        self.calls, self.fail_at, self.wire, self.error = 0, fail_at, wire, error
    def start_slot(self, directory):
        directory.mkdir()
        self.directory = directory
    def generate(self, spec, schema, tokens, seconds):
        self.calls += 1
        assert tokens == 512 and seconds == 120
        if self.error or self.calls == self.fail_at:
            raise self.error or ModelResponseValidationError({"input_tokens": 12, "output_tokens": 0}, {"output_usage_unknown": True})
        value = self.wire
        if value is None:
            value = next((expected for case, expected in preflight_cases() if spec == case), None)
        if value is None:
            key = "relation" if spec["component"] == "relation" else "sentence_ids" if spec["component"] == "rationale" else "document_ids"
            value = {"decision": "abstain", key: None if key == "relation" else []}
        raw = value if isinstance(value, str) else json.dumps(value)
        blob = raw.encode()
        (self.directory / "synthetic-response.txt").write_bytes(blob)
        def receipt(payload):
            return {"attempted_bytes": len(payload), "stored_bytes": len(payload), "dropped_bytes": 0,
                    "truncated": False, "io_failed": False, "sha256": sha(payload), "stored_prefix_sha256": sha(payload)}
        return {"raw": raw, "usage": {"input_tokens": packing(self.tokenizer, spec)["input_tokens"], "output_tokens": 40},
                "diagnostics": {"private_attachment": receipt(blob), "grammar_log": receipt(b""),
                    "actual_component_packing": packing(self.tokenizer, spec), "decoder": decoder_identity(schema),
                    "output_tokens": 40, "output_sha256": sha(blob), "output_bytes": len(blob),
                    "eos_observed": True, "reached_max_new_tokens": False}}


def matrix():
    rows = []
    for i in range(12):
        corpus, gold, result = example()
        gold["id"] += i
        if i >= 9:
            gold["evidence"] = {}
        rows.extend(prepare_claim(gold, result, corpus, TokenizerFixture()))
    slots = [{"slot": i, "input": r["input"], "packing": r["packing"]} for i, r in enumerate(rows)]
    targets = [{"slot": i, "claim_id": r["claim_id"], "component": r["component"], "target": r["target"]}
               for i, r in enumerate(rows)]
    return slots, targets


def character_accepts(schema, raw):
    from lmformatenforcer import JsonSchemaParser
    parser = JsonSchemaParser(schema, config=fixed_grammar_config())
    for char in raw:
        if char not in parser.get_allowed_characters():
            return False
        parser = parser.add_character(char)
    return parser.can_end()


@pytest.mark.parametrize("raw", ['{"decision":"select","document_ids":[2,10,1]}',
    '{"decision":"select","document_ids":[1 , 10]}', '{"document_ids":[1,10,2],"decision":"select"}'])
def test_numeric_enum_decoder_repair_without_frozen_prompt_change(raw):
    spec = preflight_cases()[0][0]
    spec["documents"].append(dict(spec["documents"][0], document_id=2))
    original, prompt = schema_for(spec), messages(spec)
    saved = copy.deepcopy(original)
    relaxed = decoder_schema(original)
    assert character_accepts(relaxed, raw) and parse(raw, spec)
    assert original == saved and messages(spec) == prompt
    active = relaxed["anyOf"][0]["properties"]
    assert active["decision"] == {"const": "select"}
    assert active["document_ids"]["uniqueItems"] is True


def test_eight_ordered_rationale_and_nine_rejected():
    spec = preflight_cases()[2][0]
    spec["documents"][0]["sentences"] = [{"sentence_id": i, "text": "Synthetic."} for i in range(10)]
    schema = decoder_schema(schema_for(spec))
    for count, accepted in ((8, True), (9, False)):
        raw = json.dumps({"decision": "select", "sentence_ids": list(reversed(range(count)))})
        assert character_accepts(schema, raw) is accepted


@pytest.mark.parametrize("value", [[1, 1], [999], [1.0], [True]])
def test_relaxed_decoder_never_replaces_strict_validation(value):
    spec = preflight_cases()[0][0]
    with pytest.raises(ContractError):
        parse(json.dumps({"decision": "select", "document_ids": value}), spec)


def test_preflight_four_distinct_inputs_not_four_scripted_responses():
    assert len({sha(encoded(spec)) for spec, _ in preflight_cases()}) == 4


@pytest.mark.parametrize("mutation", ["count", "receipt", "truncated"])
def test_provider_post_call_failure_retains_actual_cost(tmp_path, monkeypatch, mutation):
    spec = preflight_cases()[0][0]
    fake = Fake()
    fake.start_slot(tmp_path / "fake")
    response = fake.generate(spec, schema_for(spec), 512, 120)
    if mutation == "count":
        response["usage"]["input_tokens"] += 1
    elif mutation == "receipt":
        del response["diagnostics"]["private_attachment"]
    else:
        response["diagnostics"]["private_attachment"]["truncated"] = True
    calls = []
    def generated(*args):
        calls.append(1)
        return response
    monkeypatch.setattr(LocalQwenSciFactProvider, "generate", generated)
    provider = LocalQwenComponentProvider.__new__(LocalQwenComponentProvider)
    provider.base = types.SimpleNamespace(tokenizer=fake.tokenizer)
    provider._slot_ready = True
    with pytest.raises(ModelResponseValidationError) as exc:
        provider.generate(spec, schema_for(spec), 512, 120)
    assert exc.value.usage == response["usage"] and calls == [1] and not provider._slot_ready
    with pytest.raises(ContractError, match="single_attempt"):
        provider.generate(spec, schema_for(spec), 512, 120)


def test_durable_reservation_precedes_generate_and_no_second_slot_path(tmp_path):
    provider = Fake()
    original = provider.generate
    def generated(*args):
        assert (tmp_path / "diagnostic-00/started.json").exists()
        return original(*args)
    provider.generate = generated
    result = execute_slot(slot(), provider, tmp_path, "diagnostic")
    assert result["status"] == "valid" and provider.calls == 1
    with pytest.raises(FileExistsError):
        execute_slot(slot(), provider, tmp_path, "diagnostic")
    assert provider.calls == 1
    assert reconcile(slot(), tmp_path, "diagnostic") == result


def test_full_37_mock_calls_and_scoring_after_raw_verification(tmp_path):
    slots, targets = matrix()
    provider = Fake()
    report = execute_matrix(slots, provider, tmp_path / "run")
    assert provider.calls == 37 and report["status"] == "complete"
    records = [reconcile(s, tmp_path / "run", "diagnostic") for s in slots]
    pre = [reconcile(slot(spec, i), tmp_path / "run", "preflight") for i, (spec, _) in enumerate(preflight_cases())]
    summary, scored = aggregate(slots, targets, records, pre)
    assert summary["total_attempted"] == 37 and len(scored) == 33
    assert summary["denominators"]["relation_nei_control"]["planned"] == 3
    assert summary["denominators"]["rationale"]["planned"] == 9
    assert sum(summary["relation_rationale_pairs"].values()) == 9
    assert summary["screening_detail"]["valid_samples"] == 12
    assert summary["screening_detail"]["candidate_missing"] > 0
    assert summary["screening_detail"]["all_annotated_gold_selected"] == 0
    assert summary["rationale_detail"]["valid_samples"] == 9
    assert "claim_id" not in json.dumps(summary)


@pytest.mark.parametrize("failure_at", [1, 5, 7])
def test_stop_keeps_all_planned_and_nei_strata(tmp_path, failure_at):
    slots, targets = matrix()
    provider = Fake(fail_at=failure_at)
    report = execute_matrix(slots, provider, tmp_path / "run")
    assert provider.calls == failure_at and report["status"] == "stopped"
    summary, _ = aggregate(slots, targets, report["diagnostic"], report["preflight"])
    assert summary["unknown_cost_attempts"] == 1
    assert sum(summary["denominators"][c]["planned"] for c in ("screening", "relation", "rationale")) == 33
    assert summary["denominators"]["relation_nei_control"]["planned"] == 3
    assert summary["status"] == "stopped_no_semantic_effect_claim"
    assert any(r["status"] == "not_attempted_after_stop" for r in report["diagnostic"])


def test_invalid_id_is_paid_schema_failure_no_retry(tmp_path):
    result = execute_slot(slot(), Fake(wire={"decision": "select", "document_ids": [999]}), tmp_path, "diagnostic")
    assert result["status"] == "schema_failed" and result["usage_known"] and result["usage"]["output_tokens"] == 40
    assert reconcile(slot(), tmp_path, "diagnostic")["status"] == "schema_failed"


def test_exception_exact_usage_not_lost(tmp_path):
    cost = {"input_tokens": 100, "output_tokens": 17}
    result = execute_slot(slot(), Fake(error=ModelResponseValidationError(cost, {"category": "diagnostic"})), tmp_path, "diagnostic")
    assert result["status"] == "provider_failed" and result["usage"] == cost and result["usage_known"]
    assert reconcile(slot(), tmp_path, "diagnostic")["usage"] == cost


def test_interrupted_started_only_remains_unknown(tmp_path):
    provider = Fake(error=KeyboardInterrupt())
    with pytest.raises(KeyboardInterrupt):
        execute_slot(slot(), provider, tmp_path, "diagnostic")
    recovered = reconcile(slot(), tmp_path, "diagnostic")
    assert recovered["attempted"] and recovered["status"] == "provider_failed" and recovered["usage"] is None


def test_no_call_if_started_cannot_be_persisted(tmp_path, monkeypatch):
    import climate_rag.component_execution as execution
    original = durable
    def fail(path, value):
        if path.name == "started.json":
            raise OSError("synthetic")
        original(path, value)
    monkeypatch.setattr(execution, "durable", fail)
    provider = Fake()
    with pytest.raises(OSError):
        execute_slot(slot(), provider, tmp_path, "diagnostic")
    assert provider.calls == 0


def test_completed_persistence_failure_cost_not_zero(tmp_path, monkeypatch):
    import climate_rag.component_execution as execution
    original = durable
    def fail(path, value):
        if path.name == "completed.json":
            raise OSError("synthetic")
        original(path, value)
    monkeypatch.setattr(execution, "durable", fail)
    with pytest.raises(ComponentPersistenceError) as exc:
        execute_slot(slot(), Fake(), tmp_path, "diagnostic")
    assert exc.value.partial_report["usage"]["output_tokens"] == 40
    recovered = reconcile(slot(), tmp_path, "diagnostic")
    assert recovered["status"] == "provider_failed" and recovered["usage_known"] and recovered["usage"]["output_tokens"] == 40


@pytest.mark.parametrize("target", ["raw", "prediction", "wire"])
def test_scorer_does_not_trust_result_prediction_or_receipts(tmp_path, target):
    execute_slot(slot(), Fake(), tmp_path, "diagnostic")
    root = tmp_path / "diagnostic-00"
    if target == "raw":
        (root / "private/synthetic-response.txt").write_text("altered")
    elif target == "prediction":
        record = json.loads((root / "completed.json").read_bytes())
        record["prediction"]["document_ids"] = [10]
        (root / "completed.json").write_bytes(encoded(record))
    else:
        (root / "wire.json").write_text("{}")
    with pytest.raises(ValueError):
        reconcile(slot(), tmp_path, "diagnostic")


def test_gap_preserved_without_replacement_or_calls(tmp_path):
    provider = Fake()
    row = {"slot": 0, "input": None, "packing": {"status": "coverage_gap"}}
    record = execute_slot(row, provider, tmp_path, "diagnostic")
    assert record["status"] == "preparation_gap" and provider.calls == 0


def test_operator_reserves_once_and_targets_after_child_exit(tmp_path, monkeypatch):
    import run_scifact_component_operator as operator
    work = tmp_path / "climate-component-mock"
    source = work / "source"
    (source / "scripts").mkdir(parents=True)
    (source / "SOURCE_REVISION").write_text("a" * 40)
    root = tmp_path / "project"
    (root / "runs").mkdir(parents=True)
    (root / "envs").mkdir()
    monkeypatch.setattr(operator, "__file__", str(source / "scripts/run_scifact_component_operator.py"))
    monkeypatch.setattr(operator, "ROOT", root)
    monkeypatch.setattr(operator, "os", types.SimpleNamespace(name="posix", environ=operator.os.environ))
    monkeypatch.setattr(operator.signal, "signal", lambda *args: None)
    for key, value in {"SLURM_JOB_ID": "synthetic", "CUDA_VISIBLE_DEVICES": "0", "CLIMATE_COMPONENT_RELEASE": operator.RELEASE,
                       "CLIMATE_COMPONENT_WORK": str(work), "CLIMATE_SOURCE_GIT": "a" * 40, "CLIMATE_SOURCE_SHA256": "b" * 64}.items():
        monkeypatch.setenv(key, value)
    actual_digest = operator.digest
    def digest(path):
        if path.parent == root / "envs":
            return next((expected for filename, expected in operator.ARCHIVES.values() if path.name == filename), operator.GRAMMAR_SHA)
        return actual_digest(path)
    monkeypatch.setattr(operator, "digest", digest)
    events = []
    def checked(path, expected):
        if path.name == "targets.json":
            assert events == ["inference_exited"]
            events.append("targets_read")
        return b"[]"
    monkeypatch.setattr(operator, "checked", checked)
    def extract(archive, target, *args, **kwargs):
        target.mkdir()
        return 0
    monkeypatch.setattr(operator, "safe_extract", extract)
    monkeypatch.setattr(operator, "generator_only", extract)
    monkeypatch.setattr(operator, "v3_runtime_environment", lambda *args: {})
    def execute(command, **kwargs):
        result = root / "runs" / operator.RELEASE
        if command[1].endswith("run_scifact_components.py"):
            assert not events and not (work / "component-scoring").exists()
            durable(result / "worker-identity.json", {"synthetic": True})
            events.append("inference_exited")
            return {"child_reaped": True, "returncode": 0, "worker_failure": None}
        elif command[1].endswith("score_scifact_components.py"):
            assert events == ["inference_exited", "targets_read"]
            durable(result / "compact.json", {"status": "complete"})
    monkeypatch.setattr(operator, "execute", execute)
    monkeypatch.setattr(operator, "inference_exit", execute)
    operator.main()
    assert events == ["inference_exited", "targets_read"]
    with pytest.raises(FileExistsError):
        operator.main()


def test_generator_staging_excludes_reranker(tmp_path):
    import io
    import tarfile
    from run_scifact_component_operator import generator_only
    archive = tmp_path / "models.tar"
    with tarfile.open(archive, "w") as output:
        for name in ("models/generator/model/config.json", "models/generator/model_manifest.json", "models/reranker/model/config.json"):
            member = tarfile.TarInfo(name)
            member.size = 2
            output.addfile(member, io.BytesIO(b"{}"))
    destination = tmp_path / "out"
    assert generator_only(archive, destination, sha(archive.read_bytes())) == 4
    assert not (destination / "models/reranker").exists()


def test_repeated_signals_masked_until_child_is_reaped(tmp_path, monkeypatch):
    import run_scifact_component_operator as operator
    events = []
    class Child:
        pid, calls, returncode = 123, 0, None
        def wait(self, **kwargs):
            self.calls += 1
            if self.calls == 1:
                raise KeyboardInterrupt()
            assert "mask" in events and "kill" in events
            events.append("reaped")
            self.returncode = -15
            return self.returncode
        def poll(self):
            return self.returncode
    monkeypatch.setattr(operator.subprocess, "Popen", lambda *args, **kwargs: Child())
    monkeypatch.setattr(operator, "os", types.SimpleNamespace(killpg=lambda *args: events.append("kill")))
    def signal(sig, handler):
        events.append("mask" if handler == operator.signal.SIG_IGN else "restore")
        return operator.signal.SIG_DFL
    monkeypatch.setattr(operator.signal, "signal", signal)
    proof = operator.inference_exit(["synthetic"], cwd=tmp_path, env={}, log=tmp_path / "out.log")
    assert proof["child_reaped"] is True and proof["returncode"] == -15
    assert events.index("reaped") < events.index("restore")


@pytest.mark.parametrize("partial", [False, True])
def test_interrupted_response_recovers_cost_without_quality(tmp_path, partial):
    result = execute_slot(slot(), Fake(), tmp_path, "diagnostic")
    completed = tmp_path / "diagnostic-00/completed.json"
    if partial:
        completed.write_bytes(b'{"status":')
    else:
        completed.unlink()
    record = reconcile(slot(), tmp_path, "diagnostic")
    assert record["status"] == "provider_failed" and record["usage"] == result["usage"]
    assert "prediction" not in record


def test_partial_response_write_preserves_paid_exception_receipt(tmp_path, monkeypatch):
    import climate_rag.component_execution as execution
    original = durable
    def fail(path, value):
        if path.name == "response.json":
            path.write_bytes(b'{"raw":')
            raise OSError("synthetic")
        original(path, value)
    monkeypatch.setattr(execution, "durable", fail)
    result = execute_slot(slot(), Fake(), tmp_path, "diagnostic")
    recovered = reconcile(slot(), tmp_path, "diagnostic")
    assert recovered["status"] == "provider_failed" and recovered["usage"] == result["usage"]
