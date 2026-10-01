"""Full synthetic runner->disk->physical audit->original score->redacted export."""
import copy
import hashlib
import json

import jsonschema
import pytest

from climate_rag.scifact_evidence_bottleneck import (
    BottleneckProvider, PROTOCOL, ROUTES, inputs as call_inputs,
    parse_selection, parse_stage, render_prompt,
)
from climate_rag.scifact_generation import frozen_contract
from climate_rag.scifact_grounding import GoldClaim, Rationale
from climate_rag.scifact_read_continuation import ordered_write
from climate_rag.scifact_semantic_contract import MODEL_SHA
from climate_rag.scifact_utility_contract import identity
from climate_rag.scifact_utility_runtime import ledger_cost
from run_scifact_evidence_bottleneck import run_suite
from score_scifact_evidence_bottleneck import score_after_exit, export_compact
from test_scifact_document_verifier import Backend as OldBackend, inputs
from test_scifact_relation_verifier import assessment


class Backend(OldBackend):
    def render(self, obs, schema):
        return render_prompt(self.base.tokenizer, obs, schema)


SELECT = {"source_id": "c7", "sentence_ids": ["c7:7", "c7:2"]}
LABEL = {"source_id": "c7", "label": "SUPPORTS"}


def execute(tmp_path, *, actions=None, n=1, backend=None, contract=None, faults=None):
    frame, corpus = inputs()
    claims = [{"id": i, "claim": frame["observation"]["immutable_claim"]} for i in range(1, n+1)]
    frames = [copy.deepcopy(frame) for _ in claims]
    release = {"protocol": PROTOCOL, "scope": "synthetic_fixture", "max_generations": 96,
        "routes": list(ROUTES), "source_git": "a"*40, "frames_identity": identity(frames), "generation_contract": contract,
        "ordered_ids_sha256": identity([c["id"] for c in claims]), "model_sha256": MODEL_SHA,
        "planned_route_results": 3*n}
    path = tmp_path / "authorization.json"
    ordered_write(path, release)
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    backend = backend or Backend(actions if actions is not None else [assessment(), SELECT, LABEL, LABEL]*n, faults)
    out = tmp_path / "inference"
    rows = run_suite(claims, frames, corpus, backend, out, json.loads(path.read_bytes()), digest)
    ordered_write(out / "worker-exit.json", {"child_reaped": True, "returncode": 0, "interrupted": None,
                                           "release_sha256": digest})
    return out, rows, backend, frames, corpus, path, digest


def score(tmp_path, bundle, forbidden=False):
    out, _, backend, frames, corpus, release, digest = bundle
    def gold(ids):
        if forbidden:
            pytest.fail("must not read gold before complete audit")
        return [GoldClaim(i, "Synthetic claim", {77: (Rationale("SUPPORT", (2, 7)),)}, (77,)) for i in ids]
    return score_after_exit(out, release, digest, frames, backend.base.tokenizer, corpus, gold,
                            reports=tmp_path / "scored")


def test_24_all_routes_physical_vs_independent_cost_and_export(tmp_path):
    bundle = execute(tmp_path, n=24)
    out, rows, backend, *_ = bundle
    assert backend.calls == 96 and sum(len(r["routes"]) for r in rows) == 72
    report = score(tmp_path, bundle)
    assert report["status"] == "scored", report
    assert report["mechanism"] == "not_supported"
    accounting = report["accounting"]
    assert accounting["physical"]["unique_physical_calls"] == 96
    assert [accounting["independent_deployment"][r]["unique_physical_calls"] for r in ROUTES] == [24, 48, 48]
    assert all(r["routes"]["B"]["selected_sentence_ids"] == ["c7:2", "c7:7"] for r in rows)
    assert report["routes"]["B"]["official_micro"]["metrics"]["abstract_rationalized"]["f1"] == 1.0
    compact = export_compact(tmp_path / "scored", tmp_path / "compact.json")
    text = json.dumps(compact)
    assert not any(s in text for s in ("Complete immutable", "Original sentence", "c7:", '"claim_id"', '"cases"'))
    assert ledger_cost(out / "ledger")["unknown_usage_attempts"] == 0
    assert compact["scope"] == "synthetic_fixture" and not compact["Agent_gain_established"]


def test_real_provider_numeric_release_disk_audit(tmp_path, monkeypatch):
    from test_scifact_paired_comparison import bound_provider
    contract = copy.deepcopy(frozen_contract())
    for key, value in contract["expanded_model_defaults"].items():
        if type(value) is float and value.is_integer():
            contract["expanded_model_defaults"][key] = int(value)
    provider, calls = bound_provider(tmp_path, monkeypatch, [assessment(), SELECT, LABEL, LABEL], contract=contract)
    provider.__class__ = BottleneckProvider
    monkeypatch.setattr("lmformatenforcer.integrations.transformers.build_token_enforcer_tokenizer_data",
                        lambda tokenizer, use_bitmask: provider.tokenizer_data)
    bundle = execute(tmp_path, backend=provider, contract=contract)
    report = score(tmp_path, bundle)
    assert report["status"] == "scored", report
    assert len(calls) == 4
    out = bundle[0]
    configs = [json.loads((out / "ledger" / f"g{i:02d}.generation.json").read_bytes()) for i in range(4)]
    assert all(c["contract"] == contract and c["contract_sha256"] == identity(contract) for c in configs)
    assert configs[2]["effective"] == configs[3]["effective"] or (
        configs[2]["effective"]["max_length"] != configs[3]["effective"]["max_length"])
    assert all(not c["effective_parser_config"]["force_json_field_order"] for c in configs)


def test_B_C_only_difference_full_text_and_schema_identical():
    frame, _ = inputs()
    selected = parse_selection({"source_id": "c7", "sentence_ids": ["c7:7"]}, frame, "c7")
    b, bs = call_inputs(frame, "B", selected)
    c, cs = call_inputs(frame, "C", selected)
    assert bs == cs and {k for k in b if b[k] != c[k]} == {"full_document"}
    assert c["full_document"] == [] and c["selected_sentences"] == [{"sentence_id": "c7:7", "text": "Original sentence 7."}]
    assert "Original sentence 2." not in render_prompt(Backend([]).base.tokenizer, c, cs)
    assert "Original sentence 2." in render_prompt(Backend([]).base.tokenizer, b, bs)
    assert not any(k in c for k in ("remaining", "feedback", "current_citable", "arm", "route"))
    with pytest.raises(ValueError, match="locked_label_schema"):
        parse_stage(dict(LABEL, sentence_ids=["c7:2"]), frame, "C", selected)


@pytest.mark.parametrize("selection,reason", [
    ({"source_id": "c7", "sentence_ids": []}, "selection_empty"),
    ({"source_id": "c7", "sentence_ids": ["c8:2"]}, "selection_invalid"),
    ({"source_id": "c7", "sentence_ids": ["c7:2", "c7:2"]}, "selection_invalid"),
    ({"bad": True}, "selection_format"),
])
def test_selector_failure_both_routes_no_calls_no_NEI(tmp_path, selection, reason):
    bundle = execute(tmp_path, actions=[assessment(), selection])
    out, rows, backend, *_ = bundle
    assert backend.calls == 2
    assert reason in rows[0]["steps"]["selector"]["failure"]
    for r in ("B", "C"):
        assert rows[0]["routes"][r]["prediction"] is None and rows[0]["routes"][r]["label"] is None
        assert not (out / "1" / r).exists()
    assert score(tmp_path, bundle)["status"] == "scored"


@pytest.mark.parametrize("fault", ["no_eos", "bad_label", "unknown", "missing_finished", "missing_case", "worker_failed"])
def test_failure_cost_retained_and_no_illegal_gold_access(tmp_path, fault, monkeypatch):
    actions = [assessment(), SELECT, dict(LABEL, extra="change") if fault == "bad_label" else LABEL, LABEL]
    bundle = execute(tmp_path, actions=actions, faults={1: "no_eos"} if fault == "no_eos" else None)
    out, rows, backend, *_ = bundle
    forbidden = fault not in ("no_eos", "bad_label")
    if fault in ("unknown", "missing_finished"):
        p = out / "ledger/g00.finished.json"
        if fault == "missing_finished":
            p.unlink()  # synthetic fixture only
        else:
            data = json.loads(p.read_bytes())
            data["usage_known"] = False
            p.write_text(json.dumps(data), encoding="utf-8")
    if fault == "missing_case":
        (out / "1/result.json").unlink()
    if fault == "worker_failed":
        (out / "worker-exit.json").write_text(json.dumps({"child_reaped": True, "returncode": 2, "interrupted": None}))
    report = score(tmp_path, bundle, forbidden=forbidden)
    assert (report["status"] == "no_quality") == forbidden
    assert json.loads((tmp_path / "scored/cost-before-gold.json").read_bytes())["physical"]["unique_physical_calls"] == backend.calls
    if fault == "bad_label":
        assert rows[0]["routes"]["B"]["failure"] and rows[0]["routes"]["C"]["failure"] is None


@pytest.mark.parametrize("target", ["selection", "branch", "frame", "raw", "reservation", "decision"])
def test_tampering_fails_before_gold(tmp_path, target):
    bundle = execute(tmp_path)
    out, _, _, frames, *_ = bundle
    if target == "frame":
        frames[0]["observation"]["immutable_claim"] = "Changed claim"
    else:
        filename = {"selection": "1/shared-selection.json", "branch": "1/B/result.json", "raw": "ledger/g01.finished.json",
                    "reservation": "1/C/reserved.json", "decision": "1/result.json"}[target]
        path = out / filename
        data = json.loads(path.read_bytes())
        if target == "selection":
            data["selector_attempt_id"] = "g00"
        elif target == "branch":
            data["observation"]["full_document"] = []
        elif target == "raw":
            data["response"]["raw"] = json.dumps({"source_id": "c7", "sentence_ids": ["c7:2"]})
        elif target == "reservation":
            data["selection_sha256"] = "b"*64
        else:
            data["routes"]["C"]["selected_sentence_ids"] = ["c7:2"]
        path.write_text(json.dumps(data), encoding="utf-8")
    assert score(tmp_path, bundle, forbidden=True)["status"] == "no_quality"


def test_A_schema_same_language_not_forced_order():
    from climate_rag.scifact_relation_verifier import assessment_schema
    frame, _ = inputs()
    _, a = call_inputs(frame, "A")
    original = assessment_schema(frame, "c7")
    for old, new in zip(original["anyOf"], a["anyOf"], strict=True):
        assert old["properties"] == new["properties"]
        assert set(old["required"]) == set(new["required"])
    jsonschema.validate(assessment(), a)


def test_unknown_stage_and_never_retry(tmp_path):
    from climate_rag.scifact_evidence_bottleneck import BottleneckJournal
    frame, _ = inputs()
    with pytest.raises(ValueError):
        call_inputs(frame, "D")
    backend = Backend([SELECT, SELECT])
    backend.start_slot(tmp_path / "private")
    journal = BottleneckJournal(backend, tmp_path / "ledger", release_sha="a"*64)
    journal.slot = "1-selector"
    obs, schema = call_inputs(frame, "selector")
    journal.generate(obs, schema, 512, 120)
    with pytest.raises(ValueError, match="one_call"):
        journal.generate(obs, schema, 512, 120)
    assert ledger_cost(journal.directory)["unique_physical_calls"] == 1


def test_selector_overflow_retains_both_branch_failures(tmp_path, monkeypatch):
    class CountedTokenizer:
        def __init__(self, original):
            self.original = original
        def __getattr__(self, key):
            return getattr(self.original, key)
        def encode(self, text, **kwargs):
            if "Select only original sentence IDs" in text:
                return [1]*8193
            return self.original.encode(text, **kwargs)
    backend = Backend([assessment()])
    backend.base.tokenizer = CountedTokenizer(backend.base.tokenizer)
    bundle = execute(tmp_path, backend=backend)
    out, rows, *_ = bundle
    assert rows[0]["steps"]["selector"]["status"] == "not_called" and backend.calls == 1
    assert score(tmp_path, bundle)["status"] == "scored"
    assert not (out / "1/B").exists() and not (out / "1/C").exists()


def test_known_timeout_and_unknown_backend_call_keep_cost(tmp_path):
    from climate_rag.agent_protocol import ModelResponseValidationError
    class Failing(Backend):
        def generate(self, obs, schema, maximum, seconds):
            if obs["stage"] == "selector":
                raise ModelResponseValidationError(
                    usage={"input_tokens": self.count_prompt(obs, schema), "output_tokens": 0}, diagnostics={})
            return super().generate(obs, schema, maximum, seconds)
    bundle = execute(tmp_path, backend=Failing([assessment()]))
    assert ledger_cost(bundle[0]/"ledger")["unique_physical_calls"] == 2
    assert score(tmp_path, bundle)["status"] == "scored"


def test_no_sensitive_exception_text_in_public_export(tmp_path):
    from climate_rag.scifact_evidence_bottleneck import failure_code
    assert failure_code(ValueError("PRIVATE_CLAIM_SENTINEL")) == "ValueError"


def test_preflight_gold_free_prompt_stress():
    from preflight_scifact_evidence_bottleneck import probe
    frame, _ = inputs()
    result = probe(frame, Backend([]).base.tokenizer)
    assert not result["overflow"] and set(result["maximum_prompt_tokens"]) == {"A", "selector", "B", "C"}


@pytest.mark.parametrize("target", ["roster", "model", "exit", "failure", "usage"])
def test_review_regressions_binding_and_failure_denominators(tmp_path, target):
    bundle = execute(tmp_path, actions=[assessment(), SELECT, {"bad": True}, LABEL])
    out = bundle[0]
    def change(path, edit):
        data = json.loads(path.read_bytes())
        edit(data)
        path.write_text(json.dumps(data), encoding="utf-8")
    if target == "roster":
        change(out/"planned.json", lambda d: d["slots"].__setitem__(0, {"claim_id": 99, "route": "A"}))
    elif target == "model":
        change(out/"planned.json", lambda d: d.__setitem__("model_sha256", "b"*64))
    elif target == "exit":
        change(out/"worker-exit.json", lambda d: d.__setitem__("release_sha256", "b"*64))
    elif target == "failure":
        change(out/"1/B/result.json", lambda d: d.__setitem__("failure", None))
        change(out/"1/result.json", lambda d: d["steps"]["B"].__setitem__("failure", None))
    else:
        change(out/"ledger/g02.finished.json", lambda d: d["usage"].__setitem__("output_tokens", 0))
        change(out/"1/B/result.json", lambda d: d["usage"].__setitem__("output_tokens", 0))
        change(out/"1/result.json", lambda d: d["steps"]["B"]["usage"].__setitem__("output_tokens", 0))
    report = score(tmp_path, bundle, forbidden=True)
    assert report["status"] == "no_quality" and report["planned_route_results"] == 3
    assert report["unscored_route_results"] == 3 and report["physical"]["unique_physical_calls"] == 4
