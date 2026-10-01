"""Synthetic transport/trajectory tests; no real data/model inference."""
import json
import os
import stat
import types

import pytest

from climate_rag.private_diagnostics_v3 import PrivateDiagnosticStore
from climate_rag.agent_protocol import ModelResponseValidationError
from climate_rag.scifact_grounding import Abstract
from climate_rag.scifact_natural_contract import base_state, valid_physical_prefix
from climate_rag.scifact_natural_selection import PROTOCOL
from climate_rag.scifact_terminal import source_from_abstract
from climate_rag.scifact_utility_runtime import JournalProvider, ledger_cost, run_matrix
from test_bounded_scifact_runtime import ABSTAIN, SyntheticBackend


class Model:
    training = False

    def named_parameters(self):
        return [("weight", object())]

    def named_modules(self):
        return [("", self)]


class Backend(SyntheticBackend):
    def __init__(self, actions, faults=None):
        super().__init__(actions, False)
        self.base.model = Model()
        self.faults = faults or {}
        self.calls = 0

    def start_slot(self, path):
        path.mkdir(mode=0o700)
        self.store = PrivateDiagnosticStore(path, max_files=10, max_bytes=100000)

    def generate(self, observation, schema, max_output_tokens, remaining_seconds):
        response = super().generate(observation, schema, max_output_tokens, remaining_seconds)
        raw = response["raw"]
        sink = self.store.sink("response", 10000)
        sink.write(raw)
        empty = self.store.sink("grammar", 1000)
        d = response["diagnostics"]
        d.update(private_attachment=sink.receipt(), grammar_log=empty.receipt(), grammar_log_nonempty=False,
                 output_sha256=sink.receipt()["sha256"], eos_observed=True,
                 output_tokens=response["usage"]["output_tokens"], generation_elapsed_ms=1)
        if self.faults.get(self.calls) == "no_eos":
            d["eos_observed"] = False
        self.calls += 1
        return response


def test_synthetic_private_store_matches_production_permissions(tmp_path):
    path = tmp_path / "private-responses"
    backend = Backend([])
    backend.start_slot(path)
    assert backend.store.root == path.resolve()
    if os.name == "posix":
        assert stat.S_IMODE(path.stat().st_mode) & 0o077 == 0


def matrix(tmp_path, actions, faults=None):
    corpus = {i: Abstract(i, "Synthetic", ("First fixture sentence.", "Second."), False) for i in range(100, 120)}
    sources = [source_from_abstract(a) for a in corpus.values()]
    backend = Backend(actions, faults)
    out = tmp_path / "inference"
    report = run_matrix([{"id": 1, "claim": "Fixture claim"}], backend,
                        lambda q, k: sources[:k], lambda q, c: list(reversed(c)), corpus, out, natural_fit=True)
    return report, backend, out


def test_full_natural_multistep_and_scripted_single_call_failures(tmp_path):
    report, backend, out = matrix(tmp_path, [{"action": "read", "source_ids": ["c8"]}, ABSTAIN,
                                          {"action": "rerank"}, {"unexpected": True}])
    a, b, c = report["runs"]
    assert a["result"]["model_calls"] == 2 and a["result"]["events"][1]["model_selected"]
    assert a["audit_status"] == "valid_terminal" and a["direct_attempt0"] is None
    assert a["origin"] == "natural_model_trajectory"
    assert all(r["audit_status"] == "unresolved" and r["prediction"] is None for r in (b, c))
    assert all(r["result"]["new_model_calls"] == 1 for r in (b, c))
    assert all(r["result"]["validation_repairs"] == 0 for r in (b, c))
    assert b["result"]["outcome"] == "proposed_not_executed"
    assert c["result"]["outcome"] == "continuation_invalid"
    assert b["result"]["events"][1]["requested_context"] == ["c5"]  # not A's c8
    assert b["result"]["events"][1]["origin"] == "scripted_intervention"
    assert not b["result"]["events"][1]["model_selected"]
    assert backend.observations[2]["feedback"].startswith("tool_completed:")
    assert backend.observations[2]["current_citable"][0]["sentence_id"].startswith("c5:")
    assert report["physical_generation_cost"]["unique_physical_calls"] == 4
    assert sum(r["logical_generation_cost"]["unique_physical_calls"] for r in report["runs"]) == 6
    assert json.loads((out / "ledger/g00.reserved.json").read_bytes())["actual_base_state"]["adapter_loaded"] is False


@pytest.mark.parametrize("first,faults", [(ABSTAIN, {0: "no_eos"}), ({"bad": True}, {})])
def test_invalid_initial_physical_prefix_never_replaced(tmp_path, first, faults):
    report, backend, _ = matrix(tmp_path, [first, ABSTAIN], faults)
    a, b, c = report["runs"]
    assert b["status"] == c["status"] == "not_run"
    assert b["prediction"] is c["prediction"] is None
    assert b["physical_generation_ids"] == c["physical_generation_ids"] == []
    assert a["direct_attempt0"] is None
    assert backend.calls == (1 if faults else 2)
    assert len(report["runs"]) == report["planned_slots"] == 3


def test_no_eos_terminal_is_unresolved_not_nei_success(tmp_path):
    report, _, _ = matrix(tmp_path, [ABSTAIN] * 3, {1: "no_eos"})
    a, b, c = report["runs"]
    assert a["audit_status"] == c["audit_status"] == "valid_terminal"
    assert b["audit_status"] == "unresolved" and b["prediction"] is None
    assert b["unaudited_prediction"]["evidence"] == {}
    assert report["physical_generation_cost"]["unique_physical_calls"] == 3


@pytest.mark.parametrize("fault", ["raw", "prompt", "usage", "id"])
def test_prefix_physical_evidence_tampering_fails(tmp_path, fault):
    _, backend, out = matrix(tmp_path, [ABSTAIN] * 3)
    prefix = json.loads((out / "1-A/prefix.json").read_bytes())
    if fault == "raw":
        next((out / "1-A/private-responses").glob("*-response.txt")).write_text("tampered")
    elif fault == "prompt":
        path = out / "ledger/g00.reserved.json"
        row = json.loads(path.read_bytes())
        row["prompt_sha256"] = "0" * 64
        path.write_text(json.dumps(row))
    else:
        path = out / "ledger/g00.finished.json"
        row = json.loads(path.read_bytes())
        if fault == "usage":
            row["response"]["usage"]["input_tokens"] = True
        else:
            row["physical_attempt_id"] = "g01"
        path.write_text(json.dumps(row))
    with pytest.raises(ValueError):
        valid_physical_prefix(prefix, out / "ledger", out / "1-A/private-responses", backend.base.tokenizer, "1-A")


def test_new_capacity_does_not_change_legacy_default(tmp_path):
    backend = Backend([ABSTAIN])
    journal = JournalProvider(backend, tmp_path / "legacy")
    assert journal.max_generations == 56
    other = JournalProvider(backend, tmp_path / "natural", max_generations=168, protocol=PROTOCOL)
    assert other.max_generations == 168 and other.protocol == PROTOCOL
    assert ledger_cost(other.directory)["unique_physical_calls"] == 0
    with pytest.raises(ValueError, match="capacity"):
        JournalProvider(backend, tmp_path / "bad", max_generations=169)


def test_actual_peft_or_lora_model_rejected_not_merely_gap_false():
    backend = Backend([])
    backend.base.model.peft_config = {}
    with pytest.raises(ValueError, match="unwrapped"):
        base_state(backend)
    backend.base.model = types.SimpleNamespace(training=False,
        named_parameters=lambda: [("layer.lora_A.weight", object())], named_modules=lambda: [])
    with pytest.raises(ValueError, match="unwrapped"):
        base_state(backend)


def test_unknown_earlier_usage_stays_unresolved_after_natural_repair(tmp_path, monkeypatch):
    class UnknownBackend(Backend):
        def generate(self, *args):
            if self.calls == 0:
                self.calls += 1
                raise ModelResponseValidationError({}, {"output_usage_unknown": True})
            return super().generate(*args)
    monkeypatch.setitem(matrix.__globals__, "Backend", UnknownBackend)
    report, _, _ = matrix(tmp_path, [ABSTAIN])
    a = report["runs"][0]
    assert a["result"]["model_calls"] == 2
    assert a["result"]["outcome"].startswith("model_abstention:")
    assert a["audit_status"] == "unresolved" and a["prediction"] is None
    assert a["incremental_generation_cost"]["unknown_usage_attempts"] == 1
    assert a["incremental_generation_cost"]["total_tokens"] is None
