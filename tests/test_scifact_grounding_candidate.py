from __future__ import annotations

import copy
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from climate_rag.scifact_grounding import Abstract, GoldClaim, Rationale
from climate_rag.scifact_grounding_eval import audit_arm, evaluate_arm, score_arm
from climate_rag.scifact_grounding_ledger import cumulative_ledger
from climate_rag.scifact_grounding_sft import (
    advancement, canonical_context, context, fit_records, prediction,
    select_components, token_record, training_steps,
)


class Tokenizer:
    eos_token = "\x00"
    eos_token_id = 0

    def encode(self, text: str, **kwargs: object) -> list[int]:
        return [ord(c) for c in text]

    def apply_chat_template(self, messages: list[dict[str, str]], **kwargs: object) -> str:
        return "\n".join(m["role"] + ":" + m["content"] for m in messages) + "\nassistant:"


@pytest.fixture
def corpus() -> dict[int, Abstract]:
    return {8: Abstract(8, "First", ("零", "One", "Two", "Three"), False),
            12: Abstract(12, "Second", ("No", "Maybe", "Yes"), False)}


def test_production_joint_wire_and_alias_order(corpus: dict[int, Abstract]) -> None:
    ctx = context("Claim", [corpus[12], corpus[8]])
    action = {"action": "answer", "documents": [
        {"source_id": "c1", "label": "REFUTES", "sentence_ids": ["c1:2"]},
        {"source_id": "c2", "label": "SUPPORTS", "sentence_ids": ["c2:3", "c2:0"]}]}
    result = prediction(1, action, ctx, corpus)["prediction"]
    assert result == {"id": 1, "evidence": {
        "12": {"label": "CONTRADICT", "sentences": [2]},
        "8": {"label": "SUPPORT", "sentences": [3, 0]}}}
    assert prediction(1, {"action": "abstain", "reason": "insufficient_evidence"}, ctx, corpus)["prediction"]["evidence"] == {}
    assert canonical_context(json.loads(json.dumps(ctx, sort_keys=True)), corpus) == ctx
    with pytest.raises(ValueError, match="hidden_fields"):
        canonical_context(ctx | {"oracle_relation": "SUPPORT"}, corpus)


def test_official_alternatives_not_union_or_negative(corpus: dict[int, Abstract]) -> None:
    gold = GoldClaim(1, "Claim", {8: (Rationale("SUPPORT", (0, 2)), Rationale("SUPPORT", (3,)))}, (8,))
    rows = fit_records(gold, corpus, "group")
    assert [r["target"]["documents"][0]["sentence_ids"] for r in rows] == [["c1:0", "c1:2"], ["c1:3"]]
    assert {r["provenance"] for r in rows} == {"official_annotated_document_alternative"}
    assert len(rows[0]["context"]["observation"]["current_citable"]) == 4
    assert "SUPPORT" not in rows[0]["context"]["observation"]["immutable_claim"]
    assert fit_records(GoldClaim(2, "NEI", {}, (8,)), corpus, "other") == []


@pytest.mark.parametrize("sentence_ids", [["c2:0"], ["c1:0", "c1:0"], ["c1:55"]])
def test_production_invalid_citations_rejected(corpus: dict[int, Abstract], sentence_ids: list[str]) -> None:
    ctx = context("Claim", list(corpus.values()))
    with pytest.raises(ValueError):
        prediction(1, {"action": "answer", "documents": [{"source_id": "c1", "label": "SUPPORTS", "sentence_ids": sentence_ids}]}, ctx, corpus)


def test_assistant_only_mask_unicode_roles_and_real_eos(corpus: dict[int, Abstract]) -> None:
    ctx = context('assistant: {"action":"answer"} \u96f6', [corpus[8]])
    target = {"action": "answer", "documents": [{"source_id": "c1", "label": "SUPPORTS", "sentence_ids": ["c1:0"]}]}
    row = token_record(Tokenizer(), ctx, target)
    n = row["input_tokens"]
    assert row["labels"][:n] == [-100] * n
    assert row["labels"][n:] == row["input_ids"][n:]
    assert row["labels"][n] == ord("{") and row["labels"][-1] == 0
    # Batch size is frozen at one: no padding, no EOS==PAD mask ambiguity.
    assert row["attention_mask"] == [1] * len(row["input_ids"])
    saved_target = json.loads(json.dumps(target, sort_keys=True))
    assert token_record(Tokenizer(), ctx, saved_target) == row


def test_full_context_budget_never_crops(corpus: dict[int, Abstract]) -> None:
    long = Abstract(99, "Long", ("x" * 9000,), False)
    with pytest.raises(ValueError, match="full_context_over_budget"):
        token_record(Tokenizer(), context("Claim", [long]))
    assert long.sentences == ("x" * 9000,)


def test_steps_one_epoch_and_ceiling() -> None:
    assert training_steps(1) == 1 and training_steps(192) == 48
    for count in (0, 193, True):
        with pytest.raises(ValueError):
            training_steps(count)


def test_split_whole_component_deterministic_and_unconsumed() -> None:
    ids = list(range(450))
    assignment = {"eligible_train_ids": ids, "claim_component": {str(i): str(i // 3) for i in ids}}
    labels = {i: ("SUPPORTS", "REFUTES", "NOT_ENOUGH_INFO")[(i // 3) % 3] for i in ids}
    excluded = ["0", "1", "2"]
    split = select_components(assignment, labels, excluded)
    assert {k: len(v) for k, v in split.items()} == {"fit": 48, "tune": 12, "validation": 12}
    assert len({i // 3 for v in split.values() for i in v}) == 72
    assert all(str(i // 3) not in excluded for p in ("tune", "validation") for i in split[p])
    assignment["eligible_train_ids"].reverse()
    assert select_components(assignment, labels, excluded) == split
    with pytest.raises(ValueError, match="insufficient"):
        select_components(assignment, labels, [str(i) for i in range(150)])


def test_cumulative_unknown_sibling_and_carried_dedup() -> None:
    assignment = {"eligible_train_ids": [1, 2, 3, 4], "claim_component": {"1": "a", "2": "a", "3": "b", "4": "c"}}
    historical = {"model_consumed_ids": [], "uncertain_ids": []}
    event = {"physical_id": "old/synthetic", "claim_id": None, "receipt_sha256": "a" * 64, "status": "completed", "carried": False}
    run = {"run_id": "old", "protocol_sha256": "b" * 64, "receipt_sha256": "c" * 64,
           "planned_claim_ids": [1], "state": "failed", "attempts": [event], "proven_unattempted_claim_ids": []}
    second = {**run, "run_id": "new", "planned_claim_ids": [3], "state": "completed",
              "attempts": [{**event, "carried": True}, {**event, "physical_id": "new/real", "claim_id": 3, "status": "unknown"}]}
    ledger = cumulative_ledger(assignment, "d" * 64, historical, "e" * 64, [run, second])
    assert ledger["model_consumed_ids"] == [3]
    assert ledger["uncertain_ids"] == [1]
    assert ledger["component_excluded_eligible_ids"] == [1, 2, 3]
    assert ledger["new_physical_calls"] == 2 and ledger["carried_references_deduplicated"] == 1
    broken = copy.deepcopy(second)
    broken["attempts"][0]["receipt_sha256"] = "f" * 64
    with pytest.raises(ValueError, match="carry_identity"):
        cumulative_ledger(assignment, "d" * 64, historical, "e" * 64, [run, broken])


class Provider:
    def __init__(self, fail: bool = False):
        self.base = SimpleNamespace(tokenizer=Tokenizer())
        self.calls = 0
        self.fail = fail

    def generate(self, observation: object, schema: object, tokens: int, seconds: int) -> dict[str, object]:
        self.calls += 1
        if self.fail:
            raise RuntimeError("paid_unknown")
        from climate_rag.scifact_terminal import render_scifact_prompt
        n = len(self.base.tokenizer.encode(render_scifact_prompt(self.base.tokenizer, observation, schema)))
        return {"raw": '{"action":"abstain","reason":"insufficient_evidence"}',
                "usage": {"input_tokens": n, "output_tokens": 20},
                "diagnostics": {"eos_observed": True, "reached_max_new_tokens": False}}


def test_paired_inputs_call_cap_and_failure_retention(tmp_path: Path, corpus: dict[int, Abstract]) -> None:
    ctx = context("Claim", [corpus[8]])
    rows = [{"claim_id": i, "context": ctx, "packing": token_record(Tokenizer(), ctx)} for i in range(12)]
    provider = Provider()
    base = evaluate_arm(rows, corpus, provider, tmp_path / "base", "base")
    adapted = evaluate_arm(rows, corpus, provider, tmp_path / "adapted", "adapted")
    assert provider.calls == 24 and base["input_identity"] == adapted["input_identity"]
    assert audit_arm(tmp_path / "base", rows, corpus) == base
    bad = Provider(fail=True)
    result = evaluate_arm(rows, corpus, bad, tmp_path / "bad", "base")
    assert bad.calls == 1 and result["records"][0]["usage_known"] is False
    assert (tmp_path / "bad/slot-00/started.json").is_file()
    with pytest.raises(FileExistsError):
        evaluate_arm(rows, corpus, bad, tmp_path / "bad", "base")
    assert bad.calls == 1


@pytest.mark.parametrize("field", ["prediction", "usage", "attempts", "input_identity"])
def test_audit_rejects_changed_summary(tmp_path: Path, corpus: dict[int, Abstract], field: str) -> None:
    ctx = context("Claim", [corpus[8]])
    rows = [{"claim_id": i, "context": ctx, "packing": token_record(Tokenizer(), ctx)} for i in range(12)]
    result = evaluate_arm(rows, corpus, Provider(), tmp_path / "base", "base")
    if field in {"attempts", "input_identity"}:
        result[field] = 11 if field == "attempts" else "changed"
    else:
        result["records"][0][field] = {}
    path = tmp_path / "base/complete.json"
    path.write_text(json.dumps(result), encoding="utf-8")
    with pytest.raises(ValueError):
        audit_arm(tmp_path / "base", rows, corpus)


def test_partial_entrypoint_score_retains_whole_planned_denominator(tmp_path: Path, corpus: dict[int, Abstract], monkeypatch: pytest.MonkeyPatch) -> None:
    import run_scifact_grounding_candidate as entry
    from climate_rag.scifact_semantic_contract import encoded, sha, write_once
    bundle, output = tmp_path / "bundle", tmp_path / "output"
    (bundle / "inference").mkdir(parents=True)
    (bundle / "scoring").mkdir()
    output.mkdir()
    ctx = context("Claim", [corpus[8]])
    rows = [{"claim_id": i, "context": ctx, "packing": token_record(Tokenizer(), ctx)} for i in range(12)]
    gold = [GoldClaim(i, "Claim", {}, ()).gold_row() for i in range(12)]
    write_once(bundle / "inference/tune.json", rows)
    write_once(bundle / "scoring/tune.json", gold)
    manifest = {"files": {"inference/tune.json": sha(encoded(rows)), "scoring/tune.json": sha(encoded(gold))}}
    monkeypatch.setattr(entry, "load_bundle", lambda *args: (manifest, corpus))
    evaluate_arm(rows, corpus, Provider(fail=True), output / "base", "base")
    entry.terminate_inference(output, "tune", "data", "adapter")
    execution = output.with_name("output-execution")
    execution.mkdir()
    write_once(execution / "worker-exit.json", {"child_reaped": True, "returncode": 0,
        "terminated_sha256": sha((output / "inference-terminated.json").read_bytes())})
    entry.score(SimpleNamespace(bundle=bundle, output=output, data_sha="data", partition="tune"))
    results = json.loads((output / "score.json").read_bytes())
    assert results["base"]["attempts"] == 1 and results["base"]["not_attempted"] == 11
    assert results["adapted"]["attempts"] == 0 and results["adapted"]["not_attempted"] == 12
    assert results["base"]["planned_unsuccessful"] == results["adapted"]["planned_unsuccessful"] == 12
    assert json.loads((output / "gate.json").read_bytes())["passed"] is False


@pytest.mark.parametrize("usage", [None, "invalid", {"input_tokens": "invalid", "output_tokens": 1}, {"input_tokens": -1}])
def test_bad_usage_preserved_as_unknown_lower_bound(corpus: dict[int, Abstract], usage: object) -> None:
    gold = [GoldClaim(i, "NEI", {}, ()) for i in range(12)]
    result = {"input_identity": "same", "attempts": 1, "records": [{
        "claim_id": 0, "status": "failed", "usage_known": False,
        "usage": usage, "elapsed_ms": 2, "prediction": None}]}
    score = score_arm(result, gold, corpus)
    assert score["planned_unsuccessful"] == 12 and score["unknown_usage"] == 1
    assert score["valid_nei_abstention"] == 0
    assert score["cost_including_failures"]["totals_are_lower_bounds"] is True


@pytest.mark.parametrize("unknown", [False, True])
def test_exception_usage_status_is_truthful(tmp_path: Path, corpus: dict[int, Abstract], unknown: bool) -> None:
    from climate_rag.agent_protocol import ModelResponseValidationError
    class ErrorProvider(Provider):
        def generate(self, *args: object) -> dict[str, object]:
            self.calls += 1
            raise ModelResponseValidationError({"input_tokens": 100, "output_tokens": 17},
                                              {"output_usage_unknown": unknown})
    ctx = context("Claim", [corpus[8]])
    rows = [{"claim_id": i, "context": ctx, "packing": token_record(Tokenizer(), ctx)} for i in range(12)]
    provider = ErrorProvider()
    result = evaluate_arm(rows, corpus, provider, tmp_path / "errors", "base")
    assert provider.calls == 1
    first = result["records"][0]
    assert first["status"] == "failed" and first["usage_known"] is not unknown
    assert first["usage"] == {"input_tokens": 100, "output_tokens": 17}


def test_paid_response_persistence_failure_retains_cost(tmp_path: Path, corpus: dict[int, Abstract], monkeypatch: pytest.MonkeyPatch) -> None:
    import climate_rag.scifact_grounding_eval as module
    original = module.write_once
    def broken(path: Path, value: object) -> None:
        if path.name == "response.json":
            path.write_bytes(b'{"raw":')
            raise OSError("disk_fixture")
        original(path, value)
    monkeypatch.setattr(module, "write_once", broken)
    ctx = context("Claim", [corpus[8]])
    rows = [{"claim_id": i, "context": ctx, "packing": token_record(Tokenizer(), ctx)} for i in range(12)]
    result = evaluate_arm(rows, corpus, Provider(), tmp_path / "disk", "base")
    first = result["records"][0]
    assert first["status"] == "failed" and first["usage_known"] is True
    assert first["usage"]["output_tokens"] == 20
    assert result["attempts"] == 1
    audited = audit_arm(tmp_path / "disk", rows, corpus)
    assert audited["physical_files_sha256"]["slot-00/response.json"]
    score = score_arm(audited, [GoldClaim(i, "Claim", {}, ()) for i in range(12)], corpus)
    assert score["attempts"] == 1 and score["not_attempted"] == 11
    assert score["planned_unsuccessful"] == 12 and score["valid_nei_abstention"] == 0
    assert score["cost_including_failures"]["output_tokens_known_lower_bound"] == 20
    assert score["cost_including_failures"]["totals_are_lower_bounds"] is False


def test_count_gate(corpus: dict[int, Abstract]) -> None:
    base = {"input_identity": "same", "claims": 12, "attempts": 12,
            "correctly_rationalized_documents": 1, "nei_false_evidence": 0,
            "planned_unsuccessful": 0, "unknown_usage": 0, "stop_required": 0}
    assert advancement(base, base | {"correctly_rationalized_documents": 2})
    assert not advancement(base, base | {"correctly_rationalized_documents": 2, "nei_false_evidence": 1})
    assert not advancement(base, base)
    with pytest.raises(ValueError, match="paired_input"):
        advancement(base, base | {"input_identity": "changed"})


def test_actual_tiny_causal_adapter_restoration_no_download(tmp_path: Path) -> None:
    import torch
    from peft import LoraConfig, get_peft_model
    from transformers.models.qwen3.configuration_qwen3 import Qwen3Config
    from transformers.models.qwen3.modeling_qwen3 import Qwen3ForCausalLM
    from run_scifact_grounding_candidate import restore_causal_adapter
    config = Qwen3Config(vocab_size=32, hidden_size=16, intermediate_size=24,
                        num_hidden_layers=1, num_attention_heads=2, num_key_value_heads=2, head_dim=8)
    model = get_peft_model(Qwen3ForCausalLM(config), LoraConfig(
        r=8, lora_alpha=16, lora_dropout=0.0, target_modules=["q_proj", "v_proj"], task_type="CAUSAL_LM"))
    with torch.no_grad():
        for name, param in model.named_parameters():
            if "lora_" in name:
                param.fill_(0.125)
    model.save_pretrained(tmp_path / "adapter", safe_serialization=True)
    restored, audit = restore_causal_adapter(Qwen3ForCausalLM(config), tmp_path / "adapter")
    assert audit["all_checkpoint_values_equal"] is True and audit["tensor_count"] == 4
    with restored.disable_adapter():
        assert all(not p.requires_grad for p in restored.parameters())
