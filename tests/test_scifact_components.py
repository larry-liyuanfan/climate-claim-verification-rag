from __future__ import annotations

import copy
import importlib.util
import io
import json
import os
import sys
import tarfile
import types
from pathlib import Path
from typing import Any

import pytest
from jsonschema import validate

from climate_rag.scifact_component_contract import (
    ContractError, packing, parse, protocol, render, schema_for,
)
from climate_rag.scifact_component_preparation import actual_documents, coverage, prepare_claim, source_hash
from climate_rag.scifact_component_runtime import ComponentPersistenceError, fixture_slot
from climate_rag.scifact_component_scoring import denominator_summary, localization, score, scored_row
from climate_rag.scifact_semantic_contract import encoded, sha


class TokenizerFixture:
    def apply_chat_template(self, messages: Any, **kwargs: Any) -> str:
        assert kwargs == {"tokenize": False, "add_generation_prompt": True, "enable_thinking": False}
        return json.dumps(messages)

    def encode(self, text: str, **kwargs: Any) -> list[int]:
        assert kwargs == {"add_special_tokens": False}
        return [1] * len(text)


def example() -> tuple[dict[int, Any], dict[str, Any], dict[str, Any]]:
    corpus = {i: {"doc_id": i, "title": "Synthetic document", "abstract": [f"Synthetic sentence {n}." for n in range(10)]}
              for i in (2, 10, 30)}
    visible = [{"alias": alias, "doc_id": i, "sentence_index": n, "source_text_sha256": source_hash(corpus[i]),
                "text_sha256": sha(corpus[i]["abstract"][n].encode())} for alias, i, n in (("c9", 10, 1), ("c2", 2, 0), ("c2", 2, 2))]
    projected = [{"sentence_id": f'{v["alias"]}:{v["sentence_index"]}', "sha256": v["text_sha256"]} for v in visible]
    identity = {"visible": projected}
    result = {"visible_attempts": [{"attempt_index": 0, "visible": visible, "actual_observation": copy.deepcopy(identity)}],
              "initial_context_identity": copy.deepcopy(identity),
              "generation_attempts": [{"requested_context": ["c9", "c2", "c0"], "diagnostics": {"actual_observation": copy.deepcopy(identity)}}]}
    gold = {"id": 7000, "claim": "Synthetic claim.", "cited_doc_ids": [30, 2],
            "evidence": {"2": [{"label": "SUPPORT", "sentences": [0, 2]}, {"label": "SUPPORT", "sentences": [4]}],
                         "30": [{"label": "CONTRADICT", "sentences": [1]}]}}
    return corpus, gold, result


def prepared() -> list[dict[str, Any]]:
    corpus, gold, result = example()
    return prepare_claim(gold, result, corpus, TokenizerFixture())


def test_actual_visible_order_full_abstract_not_requested_or_sorted() -> None:
    corpus, _, result = example()
    recovered = actual_documents(result, corpus)
    assert recovered["candidate_doc_ids"] == [10, 2]
    assert recovered["requested_without_visible_document"] == 1
    assert len(recovered["documents"][0]["sentences"]) == 10
    assert recovered["documents"][0]["sentences"][0]["sentence_id"] == 0


@pytest.mark.parametrize("change", ["source_hash", "text_hash", "alias", "repeat_doc", "duplicate_sid", "identity_order", "missing_identity"])
def test_corrupt_reconstruction_fails(change: str) -> None:
    corpus, _, result = example()
    rows = result["visible_attempts"][0]["visible"]
    if change == "source_hash":
        rows[0]["source_text_sha256"] = "0" * 64
    elif change == "text_hash":
        rows[0]["text_sha256"] = "0" * 64
    elif change == "alias":
        rows[1]["alias"] = "c9"
    elif change == "repeat_doc":
        rows[2]["alias"] = "c8"
    elif change == "duplicate_sid":
        rows.append(copy.deepcopy(rows[-1]))
    elif change == "identity_order":
        result["initial_context_identity"]["visible"].reverse()
    else:
        result["visible_attempts"][0]["actual_observation"]["visible"] = []
    with pytest.raises(ContractError):
        actual_documents(result, corpus)


def test_component_selection_min_numeric_and_no_label_leak() -> None:
    corpus, gold, result = example()
    gold["evidence"]["10"] = [{"label": "CONTRADICT", "sentences": [1]}]
    rows = prepare_claim(gold, result, corpus, TokenizerFixture())
    assert rows[1]["input"]["documents"][0]["document_id"] == 2
    assert "oracle_relation" not in rows[1]["input"]
    assert rows[2]["input"]["oracle_relation"] == "SUPPORT"
    assert "alternatives" not in json.dumps(rows[2]["input"])
    gold["evidence"]["2"] = [{"label": "CONTRADICT", "sentences": [8]}]
    changed = prepare_claim(gold, result, corpus, TokenizerFixture())
    assert changed[0]["input"] == rows[0]["input"]
    assert changed[1]["input"] == rows[1]["input"]
    assert changed[2]["input"] | {"oracle_relation": "SUPPORT"} == rows[2]["input"]


def test_nei_control_uses_official_cited_context_not_empty_context() -> None:
    corpus, gold, result = example()
    gold["evidence"] = {}
    rows = prepare_claim(gold, result, corpus, TokenizerFixture())
    assert len(rows) == 2
    assert rows[1]["input"]["documents"][0]["document_id"] == 2
    assert rows[1]["target"]["label_source"] == "official_code_derived_cited_context_NEI"
    assert rows[1]["target"]["relation"] == "NOT_ENOUGH_INFO"
    gold["cited_doc_ids"] = []
    missing = prepare_claim(gold, result, corpus, TokenizerFixture())
    assert missing[1]["packing"]["status"] == "missing_nei_source"
    assert missing[1]["input"] is None


def test_unreachable_or_inconsistent_never_changes_document() -> None:
    corpus, gold, result = example()
    gold["evidence"]["2"] = [{"label": "SUPPORT", "sentences": [0, 1, 2, 3]}]
    rows = prepare_claim(gold, result, corpus, TokenizerFixture())
    assert rows[2]["packing"]["status"] == "first3_contract_unreachable"
    assert rows[2]["target"]["target_doc_id"] == 2
    gold["evidence"]["2"].append({"label": "CONTRADICT", "sentences": [0]})
    rows = prepare_claim(gold, result, corpus, TokenizerFixture())
    assert rows[1]["packing"]["status"] == rows[2]["packing"]["status"] == "inconsistent_target_labels"


def test_no_visible_context_is_gap_not_backfilled() -> None:
    corpus, gold, result = example()
    result["visible_attempts"][0]["visible"] = []
    result["visible_attempts"][0]["actual_observation"]["visible"] = []
    result["initial_context_identity"]["visible"] = []
    result["generation_attempts"][0]["diagnostics"]["actual_observation"]["visible"] = []
    rows = prepare_claim(gold, result, corpus, TokenizerFixture())
    assert rows[0]["packing"]["status"] == "no_actual_visible_context"
    assert rows[0]["target"]["gold_missing_from_pool"] == 2


@pytest.mark.parametrize("tokens,status", [(8192, "prepared"), (8193, "full_abstract_over_budget")])
def test_complete_prompt_budget_no_cropping(tokens: int, status: str) -> None:
    class Counter(TokenizerFixture):
        def encode(self, text: str, **kwargs: Any) -> list[int]:
            return [1] * tokens
    spec = prepared()[0]["input"]
    before = copy.deepcopy(spec)
    p = packing(Counter(), spec)
    assert p["status"] == status and spec == before
    assert p == packing(Counter(), spec)
    assert p["prompt_sha256"] == sha(render(Counter(), spec).encode())


@pytest.mark.parametrize("index,wire", [
    (0, {"decision": "select", "document_ids": [2, 10]}),
    (1, {"decision": "classify", "relation": "SUPPORT"}),
    (2, {"decision": "select", "sentence_ids": [2, 0]}),
    (0, {"decision": "abstain", "document_ids": []}),
    (1, {"decision": "classify", "relation": "NOT_ENOUGH_INFO"}),
    (1, {"decision": "abstain", "relation": None}),
])
def test_three_explicit_schemas(index: int, wire: dict[str, Any]) -> None:
    spec = prepared()[index]["input"]
    validate(wire, schema_for(spec))
    assert parse(json.dumps(wire), spec) == wire


@pytest.mark.parametrize("raw,category", [
    ('{"decision":"select","document_ids":[2,2]}', "duplicate_index"),
    ('{"decision":"select","document_ids":[30]}', "unknown_index"),
    ('{"decision":"select","document_ids":[true]}', "index_type"),
    ('{"decision":"select","document_ids":[]}', "selection_output"),
    ('{"decision":"abstain","document_ids":[2]}', "selection_output"),
    ('{"decision":"select","decision":"abstain","document_ids":[]}', "duplicate_key"),
    ('{"decision":"abstain","document_ids":[],"reason":"x"}', "output_fields"),
    ('{"decision":', "invalid_json"),
])
def test_invalid_not_silently_repaired(raw: str, category: str) -> None:
    with pytest.raises(ContractError, match=category):
        parse(raw, prepared()[0]["input"])


def test_overbudget_duplicate_and_out_of_range_sentences() -> None:
    spec = prepared()[2]["input"]
    for values, category in [([0, 0], "duplicate_index"), ([10], "unknown_index"), (list(range(9)), "output_over_budget")]:
        with pytest.raises(ContractError, match=category):
            parse(json.dumps({"decision": "select", "sentence_ids": values}), spec)


def test_alternative_OR_order_only_matters_at_first_three() -> None:
    target = prepared()[2]["target"]
    for ids in ([2, 0], [0, 2], [4]):
        s = score("rationale", {"decision": "select", "sentence_ids": ids}, target)
        assert s["correct"] and s["exact_any_alternative"] and s["alternative_complete"]
    s = score("rationale", {"decision": "select", "sentence_ids": [1, 3, 5, 4]}, target)
    assert s["alternative_complete"] and not s["first3_complete"]
    assert s["nonannotated_extra_sentences"] == 3
    assert s["minimum_redundancy_after_complete_alternative"] == 3


def test_missing_candidate_not_model_omission_and_nei_not_full_recall() -> None:
    target = prepared()[0]["target"]
    scored = score("screening", {"decision": "select", "document_ids": [2, 10]}, target)
    assert scored["candidate_missing"] == 1 and scored["model_omitted_from_full_context"] == 0
    assert scored["selected_unannotated_documents"] == 1
    nei = score("screening", {"decision": "abstain", "document_ids": []}, target | {"relevant_doc_ids": []})
    assert nei["correct"] and nei["recall_in_pool"] is None
    summary = coverage(prepared())
    assert summary["gold_document_pool_coverage"] == 0.5
    assert summary["old_context_first3_complete_gold_documents"] == 1


class FixtureProvider:
    kind = "fixture"
    tokenizer = TokenizerFixture()

    def __init__(self, raw: str, *, unknown: bool = False, fail: bool = False) -> None:
        self.raw, self.unknown, self.fail, self.calls = raw, unknown, fail, 0

    def generate(self, spec: Any, schema: Any, max_tokens: int, seconds: float) -> dict[str, Any]:
        assert max_tokens == 512 and seconds == 120
        self.calls += 1
        if self.fail:
            raise RuntimeError("fixture_failure")
        return {"raw": self.raw, "usage": None if self.unknown else {
            "input_tokens": packing(self.tokenizer, spec)["input_tokens"], "output_tokens": 10},
            "diagnostics": {"eos_observed": True, "reached_max_new_tokens": False}}


@pytest.mark.parametrize("index,wire", [(0, {"decision": "select", "document_ids": [2]}),
    (1, {"decision": "classify", "relation": "SUPPORT"}),
    (2, {"decision": "select", "sentence_ids": [0, 2]}),
    (0, {"decision": "abstain", "document_ids": []})])
def test_four_synthetic_fixture_slots_single_attempt(tmp_path: Path, index: int, wire: dict[str, Any]) -> None:
    provider = FixtureProvider(json.dumps(wire))
    record = fixture_slot(prepared()[index]["input"], provider, tmp_path / "slot")
    assert provider.calls == 1 and record["status"] == "valid" and record["model_calls"] == 0
    assert (tmp_path / "slot/response.json").is_file()
    assert record["response_sha256"] == sha((tmp_path / "slot/response.json").read_bytes())
    with pytest.raises(FileExistsError):
        fixture_slot(prepared()[index]["input"], provider, tmp_path / "slot")
    assert provider.calls == 1


@pytest.mark.parametrize("unknown,fail,status", [(True, False, "provider_failed"), (False, True, "provider_failed"),
                                                  (False, False, "schema_failed")])
def test_failed_calls_keep_cost_unknown_no_retry(tmp_path: Path, unknown: bool, fail: bool, status: str) -> None:
    provider = FixtureProvider("{", unknown=unknown, fail=fail)
    record = fixture_slot(prepared()[0]["input"], provider, tmp_path / "slot")
    assert record["status"] == status and provider.calls == 1
    assert record["usage_known"] == (not unknown and not fail)
    assert "prediction" not in record
    assert json.loads((tmp_path / "slot/completed.json").read_bytes())["status"] == status


def test_real_provider_and_incomplete_prompt_refused(tmp_path: Path) -> None:
    provider = FixtureProvider("{}")
    provider.kind = "local_model"
    with pytest.raises(ContractError, match="real_model_not_authorized"):
        fixture_slot(prepared()[0]["input"], provider, tmp_path / "slot")
    assert provider.calls == 0 and not (tmp_path / "slot").exists()


def test_persistence_failure_retains_cost_without_second_call(tmp_path: Path, monkeypatch: Any) -> None:
    import climate_rag.scifact_component_runtime as module
    write = module.write_once

    def broken(path: Path, obj: Any) -> None:
        if path.name == "completed.json":
            raise OSError("fixture_disk_full")
        write(path, obj)
    monkeypatch.setattr(module, "write_once", broken)
    provider = FixtureProvider('{"decision":"abstain","document_ids":[]}')
    with pytest.raises(ComponentPersistenceError) as exc:
        fixture_slot(prepared()[0]["input"], provider, tmp_path / "slot")
    assert exc.value.partial_report["usage_known"] and provider.calls == 1


def test_denominators_and_localization_do_not_promote_agent() -> None:
    positive = {"claim_id": 7000, "target_doc_id": 2, "status": "valid", "score": {"correct": True, "explicit_abstention": False, "exact_any_alternative": True}}
    negative = {"claim_id": 7000, "target_doc_id": 2, "status": "valid", "score": {"correct": False, "explicit_abstention": False}}
    assert localization(negative, positive, None).startswith("relation_candidate")
    assert localization(positive, negative, None).startswith("sentence_selection")
    assert localization(negative, negative, None).startswith("joint_grounding")
    assert localization(positive, positive, negative).startswith("document_discrimination")
    assert "not_Agent" in localization(positive, positive, positive)
    overselected = positive | {"score": positive["score"] | {"exact_any_alternative": False}}
    assert localization(positive, overselected, positive).startswith("rationale_overselection")
    assert localization({"claim_id": 7000, "status": "schema_failed"}, positive, None).startswith("format_coverage")
    summary = denominator_summary([positive | {"component": "relation", "nei_control": False, "attempted": True},
        {"component": "relation", "status": "schema_failed", "nei_control": True, "attempted": True},
        {"component": "relation", "status": "preparation_gap", "nei_control": True}])
    assert summary["relation"]["task_success_all_planned"] == 1 / 3
    assert summary["relation"]["accuracy_given_valid"] == 1
    assert summary["relation_nei_control"]["explicit_abstention"] == 0
    p = protocol()
    assert not p["gpu_authorized"] and p["limits"]["diagnostic_calls"] == 33
    assert p["limits"]["preflight_calls"] == 4
    assert sha(encoded(p)) == sha(encoded(protocol()))


def test_scored_row_join_keeps_nei_failure_denominators() -> None:
    relation = prepared()[1]
    valid = {"component": "relation", "status": "valid", "attempted": True,
             "packing": relation["packing"], "prediction": {"decision": "classify", "relation": "SUPPORT"}}
    failed = {"component": "relation", "status": "provider_failed", "attempted": True, "packing": relation["packing"]}
    nei_prepared = relation | {"target": relation["target"] | {"nei_control": True}}
    rows = [scored_row(relation, valid), scored_row(nei_prepared, failed)]
    summary = denominator_summary(rows)
    assert summary["relation_gold_document"]["planned"] == 1
    assert summary["relation_nei_control"]["planned"] == 1
    assert summary["relation_nei_control"]["provider_failed"] == 1
    assert summary["relation"]["correct"] == 1
    with pytest.raises(ContractError, match="relation_stratum_missing"):
        denominator_summary([failed])
    with pytest.raises(ContractError, match="failed_row_must_not_have_score"):
        denominator_summary([rows[1] | {"score": {"correct": True, "explicit_abstention": True}}])


def test_started_write_failure_makes_zero_calls(tmp_path: Path, monkeypatch: Any) -> None:
    import climate_rag.scifact_component_runtime as module
    write = module.write_once

    def broken(path: Path, obj: Any) -> None:
        if path.name == "started.json":
            raise OSError("fixture_disk_full")
        write(path, obj)
    monkeypatch.setattr(module, "write_once", broken)
    provider = FixtureProvider('{"decision":"abstain","document_ids":[]}')
    with pytest.raises(OSError):
        fixture_slot(prepared()[0]["input"], provider, tmp_path / "slot")
    assert provider.calls == 0


def test_start_reservation_is_attempted_unknown_on_interrupt(tmp_path: Path) -> None:
    class Interrupt(FixtureProvider):
        def generate(self, *args: Any, **kwargs: Any) -> dict[str, Any]:
            self.calls += 1
            raise KeyboardInterrupt()
    provider = Interrupt("{}")
    with pytest.raises(KeyboardInterrupt):
        fixture_slot(prepared()[0]["input"], provider, tmp_path / "slot")
    started = json.loads((tmp_path / "slot/started.json").read_bytes())
    assert started["attempted"] and not started["usage_known"] and started["usage"] is None
    assert provider.calls == 1 and not (tmp_path / "slot/completed.json").exists()


def test_score_input_swap_and_cross_claim_localization_rejected() -> None:
    row = prepared()[1]
    valid = {"component": "relation", "status": "valid", "attempted": True,
             "packing": row["packing"] | {"input_sha256": "0" * 64},
             "prediction": {"decision": "classify", "relation": "SUPPORT"}}
    with pytest.raises(ContractError, match="scoring_input_identity_mismatch"):
        scored_row(row, valid)
    with pytest.raises(ContractError, match="localization_claim_pair"):
        localization({"claim_id": 1}, {"claim_id": 2}, None)
    with pytest.raises(ContractError, match="localization_document_pair"):
        localization({"claim_id": 1, "status": "valid", "target_doc_id": 2},
                     {"claim_id": 1, "status": "valid", "target_doc_id": 3}, None)


def test_partial_usage_survives_invalid_response(tmp_path: Path) -> None:
    class Partial(FixtureProvider):
        def generate(self, *args: Any, **kwargs: Any) -> dict[str, Any]:
            self.calls += 1
            return {"raw": "{", "usage": {"input_tokens": 37}}
    provider = Partial("{")
    record = fixture_slot(prepared()[0]["input"], provider, tmp_path / "slot")
    assert record["known_usage_lower_bound"] == {"input_tokens": 37}
    assert record["status"] == "provider_failed" and not record["usage_known"]
    assert (tmp_path / "slot/response.json").is_file()
    assert provider.calls == 1


def test_frozen_preparation_entry_full_fixture_no_model(tmp_path: Path, monkeypatch: Any) -> None:
    monkeypatch.delitem(sys.modules, "torch", raising=False)
    for key in ("USE_TORCH", "USE_TF", "HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE", "HF_HUB_DISABLE_TELEMETRY", "TOKENIZERS_PARALLELISM"):
        monkeypatch.setenv(key, os.environ.get(key, ""))
    location = Path(__file__).resolve().parents[1] / "scripts/prepare_scifact_components.py"
    module_spec = importlib.util.spec_from_file_location("component_preparation_fixture", location)
    assert module_spec and module_spec.loader
    module = importlib.util.module_from_spec(module_spec)
    module_spec.loader.exec_module(module)
    monkeypatch.setattr(module, "ROOT", tmp_path)
    source = tmp_path / "source"
    source.mkdir()
    source_git = "1" * 40
    (source / "SOURCE_REVISION").write_text(source_git)
    monkeypatch.setattr(module, "__file__", str(source / "scripts/prepare.py"))
    corpus, gold, result = example()
    for i in range(100, 5280):
        corpus[i] = {"doc_id": i, "title": "Fixture", "abstract": ["Synthetic unused fixture."]}
    assert len(corpus) == 5183
    gold_rows = []
    for index in range(12):
        row = copy.deepcopy(gold)
        row["id"] += index
        if index >= 9:
            row["evidence"] = {}
        gold_rows.append(row)
    claims = [{"id": r["id"], "claim": r["claim"]} for r in gold_rows]
    corpus_raw = b"\n".join(encoded(v).replace(b"\n", b"") for v in corpus.values()) + b"\n"
    claims_raw = b"\n".join(json.dumps(r).encode() for r in claims) + b"\n"
    gold_raw = b"\n".join(json.dumps(r).encode() for r in gold_rows) + b"\n"
    old_protocol_raw = encoded({"ordered_claim_ids": [r["id"] for r in claims]})
    monkeypatch.setattr(module, "CORPUS_SHA", sha(corpus_raw))
    monkeypatch.setattr(module, "OLD_PROTOCOL_SHA", sha(old_protocol_raw))
    bundle_dir = tmp_path / "envs/scifact-semantic-bundles-99cd9ff"
    bundle_dir.mkdir(parents=True)
    hashes = {}
    for name, members in {"inference": {"corpus.jsonl": corpus_raw, "claims.jsonl": claims_raw, "protocol.json": old_protocol_raw},
                          "scoring": {"gold.jsonl": gold_raw}}.items():
        path = bundle_dir / (name + ".tar")
        with tarfile.open(path, "w") as archive:
            for filename, payload in members.items():
                info = tarfile.TarInfo(filename)
                info.size = len(payload)
                archive.addfile(info, io.BytesIO(payload))
        hashes[name] = sha(path.read_bytes())
    monkeypatch.setattr(module, "BUNDLES", hashes)
    run_hashes = {}
    for policy in module.RUNS:
        directory = tmp_path / "runs" / (module.RELEASE + "-" + policy) / "inference"
        directory.mkdir(parents=True)
        rows = [{"claim_id": c["id"], "arm": policy, "source_git": module.OLD_SOURCE, "route": "fixed_rerank",
                 "result": copy.deepcopy(result) | {"claim_id": c["id"], "arm": policy, "source_git": module.OLD_SOURCE,
                                                    "route": "fixed_rerank"}} for c in claims]
        payload = encoded({"source_git": module.OLD_SOURCE, "runs": rows})
        (directory / "run.json").write_bytes(payload)
        run_hashes[policy] = sha(payload)
    monkeypatch.setattr(module, "RUNS", run_hashes)
    score_path = tmp_path / "runs" / (module.RELEASE + "-scifact-gap-semantic-policy-v1") / "score.json"
    score_path.write_bytes(b"{}")
    monkeypatch.setattr(module, "SCORE_SHA", sha(b"{}"))
    token_dir = tmp_path / "tokenizer"
    token_dir.mkdir()
    (token_dir / "tokenizer.json").write_bytes(b"{}")
    monkeypatch.setattr(module, "TOKENIZER_SHA", {"tokenizer.json": sha(b"{}")})
    fake = types.ModuleType("transformers.models.auto.tokenization_auto")

    class Loader:
        @staticmethod
        def from_pretrained(*args: Any, **kwargs: Any) -> TokenizerFixture:
            assert kwargs == {"local_files_only": True, "trust_remote_code": False}
            return TokenizerFixture()
    fake.AutoTokenizer = Loader  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "transformers.models.auto.tokenization_auto", fake)
    (tmp_path / "posthoc").mkdir()
    output = tmp_path / "posthoc" / ("scifact-component-preparation-" + source_git[:12])
    receipt = module.prepare(output, token_dir, source_git)
    assert receipt["coverage"]["model_calls"] == 0
    assert receipt["coverage"]["components"]["screening"]["planned"] == 12
    assert receipt["coverage"]["components"]["relation"]["planned"] == 12
    assert receipt["coverage"]["components"]["rationale"]["planned"] == 9
    assert "Synthetic" not in (output / "compact.json").read_text()
    inputs = json.loads((output / "inference/slots.json").read_bytes())
    assert all("target" not in row and "claim_id" not in row for row in inputs)
    with pytest.raises(ContractError, match="exclusive_preparation"):
        module.prepare(output, token_dir, source_git)


@pytest.mark.parametrize("member_name,link", [("../bad.py", False), ("/bad.py", False), ("lib/python3.10/site-packages/tokenizers/link", True)])
def test_tokenizer_stage_rejects_traversal_and_links(tmp_path: Path, member_name: str, link: bool) -> None:
    location = Path(__file__).resolve().parents[1] / "scripts/stage_scifact_component_tokenizer.py"
    spec = importlib.util.spec_from_file_location("tokenizer_stager_fixture", location)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    archive_path = tmp_path / "bad.tar"
    with tarfile.open(archive_path, "w") as archive:
        info = tarfile.TarInfo(member_name)
        if link:
            info.type, info.linkname = tarfile.SYMTYPE, "/outside"
            archive.addfile(info)
        else:
            info.size = 1
            archive.addfile(info, io.BytesIO(b"x"))
    with pytest.raises(ValueError, match="tokenizer_stage_contract_failed"):
        module.unpack(archive_path, tmp_path / "unpacked", "runtime")
    assert not (tmp_path / "unpacked").exists()


def test_runtime_tar_prefix_and_unselected_python_link_not_extracted(tmp_path: Path) -> None:
    location = Path(__file__).resolve().parents[1] / "scripts/stage_scifact_component_tokenizer.py"
    spec = importlib.util.spec_from_file_location("tokenizer_stager_fixture", location)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    archive_path = tmp_path / "runtime.tar"
    with tarfile.open(archive_path, "w") as archive:
        root = tarfile.TarInfo(".")
        root.type = tarfile.DIRTYPE
        archive.addfile(root)
        link = tarfile.TarInfo("./bin/python")
        link.type, link.linkname = tarfile.SYMTYPE, "/system/python"
        archive.addfile(link)
        for path in ("./lib/python3.10/site-packages/tokenizers/__init__.py", "./lib/python3.10/site-packages/torch/__init__.py"):
            info = tarfile.TarInfo(path)
            info.size = 1
            archive.addfile(info, io.BytesIO(b"x"))
    receipt = module.unpack(archive_path, tmp_path / "site", "runtime")
    assert receipt["files"] == 1
    assert (tmp_path / "site/tokenizers/__init__.py").read_bytes() == b"x"
    assert not (tmp_path / "site/torch").exists()


def test_fixture_cli_nested_output_is_exclusive(tmp_path: Path) -> None:
    location = Path(__file__).resolve().parents[1] / "scripts/check_scifact_components.py"
    spec = importlib.util.spec_from_file_location("component_cli_fixture", location)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    output = tmp_path / "new-parent/fixture-run"
    receipt = module.check(output)
    assert receipt["valid_fixtures"] == 4 and receipt["model_calls"] == 0
    with pytest.raises(FileExistsError):
        module.check(output)
