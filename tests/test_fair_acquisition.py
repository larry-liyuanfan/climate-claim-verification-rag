"""SYNTHETIC CPU integration: no model inference or empirical benefit claim."""
import copy
import hashlib
import json

import pytest

from climate_rag import fair_acquisition as fair
from climate_rag.agent_v3 import SentenceAgentV3, Source, V3Budget
from climate_rag.fair_replay import STUDY, identity, score, validate_binding, validate_tasks, policy
from climate_rag.targeted_replay import CORPUS_SHA, MODEL_SHA, RERANKER_SHA, run_matrix
from climate_rag.targeted_score import audit
from test_targeted_replay import FixtureProvider, FixtureRerank, ABSTAIN, QUERY
from cloud_replay_contract import draft, validate_release, worker_projection
from prepare_fair_cohort import freeze, components
from climate_rag.local_agent_v3 import render_v3_prompt

TASKS = [{"id": "synthetic-fair-1", "claim_text": "Example glaciers have advanced."}]
DOCS = {f"old{i}": Source(f"old{i}", "Synthetic", (f"Synthetic background {i}.",)) for i in range(8)}
DOCS["new"] = Source("new", "Synthetic", ("Synthetic observations refute advance.",))


def binding(tasks=TASKS):
    return {"protocol": fair.PROTOCOL, "study_kind": STUDY, "task_count": len(tasks),
        "cohort_sha256": "a" * 64, "tasks_sha256": identity(tasks),
        "exposure_audit_sha256": "b" * 64, "initial_contract_sha256": "c" * 64,
        "scoring_contract_sha256": "c" * 64, "corpus_sha256": CORPUS_SHA,
        "model_sha256": MODEL_SHA, "reranker_sha256": RERANKER_SHA,
        "eligible": True, "selection_rule": "SYNTHETIC fixture only"}


class FairFixture(FixtureProvider):
    def __init__(self, mode="query"):
        super().__init__()
        self.mode = mode

    def generate(self, obs, schema, max_output_tokens, remaining_seconds):
        self.calls += 1
        if obs["phase"] == "plan":
            action = {"action": "plan_queries", "queries": [QUERY], "read_source_ids": obs["readable_source_ids"][:5]}
        elif obs["phase"] == "gate":
            if self.calls == 1 and self.mode != "stop":
                action = {"action": "acquire", "tool": "rerank"} if self.mode == "rerank" else {"action": "acquire", "tool": "query", **QUERY}
            elif self.mode == "feedback" and self.calls == 2 and not any("refute" in v["text"] for v in obs["current_citable"]):
                action = {"action": "acquire", "tool": "query", "query": "glacier coastal retreat study", "purpose": "counter_evidence"}
            else:
                action = {"action": "stop"}
        else:
            good = [v for v in obs["current_citable"] if "refute" in v["text"]]
            action = {"action": "answer", "label": "REFUTES", "sentence_ids": [good[0]["sentence_id"]]} if good else ABSTAIN
        return {"raw": json.dumps(action), "usage": {"input_tokens": self.count_prompt(obs, schema), "output_tokens": 32}}


def matrix(tmp_path, mode="query", failure=None):
    def retrieve(query, width):
        if failure == "all_empty":
            return []
        if query == QUERY["query"]:
            if failure == "empty":
                return []
            if failure == "timeout":
                raise TimeoutError("SYNTHETIC")
            return [DOCS["new"]]
        if query == "glacier coastal retreat study":
            return [DOCS["new"]]
        return list(DOCS.values())[:8]
    class Ranker(FixtureRerank):
        def __call__(self, query, rows):
            assert rows, "empty rerank must be skipped"
            returned = super().__call__(query, rows)
            return list(reversed(returned)) if mode == "rerank" else returned
    result = run_matrix(TASKS, FairFixture(mode), retrieve, Ranker(tmp_path / "reranker-ledger"),
        tmp_path / "inference", protocol=fair.PROTOCOL, binding=binding())
    audit(result, TASKS, DOCS, tmp_path / "inference/ledger", synthetic=True,
          reranker_directory=tmp_path / "reranker-ledger")
    return result


@pytest.mark.parametrize("mode,failure", [("stop", None), ("query", None), ("rerank", None), ("query", "empty"), ("query", "timeout"), ("stop", "all_empty")])
def test_actual_matrix_shared_frame_verifier_receipts_cost(tmp_path, mode, failure):
    run = matrix(tmp_path, mode, failure)
    assert len(run["runs"]) == 3
    assert all(r["initial_frame"] == run["runs"][0]["initial_frame"] for r in run["runs"])
    for r in run["runs"]:
        assert r["model_calls"] == r["physical_cost"]["unique_physical_calls"] <= 5
        assert r["tool_calls"] <= 5
        last = r["generation_attempts"][-1]
        assert last["stage"] == "verdict", r["outcome"]
        assert all(e["trigger_attempt"] < len(r["generation_attempts"]) - 1 for e in r["events"] if "tool" in e)
        assert last["schema"] == run["runs"][0]["generation_attempts"][-1]["schema"] or last["observation"]["current_citable"] != run["runs"][0]["generation_attempts"][-1]["observation"]["current_citable"]
        if failure == "all_empty":
            assert r["outcome"] == "model_abstention:insufficient_evidence"
            assert not any(e.get("tool") == "rerank" for e in r["events"])
    auto = run["runs"][-1]
    if mode == "stop":
        assert auto["tool_calls"] == 1 and auto["model_calls"] == 2
    if mode == "rerank":
        receipt = next(e for e in auto["events"] if e.get("tool") == "rerank")
        assert receipt["delivery"]["new_sentence_sha256"]
        assert 1 in auto["generation_attempts"][1]["received_tool_events"]


def test_different_feedback_changes_legal_later_verdict(tmp_path):
    (tmp_path / "good").mkdir()
    (tmp_path / "empty").mkdir()
    good = matrix(tmp_path / "good")["runs"][-1]
    empty = matrix(tmp_path / "empty", failure="empty")["runs"][-1]
    assert good["initial_frame"] == empty["initial_frame"]
    assert good["answer"]["label"] == "REFUTES" and empty["answer"] is None
    assert good["generation_attempts"][1]["observation"] != empty["generation_attempts"][1]["observation"]


def test_feedback_changes_subsequent_acquisition_not_call_counter(tmp_path):
    (tmp_path / "good").mkdir()
    (tmp_path / "empty").mkdir()
    good = matrix(tmp_path / "good", mode="feedback")["runs"][-1]
    empty = matrix(tmp_path / "empty", mode="feedback", failure="empty")["runs"][-1]
    assert good["initial_frame"] == empty["initial_frame"]
    assert good["generation_attempts"][0]["decision"] == empty["generation_attempts"][0]["decision"]
    assert good["generation_attempts"][1]["decision"] == {"action": "stop"}
    assert empty["generation_attempts"][1]["decision"]["tool"] == "query"
    assert empty["generation_attempts"][2]["decision"] == {"action": "stop"}
    assert empty["tool_calls"] == 3 and good["tool_calls"] == 2


@pytest.mark.parametrize("cap", [4000, 5000, 6000])
def test_common_initial_payload_at_capacity_including_unicode(cap):
    frames = []
    for route in fair.ROUTES:
        p = FairFixture("stop")
        p.calls = 0
        result = SentenceAgentV3(p, lambda q, k: list(DOCS.values())[:8], rerank=lambda q, rows: rows,
            budget=V3Budget(max_input_tokens=cap), protocol=fair.PROTOCOL).run("Synthetic °C glaciers — advance.", route)
        frames.append(result["initial_frame"])
        if result["initial_frame"] is not None:
            assert identity(result["initial_frame"]) == result["initial_frame_sha256"]
        if result["generation_attempts"] and route != "deterministic_workflow":
            first = result["generation_attempts"][0]["observation"]
            assert {k: first[k] for k in result["initial_frame"]} == result["initial_frame"]
    assert all(f == frames[0] for f in frames)


def test_tampered_abstention_and_feedback_audit_refuses(tmp_path):
    run = matrix(tmp_path)
    changed = copy.deepcopy(run)
    row = changed["runs"][-1]
    row["answer"] = None
    row["outcome"] = "model_abstention:insufficient_evidence"
    with pytest.raises(ValueError, match="abstention"):
        audit(changed, TASKS, DOCS, tmp_path / "inference/ledger", synthetic=True,
              reranker_directory=tmp_path / "reranker-ledger")
    changed = copy.deepcopy(run)
    changed["runs"][-1]["generation_attempts"][1]["observation"]["tool_feedback_history"][0]["status"] = "fabricated"
    with pytest.raises(ValueError, match="feedback_history"):
        audit(changed, TASKS, DOCS, tmp_path / "inference/ledger", synthetic=True,
              reranker_directory=tmp_path / "reranker-ledger")


def test_unbound_identity_and_gold_fields_refuse_before_output(tmp_path):
    with pytest.raises(ValueError, match="unbound"):
        run_matrix(TASKS, None, None, None, tmp_path / "never", protocol=fair.PROTOCOL)
    bad = copy.deepcopy(binding())
    bad["eligible"] = False
    with pytest.raises(ValueError, match="not_eligible"):
        validate_binding(bad)
    with pytest.raises(ValueError, match="gold_leak"):
        validate_tasks([{**TASKS[0], "label": "REFUTES"}], binding())
    assert not (tmp_path / "never").exists()


def test_post_exit_score_proxy_not_semantic_and_complete_denominators(tmp_path):
    run = matrix(tmp_path)
    gold = {"tasks_sha256": identity(TASKS), "claims": {TASKS[0]["id"]: {
        "claim_sha256": hashlib.sha256(TASKS[0]["claim_text"].encode()).hexdigest(), "label": "REFUTES", "evidence_ids": ["new"]}}}
    scored = score(run, gold)
    assert "unmeasured" in scored["semantic_citation_support"]
    assert "NOT semantic" in scored["citation_proxy"]
    assert all(r["all_task_denominator"] == 1 for r in scored["routes"].values())
    assert scored["paired_bootstrap"]["autonomous-minus-fixed_multiquery"]["official_task_correct"]["samples"] == 5000
    assert scored["routes"]["autonomous"]["retrieval"]["denominator"] == 1
    assert scored["paired_bootstrap"]["autonomous-minus-fixed_multiquery"]["retrieval_secondary"]["evidence_f1"]["samples"] == 5000


def test_label_blind_component_exclusion_no_result_selection():
    metadata = {"used": {"claim_text": "Example same claim", "annotated_candidate_ids": ["shared"]},
                "same": {"claim_text": "Example same claim!", "annotated_candidate_ids": ["shared"]},
                "other": {"claim_text": "A separate topic", "annotated_candidate_ids": ["other"]}}
    evidence = {"shared": "Synthetic shared text", "other": "Unrelated evidence content"}
    assert ["same", "used"] in components(metadata, evidence)
    tasks, audit_data = freeze(metadata, evidence, {"used"})
    assert [t["id"] for t in tasks] == ["other"] and audit_data["labels_used_for_selection"] is False
    metadata["other"]["label"] = "SUPPORTS"
    with pytest.raises(ValueError, match="gold_or_results"):
        freeze(metadata, evidence, {"used"})


def test_cloud_fair_draft_and_projection_share_runtime_without_gold():
    value = draft("a" * 40, "b" * 64, "c" * 64, root="/workspace/fair-climate", run_id="fair-three-arm-20261003",
                  fair_binding=binding(), fair_gold_sha256="d" * 64)
    validate_release(value, execution=False)
    assert value["policy"] == policy(binding())
    assert value["policy"]["planned_slots"] == 3
    with pytest.raises(ValueError, match="unauthorized"):
        validate_release(value)
    value.update(authorization="standalone_exact_hash_release", model_execution_authorized=True,
                 runtime_receipt_sha256="e" * 64, asset_receipt_sha256="f" * 64)
    projected = worker_projection(value, "1" * 64)
    assert "gold" not in json.dumps(projected)
    assert projected["fair_binding"] == binding()


def test_exact_normalised_empty_variants_remain_in_consumed_component():
    metadata = {"used": {"claim_text": "!!!", "annotated_candidate_ids": ["x"]},
                "claim_variant": {"claim_text": "???", "annotated_candidate_ids": ["y"]},
                "evidence_variant": {"claim_text": "Different content", "annotated_candidate_ids": ["z"]}}
    assert components(metadata, {"x": "", "y": "Text", "z": "!!!"}) == [
        ["claim_variant", "evidence_variant", "used"]]
    tasks, _ = freeze(metadata, {"x": "", "y": "Text", "z": "!!!"}, {"used"})
    assert tasks == []


def test_delivery_increment_cannot_fabricate_feedback_utilisation(tmp_path):
    from climate_rag.fair_replay import audit_state
    run = matrix(tmp_path)
    row = copy.deepcopy(run["runs"][-1])
    event = next(e for e in row["events"] if e.get("delivery") and e["tool"] == "rewrite")
    event["delivery"]["new_sentence_sha256"] = {"fake": "x"}
    with pytest.raises(ValueError, match="increment_not_actual"):
        audit_state(row, DOCS)
    row = copy.deepcopy(run["runs"][-1])
    event = next(e for e in row["events"] if e.get("delivery") and e["tool"] == "rewrite")
    event["delivery"]["next_attempt"] = 2
    with pytest.raises(ValueError, match="receipt_time"):
        audit_state(row, DOCS)
    row = copy.deepcopy(run["runs"][-1])
    row["generation_attempts"].append(copy.deepcopy(row["generation_attempts"][-1]))
    with pytest.raises(ValueError, match="verdict_must_terminate"):
        audit_state(row, DOCS)
    row = copy.deepcopy(run["runs"][-1])
    bootstrap = row["events"][0]["delivery"]
    removed = next(iter(bootstrap["visible_sentence_sha256"]))
    fingerprint = bootstrap["visible_sentence_sha256"].pop(removed)
    bootstrap["new_sentence_sha256"].pop(removed)
    next(e for e in row["events"] if e.get("delivery") and e["tool"] == "rewrite")["delivery"]["new_sentence_sha256"][removed] = fingerprint
    with pytest.raises(ValueError, match="complete_context_commit"):
        audit_state(row, DOCS)


def test_fair_package_binds_cohort_and_refuses_missing_scorer_hash(tmp_path, monkeypatch):
    import package_cloud_replay as pack
    monkeypatch.setattr(pack, "require_clean_source", lambda *a: None)
    monkeypatch.setattr(pack, "git", lambda *a: b"synthetic archive")
    monkeypatch.setattr(pack, "validate_archive", lambda *a, **kw: {
        "source_archive_sha256": "b" * 64, "wrapper_sha256": "c" * 64})
    with pytest.raises(ValueError, match="requires_scorer_hash"):
        pack.package(tmp_path, "a" * 40, tmp_path / "missing", root="/workspace/fair",
                     run_id="synthetic-fair-package", fair_binding=binding())
    assert not (tmp_path / "missing").exists()
    output = tmp_path / "package"
    pack.package(tmp_path, "a" * 40, output, root="/workspace/fair",
                 run_id="synthetic-fair-package", fair_binding=binding(), fair_gold_sha256="d" * 64)
    value = json.loads((output / "release.unauthorized.json").read_bytes())
    assert value["fair_binding"] == binding() and value["gold_sha256"] == "d" * 64
    assert value["policy"]["planned_slots"] == 3 and not value["model_execution_authorized"]


def test_asset_inventory_preserves_legacy_hash_and_explicit_fair_override(tmp_path, monkeypatch):
    import preflight_cloud_assets as assets
    archive, gold = tmp_path / "input.tar", tmp_path / "scorer.json"
    archive.write_bytes(b"SYNTHETIC input")
    gold.write_bytes(b"SYNTHETIC scorer hash only")
    fingerprint = assets.digest(gold)
    monkeypatch.setattr(assets, "GOLD_SHA", fingerprint)
    monkeypatch.setattr(assets, "SELECTION", "selection.json")
    selection = tmp_path / "selection.json"
    selection.write_bytes(b"SYNTHETIC receipt")
    monkeypatch.setattr(assets, "SELECTION_SHA", assets.digest(selection))
    monkeypatch.setattr(assets, "inspect_input", lambda *a, **kw: {"status": "verified"})
    assert assets.inventory(archive, gold, tmp_path)["asset_status"] == "verified_local_assets"
    monkeypatch.setattr(assets, "GOLD_SHA", "0" * 64)
    assert assets.inventory(archive, gold, tmp_path)["asset_status"] == "needs_assets"
    assert assets.inventory(archive, gold, tmp_path, gold_sha=fingerprint)["asset_status"] == "verified_local_assets"


def test_final_renderer_instructions_are_route_independent():
    class Tokenizer:
        def apply_chat_template(self, messages, **kwargs):
            return json.dumps(messages)
    obs = {"protocol": fair.PROTOCOL, "phase": "verdict"}
    assert render_v3_prompt(Tokenizer(), obs, {}) == render_v3_prompt(Tokenizer(), dict(obs), {})


def test_production_worker_entry_connects_actual_three_arm_matrix_with_fake_models(tmp_path, monkeypatch):
    """No weights/inference: only injected fixture providers replace model boundaries."""
    import run_targeted_replay as worker
    from climate_rag.models import EvidenceDocument
    out = tmp_path / "run"
    out.mkdir()
    inputs = tmp_path / "input"
    for kind in ("generator", "reranker"):
        directory = inputs / "models" / kind
        directory.mkdir(parents=True)
        (directory / "model_manifest.json").write_bytes(json.dumps({"kind": kind}).encode())
    monkeypatch.setattr(worker, "load_inputs", lambda *a: (TASKS, [EvidenceDocument(s.source_id, s.sentences[0]) for s in DOCS.values()], DOCS))
    monkeypatch.setattr(worker, "sha", lambda raw: worker.MANIFEST_BYTES[json.loads(raw)["kind"]])
    monkeypatch.setattr(worker, "verify_model_files", lambda path, manifest: MODEL_SHA if manifest["kind"] == "generator" else RERANKER_SHA)
    monkeypatch.setattr(worker, "LocalTargetedProvider", lambda *a, **kw: FairFixture("stop"))
    monkeypatch.setattr(worker, "GenerationBinding", lambda *a: None)
    monkeypatch.setattr(worker, "base_state", lambda *a: {})
    monkeypatch.setattr(worker, "Qwen3CausalLMReranker", lambda *a, **kw: None)
    monkeypatch.setattr(worker, "TargetedRerank", lambda backend, ranker, directory, **kw: FixtureRerank(directory))
    release = {"purpose": fair.PROTOCOL, "authorization": "validated_worker_projection", "source_git": "a" * 40,
               "source_archive_sha256": "b" * 64, "run_id": "synthetic-fair-entry", "output": str(out),
               "generation_contract": {}, "fair_binding": binding()}
    worker.worker_body(release, inputs)
    run = json.loads((out / "inference/run.json").read_bytes())
    assert [r["route"] for r in run["runs"]] == list(fair.ROUTES)
    assert all(r["generation_attempts"][-1]["stage"] == "verdict" for r in run["runs"])
    release.pop("source_git")
    with pytest.raises(ValueError, match="unbound_run"):
        worker.worker_body(release, inputs)
