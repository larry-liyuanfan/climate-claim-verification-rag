"""Coverage v2 integration is SYNTHETIC, not evidence of real model behavior."""
import copy
import json

import pytest

from climate_rag import coverage_acquisition as cov, fair_acquisition as fair
from climate_rag.agent_v3 import SentenceAgentV3
from climate_rag.fair_replay import REGRESSION_STUDY
from climate_rag.targeted_replay import run_matrix
from climate_rag.targeted_score import audit
from test_fair_acquisition import TASKS, DOCS, FairFixture, binding
from test_targeted_replay import QUERY, FixtureRerank


def assessed(decision, observation):
    good = [s["sentence_id"] for s in observation["current_citable"] if "refute" in s["text"]]
    return {"coverage": [{"claim_span": "have advanced", "kind": "relation",
        "status": "covered" if good else "missing", "sentence_ids": good[:1]}],
        "decision": decision, "stop_reason": "not_stopping" if decision["action"] != "stop" else "no_useful_action"}


class AssessedFixture(FairFixture):
    def generate(self, obs, schema, max_output_tokens, remaining_seconds):
        result = super().generate(obs, schema, max_output_tokens, remaining_seconds)
        if obs["phase"] == "gate":
            result["raw"] = json.dumps(assessed(json.loads(result["raw"]), obs))
        return result


def run(tmp_path, empty=False, mode="feedback"):
    def retrieve(query, width):
        if query in (QUERY["query"], "glacier coastal retreat study"):
            return [] if empty else [DOCS["new"]]
        return list(DOCS.values())[:8]
    bound = binding()
    bound.update(protocol=cov.PROTOCOL, study_kind=REGRESSION_STUDY)
    class Ranker(FixtureRerank):
        def __call__(self, query, rows):
            values = super().__call__(query, rows)
            return list(reversed(values)) if mode == "rerank" else values
    result = run_matrix(TASKS, AssessedFixture(mode), retrieve,
        Ranker(tmp_path / "ranker"), tmp_path / "inference", protocol=cov.PROTOCOL, binding=bound)
    audit(result, TASKS, DOCS, tmp_path / "inference/ledger", synthetic=True,
        reranker_directory=tmp_path / "ranker")
    return result


def test_worker_ledger_audit_shared_contract_actual_feedback(tmp_path):
    (tmp_path / "good").mkdir()
    (tmp_path / "empty").mkdir()
    good, empty = run(tmp_path / "good"), run(tmp_path / "empty", empty=True)
    assert all(r["initial_frame"] == good["runs"][0]["initial_frame"] for r in good["runs"])
    a, b = good["runs"][-1], empty["runs"][-1]
    assert a["generation_attempts"][0]["decision"] == b["generation_attempts"][0]["decision"]
    assert a["generation_attempts"][1]["decision"] == {"action": "stop"}
    assert b["generation_attempts"][1]["decision"]["tool"] == "query"
    assert a["generation_attempts"][1]["proposed_decision"]["coverage"][0]["status"] == "covered"
    assert a["answer"]["label"] == "REFUTES" and b["answer"] is None
    assert all(r["physical_cost"]["unique_physical_calls"] == r["model_calls"] <= 5 for r in good["runs"])


def test_stop_not_forced_to_acquire_and_rerank_delivery(tmp_path):
    (tmp_path / "stop").mkdir()
    (tmp_path / "rank").mkdir()
    stopped = run(tmp_path / "stop", mode="stop")["runs"][-1]
    ranked = run(tmp_path / "rank", mode="rerank")["runs"][-1]
    assert stopped["model_calls"] == 2 and stopped["tool_calls"] == 1
    event = next(e for e in ranked["events"] if e.get("tool") == "rerank")
    assert event["delivery"]["new_sentence_sha256"]
    assert 1 in ranked["generation_attempts"][1]["received_tool_events"]
    assert cov.system_prompt("verdict") == fair.system_prompt("verdict")
    assert cov.system_prompt("plan") == fair.system_prompt("plan")


@pytest.mark.parametrize("fault", ["span", "preview", "false_covered", "stop", "extra"])
def test_claim_gap_and_sentence_contract_refuses(fault):
    obs = {"immutable_claim": TASKS[0]["claim_text"], "current_citable": [],
        "acquisition_available": True, "query_available": True, "rerank_available": True,
        "readable_source_ids": [], "prior_queries": []}
    value = assessed({"action": "stop"}, obs)
    if fault == "span":
        value["coverage"][0]["claim_span"] = "a different claim"
    elif fault == "preview":
        value["coverage"][0]["sentence_ids"] = ["preview:0"]
    elif fault == "false_covered":
        value["coverage"][0]["status"] = "covered"
    elif fault == "stop":
        value["stop_reason"] = "not_stopping"
    else:
        value["gold"] = "REFUTES"
    with pytest.raises(ValueError):
        cov.parse(value, obs, fully_shown=[])


def test_version_binding_refuses_mismatch_before_output(tmp_path):
    with pytest.raises(ValueError, match="protocol_mismatch"):
        run_matrix(TASKS, None, None, None, tmp_path / "never", protocol=cov.PROTOCOL, binding=binding())
    assert not (tmp_path / "never").exists()
    changed = copy.deepcopy(binding())
    changed.update(protocol=cov.PROTOCOL, study_kind=REGRESSION_STUDY)
    # Constructor runs the actual renderer/controller path, not only schemas.
    assert SentenceAgentV3(AssessedFixture("stop"), lambda q, k: [], rerank=lambda q, r: r,
                           protocol=cov.PROTOCOL).run(TASKS[0]["claim_text"], "autonomous")["protocol"] == cov.PROTOCOL


def test_production_loader_refuses_other_consumed_cohort_before_read(tmp_path):
    from run_targeted_replay import load_inputs
    bound = binding()
    bound.update(protocol=cov.PROTOCOL, study_kind=REGRESSION_STUDY)
    with pytest.raises(ValueError, match="coverage_frozen_consumed_cohort_required"):
        load_inputs(tmp_path, {"purpose": cov.PROTOCOL, "fair_binding": bound})


def test_frozen_raw_bytes_and_canonical_tasks_have_separate_hashes():
    accepted = {"cohort_sha256": cov.COHORT_SHA, "tasks_sha256": cov.TASKS_SHA, "task_count": 32}
    cov.validate_frozen_cohort(accepted)
    for key in accepted:
        value = dict(accepted)
        value[key] = 31 if key == "task_count" else "a" * 64
        with pytest.raises(ValueError, match="frozen_consumed_cohort"):
            cov.validate_frozen_cohort(value)


def test_audit_and_score_refuse_different_execution_protocol(tmp_path):
    from climate_rag.fair_replay import score
    result = run(tmp_path)
    result["fair_binding"] = binding()
    with pytest.raises(ValueError, match="protocol_mismatch"):
        audit(result, TASKS, DOCS, tmp_path / "inference/ledger", synthetic=True,
              reranker_directory=tmp_path / "ranker")
    with pytest.raises(ValueError, match="protocol_mismatch"):
        score(result, {})


def test_coverage_only_tamper_without_changing_action_refused(tmp_path):
    result = run(tmp_path, mode="stop")
    result["runs"][-1]["generation_attempts"][0]["proposed_decision"]["coverage"][0]["kind"] = "time"
    with pytest.raises(ValueError):
        audit(result, TASKS, DOCS, tmp_path / "inference/ledger", synthetic=True,
              reranker_directory=tmp_path / "ranker")


def test_production_worker_v2_with_injected_model_boundaries(tmp_path, monkeypatch):
    import run_targeted_replay as worker
    from climate_rag.models import EvidenceDocument
    out, inputs = tmp_path / "run", tmp_path / "input"
    out.mkdir()
    for kind in ("generator", "reranker"):
        directory = inputs / "models" / kind
        directory.mkdir(parents=True)
        (directory / "model_manifest.json").write_bytes(json.dumps({"kind": kind}).encode())
    monkeypatch.setattr(worker, "load_inputs", lambda *a: (TASKS, [EvidenceDocument(s.source_id, s.sentences[0]) for s in DOCS.values()], DOCS))
    monkeypatch.setattr(worker, "sha", lambda raw: worker.MANIFEST_BYTES[json.loads(raw)["kind"]])
    monkeypatch.setattr(worker, "verify_model_files", lambda path, manifest: worker.MODEL_SHA if manifest["kind"] == "generator" else worker.RERANKER_SHA)
    monkeypatch.setattr(worker, "LocalTargetedProvider", lambda *a, **kw: AssessedFixture("stop"))
    monkeypatch.setattr(worker, "GenerationBinding", lambda *a: None)
    monkeypatch.setattr(worker, "base_state", lambda *a: {})
    monkeypatch.setattr(worker, "Qwen3CausalLMReranker", lambda *a, **kw: None)
    monkeypatch.setattr(worker, "TargetedRerank", lambda backend, ranker, directory, **kw: FixtureRerank(directory))
    bound = binding()
    bound.update(protocol=cov.PROTOCOL, study_kind=REGRESSION_STUDY)
    worker.worker_body({"purpose": cov.PROTOCOL, "authorization": "validated_worker_projection", "source_git": "a" * 40,
        "source_archive_sha256": "b" * 64, "run_id": "synthetic-coverage-entry", "output": str(out),
        "generation_contract": {}, "fair_binding": bound}, inputs)
    result = json.loads((out / "inference/run.json").read_bytes())
    assert result["protocol"] == cov.PROTOCOL
    assert [r["route"] for r in result["runs"]] == list(fair.ROUTES)
    assert result["runs"][-1]["generation_attempts"][0]["proposed_decision"]["coverage"]
    assert all(r["generation_attempts"][-1]["stage"] == "verdict" for r in result["runs"])
