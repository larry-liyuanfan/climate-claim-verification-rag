"""Synthetic protocol/gold fixtures only; no real evaluation predictions are opened."""
import copy
import hashlib
import importlib.util
import json
import sys
from pathlib import Path

import pytest


spec = importlib.util.spec_from_file_location(
    "agent_scoring", Path(__file__).resolve().parents[1] / "scripts/score_budget_agent.py"
)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def packed(value):
    return json.dumps(value, sort_keys=True).encode()


def digest(value):
    return hashlib.sha256(value).hexdigest()


def fixtures(phase="validation", count=2, evidence_count=1, label_count=1):
    tasks = [{"id": f"synthetic-{i}", "claim_text": f"Fixture claim {i}"} for i in range(count)]
    protocol = {"corpus_sha256": "synthetic-corpus", "study_kind": "synthetic-fixture-only",
                "strategies": list(module.STRATEGIES), phase: tasks,
                "official_gold_available": phase == "validation"}
    protocol_bytes = packed(protocol)
    protocol_sha = digest(protocol_bytes)
    run = {"protocol_sha256": protocol_sha, "corpus_sha256": protocol["corpus_sha256"],
           "study_kind": protocol["study_kind"], "phase": phase, "code_sha": "inference-fixture",
           "runs": []}
    for strategy in module.STRATEGIES:
        for task in tasks:
            run["runs"].append({
                "strategy": strategy, "task_id": task["id"], "claim_text": task["claim_text"],
                "provider_kind": "local_model", "provider": "synthetic-local-provider",
                "answer": None, "status": "abstained", "reason": "deadline_exceeded",
                "events": [{"stage": "failure", "error_type": "TimeoutError"}],
                "candidate_evidence_ids": ["fixture-evidence"], "delivered_evidence_ids": [],
                "tool_calls": 2, "model_calls": 1, "generation_calls": 1,
                "retrieval_calls": 1, "rerank_calls": 1, "rerank_candidate_pairs": 20,
                "elapsed_ms": 20.0, "usage_known": False,
                "usage": {"input_tokens": 10, "output_tokens": 2},
            })
    gold = {"protocol_sha256": protocol_sha,
            "provenance": {"annotation_kind": "official_CLIMATE_FEVER_evidence_and_label",
                           "split": "repeated_validation_replay", "archive_sha256": "fixture-archive",
                           "member_sha256": "fixture-member"},
            "claims": {x["id"]: {"claim_sha256": digest(x["claim_text"].encode()),
                                  "evidence_ids": ["fixture-evidence"] if i < evidence_count else [],
                                  "label": "SUPPORTS" if i < label_count else "NOT_ENOUGH_INFO"}
                       for i, x in enumerate(tasks)}}
    gold_bytes = packed(gold)
    selection = {"protocol_sha256": protocol_sha, "gold_sha256": digest(gold_bytes),
                 "source_archive_sha256": "fixture-archive", "member_sha256": "fixture-member",
                 "selected_ids": [x["id"] for x in tasks], "retrieval_denominator": evidence_count,
                 "selected_decisive": evidence_count, "selected_nondecisive": count - evidence_count,
                 "call_latency_denominator_per_strategy": count}
    kwargs = {"phase": phase, "expected_protocol_sha256": protocol_sha}
    if phase == "validation":
        kwargs.update(gold_bytes=gold_bytes, selection=selection)
    return run, protocol_bytes, kwargs


def test_complete_96_slot_fixture_keeps_32_calls_24_retrieval_and_actual_label_denominator():
    run, protocol, kwargs = fixtures(count=32, evidence_count=24, label_count=20)
    before = copy.deepcopy(run)
    scored = module.score(run, protocol, **kwargs)
    assert run == before
    assert scored["matrix"]["expected_slots"] == scored["matrix"]["actual_slots"] == 96
    for value in scored["aggregates"].values():
        assert (value["count"], value["retrieval_denominator"], value["label_denominator"]) == (32, 24, 20)
        assert value["evidence_metrics"]["recall@5"] == 0  # timed-out retrieval earns no credit
        assert value["official_decisive_label_accuracy"] == 0
    pair = scored["paired_bootstrap"]["adaptive"]["recall@5"]
    assert pair["pair_count"] == 24 and pair["samples"] == 5000
    assert scored["semantic_supportability"] is None


@pytest.mark.parametrize("defect", ["common_missing_gold", "common_missing_nongold", "duplicate",
                                   "extra_task", "missing_route", "unknown_route", "empty"])
def test_full_matrix_rejects_shared_missing_nongold_and_every_slot_defect(defect):
    run, protocol, kwargs = fixtures()
    if defect.startswith("common_missing"):
        missing = "synthetic-0" if defect.endswith("_gold") else "synthetic-1"
        run["runs"] = [x for x in run["runs"] if x["task_id"] != missing]
    elif defect == "duplicate":
        run["runs"].append(copy.deepcopy(run["runs"][0]))
    elif defect == "extra_task":
        run["runs"][0]["task_id"] = "extra"
    elif defect == "missing_route":
        run["runs"].pop()
    elif defect == "unknown_route":
        run["runs"][0]["strategy"] = "new-route"
    else:
        run["runs"] = []
    with pytest.raises(ValueError, match="matrix"):
        module.score(run, protocol, **kwargs)


@pytest.mark.parametrize("field,value", [
    ("phase", "vnext"), ("protocol_sha256", "wrong"), ("corpus_sha256", "wrong"),
    ("study_kind", "wrong"),
])
def test_run_identity_mismatch_is_rejected(field, value):
    run, protocol, kwargs = fixtures()
    run[field] = value
    with pytest.raises(ValueError, match="mismatch"):
        module.score(run, protocol, **kwargs)


def test_protocol_bytes_claim_text_and_duplicate_task_are_rejected():
    run, protocol, kwargs = fixtures("vnext")
    with pytest.raises(ValueError, match="protocol SHA"):
        module.score(run, protocol + b" ", **kwargs)
    run["runs"][0]["claim_text"] += " modified"
    with pytest.raises(ValueError, match="claim mismatch"):
        module.score(run, protocol, **kwargs)
    run, protocol, kwargs = fixtures("vnext")
    value = json.loads(protocol)
    value["vnext"].append(value["vnext"][0])
    changed = packed(value)
    kwargs["expected_protocol_sha256"] = run["protocol_sha256"] = digest(changed)
    with pytest.raises(ValueError, match="duplicate protocol"):
        module.score(run, changed, **kwargs)


def test_wrong_gold_file_rejected_even_when_protocol_hash_matches():
    run, protocol, kwargs = fixtures()
    gold = json.loads(kwargs["gold_bytes"])
    gold["claims"]["synthetic-0"]["evidence_ids"] = ["different"]
    kwargs["gold_bytes"] = packed(gold)
    with pytest.raises(ValueError, match="gold SHA"):
        module.score(run, protocol, **kwargs)


@pytest.mark.parametrize("defect", ["protocol", "claim", "extra", "missing", "provenance", "denominator"])
def test_gold_content_and_selection_cross_checks(defect):
    run, protocol, kwargs = fixtures()
    gold = json.loads(kwargs["gold_bytes"])
    if defect == "protocol":
        gold["protocol_sha256"] = "wrong"
    elif defect == "claim":
        gold["claims"]["synthetic-0"]["claim_sha256"] = "wrong"
    elif defect == "extra":
        gold["claims"]["extra"] = gold["claims"]["synthetic-0"]
    elif defect == "missing":
        del gold["claims"]["synthetic-1"]
    elif defect == "provenance":
        gold["provenance"]["member_sha256"] = "wrong"
    else:
        kwargs["selection"]["retrieval_denominator"] = 2
    kwargs["gold_bytes"] = packed(gold)
    # Synthetic selection setup isolates each additional guard after byte-identity checks.
    kwargs["selection"]["gold_sha256"] = digest(kwargs["gold_bytes"])
    with pytest.raises(ValueError, match="mismatch"):
        module.score(run, protocol, **kwargs)


def test_missing_validation_gold_or_selection_rejected():
    run, protocol, kwargs = fixtures()
    for missing in ("gold_bytes", "selection"):
        modified = dict(kwargs)
        modified.pop(missing)
        with pytest.raises(ValueError, match="validation requires"):
            module.score(run, protocol, **modified)


def test_24_slot_authored_fixture_has_null_official_quality_and_rejects_injected_gold():
    run, protocol, kwargs = fixtures("vnext", count=8)
    score = module.score(run, protocol, **kwargs)
    assert score["matrix"]["expected_slots"] == 24
    assert score["gold_source"] is score["paired_bootstrap"] is score["semantic_supportability"] is None
    for route in score["aggregates"].values():
        assert route["evidence_metrics"] is route["official_decisive_label_accuracy"] is None
        assert route["retrieval_denominator"] is route["label_denominator"] is None
    with pytest.raises(ValueError, match="authored phase forbids"):
        module.score(run, protocol, gold_bytes=b"{}", **kwargs)


def test_heuristic_validation_never_scores_unconfigured_classifier():
    run, protocol, kwargs = fixtures()
    for row in run["runs"]:
        row["provider_kind"] = "heuristic_no_model"
        row["model_calls"] = row["generation_calls"] = 0
    score = module.score(run, protocol, **kwargs)
    for route in score["aggregates"].values():
        assert route["label_denominator"] == 0
        assert route["official_decisive_label_accuracy"] is None


def test_partial_usage_tokens_and_separate_work_survive_with_no_cost_comparison():
    run, protocol, kwargs = fixtures("vnext")
    for row in run["runs"]:
        if row["task_id"] == "synthetic-1":
            row["usage_known"] = True
            row["usage"] = {"input_tokens": 20, "output_tokens": 3}
    score = module.score(run, protocol, **kwargs)
    for route in score["aggregates"].values():
        assert route["recorded_generation_input_tokens"] == 30
        assert route["recorded_generation_output_tokens"] == 5
        assert route["complete_accounting_generation_input_tokens"] == 20
        assert route["partial_accounting_recorded_input_tokens"] == 10
        assert route["unknown_token_accounting_runs"] == 1
        assert not route["generation_token_totals_exact"]
        assert route["retrieval_calls_sum"] == route["rerank_calls_sum"] == 2
        assert route["rerank_candidate_pairs_sum"] == 40
        assert route["generation_calls_sum"] == 2 and route["tool_calls_sum"] == 4
    for pair in score["cost_comparison"].values():
        assert not pair["generation_token_comparison_available"]
        assert pair["unavailable_reason"] == "unknown_token_accounting"
        assert not pair["monetary_cost_comparison_available"]


def test_comparison_only_unavailable_for_pairs_with_unknown_usage():
    run, protocol, kwargs = fixtures("vnext")
    for row in run["runs"]:
        row["usage_known"] = True
    run["runs"][-1]["usage_known"] = False
    score = module.score(run, protocol, **kwargs)
    assert score["cost_comparison"]["fixed_rerank"]["generation_token_comparison_available"]
    assert not score["cost_comparison"]["adaptive"]["generation_token_comparison_available"]


@pytest.mark.parametrize("case,expected", [
    ("model_abstain", "model_requested_abstention"),
    ("model_reason_matches_failure_word", "model_requested_abstention"),
    ("heuristic_abstain", "heuristic_control_abstention"),
    ("schema_failure", "controller_failure"),
    ("deadline", "controller_failure"),
    ("quote_rejected", "controller_rejection"),
    ("disallowed", "controller_rejection"),
    ("answer", "mechanically_validated_answer_semantics_unverified"),
])
def test_status_reason_are_not_collapsed_into_model_abstention(case, expected):
    run, protocol, kwargs = fixtures("vnext")
    row = run["runs"][0]
    if case in {"model_abstain", "model_reason_matches_failure_word", "heuristic_abstain"}:
        row["reason"] = "stage_failed" if case == "model_reason_matches_failure_word" else "Fixture abstention"
        row["events"] = [{"stage": "decision", "decision": {"action": "abstain", "reason": row["reason"]}}]
        if case == "heuristic_abstain":
            row["provider_kind"] = "heuristic_no_model"
    elif case == "schema_failure":
        row["reason"] = "stage_failed"
        row["events"] = [{"stage": "failure", "error_type": "GeneratedResponseError"}]
    elif case == "quote_rejected":
        row["reason"] = "answer_validation:quote_not_exact"
        row["events"] = [{"stage": "decision", "decision": {"action": "answer"}}]
    elif case == "disallowed":
        row["reason"] = "disallowed_action"
        row["events"] = [{"stage": "decision", "decision": {"action": "rewrite"}}]
    elif case == "answer":
        row.update(status="answered", answer={"label": "SUPPORTS"},
                   reason="source_and_numeric_checks_passed_semantics_unverified", events=[])
    result = module.score(run, protocol, **kwargs)["aggregates"]["fixed_retrieval"]
    assert result["outcome_counts"][expected] >= 1
    public_reason = expected if expected.endswith("abstention") else row["reason"]
    assert public_reason in result["reason_counts"]
    assert "Fixture abstention" not in json.dumps(result)
    assert row["status"] in result["status_counts"]
    assert result["operational_failures"] == (2 if expected == "controller_failure" else 1)


@pytest.mark.parametrize("field,value", [("status", "unknown"), ("reason", "unexplained"),
                                       ("usage_known", "false"), ("elapsed_ms", float("nan")),
                                       ("generation_calls", -1), ("provider_kind", "unconfigured")])
def test_invalid_status_or_accounting_fails_closed(field, value):
    run, protocol, kwargs = fixtures("vnext")
    run["runs"][0][field] = value
    with pytest.raises(ValueError):
        module.score(run, protocol, **kwargs)


def test_free_text_disguised_as_controller_error_is_not_exported():
    run, protocol, kwargs = fixtures("vnext")
    run["runs"][0].update(reason="answer_validation:private content", events=[])
    with pytest.raises(ValueError, match="unrecognized answer-validation"):
        module.score(run, protocol, **kwargs)


def test_production_selection_loader_locks_existing_manifest_identity_and_counts(tmp_path, monkeypatch):
    selection = module.load_selection_manifest()  # metadata only; no prediction/gold read
    assert len(selection["selected_ids"]) == 32
    assert selection["retrieval_denominator"] == 24 and selection["selected_nondecisive"] == 8
    altered = copy.deepcopy(selection)
    altered["selected_decisive"] = altered["retrieval_denominator"] = 25
    altered["selected_nondecisive"] = 7
    path = tmp_path / "altered-selection.json"
    path.write_bytes(packed(altered))
    monkeypatch.setattr(module, "SELECTION_PATH", path)
    with pytest.raises(ValueError, match="selection manifest SHA"):
        module.load_selection_manifest()
    # Isolate the fixed cardinality gate even if a future edit changed the hash constant.
    monkeypatch.setattr(module, "SELECTION_SHA", digest(path.read_bytes()))
    with pytest.raises(ValueError, match="32/24/8"):
        module.load_selection_manifest()


def test_cli_reuses_frozen_selection_and_records_separate_scorer_identity(tmp_path, monkeypatch):
    run, protocol, kwargs = fixtures()
    files = {"run": packed(run), "protocol": protocol, "gold": kwargs["gold_bytes"],
             "selection": packed(kwargs["selection"])}
    for name, raw in files.items():
        (tmp_path / f"{name}.json").write_bytes(raw)
    monkeypatch.setattr(module, "SELECTION_PATH", tmp_path / "selection.json")
    monkeypatch.setattr(module, "load_selection_manifest", lambda: kwargs["selection"])
    monkeypatch.setattr(module, "scorer_identity", lambda: {"git_sha": "scorer-fixture"})
    output = tmp_path / "score.json"
    args = ["score_budget_agent.py", "--run", str(tmp_path / "run.json"),
            "--protocol", str(tmp_path / "protocol.json"), "--expected-protocol-sha256",
            kwargs["expected_protocol_sha256"], "--phase", "validation",
            "--gold", str(tmp_path / "gold.json"), "--output", str(output)]
    monkeypatch.setattr(sys, "argv", args)
    assert module.main() == 0
    result = json.loads(output.read_text())
    assert result["inference_source_git_sha"] == "inference-fixture"
    assert result["scorer_identity"]["git_sha"] == "scorer-fixture"
    assert result["selection_manifest_sha256"] == digest(files["selection"])
    assert result["gold_sha256"] == digest(files["gold"])
    with pytest.raises(ValueError, match="overwrite"):
        module.main()
