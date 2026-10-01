"""Affected complete chain; all claims, corpus and gold here are synthetic."""

import copy
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from climate_rag.agent_v3 import Source
from climate_rag.targeted_query import PROTOCOL, ROUTES, planning_schema
from climate_rag.targeted_replay import run_matrix, sha, policy
from climate_rag.targeted_score import audit, score
from climate_rag.scifact_read_continuation import ordered_write
import run_targeted_replay_operator as operator
from run_targeted_replay import TargetedRerank

TASKS = [
    {"id": "fixture1", "claim_text": "Example glaciers have advanced."},
    {"id": "fixture2", "claim_text": "Unanswerable example."},
]
DOCS = {
    "old": Source("old", "fixture", ("Example glaciers background only.",)),
    "new": Source(
        "new",
        "fixture",
        ("Ice cores refute glacier advance in this SYNTHETIC example.",),
    ),
    "noise": Source(
        "noise",
        "fixture",
        ("Ice cores are a dessert in this irrelevant SYNTHETIC text.",),
    ),
}
QUERY = {"purpose": "counter_evidence", "query": "ice cores retreat measurements"}
ABSTAIN = {"action": "abstain", "reason": "insufficient_evidence"}


class FixtureProvider:
    name, kind, gap, terminal_protocol = (
        "synthetic-script-not-LLM",
        "fixture",
        False,
        PROTOCOL,
    )

    def __init__(self, *, failure=False):
        self.base = SimpleNamespace(
            tokenizer=SimpleNamespace(encode=lambda text, **kw: list(text))
        )
        self.failure = failure

    def start_slot(self, path):
        path.mkdir()
        self.calls = 0

    def render(self, observation, schema):
        return json.dumps([observation, schema], sort_keys=True)

    def count_prompt(self, observation, schema):
        return len(self.render(observation, schema))

    def count_text(self, text):
        return len(text)

    def generate(self, observation, schema, max_output_tokens, remaining_seconds):
        self.calls += 1
        if self.failure and "rewrite" in observation["allowed_actions"]:
            raise RuntimeError("synthetic physical failure")
        visible = observation["current_citable"]
        if observation["allowed_actions"] == ["plan_queries"]:
            action = {"action": "plan_queries", "queries": [QUERY]}
        elif self.calls == 1 and "rewrite" in observation["allowed_actions"]:
            action = {"action": "rewrite", **QUERY}
        else:
            good = [s for s in visible if "refute" in s["text"]]
            action = (
                {
                    "action": "answer",
                    "label": "REFUTES",
                    "sentence_ids": [good[0]["sentence_id"]],
                }
                if good
                else ABSTAIN
            )
        return {
            "raw": json.dumps(action),
            "usage": {
                "input_tokens": self.count_prompt(observation, schema),
                "output_tokens": 32,
            },
        }


class FixtureRerank(TargetedRerank):
    def __init__(self, directory):
        self.directory = directory
        directory.mkdir()

    def __call__(self, query, rows):
        key = f"r{len(list(self.directory.glob('r*.reserved.json'))):02d}"
        ordered_write(
            self.directory / (key + ".reserved.json"),
            {**self.request_context, "requested_pairs": len(rows), "synthetic": True},
        )
        ordered_write(
            self.directory / (key + ".finished.json"),
            {
                "requested_pairs": len(rows),
                "completed_pairs": len(rows),
                "observed_nonpadding_tokens": len(rows) * 100,
                "elapsed_ms_including_swaps": 1.0,
                "batches": [{"status": "completed"}],
            },
        )
        return rows


def synthetic(tmp_path, *, failure=False, noise=False):
    observed = []

    def retrieve(query, k):
        observed.append(query)
        if query == QUERY["query"]:
            return [DOCS["noise" if noise else "new"]]
        return [DOCS["old"]]

    run = run_matrix(
        TASKS,
        FixtureProvider(failure=failure),
        retrieve,
        FixtureRerank(tmp_path / "reranker-ledger"),
        tmp_path / "inference",
    )
    gold = {
        "claims": {
            "fixture1": {
                "claim_sha256": sha(TASKS[0]["claim_text"].encode()),
                "evidence_ids": ["new"],
                "label": "REFUTES",
            },
            "fixture2": {
                "claim_sha256": sha(TASKS[1]["claim_text"].encode()),
                "evidence_ids": [],
                "label": "NOT_ENOUGH_INFO",
            },
        }
    }
    audit(
        run,
        TASKS,
        DOCS,
        tmp_path / "inference/ledger",
        synthetic=True,
        reranker_directory=tmp_path / "reranker-ledger",
    )
    return run, gold, observed


def test_actual_matrix_feedback_citation_scoring_cost_and_compact(tmp_path):
    run, gold, queries = synthetic(tmp_path)
    assert len(run["runs"]) == 10
    assert QUERY["query"] in queries
    result = score(run, gold, synthetic=True)
    assert result["study_kind"] == "synthetic_fixture_not_quality_evidence"
    assert result["routes"]["adaptive"]["official_decisive_label_accuracy"] == 1
    assert result["routes"]["fixed_rerank"]["official_decisive_label_accuracy"] == 0
    assert result["routes"]["fixed_multiquery"]["official_decisive_label_accuracy"] == 1
    # Strong fixed multiquery solves it too: no unsupported adaptive advantage.
    assert result["quality_gate"] is False and result["resume_promotion"] is False
    for route in ROUTES:
        assert result["routes"][route]["slots"] == 2
        assert result["routes"][route]["retrieval_denominator"] == 1
        assert result["routes"][route]["label_denominator"] == 1
    assert (
        result["comparisons"]["fixed_multiquery"]["paired_bootstrap_5000"][
            "evidence_f1"
        ]["samples"]
        == 5000
    )
    ordered_write(tmp_path / "compact.json", result)
    assert "Ice cores" not in (tmp_path / "compact.json").read_text()


def test_failure_keeps_denominator_unknown_cost_and_no_citations(tmp_path):
    run, gold, _ = synthetic(tmp_path, failure=True)
    result = score(run, gold, synthetic=True)
    row = result["routes"]["adaptive"]
    assert row["slots"] == 2 and row["physical_calls"] == 2
    assert (
        row["unknown_usage_attempts"] == 2
        and row["mean_total_generator_tokens"] is None
    )
    assert (
        row["operational_failures"] == 2
        and row["official_decisive_label_accuracy"] == 0
    )
    assert not result["comparisons"]["fixed_rerank"]["cost_gate"]


def test_new_id_is_not_automatically_relevant_or_true(tmp_path):
    run, gold, _ = synthetic(tmp_path, noise=True)
    assert all(r["answer"] is None for r in run["runs"])
    result = score(run, gold, synthetic=True)
    assert result["routes"]["adaptive"]["evidence_metrics"]["recall@5"] == 0


@pytest.mark.parametrize(
    "mutation",
    [
        "drop",
        "claim",
        "quote",
        "physical",
        "rank",
        "provider",
        "label",
        "id_missing",
        "id_reuse",
        "rank_order",
    ],
)
def test_complete_chain_audit_rejects_identity_or_denominator_drift(tmp_path, mutation):
    run, _, _ = synthetic(tmp_path)
    if mutation == "drop":
        run["runs"].pop()
    elif mutation == "claim":
        run["runs"][0]["immutable_claim_sha256"] = "0" * 64
    elif mutation == "quote":
        next(r for r in run["runs"] if r["answer"])["answer"]["citations"][0][
            "text"
        ] = "fabricated"
    elif mutation == "physical":
        run["runs"][0]["physical_cost"]["unique_physical_calls"] += 1
    elif mutation == "rank":
        run["runs"][0]["delivered_evidence_ids"] = ["unknown"]
    elif mutation == "label":
        row = next(r for r in run["runs"] if r["answer"])
        row["answer"]["label"] = "SUPPORTS"
        row["generation_attempts"][-1]["decision"]["label"] = "SUPPORTS"
    elif mutation == "id_missing":
        run["runs"][0]["generation_attempts"][0]["diagnostics"].pop(
            "physical_attempt_id"
        )
    elif mutation == "id_reuse":
        row = next(r for r in run["runs"] if len(r["generation_attempts"]) > 1)
        row["generation_attempts"][1]["diagnostics"]["physical_attempt_id"] = row[
            "generation_attempts"
        ][0]["diagnostics"]["physical_attempt_id"]
    elif mutation == "rank_order":
        row = next(r for r in run["runs"] if len(r["delivered_evidence_ids"]) > 1)
        row["delivered_evidence_ids"].reverse()
    if mutation == "provider":
        with pytest.raises(ValueError, match="unconfigured"):
            audit(run, TASKS, DOCS, tmp_path / "inference/ledger")
    else:
        with pytest.raises(ValueError):
            audit(run, TASKS, DOCS, tmp_path / "inference/ledger", synthetic=True)


def test_missing_rerank_receipt_blocks_efficiency_not_free_compute(tmp_path):
    run, gold, _ = synthetic(tmp_path)
    row = next(r for r in run["runs"] if r["route"] == "fixed_rerank")
    # Simulate interruption/unknown measurement without deleting a file.
    row["reranker_cost"]["unknown_requests"] = 1
    row["reranker_cost"]["nonpadding_tokens"] = None
    with pytest.raises(ValueError, match="rerank_slot_accounting"):
        audit(
            run,
            TASKS,
            DOCS,
            tmp_path / "inference/ledger",
            synthetic=True,
            reranker_directory=tmp_path / "reranker-ledger",
        )
    result = score(run, gold, synthetic=True)
    assert not result["routes"]["fixed_rerank"]["reranker_accounting_complete"]
    assert not result["comparisons"]["fixed_rerank"]["cost_gate"]


def test_rank_and_event_dual_change_cannot_replace_physical_feedback(tmp_path):
    run, _, _ = synthetic(tmp_path)
    row = next(r for r in run["runs"] if len(r["delivered_evidence_ids"]) > 1)
    row["delivered_evidence_ids"].reverse()
    event = next(e for e in reversed(row["events"]) if e.get("status") == "completed")
    event["candidate_ids"] = list(reversed(event["candidate_ids"]))
    with pytest.raises(ValueError, match="physical_feedback"):
        audit(run, TASKS, DOCS, tmp_path / "inference/ledger", synthetic=True)


@pytest.mark.parametrize(
    "exhausted", ["validation_repair_exhausted", "generation_budget_exhausted"]
)
def test_nondecisive_exhaustion_blocks_false_efficiency_with_full32_denominator(
    tmp_path, exhausted
):
    # Pure scorer fixture, NOT a fabricated model run: quality deliberately wins
    # on all24 decisive items so only the operational-failure gate can catch this.
    run, gold, _ = synthetic(tmp_path)
    template = run["runs"][0]
    rows = []
    labels = {}
    for i in range(32):
        key = f"synthetic-{i}"
        labels[key] = {
            "claim_sha256": template["immutable_claim_sha256"],
            "evidence_ids": ["new"] if i < 24 else [],
            "label": "REFUTES" if i < 24 else "NOT_ENOUGH_INFO",
        }
        for route in ROUTES:
            row = copy.deepcopy(template)
            row.update(
                task_id=key,
                route=route,
                elapsed_ms=1,
                tool_calls=1,
                rerank_pairs=0,
                delivered_evidence_ids=["new"]
                if route == "adaptive" and i < 24
                else [],
                answer={"label": "REFUTES", "citations": [{"source_id": "new"}]}
                if route == "adaptive" and i < 24
                else None,
                outcome="ids_validated_semantics_unmeasured"
                if route == "adaptive" and i < 24
                else "model_abstention:budget",
            )
            if route == "adaptive" and i >= 24:
                row["outcome"] = exhausted
            rows.append(row)
    result = score({"runs": rows}, {"claims": labels}, synthetic=True)
    assert result["quality_gate"] is True
    for control in ("fixed_rerank", "fixed_multiquery"):
        assert result["comparisons"][control]["cost_gate"] is False
    adaptive = result["routes"]["adaptive"]
    assert (
        adaptive["slots"] == 32
        and adaptive["label_denominator"] == 24
        and adaptive["retrieval_denominator"] == 24
    )
    assert adaptive["operational_failures"] == 8
    assert adaptive["terminal_categories"]["execution_exhausted"] == 8
    assert adaptive["physical_calls"] == 32  # previously consumed work retained
    assert result["efficiency_gate"] is False


def test_real_decoder_parses_new_upfront_array_schema_without_model():
    from lmformatenforcer import JsonSchemaParser
    from climate_rag.local_agent_v3 import fixed_grammar_config

    text = json.dumps(
        {"action": "plan_queries", "queries": [QUERY]}, separators=(",", ":")
    )
    parser = JsonSchemaParser(planning_schema(), config=fixed_grammar_config())
    for char in text:
        assert char in parser.get_allowed_characters()
        parser = parser.add_character(char)
    assert parser.can_end()


def test_draft_stops_before_any_output_preparation_or_model(tmp_path, monkeypatch):
    monkeypatch.setattr(operator, "ROOT", tmp_path)
    value = operator.draft("a" * 40, "b" * 64, "c" * 64)
    assert value["policy"] == policy()

    def never(*args, **kwargs):
        raise AssertionError("must never execute")

    with pytest.raises(ValueError, match="unauthorized"):
        operator.run_supervised(
            value,
            tmp_path / "draft",
            "d" * 64,
            tmp_path,
            tmp_path,
            prepare_fn=never,
            worker_fn=never,
            scorer_fn=never,
        )
    assert list(tmp_path.iterdir()) == []


@pytest.mark.parametrize(
    "phase", ["success", "prepare_fail", "worker_fail", "score_fail"]
)
def test_supervised_cli_connections_reaped_then_cost_then_quality(
    tmp_path, monkeypatch, phase
):
    monkeypatch.setattr(operator, "ROOT", tmp_path)
    (tmp_path / "runs").mkdir()
    value = operator.draft("a" * 40, "b" * 64, "c" * 64)
    value.update(
        authorization="coordinator_exact_hash_release", model_execution_authorized=True
    )
    output = Path(value["output"])
    sequence = []

    def prepare(*args):
        sequence.append("prepare")
        if phase == "prepare_fail":
            raise OSError("synthetic preparation")

    def worker(command, allocation, inference, seconds, **kwargs):
        sequence.append("worker")
        assert command[2] == "worker" and "--input" in command
        assert kwargs["watchdog"] is operator.slot_watchdog
        assert seconds > 0
        synthetic(output)
        return {
            "child_started": True,
            "child_reaped": True,
            "returncode": 2 if phase == "worker_fail" else 0,
            "interrupted": None,
        }

    def scorer(command, **kwargs):
        sequence.append("score")
        assert command[2] == "score"
        assert (output / "allocation/worker-exit.json").exists()
        assert (output / "cost-before-quality.json").exists()
        if phase == "score_fail":
            raise OSError("synthetic scorer")
        ordered_write(output / "compact.json", {"synthetic": True})

    result = operator.run_supervised(
        value,
        tmp_path / "release.json",
        "d" * 64,
        tmp_path,
        tmp_path,
        prepare_fn=prepare,
        worker_fn=worker,
        scorer_fn=scorer,
    )
    assert (result["status"] == "completed") == (phase == "success")
    assert sequence == (
        ["prepare"]
        if phase == "prepare_fail"
        else ["prepare", "worker"]
        if phase == "worker_fail"
        else ["prepare", "worker", "score"]
    )
    cost = json.loads((output / "cost-before-quality.json").read_bytes())
    assert cost["planned_slots"] == 160
    assert cost["generation"]["unique_physical_calls"] == (
        0 if phase == "prepare_fail" else 14
    )
    assert result["automatic_retry"] is False


def test_release_cannot_change_route_input_or_budget(tmp_path, monkeypatch):
    monkeypatch.setattr(operator, "ROOT", tmp_path)
    value = operator.draft("a" * 40, "b" * 64, "c" * 64)
    value.update(
        authorization="coordinator_exact_hash_release", model_execution_authorized=True
    )
    operator.validate_release(value)
    bad = copy.deepcopy(value)
    bad["policy"]["budget"]["max_calls"] = 1
    with pytest.raises(ValueError, match="frozen_release_changed"):
        operator.validate_release(bad)
