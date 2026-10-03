"""New policy whole-chain checks; all text/labels/providers are SYNTHETIC."""

import copy
import json

import pytest
from jsonschema import ValidationError, validate

from climate_rag.agent_v3 import SentenceAgentV3, Source, V3Budget
from climate_rag import stop_acquire
from climate_rag.targeted_replay import policy, run_matrix
from climate_rag.targeted_score import audit, score
from test_agent_v3 import Provider, ABSTAIN
from test_targeted_replay import FixtureProvider, FixtureRerank, TASKS, QUERY, sha
import run_targeted_replay_operator as operator

DOCS = {
    f"old{i}": Source(f"old{i}", "Synthetic", (f"Synthetic background {i}.",))
    for i in range(6)
}
DOCS["new"] = Source("new", "Synthetic", ("Synthetic observations refute advance.",))
STOP = {"action": "stop"}
READ = {"action": "acquire", "tool": "read", "source_ids": ["c5"]}
SEARCH = {"action": "acquire", "tool": "query", **QUERY}


class GatedFixture(FixtureProvider):
    def __init__(self, mode="query"):
        super().__init__()
        self.mode = mode

    def generate(self, observation, schema, max_output_tokens, remaining_seconds):
        if observation.get("protocol") != stop_acquire.PROTOCOL:
            return super().generate(
                observation, schema, max_output_tokens, remaining_seconds
            )
        self.calls += 1
        if observation["phase"] == "gate":
            action = (
                (READ if self.mode == "read" else SEARCH)
                if self.calls == 1 and self.mode != "stop"
                else STOP
            )
        else:
            found = [v for v in observation["current_citable"] if "refute" in v["text"]]
            action = (
                {
                    "action": "answer",
                    "label": "REFUTES",
                    "sentence_ids": [found[0]["sentence_id"]],
                }
                if found
                else ABSTAIN
            )
        return {
            "raw": json.dumps(action),
            "usage": {
                "input_tokens": self.count_prompt(observation, schema),
                "output_tokens": 32,
            },
        }


def matrix(tmp_path, mode="query", failure=None, protocol=stop_acquire.PROTOCOL):
    def retrieve(query, width):
        if query == QUERY["query"]:
            if failure == "empty":
                return []
            if failure == "timeout":
                raise TimeoutError("synthetic transient search failure")
            return [DOCS["new"]]
        return list(DOCS.values())[:6]

    run = run_matrix(
        TASKS,
        GatedFixture(mode),
        retrieve,
        FixtureRerank(tmp_path / "reranker-ledger"),
        tmp_path / "inference",
        protocol=protocol,
    )
    audit(
        run,
        TASKS,
        DOCS,
        tmp_path / "inference/ledger",
        synthetic=True,
        reranker_directory=tmp_path / "reranker-ledger",
    )
    return run


@pytest.mark.parametrize(
    "mode,failure",
    [
        ("stop", None),
        ("read", None),
        ("query", None),
        ("query", "empty"),
        ("query", "timeout"),
    ],
)
def test_matrix_actual_execution_feedback_scorer_and_cost(tmp_path, mode, failure):
    run = matrix(tmp_path, mode, failure)
    gold = {
        "claims": {
            t["id"]: {
                "claim_sha256": sha(t["claim_text"].encode()),
                "evidence_ids": ["new"] if i == 0 else [],
                "label": "REFUTES" if i == 0 else "NOT_ENOUGH_INFO",
            }
            for i, t in enumerate(TASKS)
        }
    }
    result = score(run, gold, synthetic=True)
    assert (
        result["protocol"] == stop_acquire.PROTOCOL
        and result["resume_promotion"] is False
    )
    for row in (r for r in run["runs"] if r["route"] == "adaptive"):
        attempts = row["generation_attempts"]
        assert [a["stage"] for a in attempts] == (
            ["gate", "verdict"] if mode == "stop" else ["gate", "gate", "verdict"]
        )
        assert (
            row["model_calls"]
            == row["physical_cost"]["unique_physical_calls"]
            == len(attempts)
        )
        assert all(e.get("status") != "running" for e in row["events"])
        if failure:
            obs = attempts[1]["observation"]
            assert obs["tool_feedback"]["status"] == (
                "failed" if failure == "timeout" else "completed"
            )
            assert obs["tool_feedback"]["search_empty"] is (
                True if failure == "empty" else None
            )
            assert obs["prior_queries"] == [QUERY["query"]]
        if mode == "read":
            assert "c5:0" in {
                x["sentence_id"] for x in attempts[1]["observation"]["current_citable"]
            }
            assert "c5" not in attempts[1]["observation"]["readable_source_ids"]
        assert attempts[0]["observation"]["current_citable"]


def test_read_capacity_schema_and_executor_match_no_dangling_event():
    action = {**READ, "source_ids": ["c2", "c3", "c4"]}
    p = Provider([action, STOP, ABSTAIN])
    r = SentenceAgentV3(
        p,
        lambda q, k: list(DOCS.values()),
        budget=V3Budget(context_k=2),
        protocol=stop_acquire.PROTOCOL,
    ).run("Synthetic claim")
    with pytest.raises(ValidationError):
        validate(action, p.observations[0][1])
    assert r["generation_attempts"][0]["status"] == "validation_failed"
    assert r["tool_calls"] == 1 and all(
        e.get("status") != "running" for e in r["events"]
    )


@pytest.mark.parametrize(
    "actions",
    [
        [{**READ, "source_ids": ["c0"]}, STOP, ABSTAIN],
        [
            STOP,
            {"action": "answer", "label": "SUPPORTS", "sentence_ids": ["c5:0"]},
            ABSTAIN,
        ],
        [STOP, {"action": "abstain", "reason": "budget"}, ABSTAIN],
    ],
)
def test_duplicate_read_preview_citation_and_budget_abstention_rejected(actions):
    p = Provider(actions)
    r = SentenceAgentV3(
        p, lambda q, k: list(DOCS.values()), protocol=stop_acquire.PROTOCOL
    ).run("Synthetic claim")
    assert any(a["status"] == "validation_failed" for a in r["generation_attempts"])
    assert r["tool_calls"] == 1 and r["answer"] is None


def test_shared_budget_reserves_verdict_not_forced_tool_or_fake_abstention():
    p = Provider(
        [
            SEARCH,
            {**SEARCH, "query": "second synthetic measurement"},
            {"action": "invalid"},
            {"action": "invalid"},
        ]
    )
    r = SentenceAgentV3(
        p, lambda q, k: list(DOCS.values()), protocol=stop_acquire.PROTOCOL
    ).run("Synthetic claim")
    assert r["model_calls"] == 4 and r["tool_calls"] == 3
    assert r["outcome"] == "generation_budget_exhausted" and r["answer"] is None
    assert r["events"][-1]["status"] == "budget_forced"
    assert r["usage"]["output_tokens"] == 100


@pytest.mark.parametrize(
    "mutation", ["phase", "mask", "feedback", "tool", "proposal", "budget", "running"]
)
def test_audit_rejects_whole_chain_drift(tmp_path, mutation):
    run = matrix(tmp_path)
    row = next(r for r in run["runs"] if r["route"] == "adaptive")
    attempt = row["generation_attempts"][1]
    if mutation == "phase":
        attempt["stage"] = "verdict"
    elif mutation == "mask":
        attempt["observation"]["readable_source_ids"].append("c0")
    elif mutation == "feedback":
        attempt["observation"]["tool_feedback"]["status"] = "failed"
    elif mutation == "tool":
        row["events"][1]["trigger_attempt"] = 1
    elif mutation == "proposal":
        row["generation_attempts"][0]["proposed_decision"] = STOP
    elif mutation == "budget":
        attempt["observation"]["remaining_calls"] = 5
    else:
        row["events"][1]["status"] = "running"
    with pytest.raises(ValueError):
        audit(run, TASKS, DOCS, tmp_path / "inference/ledger", synthetic=True)


def test_four_fixed_controls_observations_schemas_and_outcomes_unchanged(tmp_path):
    from climate_rag.targeted_query import PROTOCOL as OLD

    old, new = tmp_path / "old", tmp_path / "new"
    old.mkdir()
    new.mkdir()
    a, b = matrix(old, protocol=OLD), matrix(new)
    for left, right in zip(a["runs"], b["runs"]):
        if left["route"] == "adaptive":
            continue
        assert left["protocol"] == right["protocol"] == OLD
        assert left["outcome"] == right["outcome"]
        for x, y in zip(left["generation_attempts"], right["generation_attempts"]):
            assert x["observation"] == y["observation"] and x["schema"] == y["schema"]


def test_unapproved_draft_protocol_and_resource_cannot_drift(tmp_path, monkeypatch):
    monkeypatch.setattr(operator, "ROOT", tmp_path)
    release = operator.draft(
        "a" * 40, "b" * 64, "c" * 64, protocol=stop_acquire.PROTOCOL
    )
    assert release["policy"] == policy(stop_acquire.PROTOCOL)
    assert release["resource_cap"]["host_ram_gib"] == 48
    assert "\\" not in release["output"]
    with pytest.raises(ValueError, match="unauthorized"):
        operator.run_supervised(
            release, tmp_path / "draft", "d" * 64, tmp_path, tmp_path
        )
    assert list(tmp_path.iterdir()) == []
    allowed = copy.deepcopy(release)
    allowed.update(
        authorization="coordinator_exact_hash_release", model_execution_authorized=True
    )
    operator.validate_release(allowed)
    allowed["resource_cap"]["host_ram_gib"] = 32
    with pytest.raises(ValueError, match="frozen_release_changed"):
        operator.validate_release(allowed)


def test_cpu_torch_provider_binding_gate_and_verdict(tmp_path, monkeypatch):
    from test_scifact_paired_comparison import bound_provider
    from climate_rag.local_targeted_provider import LocalTargetedProvider
    from climate_rag.targeted_replay import TargetedJournal

    (tmp_path / "private").mkdir(mode=0o700)
    provider, calls = bound_provider(tmp_path / "private", monkeypatch, [STOP, ABSTAIN])
    provider.__class__ = LocalTargetedProvider
    provider.private_dir = tmp_path / "private"
    provider.terminal_protocol = stop_acquire.PROTOCOL
    journal = TargetedJournal(
        provider, tmp_path / "ledger", max_generations=5, protocol=stop_acquire.PROTOCOL
    )
    journal.slot = "synthetic-only"
    result = SentenceAgentV3(
        journal, lambda q, k: list(DOCS.values()), protocol=stop_acquire.PROTOCOL
    ).run("Synthetic claim")
    assert result["outcome"] == "model_abstention:insufficient_evidence", result[
        "events"
    ]
    assert (
        len(calls) == 2
        and len(list((tmp_path / "ledger").glob("*.generation.json"))) == 2
    )
    assert all(call["input_ids"].device.type == "cpu" for call in calls)


@pytest.mark.parametrize("failure", ["deadline", "mutation"])
def test_tool_failure_after_valid_gate_retains_cost_and_can_be_audited(
    tmp_path, failure
):
    from climate_rag.targeted_score import audit_acquisition

    clock = [0.0]

    def retrieve(query, width):
        if query == QUERY["query"]:
            if failure == "deadline":
                clock[0] = 121.0
                return [DOCS["new"]]
            return [Source("old0", "Synthetic", ("Mutated synthetic source.",))]
        return list(DOCS.values())[:6]

    result = SentenceAgentV3(
        Provider([SEARCH]),
        retrieve,
        protocol=stop_acquire.PROTOCOL,
        clock=lambda: clock[0],
    ).run("Synthetic claim")
    assert (
        result["outcome"].startswith("controller_failure") and result["answer"] is None
    )
    assert result["model_calls"] == 1 and result["tool_calls"] == 2
    assert (
        result["generation_attempts"][0]["execution_status"]
        == result["events"][1]["status"]
    )
    audit_acquisition(result, DOCS)


def test_execution_failure_cannot_be_relabelled_as_abstention():
    from climate_rag.targeted_score import audit_acquisition

    result = SentenceAgentV3(
        Provider([{"action": "invalid"}] * 3),
        lambda q, k: list(DOCS.values()),
        protocol=stop_acquire.PROTOCOL,
    ).run("Synthetic claim")
    assert result["outcome"] == "validation_repair_exhausted"
    audit_acquisition(result, DOCS)
    result["outcome"] = "model_abstention:insufficient_evidence"
    with pytest.raises(ValueError, match="gate_failure_not_abstention"):
        audit_acquisition(result, DOCS)


def test_deadline_between_gate_and_tool_does_not_copy_previous_success():
    from climate_rag.targeted_score import audit_acquisition

    state = {"generated": False, "post": 0}

    def clock():
        if not state["generated"]:
            return 0.0
        state["post"] += 1
        return 119.9 if state["post"] == 1 else 120.1

    p = Provider([SEARCH], advance=lambda: state.update(generated=True))
    r = SentenceAgentV3(
        p, lambda q, k: list(DOCS.values()), protocol=stop_acquire.PROTOCOL, clock=clock
    ).run("Synthetic claim")
    assert r["outcome"] == "controller_failure:RuntimeError"
    assert r["model_calls"] == 1 and r["tool_calls"] == 1
    assert r["generation_attempts"][0]["execution_status"] == "not_started"
    audit_acquisition(r, DOCS)


def test_context_overflow_fails_instead_of_hiding_old_evidence():
    p = Provider([READ, STOP, ABSTAIN])
    docs = [DOCS["old0"], Source("huge", "Synthetic", ("synthetic " * 5000,))]
    p.actions = iter([{**READ, "source_ids": ["c1"]}, STOP, ABSTAIN])
    r = SentenceAgentV3(
        p,
        lambda q, k: docs,
        budget=V3Budget(context_k=1),
        protocol=stop_acquire.PROTOCOL,
    ).run("Synthetic claim")
    assert r["outcome"].startswith("controller_failure") and r["model_calls"] == 1
    assert r["events"][-1]["failure_code"] == "selected_context_capacity"


def test_actual_worker_supervisor_score_connections_with_synthetic_160_slots(
    tmp_path, monkeypatch
):
    """Production entry functions; ONLY weights/data/allocated-host boundaries mocked."""
    from pathlib import Path
    from types import SimpleNamespace
    import run_targeted_replay as worker
    from transformers.models.auto.tokenization_auto import AutoTokenizer
    from climate_rag.scifact_read_continuation import ordered_write

    tasks = [
        {"id": f"synthetic-{i}", "claim_text": "Synthetic background claim."}
        for i in range(32)
    ]
    gold = {
        "claims": {
            t["id"]: {
                "claim_sha256": sha(t["claim_text"].encode()),
                "evidence_ids": ["new"] if i < 24 else [],
                "label": "REFUTES"
                if i < 23
                else "DISPUTED"
                if i == 23
                else "NOT_ENOUGH_INFO",
            }
            for i, t in enumerate(tasks)
        }
    }
    monkeypatch.setattr(operator, "ROOT", tmp_path)
    monkeypatch.setattr(worker, "ROOT", tmp_path)
    monkeypatch.setattr(
        worker,
        "os",
        SimpleNamespace(
            name="posix",
            environ={"SLURM_JOB_ID": "synthetic", "CUDA_VISIBLE_DEVICES": "synthetic"},
        ),
    )
    monkeypatch.setattr(worker, "load_inputs", lambda path: (tasks, [], DOCS))
    monkeypatch.setattr(
        worker, "MANIFEST_BYTES", {k: sha(b"{}\n") for k in ("generator", "reranker")}
    )
    monkeypatch.setattr(
        worker,
        "verify_model_files",
        lambda path, manifest: worker.MODEL_SHA
        if path.parent.name == "generator"
        else worker.RERANKER_SHA,
    )
    seen = []

    def provider(*args, **kwargs):
        seen.append(kwargs["protocol"])
        return GatedFixture()

    monkeypatch.setattr(worker, "LocalTargetedProvider", provider)
    monkeypatch.setattr(worker, "GenerationBinding", lambda *args: None)
    monkeypatch.setattr(worker, "base_state", lambda *args: {})
    monkeypatch.setattr(worker, "Qwen3CausalLMReranker", lambda *args, **kwargs: None)
    monkeypatch.setattr(
        worker, "TargetedRerank", lambda a, b, path, **kwargs: FixtureRerank(path)
    )

    class Index:
        def fit(self, docs):
            return self

        def search(self, query, width):
            return [
                SimpleNamespace(evidence_id=k)
                for k in (["new"] if query == QUERY["query"] else list(DOCS)[:6])
            ]

    monkeypatch.setattr(worker, "BM25Index", Index)
    monkeypatch.setattr(AutoTokenizer, "from_pretrained", lambda *args, **kwargs: None)
    original_audit = worker.audit
    monkeypatch.setattr(
        worker,
        "audit",
        lambda *args, **kwargs: original_audit(*args, **kwargs, synthetic=True),
    )
    monkeypatch.setattr(
        worker, "score", lambda run, labels: score(run, labels, synthetic=True)
    )
    monkeypatch.setattr(worker, "load_selection_manifest", lambda: {})
    monkeypatch.setattr(worker, "validate_gold", lambda *args: gold)
    (tmp_path / "runs").mkdir()
    value = operator.draft("a" * 40, "b" * 64, "c" * 64, protocol=stop_acquire.PROTOCOL)
    value.update(
        authorization="coordinator_exact_hash_release", model_execution_authorized=True
    )

    def prepare(release, work):
        for key in ("generator", "reranker"):
            (work / "input/models" / key).mkdir(parents=True)
            ordered_write(work / "input/models" / key / "model_manifest.json", {})
        ordered_write(work / "input/validation-protocol.json", {})
        (tmp_path / "envs").mkdir()
        ordered_write(
            tmp_path
            / "envs"
            / ("budget-agent-validation-gold-" + worker.GOLD_SHA + ".json"),
            {},
        )

    def execute(command, allocation, inference, seconds, **kwargs):
        worker.worker(value, tmp_path / "input")
        return {
            "child_started": True,
            "child_reaped": True,
            "returncode": 0,
            "interrupted": None,
        }

    def scoring(command, **kwargs):
        worker.score_after_exit(value, tmp_path / "input")

    result = operator.run_supervised(
        value,
        tmp_path / "release",
        "d" * 64,
        tmp_path,
        tmp_path,
        prepare_fn=prepare,
        worker_fn=execute,
        scorer_fn=scoring,
    )
    assert result["status"] == "completed", result
    assert seen == [stop_acquire.PROTOCOL]
    compact = json.loads((Path(value["output"]) / "compact.json").read_bytes())
    assert compact["protocol"] == stop_acquire.PROTOCOL
    for route in compact["routes"].values():
        assert (
            route["retrieval_denominator"],
            route["label_denominator"],
            route["slots"],
        ) == (24, 23, 32)
