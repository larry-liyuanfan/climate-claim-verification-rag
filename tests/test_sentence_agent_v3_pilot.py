"""CPU protocol/operator fixtures; no weights, CUDA or scheduler calls."""

import copy
import importlib.util
import json
import re
import sys
from pathlib import Path

import pytest

from climate_rag.agent_v3 import SentenceAgentV3, Source

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
spec = importlib.util.spec_from_file_location(
    "v3_pilot_fixture", ROOT / "scripts/run_sentence_agent_v3.py"
)
RUNNER = importlib.util.module_from_spec(spec)
spec.loader.exec_module(RUNNER)
PROTOCOL = json.loads(
    (ROOT / "configs/agent_sentence_v3_pilot_20260930.json").read_text()
)


class Provider:
    name, kind = "scripted", "fixture_not_model"

    def count_prompt(self, observation, schema):
        return 100

    def count_text(self, text):
        return len(text)

    def generate(self, observation, schema, max_output_tokens, remaining_seconds):
        if observation.get("claim") == "Synthetic protocol check":
            payload = {
                "action": "read",
                "source_ids": [observation["preview_only"][0]["source_id"]],
            }
        else:
            payload = {"action": "abstain", "reason": "insufficient_evidence"}
        truncated = max_output_tokens == 1
        return {
            "raw": "{" if truncated else json.dumps(payload),
            "usage": {"input_tokens": 100, "output_tokens": 1 if truncated else 20},
            "diagnostics": {
                "eos_observed": not truncated,
                "reached_max_new_tokens": truncated,
            },
        }


def test_same_six_already_exposed_tasks_and_four_routes():
    previous = json.loads(
        (ROOT / "configs/budget_agent_feedback_pilot_20260930.json").read_text()
    )
    assert PROTOCOL["tasks"] == previous["pilot"]
    assert PROTOCOL["corpus_sha256"] == previous["corpus_sha256"]
    assert RUNNER.validate_protocol(PROTOCOL).max_tools == 5
    for field, value in [
        ("official_gold_available", True),
        ("document_count", 5183),
        ("routes", ["adaptive"]),
        ("tasks", [{"id": "dev", "label": "SUPPORTS"}]),
    ]:
        changed = {**PROTOCOL, field: value}
        with pytest.raises(ValueError):
            RUNNER.validate_protocol(changed)


def test_runtime_preflight_checks_aba_eos_truncation_and_input_count():
    assert RUNNER.runtime_preflight(Provider())["passed"]

    class Bad(Provider):
        def generate(self, *args):
            response = super().generate(*args)
            response["diagnostics"]["eos_observed"] = False
            return response

    report = RUNNER.runtime_preflight(Bad())
    assert not report["passed"] and len(report["records"]) == 1

    class Overrun(Provider):
        def generate(self, *args):
            response = super().generate(*args)
            if args[2] == 1:
                response["usage"]["output_tokens"] = 2
            return response

    report = RUNNER.runtime_preflight(Overrun())
    assert not report["passed"] and len(report["records"]) == 4


def test_compact_requires_complete_matrix_and_exports_no_source_text():
    rows = []
    agent = SentenceAgentV3(
        Provider(),
        lambda q, w: [Source("fixture", "private title", ("private evidence marker",))],
        rerank=lambda q, rows: rows,
    )
    for task in PROTOCOL["tasks"]:
        for route in RUNNER.ROUTES:
            rows.append({"task_id": task["id"], **agent.run(task["claim_text"], route)})
    run = {
        "runs": rows,
        "source_git": "a" * 40,
        "protocol_sha256": "b" * 64,
        "model_sha256": "c" * 64,
        "reranker_sha256": "d" * 64,
        "document_count": 5240,
    }
    compact = RUNNER.compact_run(run, PROTOCOL)
    assert compact["complete_slots"] == 24
    assert "private evidence marker" not in json.dumps(compact)
    assert compact["semantic_quality"] is None
    assert compact["routes"]["adaptive"]["mechanically_accepted_answers"] == 0
    for bad_rows in (rows[:-1], [*rows[:-1], rows[0]]):
        with pytest.raises(ValueError):
            RUNNER.compact_run({**run, "runs": bad_rows}, PROTOCOL)


def test_no_default_budget_drift_or_output_overwrite(tmp_path):
    protocol = copy.deepcopy(PROTOCOL)
    protocol["budget"]["context_k"] = 4
    with pytest.raises(ValueError):
        RUNNER.validate_protocol(protocol)
    RUNNER.write_once(tmp_path / "run.json", {"fixture": True})
    with pytest.raises(FileExistsError):
        RUNNER.write_once(tmp_path / "run.json", {})


def test_preparation_operator_is_single_use_stdlib_and_one_gpu():
    wrapper = (ROOT / "hpc/agent_sentence_v3_pilot.sbatch").read_text()
    operator = (ROOT / "scripts/run_sentence_agent_v3_operator.py").read_text()
    for setting in (
        "--gres=gpu:A100:1",
        "--cpus-per-task=8",
        "--mem=32G",
        "--tmp=30G",
        "--time=01:00:00",
        "--no-requeue",
    ):
        assert "#SBATCH " + setting in wrapper
    assert not re.search(r"(?m)^\s*sbatch\s", wrapper + operator)
    assert "scancel" not in wrapper + operator
    assert "from run_sentence_agent_v3 import" not in operator
    assert "result.mkdir(mode=0o700)" in operator
    assert all(
        flag in operator
        for flag in ('"--no-index"', '"--no-deps"', '"--require-hashes"')
    )
    assert "source revision mismatch" in operator
    assert (
        "private.mkdir(mode=0o700)"
        in (ROOT / "scripts/run_sentence_agent_v3.py").read_text()
    )
