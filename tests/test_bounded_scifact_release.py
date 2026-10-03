"""Offline release gates and physical receipt checks, entirely synthetic."""
import hashlib
import importlib
import inspect
import json
from pathlib import Path

import pytest


@pytest.fixture
def scripts(monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1] / "scripts"))
    return importlib.import_module("run_scifact_bounded_operator")


def test_pair_protocol_is_frozen_96_slots_no_gpu_release(scripts):
    from run_scifact_bounded_arm import PAIR_SHA
    path = Path(__file__).resolve().parents[1] / "docs/protocols/scifact-bounded-gap-pair-20260930.json"
    assert hashlib.sha256(path.read_bytes()).hexdigest() == PAIR_SHA
    p = json.loads(path.read_text())
    assert len(set(p["ordered_claim_ids"])) == 12 and len(p["arms"]) == 2 and len(p["routes"]) == 4
    assert p["total_slots"] == 96 and p["slots_per_arm"] == 48
    assert p["budget"]["timeout_seconds"] * p["slots_per_arm"] < p["resources_per_arm"]["walltime_seconds"]
    assert p["budget"]["timeout_seconds"] * p["total_slots"] > p["resources_per_arm"]["walltime_seconds"]
    assert p["actual_gpu_submission"] is False


def test_pair_operator_requires_frozen_complete_first_arm(tmp_path, scripts):
    (tmp_path / "runs").mkdir()
    env = {"CLIMATE_SCIFACT_PAIR_RELEASE": scripts.RELEASE, "CLIMATE_SCIFACT_ARM": scripts.ARMS[0],
           "CLIMATE_SOURCE_GIT": "a" * 40, "CLIMATE_SOURCE_SHA256": "b" * 64}
    first, _ = scripts.create_attempt_result(tmp_path, env)
    with pytest.raises(FileExistsError):
        scripts.create_attempt_result(tmp_path, env)
    env["CLIMATE_SCIFACT_ARM"] = scripts.ARMS[1]
    with pytest.raises(FileNotFoundError):
        scripts.create_attempt_result(tmp_path, env)
    (first / "inference").mkdir()
    (first / "inference/run.json").write_text("{}")
    state = {"status": "complete", "source_git": "different", "source_archive_sha256": "b" * 64,
             "pair_protocol_sha256": scripts.PAIR_SHA, "inference_sha256": scripts.digest(first / "inference/run.json")}
    status = first / "operator-status.json"
    status.write_text(json.dumps(state))
    with pytest.raises(ValueError, match="different"):
        scripts.create_attempt_result(tmp_path, env)
    state["source_git"] = env["CLIMATE_SOURCE_GIT"]
    status.write_text(json.dumps(state))
    second, identity = scripts.create_attempt_result(tmp_path, env)
    assert second.exists() and identity["arm"] == scripts.ARMS[1]


def test_runner_no_gold_args_and_operator_scores_only_after_both_exits(scripts):
    import run_scifact_bounded_arm
    source = inspect.getsource(run_scifact_bounded_arm)
    assert '"--gold"' not in source and '"--selection"' not in source
    assert 'provider.start_slot' in source and 'private wire/usage audit incomplete' in source
    source = inspect.getsource(scripts.main)
    assert source.index('execute(command') < source.index('if state["arm"] == ARMS[1]') < source.index('extract_scoring_after_both')


def test_external_scoring_and_inference_cannot_replace_preregistered_assets(scripts):
    source = Path(__file__).resolve().parents[1]
    pair = json.loads((source / "docs/protocols/scifact-bounded-gap-pair-20260930.json").read_text())
    env = {"CLIMATE_INFERENCE_SHA256": pair["inference_tar_sha256"],
           "CLIMATE_SCORING_SHA256": pair["scoring_tar_sha256"],
           "CLIMATE_DIAGNOSTIC_PROTOCOL_SHA256": pair["parent_protocol_sha256"],
           "CLIMATE_GRAMMAR_SHA256": "36ed51e0a9a71ded058cc0e7290a78a7f5f638b91bff517d49ad6ea83c70e410"}
    scripts.validate_external_assets(env, source)
    for key in env:
        changed = dict(env, **{key: "0" * 64})
        with pytest.raises(ValueError, match="frozen pair"):
            scripts.validate_external_assets(changed, source)


def test_physical_multiset_duplicates_and_missing_wire(scripts, tmp_path):
    from score_scifact_bounded_pair import audit_private_wire
    raw = b'{"action":"rerank"}'
    digest = hashlib.sha256(raw).hexdigest()
    receipt = {"sha256": digest, "stored_prefix_sha256": digest, "truncated": False, "io_failed": False,
               "dropped_bytes": 0, "stored_bytes": len(raw), "attempted_bytes": len(raw)}
    attempt = {"diagnostics": {"private_attachment": receipt, "raw_wire_action": "rerank"}}
    (tmp_path / "a-response.txt").write_bytes(raw)
    with pytest.raises(ValueError, match="multiset"):
        audit_private_wire(tmp_path, [attempt, attempt], False)
    (tmp_path / "b-response.txt").write_bytes(raw)
    assert audit_private_wire(tmp_path, [attempt, attempt], False)["complete_responses"] == 2
