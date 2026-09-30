"""Synthetic-only continuation adversarial tests; never load model weights."""
from __future__ import annotations

import copy
import types

import pytest

import climate_rag.component_continuation as cont
import climate_rag.component_execution as execution
from climate_rag.component_audit import aggregate
from climate_rag.component_preflight import preflight_cases
from climate_rag.scifact_component_contract import ContractError, packing
from climate_rag.scifact_component_runtime import ComponentPersistenceError
from climate_rag.scifact_semantic_contract import encoded, sha
from test_component_execution import Fake, matrix
from test_scifact_components import TokenizerFixture


class FixedCountTokenizer(TokenizerFixture):
    def encode(self, text, **kwargs):
        return [1] * 277  # Explicit fake; no real token or quality claim.


class Provider(Fake):
    tokenizer = FixedCountTokenizer()

    def __init__(self, *, wrong_at=-1, invalid_at=-1, **kwargs):
        super().__init__(**kwargs)
        self.wrong_at, self.invalid_at = wrong_at, invalid_at

    def generate(self, spec, schema, tokens, seconds):
        if self.calls + 1 == self.wrong_at:
            self.wire = {"decision": "abstain", "relation": None}
        elif self.calls + 1 == self.invalid_at:
            self.wire = {"decision": "select", "document_ids": [999]}
        result = super().generate(spec, schema, tokens, seconds)
        self.wire = None
        result["usage"]["output_tokens"] = 15
        result["diagnostics"]["output_tokens"] = 15
        return result


def frozen_matrix():
    slots, targets = matrix()
    for slot in slots:
        slot["packing"] = packing(FixedCountTokenizer(), slot["input"])
    return slots, targets


@pytest.fixture
def predecessor(tmp_path, monkeypatch):
    """Bind all production checks to hash-closed synthetic physical receipts."""
    root = tmp_path / "project"
    prior = root / "runs" / cont.PREVIOUS_RELEASE
    prior.mkdir(parents=True)
    archive = b"Explicit synthetic source archive, not execution evidence"
    archive_sha = sha(archive)
    monkeypatch.setattr(cont, "PREVIOUS_ARCHIVE", archive_sha)
    archive_path = root / "envs" / ("component-execution-" + archive_sha) / "source.tar"
    archive_path.parent.mkdir(parents=True)
    archive_path.write_bytes(archive)
    slots, targets = frozen_matrix()
    report = execution.execute_matrix(slots, Provider(wire={"decision": "abstain", "document_ids": []}), prior / "inference")
    assert report["preflight"][0]["preflight_pass"] is False
    compact, _ = aggregate(slots, targets, report["diagnostic"], report["preflight"])
    execution.durable(prior / "compact.json", compact)
    identity = {"release": cont.PREVIOUS_RELEASE, "source_git": cont.PREVIOUS_SOURCE,
        "source_archive_sha256": archive_sha, "model_manifest_sha256": cont.MODEL_SHA,
        "slots_sha256": cont.SLOTS_SHA, "scoring_targets_loaded": False,
        "preflight": [{"slot": i, "input": spec, "packing": packing(FixedCountTokenizer(), spec)}
                      for i, (spec, _) in enumerate(preflight_cases())]}
    execution.durable(prior / "worker-identity.json", identity)
    execution.durable(prior / "inference-exited.json", {"release": cont.PREVIOUS_RELEASE,
        "child_reaped": True, "returncode": 0, "worker_identity_sha256": sha(encoded(identity))})
    execution.durable(prior / "operator-status.json", {"job_id": cont.PREVIOUS_JOB,
        "compact_sha256": sha(encoded(compact))})
    monkeypatch.setattr(cont, "FILES", {name: sha((prior / name).read_bytes()) for name in cont.FILES})
    return root, prior, slots, targets


def tree_hash(path):
    return {p.relative_to(path).as_posix(): sha(p.read_bytes()) for p in path.rglob("*") if p.is_file()}


def test_valid_wrong_semantic_continues_without_repeating_or_rewriting_carried(predecessor):
    root, prior, slots, targets = predecessor
    before = tree_hash(prior)
    provider = Provider(wrong_at=1)
    output = root / "continuation"
    report = cont.execute_continuation(slots, provider, output, prior)
    assert provider.calls == 36 and report["status"] == "complete"
    assert tree_hash(prior) == before and not (output / "preflight-00").exists()
    assert [r["semantic_match"] for r in report["preflight"]] == [False, False, True, True]
    assert all(r["technical_ready"] for r in report["preflight"])
    assert report["cost_partition"]["carried"]["tokens"] == {"input_tokens": 277, "output_tokens": 15}
    assert [report["cost_partition"][k]["attempted"] for k in ("carried", "new", "cumulative")] == [1, 36, 37]
    summary, _ = aggregate(slots, targets, report["diagnostic"], report["preflight"], preflight_policy=cont.POLICY)
    assert summary["preflight_semantic_matches"] == 2 and summary["preflight_technical_ready"] == 4
    assert summary["planned_diagnostic"] == 33 and summary["total_attempted"] == 37
    assert [summary["denominators"][k]["planned"] for k in ("screening", "relation", "rationale")] == [12, 12, 9]
    with pytest.raises(ContractError, match="preflight_after_stop"):
        aggregate(slots, targets, report["diagnostic"], report["preflight"])
    with pytest.raises(ContractError, match="unknown_preflight_policy"):
        aggregate(slots, targets, report["diagnostic"], report["preflight"], preflight_policy="forged")
    with pytest.raises(FileExistsError):
        cont.execute_continuation(slots, provider, output, prior)
    assert provider.calls == 36


@pytest.mark.parametrize("failure_at", [1, 3, 4, 19, 36])
def test_technical_unknown_cost_stops_and_keeps_all_denominators(predecessor, failure_at):
    root, prior, slots, targets = predecessor
    provider = Provider(fail_at=failure_at)
    report = cont.execute_continuation(slots, provider, root / "continuation", prior)
    summary, _ = aggregate(slots, targets, report["diagnostic"], report["preflight"], preflight_policy=cont.POLICY)
    assert report["status"] == "stopped" and provider.calls == failure_at
    assert report["cost_partition"]["new"]["unknown_cost_attempts"] == 1
    assert summary["unknown_cost_attempts"] == 1 and summary["total_attempted"] == failure_at + 1
    assert len(report["preflight"]) == 4 and len(report["diagnostic"]) == 33
    assert all(r["semantic_match"] is None for r in report["preflight"] if not r["attempted"])


@pytest.mark.parametrize("invalid_at", [1, 4])
def test_schema_failure_stops_preflight_but_paid_formal_diagnostic_continues(predecessor, invalid_at):
    root, prior, slots, targets = predecessor
    provider = Provider(invalid_at=invalid_at)
    report = cont.execute_continuation(slots, provider, root / "continuation", prior)
    assert provider.calls == (1 if invalid_at == 1 else 36)
    summary, _ = aggregate(slots, targets, report["diagnostic"], report["preflight"], preflight_policy=cont.POLICY)
    assert summary["unknown_cost_attempts"] == 0
    record = report["preflight"][1] if invalid_at == 1 else report["diagnostic"][0]
    assert record["status"] == "schema_failed" and record["usage_known"]


@pytest.mark.parametrize("name", list(cont.FILES))
def test_each_predecessor_file_hash_is_mandatory(predecessor, name):
    _, prior, _, _ = predecessor
    with (prior / name).open("ab") as stream:
        stream.write(b" ")
    with pytest.raises(ValueError, match="input_sha_mismatch"):
        cont.load_carried(prior)


@pytest.mark.parametrize("tamper", ["hidden_started", "hidden_response", "raw", "archive", "packing"])
def test_independent_physical_predecessor_checks(predecessor, tamper):
    root, prior, _, _ = predecessor
    if tamper.startswith("hidden"):
        hidden = prior / "inference/diagnostic-32"
        hidden.mkdir()
        (hidden / ("started.json" if tamper == "hidden_started" else "response.json")).write_text("{}")
    elif tamper == "raw":
        (prior / "inference/preflight-00/private/synthetic-response.txt").write_text("tampered")
    elif tamper == "archive":
        (root / "envs" / ("component-execution-" + cont.PREVIOUS_ARCHIVE) / "source.tar").write_bytes(b"tampered")
    with pytest.raises((ContractError, ValueError)):
        cont.load_carried(prior, TokenizerFixture() if tamper == "packing" else None)


def test_persistence_failure_stops_without_automatic_resume(predecessor, monkeypatch):
    root, prior, slots, _ = predecessor
    real_durable = execution.durable
    def fail_completed(path, value):
        if path.name == "completed.json":
            raise OSError("synthetic disk failure")
        return real_durable(path, value)
    monkeypatch.setattr(execution, "durable", fail_completed)
    provider = Provider()
    output = root / "continuation"
    with pytest.raises(ComponentPersistenceError):
        cont.execute_continuation(slots, provider, output, prior)
    with pytest.raises(FileExistsError):
        cont.execute_continuation(slots, provider, output, prior)
    assert provider.calls == 1 and (output / "preflight-01/started.json").exists()


@pytest.mark.parametrize("mutation", ["duplicate", "reordered", "missing", "extra"])
def test_partition_rejects_recounted_or_wrong_slots(predecessor, mutation):
    root, prior, slots, _ = predecessor
    report = cont.execute_continuation(slots, Provider(), root / "continuation", prior)
    pre, records = copy.deepcopy(report["preflight"]), report["diagnostic"]
    if mutation == "duplicate":
        pre[1]["execution_origin"] = "carried"
    elif mutation == "reordered":
        pre[0], pre[1] = pre[1], pre[0]
    elif mutation == "missing":
        records = records[:-1]
    else:
        records = records + [records[-1]]
    with pytest.raises(ContractError):
        cont.cost_partition(pre, records)


def scoring_files(root, slots, targets, monkeypatch):
    import score_scifact_component_continuation as scorer
    monkeypatch.setattr(scorer, "ROOT", root)
    slots_path, target_path = root / "slots.json", root / "targets.json"
    execution.durable(slots_path, slots)
    execution.durable(target_path, targets)
    # Test-only synthetic frozen matrix, with production digest checking intact.
    monkeypatch.setattr(execution, "SLOTS_SHA", sha(encoded(slots)))
    monkeypatch.setattr(scorer, "SLOTS_SHA", sha(encoded(slots)))
    monkeypatch.setattr(scorer, "TARGETS_SHA", sha(encoded(targets)))
    return scorer, slots_path, target_path


@pytest.mark.parametrize("tamper", [None, "duplicate", "hidden", "cost", "policy", "exit", "carried"])
def test_independent_scorer_agrees_or_rejects_tamper(predecessor, monkeypatch, tamper):
    root, prior, slots, targets = predecessor
    result = root / "runs" / cont.RELEASE
    result.mkdir()
    report = cont.execute_continuation(slots, Provider(), result / "inference", prior)
    scorer, slots_path, target_path = scoring_files(root, slots, targets, monkeypatch)
    identity = {"release": cont.RELEASE, "scoring_targets_loaded": False, "preflight_policy": cont.policy_identity(),
        "slots_sha256": scorer.SLOTS_SHA, "model_manifest_sha256": cont.MODEL_SHA,
        "carried_reference": cont.load_carried(prior)["reference"], "source_git": "f" * 40,
        "source_archive_sha256": "a" * 64,
        "preflight": [{"slot": i, "input": spec, "packing": packing(FixedCountTokenizer(), spec)}
                      for i, (spec, _) in enumerate(preflight_cases())]}
    if tamper == "policy":
        identity["preflight_policy"] = {"version": "forged"}
    if tamper == "carried":
        identity["carried_reference"]["record_sha256"] = "0" * 64
    execution.durable(result / "worker-identity.json", identity)
    marker = {"release": cont.RELEASE, "child_reaped": tamper != "exit", "returncode": 0,
        "worker_identity_sha256": sha(encoded(identity))}
    execution.durable(result / "inference-exited.json", marker)
    if tamper == "duplicate":
        (result / "inference/preflight-00").mkdir()
    elif tamper == "hidden":
        (result / "inference/other").mkdir()
        (result / "inference/other/started.json").write_text("{}")
    elif tamper == "cost":
        report["cost_partition"]["new"]["attempted"] = 0
        (result / "inference/run.json").write_bytes(encoded(report))
    if tamper:
        with pytest.raises((ContractError, ValueError)):
            scorer.score_run(result, slots_path, target_path, sha(encoded(marker)))
        assert not (result / "compact.json").exists()
    else:
        summary = scorer.score_run(result, slots_path, target_path, sha(encoded(marker)))
        assert summary["cost_partition"] == report["cost_partition"]
        assert summary["new_attempted"] == 36 and summary["cumulative_attempted"] == 37
        assert summary["semantic_measurements"][0]["semantic_match"] is False
        assert summary["original_protocol_completed"] is False


def test_bad_predecessor_checked_before_constructor(tmp_path, monkeypatch):
    import run_scifact_component_continuation as worker
    from transformers.models.auto.tokenization_auto import AutoTokenizer
    source = tmp_path / "source"
    (source / "scripts").mkdir(parents=True)
    (source / "SOURCE_REVISION").write_text("a" * 40)
    monkeypatch.setattr(worker, "__file__", str(source / "scripts/run_scifact_component_continuation.py"))
    monkeypatch.setattr(worker, "ROOT", tmp_path)
    monkeypatch.setattr(worker, "os", types.SimpleNamespace(name="posix", environ={
        "SLURM_JOB_ID": "synthetic", "CUDA_VISIBLE_DEVICES": "0", "CLIMATE_COMPONENT_RELEASE": cont.RELEASE,
        "CLIMATE_SOURCE_GIT": "a" * 40, "CLIMATE_SOURCE_SHA256": "b" * 64,
        "HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1", "PYTHONHASHSEED": "0"}))
    monkeypatch.setattr(worker.argparse.ArgumentParser, "parse_args", lambda self:
        types.SimpleNamespace(slots=tmp_path, model_manifest=tmp_path, model_dir=tmp_path))
    monkeypatch.setattr(worker, "frozen_slots", lambda _: frozen_matrix()[0])
    monkeypatch.setattr(worker, "verify_manifest", lambda *args: {})
    monkeypatch.setattr(AutoTokenizer, "from_pretrained", lambda *args, **kwargs: FixedCountTokenizer())
    constructors = []
    monkeypatch.setattr(worker, "LocalQwenComponentProvider", lambda *args, **kwargs: constructors.append(True))
    with pytest.raises(FileNotFoundError):
        worker.main()
    assert constructors == [] and not (tmp_path / "runs" / cont.RELEASE / "worker-identity.json").exists()
