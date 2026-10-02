"""Standalone contract tests. All archives/instance records here are SYNTHETIC."""
import copy
import io
import json
import os
import tarfile
from pathlib import Path

import pytest

import cloud_replay_contract as contract
import run_cloud_replay as entry
from preflight_cloud_assets import inventory
from climate_rag.targeted_replay import sha
from climate_rag.scifact_read_continuation import ordered_write
from run_targeted_replay_operator import run_supervised, draft as slurm_draft


def release():
    return contract.draft("a" * 40, "b" * 64, "c" * 64,
                          root="/workspace/climate-replay", run_id="cpu-fixture")


def approved(value):
    return {**value, "authorization": "standalone_exact_hash_release",
            "model_execution_authorized": True, "runtime_receipt_sha256": "d" * 64,
            "asset_receipt_sha256": "e" * 64}


def test_new_backend_draft_and_transport_separate_from_history():
    value = release()
    contract.validate_release(value, execution=False)
    with pytest.raises(ValueError, match="unauthorized"):
        contract.validate_release(value)
    updated = contract.draft("a" * 40, "b" * 64, "c" * 64,
                             root=value["root"], run_id="other-run", input_sha="e" * 64)
    assert updated["input_transport_sha256"] != updated["policy"]["input_archive_sha256"]
    assert updated["policy"] == value["policy"]
    assert updated["runtime_receipt_sha256"] is None
    assert "slurm" not in json.dumps(updated).lower()
    contract.validate_release(approved(updated))


@pytest.mark.parametrize("mutation", ["spartan", "root", "policy", "gold", "runtime", "budget"])
def test_frozen_release_rejects_drift(mutation):
    value = approved(release())
    if mutation == "spartan":
        value = slurm_draft("a" * 40, "b" * 64, "c" * 64)
    elif mutation == "root":
        value["root"] = "/workspace/climate/../other"
    elif mutation == "policy":
        value["policy"]["tasks"] = 300
    elif mutation == "gold":
        value["gold_path"] = value["input"] + "/gold.json"
    elif mutation == "runtime":
        value["runtime_receipt_sha256"] = None
    else:
        value["max_worker_seconds"] = 9000
    with pytest.raises((ValueError, TypeError)):
        contract.validate_release(value)


def test_draft_rejected_before_output_or_preparation(tmp_path):
    value = release()
    seen = []
    with pytest.raises(ValueError, match="unauthorized"):
        run_supervised(value, tmp_path / "release", "a" * 64, tmp_path, tmp_path,
                       validate_fn=contract.validate_release,
                       prepare_fn=lambda *args: seen.append("prepare"))
    assert not seen and not list(tmp_path.iterdir())


def test_projection_excludes_gold_and_does_not_authorize_draft():
    with pytest.raises(ValueError, match="unauthorized"):
        contract.worker_projection(release(), "e" * 64)
    value = approved(release())
    projected = contract.worker_projection(value, "e" * 64)
    assert not any("gold" in k for k in projected)
    assert value["gold_path"] not in json.dumps(projected)
    assert projected["parent_release_sha256"] == "e" * 64
    assert projected["input"] == value["input"]
    assert "SLURM_JOB_ID" not in json.dumps(projected)


def synthetic_tar(tmp_path, monkeypatch, *, extra=None):
    blobs = {"evidence.jsonl": b"synthetic corpus", "validation-protocol.json": b"synthetic tasks"}
    monkeypatch.setattr(contract, "CORPUS_SHA", sha(blobs["evidence.jsonl"]))
    monkeypatch.setattr(contract, "OLD_PROTOCOL_SHA", sha(blobs["validation-protocol.json"]))
    manifests = {}
    for key in ("generator", "reranker"):
        content = b"NOT MODEL WEIGHTS"
        raw = json.dumps({"synthetic.bin": sha(content)}).encode()
        manifests[key] = sha(raw)
        blobs[f"models/{key}/model_manifest.json"] = raw
        blobs[f"models/{key}/model/synthetic.bin"] = content
    monkeypatch.setattr(contract, "MANIFEST_BYTES", manifests)
    if extra:
        blobs[extra] = b"SYNTHETIC unrelated payload"
    path = tmp_path / "input.tar"
    with tarfile.open(path, "w") as bundle:
        for name, content in blobs.items():
            info = tarfile.TarInfo(name)
            info.size = len(content)
            bundle.addfile(info, io.BytesIO(content))
    return path


def test_inference_extraction_omits_historical_sidecars(tmp_path, monkeypatch):
    archive = synthetic_tar(tmp_path, monkeypatch, extra="args-validation.json")
    target = tmp_path / "input"
    result = contract.inspect_input(archive, contract.digest(archive), full=True, target=target)
    assert result["status"] == "verified" and not result["gold_extracted"]
    assert not (target / "args-validation.json").exists()
    assert (target / "evidence.jsonl").is_file()
    with pytest.raises(ValueError, match="new_input_target"):
        contract.inspect_input(archive, contract.digest(archive), full=True, target=target)


@pytest.mark.parametrize("extra", ["gold.json", "../escape", "/absolute", "models/generator/model/extra.py"])
def test_unapproved_input_never_extracted(tmp_path, monkeypatch, extra):
    archive = synthetic_tar(tmp_path, monkeypatch, extra=extra)
    with pytest.raises(ValueError):
        contract.inspect_input(archive, contract.digest(archive), full=True, target=tmp_path / "input")
    assert not (tmp_path / "input").exists()


def test_wrong_transport_and_missing_assets_fail_closed(tmp_path, monkeypatch):
    archive = synthetic_tar(tmp_path, monkeypatch)
    with pytest.raises(ValueError, match="transport_hash"):
        contract.inspect_input(archive, "f" * 64, full=True)
    result = inventory(tmp_path / "absent.tar", tmp_path / "absent-gold", tmp_path)
    assert result["asset_status"] == "needs_assets"
    assert result["ready_for_model_execution"] is False
    assert set(result["runtime"].values()) == {"unobserved"}


@pytest.mark.skipif(os.name != "posix", reason="native Linux path/process contract in CI")
@pytest.mark.parametrize("case", ["success", "prepare_fail", "worker_fail", "score_fail"])
def test_cloud_supervisor_real_boundary_and_gold_projection(tmp_path, case):
    value = approved(contract.draft("a" * 40, "b" * 64, "c" * 64,
                                    root=tmp_path.as_posix(), run_id="native-synthetic"))
    Path(value["work"]).parent.mkdir()
    Path(value["output"]).parent.mkdir()
    release_path = tmp_path / "authorized.json"
    ordered_write(release_path, value)
    seen = []

    def prep(value, work):
        seen.append("prepare")
        if case == "prepare_fail":
            raise RuntimeError("synthetic preparation failure")
        work.mkdir()

    def worker(command, allocation, inference, seconds, **kwargs):
        seen.append("worker")
        assert command[3] == "worker" and 0 < seconds <= 6000
        payload = json.loads(Path(command[5]).read_bytes())
        assert payload == contract.worker_projection(value, contract.digest(release_path))
        assert value["gold_path"] not in json.dumps(payload)
        return {"child_started": True, "child_reaped": True,
                "returncode": 1 if case == "worker_fail" else 0, "interrupted": None}

    def scorer(command, **kwargs):
        seen.append("score")
        assert command[3] == "score"
        output = Path(value["output"])
        assert (output / "allocation/worker-exit.json").exists()
        assert (output / "cost-before-quality.json").exists()
        if case == "score_fail":
            raise RuntimeError("synthetic scoring failure")
        ordered_write(output / "compact.json", {"synthetic": True})

    result = run_supervised(value, release_path, contract.digest(release_path), tmp_path,
                            Path(value["work"]), validate_fn=contract.validate_release,
                            prepare_fn=prep, worker_fn=worker, scorer_fn=scorer,
                            stage_command=lambda stage: entry.projection_command(value, release_path, contract.digest(release_path), stage),
                            execution_identity={"backend": contract.BACKEND, "instance_id": "synthetic-not-cloud"})
    assert result["status"] == ("completed" if case == "success" else "failed_unscored")
    assert seen == (["prepare"] if case == "prepare_fail" else ["prepare", "worker"] if case == "worker_fail" else ["prepare", "worker", "score"])
    cost = json.loads((Path(value["output"]) / "cost-before-quality.json").read_bytes())
    assert cost["planned_slots"] == 160 and cost["worker_proof"]["child_reaped"] is (case != "prepare_fail")


@pytest.mark.skipif(os.name != "posix", reason="native Linux path contract in CI")
def test_path_escape_and_reused_run(tmp_path):
    value = approved(contract.draft("a" * 40, "b" * 64, "c" * 64,
                                    root=tmp_path.as_posix(), run_id="paths-synthetic"))
    contract.check_paths(value, new_run=True)
    Path(value["output"]).mkdir(parents=True)
    with pytest.raises(ValueError, match="already_exists"):
        contract.check_paths(value, new_run=True)
    other = copy.deepcopy(value)
    other["input_archive"] = str(tmp_path / "escape" / "input.tar")
    (tmp_path / "escape").symlink_to(tmp_path.parent, target_is_directory=True)
    with pytest.raises(ValueError, match="symlink"):
        contract.check_paths(other, new_run=False)


def test_legacy_or_synthetic_runtime_rejected_before_probe(tmp_path, monkeypatch):
    path = tmp_path / "receipt.json"
    ordered_write(path, {"backend": "spartan", "synthetic": True})
    monkeypatch.setattr(entry, "observe_runtime", lambda *args: pytest.fail("must not probe"))
    with pytest.raises(ValueError, match="legacy_runtime"):
        entry.verify_runtime({"runtime_receipt": str(path), "runtime_receipt_sha256": contract.digest(path)})


def test_runtime_refuses_synthetic_instance_identity(monkeypatch):
    monkeypatch.setattr(entry.platform, "system", lambda: "Linux")
    monkeypatch.delenv("SLURM_JOB_ID", raising=False)
    with pytest.raises(ValueError, match="instance_identity"):
        entry.observe_runtime(release(), "synthetic-placeholder")


@pytest.mark.skipif(os.name != "posix", reason="native Linux projection confinement in CI")
@pytest.mark.parametrize("mutation", ["none", "output", "purpose", "generation", "parent", "extra"])
def test_projection_requires_exact_contract_and_supervisor_binding(tmp_path, mutation):
    value = approved(contract.draft("a" * 40, "b" * 64, "c" * 64,
                                    root=tmp_path.as_posix(), run_id="projection-test"))
    output = Path(value["output"])
    (output / "allocation").mkdir(parents=True)
    Path(value["work"]).mkdir(parents=True)
    parent = "e" * 64
    ordered_write(output / "reserved.json", {"release_sha256": parent, "backend": contract.BACKEND})
    command = entry.projection_command(value, tmp_path / "authorized", parent, "worker")
    path = Path(command[5])
    projected = json.loads(path.read_bytes())
    if mutation == "none":
        contract.validate_projection(projected, path, contract.digest(path))
        return
    if mutation == "output":
        projected["output"] = str(tmp_path / "arbitrary-output")
    elif mutation == "purpose":
        projected["purpose"] = "old-protocol"
    elif mutation == "generation":
        projected["generation_contract"]["seed"] = 3
    elif mutation == "parent":
        projected["parent_release_sha256"] = "f" * 64
    else:
        projected["gold_path"] = "forbidden"
    # Even a rehashed mutation cannot reuse the supervisor's reservation.
    with pytest.raises(ValueError):
        contract.validate_projection(projected, path, sha(json.dumps(projected).encode()))


def test_source_hash_and_all_importable_files_are_checked(tmp_path, monkeypatch):
    source = tmp_path / "source"
    source.mkdir()
    files = {"SOURCE_REVISION": b"a" * 40 + b"\n", contract.ENTRY: b"# synthetic entry\n",
             contract.SELECTION: b"synthetic selection", "src/climate_rag/example.py": b"# fixture\n"}
    archive = tmp_path / "source.tar"
    with tarfile.open(archive, "w") as bundle:
        for name, content in files.items():
            local = source / name
            local.parent.mkdir(parents=True, exist_ok=True)
            local.write_bytes(content)
            member = tarfile.TarInfo(name)
            member.size = len(content)
            bundle.addfile(member, io.BytesIO(content))
    monkeypatch.setattr(contract, "SELECTION_SHA", sha(files[contract.SELECTION]))
    value = {"source": str(source), "source_archive": str(archive), "source_archive_sha256": contract.digest(archive),
             "source_git": "a" * 40, "entry_sha256": sha(files[contract.ENTRY])}
    contract.verify_source(value, source)
    (source / "scripts/extra.py").write_text("# stale editable code")
    with pytest.raises(ValueError, match="extra_code"):
        contract.verify_source(value, source)
    (source / contract.ENTRY).write_text("# stale entry")
    with pytest.raises(ValueError, match="source_member_drift"):
        contract.verify_source(value, source)


def test_actual_import_must_belong_to_hashed_distribution(tmp_path, monkeypatch):
    from types import SimpleNamespace

    path = tmp_path / "shadowed.py"
    monkeypatch.setattr(entry.importlib, "import_module", lambda name: SimpleNamespace(__file__=str(path)))
    with pytest.raises(ValueError, match="outside_inventory"):
        entry.imported_runtime({str(tmp_path / "pristine.py"): "a" * 64})


@pytest.mark.parametrize("failure", ["unreaped", "no_cost", "wire_audit", "cost_drift"])
def test_scorer_only_gold_stays_closed_until_exit_wire_and_cost_checks(tmp_path, monkeypatch, failure):
    import run_targeted_replay as scorer
    from transformers.models.auto.tokenization_auto import AutoTokenizer

    (tmp_path / "allocation").mkdir()
    ordered_write(tmp_path / "allocation/worker-exit.json", {
        "child_started": True, "child_reaped": failure != "unreaped",
        "returncode": 0, "interrupted": None})
    if failure != "no_cost":
        ordered_write(tmp_path / "cost-before-quality.json", {
            "planned_slots": 160, "completed_slots": 0, "generation": {}, "reranker": {}})
    (tmp_path / "inference").mkdir()
    ordered_write(tmp_path / "inference/run.json", {"protocol": "synthetic", "runs": []})
    monkeypatch.setattr(scorer, "load_inputs", lambda path: ([], [], {}))
    monkeypatch.setattr(AutoTokenizer, "from_pretrained", lambda *args, **kwargs: None)

    def wire_audit(*args, **kwargs):
        if failure == "wire_audit":
            raise ValueError("synthetic_wire_failure")

    monkeypatch.setattr(scorer, "audit", wire_audit)
    real_read = Path.read_bytes
    gold = tmp_path / "never_read_gold.json"

    def guarded_read(path):
        assert path != gold, "gold was read before all prerequisites passed"
        return real_read(path)

    monkeypatch.setattr(Path, "read_bytes", guarded_read)
    with pytest.raises(ValueError):
        scorer.score_after_exit({"output": str(tmp_path), "purpose": "synthetic", "generation_contract": {}},
                                tmp_path, gold_path=gold)
