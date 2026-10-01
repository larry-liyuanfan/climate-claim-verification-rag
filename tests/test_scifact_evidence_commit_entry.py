"""72-slot synthetic scorer boundary; exact release remains non-executable draft."""
import copy
import hashlib
import io
import json
from pathlib import Path
import tarfile

import pytest

from climate_rag.scifact_evidence_commit import ARMS, CALL_CAPS, PROTOCOL
from climate_rag.scifact_evidence_commit_runtime import CommitJournal, run_episode
from climate_rag.scifact_natural_contract import base_state
from climate_rag.scifact_read_continuation import ordered_write
import run_scifact_evidence_commit_operator as operator
import score_scifact_document_verifier as shared
import score_scifact_evidence_commit as scoring
from run_scifact_grounding_train_operator import sha
from test_scifact_evidence_commit import Backend, RUN, choose, inputs, verdict


def setup(output, monkeypatch):
    (output/"prepared/inference").mkdir(parents=True)
    ids = list(range(1, 25))
    ordered_write(output/"prepared/selection.json", {"selected": [{"id":i} for i in ids]})
    monkeypatch.setattr(shared, "SELECTION_SHA", sha(output/"prepared/selection.json"))
    ordered_write(output/"reserved.json", {"release_sha256": RUN})
    ordered_write(output/"worker-exit.json", {"child_reaped": True, "returncode": 0, "interrupted": None, "release_sha256":RUN})
    return ids


@pytest.mark.parametrize("fault", ["worker_failure", "missing", "unknown", "unassigned"])
def test_three_arm_gate_cost_before_any_audit_or_gold(tmp_path, monkeypatch, fault):
    ids = setup(tmp_path, monkeypatch)
    if fault == "worker_failure":
        path = tmp_path/"worker-exit.json"
        path.write_text(json.dumps({"child_reaped":True, "returncode":1, "interrupted":None, "release_sha256":RUN}))
    for i in ids:
        for arm in ARMS:
            if fault == "missing" and (i, arm) == (24, "adaptive"):
                continue
            p = tmp_path/"inference"/f"{i}-{arm}"
            p.mkdir(parents=True)
            ordered_write(p/"result.json", {})
    if fault in {"unknown", "unassigned"}:
        p = tmp_path/"inference/ledger"
        p.mkdir()
        ordered_write(p/"g00.reserved.json", {"slot":"outside" if fault == "unassigned" else "1-fixed_top1"})
        ordered_write(p/"g00.finished.json", {"usage_known":fault == "unassigned", "elapsed_ms":1,
            "usage": {"input_tokens":1, "output_tokens":1} if fault == "unassigned" else None})
    def forbidden(*a, **k):
        pytest.fail("no tokenizer, physical audit or gold allowed")
    monkeypatch.setattr(scoring, "audit_episode", forbidden)
    monkeypatch.setattr(shared.tarfile, "open", forbidden)
    report = scoring.score_after_exit(tmp_path, forbidden, {}, root=tmp_path/"absent", release_sha=RUN)
    assert report["status"] == "no_quality" and report["planned_slots"] == 72 and not report["gold_read"]
    cost = json.loads((tmp_path/"cost-before-gold.json").read_bytes())
    assert set(cost["arm_physical_generation_cost"]) == set(ARMS)
    assert not (tmp_path/"quality.json").exists()


@pytest.mark.parametrize("replay", [False, True])
def test_72_actual_journal_slots_reassembled_before_synthetic_gold(tmp_path, monkeypatch, replay):
    output = tmp_path/"run"
    reports = tmp_path/"replay" if replay else output
    ids = setup(output, monkeypatch)
    frame, corpus = inputs(5)
    frames = [copy.deepcopy(frame) for _ in ids]
    claims = [{"id":i, "claim":frame["observation"]["immutable_claim"]} for i in ids]
    ordered_write(output/"prepared/inference/claims.json", claims)
    ordered_write(output/"prepared/preparation.json", {
        "claims_sha256":sha(output/"prepared/inference/claims.json")})
    corpus_bytes = b"".join((json.dumps({"doc_id":d.doc_id, "title":d.title,
        "abstract":list(d.sentences), "structured":False})+"\n").encode() for d in corpus.values())
    (output/"prepared/inference/corpus.jsonl").write_bytes(corpus_bytes)
    monkeypatch.setattr(shared, "CORPUS_SHA", sha(output/"prepared/inference/corpus.jsonl"))
    monkeypatch.setattr(shared, "load_frames", lambda *a: frames)
    actions = []
    for _ in ids:
        actions += [verdict(), verdict(), *[verdict(f"c{i}", "INSUFFICIENT") for i in (8, 9, 10)],
                    {"action":"verify", "source_id":"c7"}, verdict(), choose]
    backend = Backend(actions)
    inference = output/"inference"
    inference.mkdir()
    ordered_write(inference/"initial-frames.json", frames)
    journal = CommitJournal(backend, inference/"ledger", max_generations=240,
                            protocol=PROTOCOL, physical_guard=base_state, run_identity=RUN)
    for claim, f in zip(claims, frames, strict=True):
        for arm in ARMS:
            run_episode(claim["id"], arm, f, journal, corpus, inference/f"{claim['id']}-{arm}", run_identity=RUN)
    assert backend.calls == 192
    root = tmp_path/"synthetic-root"
    (root/"envs").mkdir(parents=True)
    archive = root/"envs"/shared.ARCHIVE
    gold_bytes = b"".join((json.dumps(dict(c, evidence={"77":[{"label":"SUPPORT", "sentences":[7,2]}]},
        cited_doc_ids=[77]))+"\n").encode() for c in claims)
    info = tarfile.TarInfo(shared.PREP+"/gold/claims_train.jsonl")
    info.size = len(gold_bytes)
    with tarfile.open(archive, "w") as bundle:
        bundle.addfile(info, io.BytesIO(gold_bytes))
    monkeypatch.setattr(shared, "ARCHIVE_SHA", sha(archive))
    monkeypatch.setattr(shared, "TRAIN_SHA", hashlib.sha256(gold_bytes).hexdigest())
    seen = []
    original_audit, original_open = scoring.audit_episode, tarfile.open
    def physical_audit(row, *args, **kwargs):
        assert (reports/"cost-before-gold.json").is_file()
        seen.append((row["claim_id"], row["arm"]))
        return original_audit(row, *args, **kwargs)
    def after_audit(path, *args, **kwargs):
        assert Path(path) == archive and len(seen) == 72
        return original_open(path, *args, **kwargs)
    monkeypatch.setattr(scoring, "audit_episode", physical_audit)
    monkeypatch.setattr(shared.tarfile, "open", after_audit)
    if replay:
        ordered_write(output/"no-quality.json", {"status":"no_quality", "gold_read":False})
    before = {p.relative_to(output): sha(p) for p in output.rglob("*") if p.is_file()}
    report = scoring.score_after_exit(output, lambda: backend.base.tokenizer,
                                     {"initial_inventory_sha256":"synthetic"}, root, release_sha=RUN,
                                     report_directory=reports if replay else None)
    assert report["status"] == "scored" and len(seen) == 72
    assert [report["arms"][a]["positive_correct"] for a in ARMS] == [24,24,24]
    assert [report["arm_physical_generation_cost"][a]["unique_physical_calls"] for a in ARMS] == [24,96,72]
    assert report["comparison_limits"]["no_automatic_agent_gain"] is True
    if replay:
        assert before == {p.relative_to(output): sha(p) for p in output.rglob("*") if p.is_file()}
        assert (reports/"quality.json").is_file() and not (output/"quality.json").exists()
        with pytest.raises(FileExistsError):
            scoring.score_after_exit(output, lambda: backend.base.tokenizer,
                                     {}, root, release_sha=RUN, report_directory=reports)


@pytest.mark.parametrize("relative", [".", "run", "run/replay"])
def test_replay_reports_cannot_overlap_original(tmp_path, monkeypatch, relative):
    output = tmp_path/"run"
    setup(output, monkeypatch)
    with pytest.raises(ValueError, match="replay_reports_must_be_disjoint"):
        scoring.score_after_exit(output, lambda: None, {}, release_sha=RUN,
                                 report_directory=tmp_path/relative)


def test_replay_no_quality_preserves_original(tmp_path, monkeypatch):
    output, reports = tmp_path/"run", tmp_path/"reports"
    setup(output, monkeypatch)
    ordered_write(output/"no-quality.json", {"status":"original_failure"})
    before = {p.relative_to(output): sha(p) for p in output.rglob("*") if p.is_file()}
    report = scoring.score_after_exit(output, lambda: pytest.fail("no tokenizer allowed"), {},
                                     release_sha=RUN, report_directory=reports)
    assert report["status"] == "no_quality" and not report["gold_read"]
    assert (reports/"no-quality.json").is_file()
    assert before == {p.relative_to(output): sha(p) for p in output.rglob("*") if p.is_file()}


def test_cpu_candidate_release_scope_is_not_gpu_authorization(tmp_path, monkeypatch):
    release = dict(copy.deepcopy(operator.FROZEN_FIELDS), authorization="DRAFT_CPU_READY_NOT_AUTHORIZED",
        source_git="a"*40, source_archive_sha256="b"*64, wrapper_sha256="c"*64)
    with pytest.raises(ValueError, match="draft"):
        operator.validate_release(release)
    release["authorization"] = "coordinator_exact_hash_release"
    operator.validate_release(release)  # Synthetic contract only; no scheduler calls.
    assert release["planned_episodes"] == 72 and release["max_generator_calls"] == 240
    assert release["call_caps"] == CALL_CAPS
    for key, value in (("planned_episodes",48), ("call_caps",dict(CALL_CAPS, adaptive=6))):
        with pytest.raises(ValueError, match="contract"):
            operator.validate_release(dict(release, **{key:value}))
    observed = {"model_loaded":False, "generation_calls":0}
    monkeypatch.setattr(operator, "OUTPUT", tmp_path/"new-output")
    monkeypatch.setattr(operator, "verify_runtime_receipt", lambda r: observed)
    assert operator.start_attempt(release, RUN, "synthetic-not-submitted") == observed
    with pytest.raises(FileExistsError):
        operator.start_attempt(release, RUN, "duplicate")


@pytest.mark.parametrize("success", ["fixed_top1", "fixed_all", "adaptive", None])
def test_summary_all_three_positive_recovery_patterns(success):
    from climate_rag.scifact_grounding import GoldClaim, Rationale
    _, corpus = inputs(1)
    gold = GoldClaim(1, "claim", {77:(Rationale("SUPPORT", (7,2)),)}, (77,))
    bad = {"state":"unresolved", "prediction":None, "verification_feedback":[], "physical_calls":1, "elapsed_seconds":1}
    good = dict(bad, state="valid_terminal", prediction={"id":1,"evidence":{"77":{"label":"SUPPORT","sentences":[7,2]}}})
    rows = {(1,a): good if a == success else bad for a in ARMS}
    report = shared.summarize([1], [gold], rows, corpus, ARMS, "fixed_all")
    assert report["positive_recovered_arms"] == ([success] if success else [])
    assert ("remains_zero" in report["hypothesis"]) == (success is None)
    assert report["adaptive_minus_fixed_positive_correct_by_arm"] == {
        a:int(success == "adaptive")-int(success == a) for a in ("fixed_top1","fixed_all")}


def test_actual_reservation_watchdog_and_bounded_worker_termination(tmp_path, monkeypatch):
    """Real functions and actual reservation, synthetic child/signal transport."""
    import types
    import run_scifact_natural_operator as watchdog_module
    import run_scifact_evidence_note_operator as worker_module
    from test_scifact_evidence_commit import run
    original = Backend.generate
    checked = []
    def observe(self, obs, schema, maximum, seconds):
        # This is called after run_episode has written its actual reservation.
        inference = tmp_path/"inference"
        # Test run() uses 'episode'; watchdog production glob expects claim-arm.
        reservation = json.loads((tmp_path/"episode/reserved.json").read_bytes())
        slot = inference/"1-fixed_top1"
        slot.mkdir(parents=True)
        ordered_write(slot/"reserved.json", reservation)
        now = reservation["started_unix"]
        with monkeypatch.context() as patch:
            patch.setattr(watchdog_module, "time", types.SimpleNamespace(time=lambda: now+119))
            operator.slot_watchdog(inference)  # Real unchanged watchdog, not expired.
        with monkeypatch.context() as patch:
            fake = types.SimpleNamespace(pid=42, killed=False, waited=False)
            fake.poll = lambda: -9 if fake.killed else None
            def wait():
                fake.waited = True
                return -9
            fake.wait = wait
            patch.setattr(watchdog_module, "time", types.SimpleNamespace(time=lambda: now+120))
            patch.setattr(worker_module, "os", types.SimpleNamespace(name="posix",
                killpg=lambda pid, sig: setattr(fake, "killed", True)))
            patch.setattr(worker_module, "signal", types.SimpleNamespace(SIGTERM=15, SIGINT=2,
                SIGKILL=9, SIG_IGN=1, signal=lambda *a: 0))
            patch.setattr(worker_module.subprocess, "Popen", lambda *a, **k: fake)
            allocation = tmp_path/"worker"
            allocation.mkdir()
            proof = operator.bounded_worker(["synthetic-child"], allocation, inference, 300, watchdog=operator.slot_watchdog)
            assert proof["interrupted"] == "TimeoutError" and proof["child_reaped"] is True
            assert fake.killed and fake.waited and proof["returncode"] == -9
        checked.append(True)
        return original(self, obs, schema, maximum, seconds)
    monkeypatch.setattr(Backend, "generate", observe)
    row, *_ = run(tmp_path, [verdict()], "fixed_top1")
    assert checked == [True] and row["state"] == "valid_terminal"
