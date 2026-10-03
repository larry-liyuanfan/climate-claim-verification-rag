"""Synthetic scorer entry-point integration; no model or external gold/data."""
from __future__ import annotations

from functools import partial
import hashlib
import io
import json
from pathlib import Path
import tarfile

import pytest

from climate_rag.scifact_document_verifier import ARMS, DocumentJournal, PROTOCOL, run_episode
from climate_rag.scifact_natural_contract import base_state
from climate_rag.scifact_read_continuation import ordered_write
from climate_rag.scifact_terminal import source_from_abstract
from climate_rag.scifact_utility_contract import identity
from climate_rag.scifact_utility_runtime import run_matrix
import prepare_scifact_document_verifier as preparation
import score_scifact_document_verifier as scoring
from run_scifact_grounding_train_operator import sha
from test_bounded_scifact_runtime import ABSTAIN
from test_scifact_document_verifier import Backend, inputs, verdict
from test_scifact_natural_runtime import Backend as NaturalBackend


IDS = list(range(1, 25))


def selection(output, monkeypatch):
    prepared = output / "prepared"
    prepared.mkdir(parents=True)
    ordered_write(prepared / "selection.json", {"selected": [{"id": i} for i in IDS]})
    digest = sha(prepared / "selection.json")
    monkeypatch.setattr(scoring, "SELECTION_SHA", digest)
    return digest


def read(path):
    return json.loads(path.read_bytes())


@pytest.mark.parametrize("fault", ["nonzero_exit", "missing_episode", "unknown_usage", "unassigned_call"])
def test_no_quality_gate_writes_cost_without_tokenizer_or_gold(tmp_path, monkeypatch, fault):
    selection(tmp_path, monkeypatch)
    ordered_write(tmp_path / "worker-exit.json", {
        "child_reaped": True, "returncode": 1 if fault == "nonzero_exit" else 0,
        "interrupted": None,
    })
    for claim_id in IDS:
        for arm in ARMS:
            if fault == "missing_episode" and (claim_id, arm) == (IDS[-1], ARMS[-1]):
                continue
            slot = tmp_path / "inference" / f"{claim_id}-{arm}"
            slot.mkdir(parents=True)
            ordered_write(slot / "result.json", {})  # Must never be decoded on these gates.
    if fault in {"unknown_usage", "unassigned_call"}:
        ledger = tmp_path / "inference/ledger"
        ledger.mkdir()
        ordered_write(ledger / "g00.reserved.json", {
            "slot": "outside-the-48-slots" if fault == "unassigned_call" else "1-fixed",
        })
        ordered_write(ledger / "g00.finished.json", {
            "usage_known": fault == "unassigned_call",
            "usage": {"input_tokens": 11, "output_tokens": 3} if fault == "unassigned_call" else None,
            "elapsed_ms": 1,
        })

    def forbidden(*args, **kwargs):
        pytest.fail("early no-quality gate must not load tokenizer, audit rows, or open gold")

    monkeypatch.setattr(scoring, "audit_episode", forbidden)
    monkeypatch.setattr(scoring.tarfile, "open", forbidden)
    monkeypatch.setattr(scoring, "select_complete_fit", forbidden)
    result = scoring.score_after_exit(tmp_path, forbidden, {}, root=tmp_path / "unused-root")
    cost = read(tmp_path / "cost-before-gold.json")
    assert result == read(tmp_path / "no-quality.json")
    assert result["status"] == "no_quality" and result["gold_read"] is False
    assert result["reason"] == "worker_failure_missing_episode_or_unknown_cost"
    assert result["cost_sha256"] == sha(tmp_path / "cost-before-gold.json")
    assert cost["gold_read"] is False and cost["planned_slots"] == 48
    assert cost["exit_proof_sha256"] == sha(tmp_path / "worker-exit.json")
    assert cost["missing_episodes"] == int(fault == "missing_episode")
    physical = cost["physical_generation_cost"]
    assert physical["unique_physical_calls"] == int(fault in {"unknown_usage", "unassigned_call"})
    assert physical["unknown_usage_attempts"] == int(fault == "unknown_usage")
    assert cost["unassigned_physical_generation_cost"]["unique_physical_calls"] == int(fault == "unassigned_call")
    if fault == "unknown_usage":
        assert physical["total_tokens"] is None
    assert not (tmp_path / "quality.json").exists()


def test_success_audits_all_48_physical_episodes_before_synthetic_gold(tmp_path, monkeypatch):
    output = tmp_path / "run"
    digest = selection(output, monkeypatch)
    monkeypatch.setattr(preparation, "SELECTION_SHA", digest)
    claims = [{"id": i, "claim": "Complete immutable fixture claim."} for i in IDS]
    _, corpus = inputs(n=5)
    sources = [source_from_abstract(doc) for doc in corpus.values()]

    # Build real, sealed historical prefixes locally. Only synthetic actions are used.
    original = tmp_path / "original"
    (original / "prepared").mkdir(parents=True)
    (original / "prepared/selection.json").write_bytes((output / "prepared/selection.json").read_bytes())
    ordered_write(original / "reserved.json", {
        "source_git": preparation.ORIGINAL_SOURCE, "release_sha256": preparation.ORIGINAL_RELEASE,
    })
    old_backend = NaturalBackend([ABSTAIN] * 72)
    run_matrix(claims, old_backend, lambda query, k: sources[:k],
               lambda query, candidates: candidates, corpus, original / "inference", natural_fit=True)
    inventory_sha = identity(preparation.input_inventory(original))
    frames = preparation.load_frames(claims, corpus, old_backend.base.tokenizer, inventory_sha, original)
    monkeypatch.setattr(scoring, "load_frames", partial(preparation.load_frames, original=original))

    prepared = output / "prepared/inference"
    prepared.mkdir()
    ordered_write(prepared / "claims.json", claims)
    corpus_bytes = b"".join((json.dumps({
        "doc_id": doc.doc_id, "title": doc.title, "abstract": list(doc.sentences),
        "structured": doc.structured,
    }) + "\n").encode() for doc in corpus.values())
    (prepared / "corpus.jsonl").write_bytes(corpus_bytes)
    monkeypatch.setattr(scoring, "CORPUS_SHA", sha(prepared / "corpus.jsonl"))
    ordered_write(output / "prepared/preparation.json", {"claims_sha256": sha(prepared / "claims.json")})

    actions = []
    for frame in frames:
        actions.extend(verdict(doc, "INSUFFICIENT") for doc in frame["document_order"][:4])
        actions.extend([ABSTAIN, ABSTAIN])  # Fixed terminal, then direct adaptive terminal.
    backend = Backend(actions)
    inference = output / "inference"
    inference.mkdir()
    ordered_write(inference / "initial-frames.json", frames)
    journal = DocumentJournal(backend, inference / "ledger", max_generations=240,
                              protocol=PROTOCOL, physical_guard=base_state)
    for claim, frame in zip(claims, frames, strict=True):
        for arm in ARMS:
            run_episode(claim["id"], arm, frame, journal, corpus, inference / f"{claim['id']}-{arm}")
    assert backend.calls == 144
    ordered_write(output / "worker-exit.json", {"child_reaped": True, "returncode": 0, "interrupted": None})

    # This archive contains only test-written labels, never the real TRAIN member.
    root = tmp_path / "synthetic-root"
    (root / "envs").mkdir(parents=True)
    archive = root / "envs" / scoring.ARCHIVE
    gold_bytes = b"".join((json.dumps(dict(claim, evidence={}, cited_doc_ids=[])) + "\n").encode()
                          for claim in claims)
    info = tarfile.TarInfo(scoring.PREP + "/gold/claims_train.jsonl")
    info.size = len(gold_bytes)
    with tarfile.open(archive, "w") as bundle:
        bundle.addfile(info, io.BytesIO(gold_bytes))
    monkeypatch.setattr(scoring, "ARCHIVE_SHA", sha(archive))
    monkeypatch.setattr(scoring, "TRAIN_SHA", hashlib.sha256(gold_bytes).hexdigest())

    audited, gold_access = [], []
    expected_slots = [(i, arm) for i in IDS for arm in ARMS]
    real_audit, real_open = scoring.audit_episode, tarfile.open

    def audit_before_gold(row, *args, **kwargs):
        assert not gold_access
        assert read(output / "cost-before-gold.json")["gold_read"] is False
        result = real_audit(row, *args, **kwargs)
        audited.append((row["claim_id"], row["arm"]))
        return result

    def open_after_all_audits(path, *args, **kwargs):
        assert Path(path) == archive
        assert audited == expected_slots
        assert read(output / "cost-before-gold.json")["physical_generation_cost"]["unique_physical_calls"] == 144
        gold_access.append(str(path))
        return real_open(path, *args, **kwargs)

    def tokenizer_after_cost():
        assert not audited and not gold_access
        assert (output / "cost-before-gold.json").is_file()
        return backend.base.tokenizer

    monkeypatch.setattr(scoring, "audit_episode", audit_before_gold)
    monkeypatch.setattr(scoring.tarfile, "open", open_after_all_audits)
    result = scoring.score_after_exit(output, tokenizer_after_cost,
                                      {"initial_inventory_sha256": inventory_sha}, root=root)
    assert result == read(output / "quality.json")
    assert result["status"] == "scored"
    assert audited == expected_slots and gold_access == [str(archive)]
    assert not (output / "no-quality.json").exists()
    assert [result["arms"][arm]["planned"] for arm in ARMS] == [24, 24]
    assert [result["arms"][arm]["nei_correct"] for arm in ARMS] == [24, 24]
    assert [result["arm_physical_generation_cost"][arm]["unique_physical_calls"] for arm in ARMS] == [120, 24]
    cost = read(output / "cost-before-gold.json")
    assert cost["missing_episodes"] == cost["physical_generation_cost"]["unknown_usage_attempts"] == 0
    assert result["cost_sha256"] == sha(output / "cost-before-gold.json")
