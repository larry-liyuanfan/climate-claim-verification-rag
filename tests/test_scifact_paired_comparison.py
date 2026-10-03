"""Synthetic process/config contracts only: zero model weights or judgments."""
import copy
import json
from pathlib import Path
import types

import pytest
import torch
from transformers import GenerationConfig

from climate_rag.scifact_evidence_commit import EvidenceCommitProvider, CommitState
from climate_rag.scifact_evidence_commit_runtime import CommitJournal, audit_episode
from climate_rag.scifact_generation import FILE_DEFAULTS, GenerationBinding, frozen_contract, audit_generation
from climate_rag.scifact_natural_contract import base_state
from climate_rag.scifact_utility_runtime import ledger_cost
from climate_rag.scifact_read_continuation import ordered_write
from run_scifact_evidence_commit_operator import release_fields, validate_release
from run_scifact_evidence_commit_paired import run_pair
import scifact_paired_comparison as pair
import scifact_evidence_input as inputs_adapter
from test_scifact_document_decoder import real_provider
from test_scifact_evidence_commit import inputs, verdict, RUN, run


def release(tmp_path, monkeypatch):
    monkeypatch.setattr(pair, "OUTPUT", tmp_path / "pair")
    children = [dict(release_fields({"protocol": protocol, "input_protocol": inputs_adapter.PROTOCOL,
                                    "paired_comparison": pair.PAIR}),
                     source_git="a"*40, source_archive_sha256="b"*64, wrapper_sha256="c"*64,
                     authorization="coordinator_exact_hash_release") for protocol in pair.VERSIONS]
    return dict(pair.pair_fields(children), authorization="coordinator_exact_hash_release")


def quality(protocol):
    from climate_rag.scifact_scoring import score_original, ClaimPrediction
    from climate_rag.scifact_grounding import GoldClaim
    cases = [{"claim_id": i, "positive": i <= 15, "strict_whole_answer": i <= 3}
             for i in range(1,25)]
    arm = {"planned": 24, "cases": cases, "positive_correct": 3, "nei_correct": 0,
           "unresolved": 0, "official_micro": score_original(
               [GoldClaim(i,"synthetic",{},()) for i in range(1,25)],
               [ClaimPrediction(i,{}) for i in range(1,25)])}
    return {"protocol": protocol, "status": "scored", "arms": {a: copy.deepcopy(arm) for a in pair.ARMS}}


def runner(tmp_path, monkeypatch, *, failure=None, unknown=False, time_left=False):
    candidate = release(tmp_path, monkeypatch)
    events = []
    tick = [0.0]
    def extract(*args):
        events.append("extract_once")
        if failure == "extract_termination":
            raise InterruptedError("synthetic_supervisor_signal_during_extract")
        if failure == "extract":
            raise OSError("synthetic")
    def reserve(child, digest, job):
        Path(child["output"]).mkdir()
    def prepare(path, child):
        if failure == "second_prepare" and child["protocol"] == pair.VERSIONS[1]:
            raise InterruptedError("synthetic_external_termination_during_preparation")
        path.mkdir()
        (path/"inference").mkdir()
        ordered_write(path/"preparation.json", {})
    def worker(command, phase, inference, seconds, **kwargs):
        index = len([e for e in events if e.startswith("worker")])
        events.append(f"worker{index}_reaped")
        ledger = inference/"ledger"
        ledger.mkdir(parents=True)
        ordered_write(ledger/"g00.reserved.json", {"slot": "1-fixed_top1"})
        if not unknown:
            ordered_write(ledger/"g00.finished.json", {"usage_known": True,
                "usage": {"input_tokens": 10, "output_tokens": 2}, "elapsed_ms": 1})
        if time_left:
            tick[0] += 2800  # second phase cannot receive its full frozen budget
        return {"child_started": failure != "launch", "child_reaped": failure != "launch",
                "returncode": 0 if failure in (None,"score_only","second_prepare") else 2,
                "interrupted": "TimeoutError" if failure == "timeout" else None}
    def score(child, record):
        events.append("score")
        if failure in ("no_quality","score_only"):
            return {"status": "no_quality", "protocol": child["protocol"]}
        return quality(child["protocol"])
    result = run_pair(candidate, "d"*64, tmp_path/"source", tmp_path,
                      clock=lambda: tick[0], worker=worker, prepare=prepare, reserve=reserve,
                      extract=extract, readonly=lambda _: None, score=score)
    costs = json.loads((pair.OUTPUT/"cost-before-gold.json").read_bytes())
    return result, costs, events


def test_release_two_protocols_identity_budget_and_draft_refusal(tmp_path, monkeypatch):
    candidate = release(tmp_path, monkeypatch)
    pair.validate_pair(candidate)
    for child in candidate["children"]:
        validate_release(child)
        assert child["planned_episodes"] == 72 and child["max_generator_calls"] == 240
        assert child["resource_cap"]["slurm_seconds"] == 2400
        assert child["generation_contract"]["expanded_model_defaults"]["do_sample"] is True
        assert child["generation_contract"]["overrides"]["do_sample"] is False
    for key, value in (("authorization", "DRAFT_CPU_READY_NOT_AUTHORIZED"),
                       ("max_generator_calls",481), ("planned_episodes",48)):
        with pytest.raises(ValueError):
            pair.validate_pair(dict(candidate, **{key:value}))
    bad = copy.deepcopy(candidate)
    bad["children"][1]["frames_sha256"] = "e"*64
    with pytest.raises(ValueError):
        pair.validate_pair(bad)


def test_serial_supervisor_single_extract_all_reaped_before_score(tmp_path, monkeypatch):
    result, costs, events = runner(tmp_path, monkeypatch)
    assert events == ["extract_once", "worker0_reaped", "worker1_reaped", "score", "score"]
    assert result["status"] == "paired_scored"
    assert costs["planned_slots"] == 144 and costs["unique_physical_calls"] == 2
    assert costs["total_tokens"] == {"input_tokens":20,"output_tokens":4}
    assert costs["gold_read"] is False


@pytest.mark.parametrize("failure", ["extract", "launch", "timeout", "no_quality"])
def test_failures_keep_144_slots_no_retry_no_false_success(tmp_path, monkeypatch, failure):
    result, costs, events = runner(tmp_path, monkeypatch, failure=failure)
    assert result["status"] == "no_paired_quality" and result["planned_slots"] == 144
    assert costs["planned_slots"] == 144
    assert len([e for e in events if e.startswith("worker")]) <= 1
    assert costs["unique_physical_calls"] == (0 if failure == "extract" else 1)


def test_unknown_usage_is_lower_bound_and_blocks_second_phase(tmp_path, monkeypatch):
    result, costs, events = runner(tmp_path, monkeypatch, unknown=True)
    assert costs["unique_physical_calls"] == costs["unknown_usage_attempts"] == 1
    assert costs["total_tokens"] is None and costs["phases"][1]["status"] == "not_started"
    assert result["status"] == "no_paired_quality"


def test_exit_zero_but_no_quality_is_not_pair_success(tmp_path, monkeypatch):
    result,costs,events = runner(tmp_path,monkeypatch,failure="score_only")
    assert costs["unique_physical_calls"] == 2 and events.count("score") == 2
    assert result["status"] == "no_paired_quality"


def test_second_preparation_termination_preserves_first_cost_and_unstarted_slots(tmp_path, monkeypatch):
    result,costs,events = runner(tmp_path,monkeypatch,failure="second_prepare")
    assert costs["unique_physical_calls"] == 1 and costs["planned_slots"] == 144
    assert costs["phases"][0]["proof"]["child_reaped"] is True
    assert costs["phases"][1]["status"] == "infrastructure_failed"
    assert costs["phases"][1]["proof"]["child_started"] is False
    assert result["status"] == "no_paired_quality"


def test_shared_preparation_termination_keeps_full_zero_started_roster(tmp_path,monkeypatch):
    result,costs,events = runner(tmp_path,monkeypatch,failure="extract_termination")
    assert costs["planned_slots"] == 144 and costs["unique_physical_calls"] == 0
    assert costs["shared_preparation_error"] == "InterruptedError"
    assert result["status"] == "no_paired_quality" and events == ["extract_once"]


def test_interphase_termination_retains_completed_worker_ledger(tmp_path,monkeypatch):
    import run_scifact_evidence_commit_paired as entry
    original = entry.ledger_cost
    called = [False]
    def interrupted_after_reap(path):
        if not called[0]:
            called[0] = True
            raise InterruptedError("synthetic_signal_between_phases")
        return original(path)
    monkeypatch.setattr(entry,"ledger_cost",interrupted_after_reap)
    result,costs,events = runner(tmp_path,monkeypatch)
    assert costs["unique_physical_calls"] == 1 and costs["planned_slots"] == 144
    assert costs["phases"][0]["proof"]["child_reaped"] is True
    assert costs["phases"][1]["status"] == "not_started"
    assert result["status"] == "no_paired_quality"


def test_preparation_timeout_uses_no_gpu_subprocess(tmp_path,monkeypatch):
    import os
    import time
    from run_scifact_evidence_commit_paired import cpu_stage
    if os.name != "posix":
        pytest.skip("real SIGALRM validated by Linux CI, not emulated as Windows POSIX")
    with pytest.raises(TimeoutError,match="cpu_stage_deadline"):
        cpu_stage(0.01,time.sleep,1)


def test_insufficient_remaining_time_keeps_second_phase_not_started(tmp_path, monkeypatch):
    result, costs, _ = runner(tmp_path, monkeypatch, time_left=True)
    assert result["status"] == "no_paired_quality" and costs["phases"][1]["status"] == "not_started"


def test_comparison_axes_and_micro_not_mean_per_case():
    old, new = [quality(p) for p in pair.VERSIONS]
    new["arms"]["fixed_top1"].update(positive_correct=4)
    new["arms"]["fixed_top1"]["cases"][3]["strict_whole_answer"] = True
    new["arms"]["fixed_top1"]["official_micro"]["metrics"]["abstract_rationalized"]["f1"] = 0.25
    result = pair.compare_quality([old,new])
    fixed = result["fixed_policy_v3_minus_v2"]["fixed_top1"]
    assert fixed["paired_positive_wins"] == fixed["positive_correct_delta"] == 1
    assert fixed["original_official_micro_delta"]["abstract_rationalized"]["f1"] == pytest.approx(0.25)
    assert result["adaptive_minus_fixed_within_version"][pair.VERSIONS[1]]["fixed_top1"]["positive_correct_delta"] == -1
    new["arms"]["adaptive"]["cases"].pop()
    with pytest.raises(ValueError, match="roster"):
        pair.compare_quality([old,new])


def bound_provider(tmp_path, monkeypatch, actions, *, contract=None):
    provider, calls, _ = real_provider(tmp_path, [])
    provider.__class__ = EvidenceCommitProvider
    provider.base._torch = torch
    provider.base.tokenizer.eos_token_id = 151645
    provider.tokenizer_data.eos_token_id = 151645
    provider.base.model.generation_config = GenerationConfig.from_dict(copy.deepcopy(FILE_DEFAULTS))
    monkeypatch.setattr("climate_rag.scifact_generation.checked", lambda *args: b"synthetic_fixture")
    provider.generation_binding = GenerationBinding(
        provider.base, tmp_path, frozen_contract() if contract is None else contract)
    old_decode = provider.base.tokenizer.decode
    provider.base.tokenizer.decode = lambda ids, **kw: old_decode([i for i in ids if int(i) != 151645], **kw)
    iterator = iter(actions)
    def generate(**kwargs):
        calls.append(kwargs)
        assert kwargs["use_model_defaults"] is False
        config = kwargs["generation_config"]
        assert not config.do_sample and config.max_length == len(kwargs["input_ids"][0])+512
        raw = json.dumps(next(iterator), separators=(",", ":"))
        sent = kwargs["input_ids"][0].tolist()
        for token in [*provider.base.tokenizer.encode(raw),151645]:
            assert token in kwargs["prefix_allowed_tokens_fn"](0, torch.tensor(sent))
            sent.append(token)
        return [sent]
    provider.base.model.generate = generate
    return provider, calls


def journal_for(provider, path, frame):
    journal = CommitJournal(provider, path, run_identity=RUN, protocol=pair.VERSIONS[0], physical_guard=base_state)
    from climate_rag.scifact_utility_contract import identity
    journal.slot = "1-fixed_top1"
    journal.episode_identity = f"{journal.protocol}:{RUN}:1:fixed_top1"
    journal.frame_sha = identity(frame)
    return journal


def test_real_journal_effective_config_seed_lmfe_and_tamper(tmp_path, monkeypatch):
    provider, calls = bound_provider(tmp_path, monkeypatch, [verdict()])
    frame, _ = inputs(2)
    state = CommitState(1,"fixed_top1",frame,"synthetic", protocol=pair.VERSIONS[0])
    observation,schema = state.inputs("verify","c7",[])
    journal = journal_for(provider,tmp_path/"ledger",frame)
    response = journal.generate(observation,schema,512,120)
    receipt = json.loads((journal.directory/"g00.generation.json").read_bytes())
    request = json.loads((journal.directory/"g00.reserved.json").read_bytes())
    parser_config = response["diagnostics"]["effective_parser_config"]
    audit_generation(receipt,request,response["diagnostics"],parser_config)
    assert receipt["effective"]["eos_token_id"] == [151645,151643]
    assert receipt["effective"]["temperature"] == 0.6  # inactive sampling setting preserved
    assert receipt["effective_parser_config"] == response["diagnostics"]["effective_parser_config"]
    assert len(calls) == 1 and provider.generation_binding.path is None
    changed = copy.deepcopy(receipt)
    changed["effective"]["do_sample"] = True
    with pytest.raises(ValueError):
        audit_generation(changed,request,response["diagnostics"],parser_config)
    changed = copy.deepcopy(receipt)
    changed["effective_parser_config"]["max_json_array_length"] = 200
    changed_diagnostics = dict(response["diagnostics"],generation_binding=changed)
    with pytest.raises(ValueError):
        audit_generation(changed,request,changed_diagnostics,parser_config)
    # Even if both self-reported copies agree, trusted tokenizer/config wins.
    changed_diagnostics["effective_parser_config"] = changed["effective_parser_config"]
    with pytest.raises(ValueError):
        audit_generation(changed,request,changed_diagnostics,parser_config)


@pytest.mark.parametrize("protocol", pair.VERSIONS)
def test_released_numeric_serialization_full_disk_audit(tmp_path, monkeypatch, protocol):
    from climate_rag.scifact_evidence_commit_runtime import run_episode
    from climate_rag.scifact_generation import expected_parser_config
    from climate_rag.scifact_relation_verifier import QUALIFIERS
    from climate_rag.scifact_utility_contract import identity

    contract = copy.deepcopy(frozen_contract())
    for key, value in contract["expanded_model_defaults"].items():
        if type(value) is float and value.is_integer():
            contract["expanded_model_defaults"][key] = int(value)
    assert contract == frozen_contract()
    assert identity(contract) != identity(frozen_contract())
    action = verdict()
    if protocol == pair.VERSIONS[1]:
        action = {"source_id": "c7", "relation": "SUPPORTS",
                  "qualifiers": {k: "aligned" for k in QUALIFIERS},
                  "direct_sentence_ids": action["sentence_ids"],
                  "background_sentence_ids": [], "minimal_sentence_ids": action["sentence_ids"],
                  "uncertainty": "none"}
    provider, calls = bound_provider(tmp_path, monkeypatch, [action], contract=contract)
    frame, corpus = inputs(2)
    journal = CommitJournal(provider, tmp_path/"ledger", run_identity=RUN,
                            protocol=protocol, physical_guard=base_state)
    run_episode(1, "fixed_top1", frame, journal, corpus, tmp_path/"episode", run_identity=RUN)
    row = json.loads((tmp_path/"episode/result.json").read_bytes())
    assert row["state"] == "valid_terminal" and len(calls) == 1
    # Derive the trusted parser contract from provider tokenizer data, never
    # from the response's self-reported diagnostics. No model weights involved.
    monkeypatch.setattr("lmformatenforcer.integrations.transformers.build_token_enforcer_tokenizer_data",
                        lambda tokenizer, use_bitmask: provider.tokenizer_data)
    parser_config = expected_parser_config(provider.base.tokenizer)
    request = json.loads((journal.directory/"g00.reserved.json").read_bytes())
    saved = json.loads((journal.directory/"g00.generation.json").read_bytes())
    diagnostics = json.loads((journal.directory/"g00.finished.json").read_bytes())["response"]["diagnostics"]
    with pytest.raises(ValueError, match="paired_effective_generation_binding"):
        audit_generation(saved, request, diagnostics, parser_config)
    assert audit_episode(row, frame, journal.directory, tmp_path/"episode/private-responses",
                         provider.base.tokenizer, corpus, run_identity=RUN, protocol=protocol,
                         generation_contract=contract) == row
    changed = copy.deepcopy(saved)
    changed["contract"] = frozen_contract()
    changed["contract_sha256"] = identity(changed["contract"])
    with pytest.raises(ValueError, match="paired_effective_generation_binding"):
        audit_generation(changed, request, dict(diagnostics, generation_binding=changed),
                         parser_config, trusted_contract=contract)
    invalid_contract = copy.deepcopy(contract)
    invalid_contract["overrides"]["max_new_tokens"] = 1024
    with pytest.raises(ValueError, match="paired_unknown_generation_contract"):
        audit_generation(saved, request, diagnostics, parser_config, trusted_contract=invalid_contract)


@pytest.mark.parametrize("where", ["decoder", "tokenizer", "deadline"])
def test_pre_generation_failure_releases_binding_next_slot_can_run(tmp_path, monkeypatch, where):
    provider, calls = bound_provider(tmp_path,monkeypatch,[verdict()])
    frame,_ = inputs(2)
    state = CommitState(1,"fixed_top1",frame,"synthetic",protocol=pair.VERSIONS[0])
    observation,schema = state.inputs("verify","c7",[])
    journal = journal_for(provider,tmp_path/"ledger",frame)
    original_decoder, original_tokenizer = provider.build_decoder, provider.base.tokenizer
    def fail(*args, **kwargs):
        raise ValueError("synthetic_pre_generation")
    if where == "decoder":
        provider.build_decoder = fail
    if where == "tokenizer":
        class BadTokenizer:
            def __getattr__(self,name):
                return getattr(original_tokenizer,name)
            __call__ = fail
        provider.base.tokenizer = BadTokenizer()
    import time
    original_clock = time.perf_counter
    if where == "deadline":
        ticks = iter((0.0,121.0))
        monkeypatch.setattr("climate_rag.local_scifact_provider.time.perf_counter",lambda: next(ticks))
    with pytest.raises(Exception):
        journal.generate(observation,schema,512,120)
    monkeypatch.setattr("climate_rag.local_scifact_provider.time.perf_counter",original_clock)
    assert provider.generation_binding.path is None
    provider.build_decoder,provider.base.tokenizer = original_decoder,original_tokenizer
    journal.slot = "2-fixed_top1"
    journal.episode_identity = f"{journal.protocol}:{RUN}:2:fixed_top1"
    response = journal.generate(observation,schema,512,120)
    assert json.loads(response["raw"]) == verdict() and len(calls) == 1
    costs = ledger_cost(journal.directory)
    assert costs["unique_physical_calls"] == 2 and costs["unknown_usage_attempts"] == 1
    assert costs["total_tokens"] is None
    assert (journal.directory/"g00.finished.json").exists() and (journal.directory/"g01.generation.json").exists()


def test_source_manifest_config_and_loaded_defaults_fail_closed(tmp_path, monkeypatch):
    from climate_rag.scifact_generation import CONFIG_FILE_SHA
    base = types.SimpleNamespace(model=types.SimpleNamespace(generation_config=GenerationConfig.from_dict(FILE_DEFAULTS)))
    with pytest.raises((OSError, ValueError)):
        GenerationBinding(base,tmp_path,frozen_contract())
    assert len(CONFIG_FILE_SHA) == 64
    monkeypatch.setattr("climate_rag.scifact_generation.checked",lambda *a:b"fixture")
    base.model.generation_config.eos_token_id = 123
    with pytest.raises(ValueError,match="defaults_changed"):
        GenerationBinding(base,tmp_path,frozen_contract())


def test_new_generation_audit_mandatory_old_replay_unchanged(tmp_path):
    row, backend, journal, frame, corpus = run(tmp_path,[verdict()],"fixed_top1",protocol=pair.VERSIONS[0])
    args = (row,frame,journal.directory,tmp_path/"episode/private-responses",backend.base.tokenizer,corpus)
    # Existing test helper's private path comes from the recorded directory.
    private = next(tmp_path.rglob("result.json")).parent/"private-responses"
    args = (*args[:3],private,*args[4:])
    assert audit_episode(*args,run_identity=RUN,protocol=pair.VERSIONS[0]) == row
    with pytest.raises(OSError):
        audit_episode(*args,run_identity=RUN,protocol=pair.VERSIONS[0],generation_contract=frozen_contract())
