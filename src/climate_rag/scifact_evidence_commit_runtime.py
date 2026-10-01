"""Physical generation -> immutable ref -> commit -> original scorer bridge."""
from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
import time
from typing import Any

from .evidence_gap_candidate import _no_duplicate_keys
from .scifact_document_verifier import DocumentJournal, MAX_SECONDS
from .scifact_evidence_commit import (
    ARMS, ASSEMBLER, CALL_CAPS, CommitState, PROTOCOL, PROTOCOLS, render_prompt, verifier_identity,
)
from .scifact_natural_contract import complete_wire, require
from .scifact_read_continuation import ordered_write
from .scifact_semantic_contract import MODEL_SHA
from .scifact_utility_contract import identity
from .scifact_utility_runtime import ledger_cost


class CommitJournal(DocumentJournal):
    def __init__(self, backend: Any, directory: Path, *, run_identity: str,
                 max_generations: int = 240, protocol: str = PROTOCOL, physical_guard: Any) -> None:
        require(len(run_identity) == 64 and all(c in "0123456789abcdef" for c in run_identity), "release_sha_required")
        require(protocol in PROTOCOLS, "unknown_commit_protocol")
        self.run_identity, self.episode_identity, self.frame_sha = run_identity, "", ""
        self.base_guard = physical_guard
        super().__init__(backend, directory, max_generations=max_generations, protocol=protocol,
                         physical_guard=self.bound_physical_state)

    def bound_physical_state(self, backend: Any) -> dict[str, Any]:
        require(self.episode_identity == f"{self.protocol}:{self.run_identity}:{self.slot.replace('-', ':', 1)}"
                and len(self.frame_sha) == 64, "physical_episode_scope_required")
        return dict(self.base_guard(backend), run_identity=self.run_identity,
                    episode_id=self.episode_identity, frame_sha256=self.frame_sha)

    def generate(self, observation: Any, schema: Any, max_output_tokens: int,
                 remaining_seconds: float) -> dict[str, Any]:
        arm = self.slot.split("-", 1)[-1]
        own = [p for p in self.directory.glob("g*.reserved.json")
               if json.loads(p.read_bytes())["slot"] == self.slot]
        require(self.protocol in PROTOCOLS and arm in ARMS and len(own) < CALL_CAPS[arm], "per_arm_call_budget")
        return super().generate(observation, schema, max_output_tokens, remaining_seconds)


def _result(state: CommitState, frame: Any, steps: Any, corpus: Any, elapsed: float) -> dict[str, Any]:
    return {"protocol": state.protocol, "assembler": ASSEMBLER,
        "claim_id": state.claim_id, "arm": state.arm, "episode_id": state.episode_id,
        "input_sha256": identity(frame),
        "state": "valid_terminal" if state.terminal else "unresolved", "reason": state.reason,
        "terminal": state.terminal, "assembled": state.assembled, "prediction": state.prediction(corpus),
        "steps": steps, "verification_feedback": state.feedback, "verdict_refs": state.registry.records(),
        "physical_calls": state.calls, "tool_calls": state.tools, "elapsed_seconds": elapsed,
        "initial_retrieval": frame.get("initial_retrieval", {"status": "replayed_frozen_original", "tool_debit": 1,
            "cost": "historical_not_recomputed_or_claimed_free_online"}),
        "verify_is_commit_prerequisite_not_spontaneous_demand": True}


def run_episode(claim_id: int, arm: str, frame: Any, journal: CommitJournal, corpus: Any,
                output: Path, *, run_identity: str, clock: Any = time.monotonic) -> dict[str, Any]:
    require(len(run_identity) == 64 and all(c in "0123456789abcdef" for c in run_identity), "release_sha_required")
    require(run_identity == journal.run_identity, "journal_release_mismatch")
    output.mkdir(mode=0o700)
    state = CommitState(claim_id, arm, frame, f"{journal.protocol}:{run_identity}:{claim_id}:{arm}",
                        protocol=journal.protocol)
    started = clock()
    ordered_write(output / "reserved.json", {"episode_id": state.episode_id, "run_identity": run_identity,
        "claim_id": claim_id, "arm": arm, "input_sha256": identity(frame),
        "started_unix": time.time(), "deadline_seconds": MAX_SECONDS})
    journal.slot = f"{claim_id}-{arm}"
    journal.episode_identity, journal.frame_sha = state.episode_id, identity(frame)
    journal.backend.start_slot(output / "private-responses")
    steps: list[Any] = []
    while not state.stopped:
        if clock() - started >= MAX_SECONDS:
            state.reason, state.stopped = "episode_deadline", True
            break
        call = state.next_call()
        if call is None:
            break
        stage, doc, available = call
        observation, schema = state.inputs(stage, doc, available)
        prompt = journal.render(observation, schema)
        count = len(journal.base.tokenizer.encode(prompt, add_special_tokens=False))
        remaining = MAX_SECONDS - (clock() - started)
        step: dict[str, Any] = {"stage": stage, "source_id": doc, "observation": observation,
            "schema": schema, "input_prompt_tokens": count, "remaining_seconds": remaining,
            "status": "not_called", "physical_attempt_id": None, "usage": None}
        steps.append(step)
        if count != journal.count_prompt(observation, schema) or count > 8192 or remaining <= 0:
            state.reason = "prompt_or_deadline_guard_no_repacking"
            break
        before = {p.name for p in journal.directory.glob("g*.reserved.json")}
        raw_sha = ""
        try:
            response = journal.generate(observation, schema, 512, remaining)
            step.update(physical_attempt_id=response["diagnostics"]["physical_attempt_id"], usage=response["usage"])
            complete_wire(response, count, output / "private-responses")
            require(clock() - started < MAX_SECONDS, "episode_deadline")
            raw = json.loads(response["raw"], object_pairs_hook=_no_duplicate_keys)
            raw_sha = hashlib.sha256(response["raw"].encode()).hexdigest()
            step.update(status="valid", decision=state.parse(raw, doc, available))
        except Exception as exc:
            step.update(status="failed", failure=type(exc).__name__)
        created = [p for p in journal.directory.glob("g*.reserved.json") if p.name not in before]
        require(len(created) <= 1, "single_physical_call_per_step")
        if created:
            key = created[0].name.removesuffix(".reserved.json")
            require(step["physical_attempt_id"] in (None, key), "physical_response_id_mismatch")
            step["physical_attempt_id"] = key
            receipt = created[0].with_name(key + ".finished.json")
            if receipt.exists():
                step["usage"] = json.loads(receipt.read_bytes()).get("usage")
            state.accept(step, doc, verifier_identity(prompt, schema, raw_sha))
            if ledger_cost(journal.directory, [key])["unknown_usage_attempts"]:
                state.reason, state.stopped = "unknown_physical_cost", True
        else:
            state.reason, state.stopped = "unreserved_call_failure", True
        ordered_write(output / f"step-{len(steps):02d}.json", step)
        if clock() - started >= MAX_SECONDS:
            state.terminal, state.assembled = None, None
            state.reason, state.stopped = "episode_deadline", True
    elapsed = clock()-started
    if elapsed >= MAX_SECONDS:
        state.terminal, state.assembled = None, None
        state.reason, state.stopped = "episode_deadline", True
    row = _result(state, frame, steps, corpus, elapsed)
    ordered_write(output / "result.json", row)
    return row


def audit_episode(row: Any, frame: Any, ledger: Path, private: Path, tokenizer: Any,
                  corpus: Any, *, run_identity: str, protocol: str = PROTOCOL) -> dict[str, Any]:
    """Reissue from verified physical wire, never trust saved refs/feedback."""
    arm, claim_id = row["arm"], row["claim_id"]
    require(protocol in PROTOCOLS, "unknown_commit_protocol")
    episode_id = f"{protocol}:{run_identity}:{claim_id}:{arm}"
    require(row["protocol"] == protocol and row["episode_id"] == episode_id
            and row["input_sha256"] == identity(frame), "episode_identity")
    reservation = json.loads((private.parent / "reserved.json").read_bytes())
    stamp = reservation.get("started_unix")
    require(type(stamp) in {int, float} and math.isfinite(stamp) and stamp > 0, "watchdog_timestamp_required")
    require(reservation == {"episode_id":episode_id, "run_identity":run_identity, "claim_id":claim_id,
            "arm":arm, "input_sha256":identity(frame), "started_unix":stamp,
            "deadline_seconds":MAX_SECONDS}, "episode_reservation_binding")
    state = CommitState(claim_id, arm, frame, episode_id, protocol=protocol)
    own = [p.name.removesuffix(".reserved.json") for p in ledger.glob("g*.reserved.json")
           if json.loads(p.read_bytes())["slot"] == f"{claim_id}-{arm}"]
    require(len(own) <= CALL_CAPS[arm] and not ledger_cost(ledger, own)["unknown_usage_attempts"], "physical_cost_guard")
    seen: list[str] = []
    physical_seconds = 0.0
    elapsed = row["elapsed_seconds"]
    require(type(elapsed) in {float, int} and math.isfinite(elapsed) and elapsed >= 0, "elapsed_invalid")
    for pos, step in enumerate(row["steps"]):
        call = state.next_call()
        require(call is not None, "call_after_completion")
        assert call is not None
        stage, doc, available = call
        obs, schema = state.inputs(stage, doc, available)
        require(step["stage"] == stage and step["source_id"] == doc and step["observation"] == obs
                and step["schema"] == schema, "policy_or_ref_catalog_changed")
        prompt = render_prompt(tokenizer, obs, schema)
        ids = tokenizer.encode(prompt, add_special_tokens=False)
        require(step["input_prompt_tokens"] == len(ids), "prompt_count_changed")
        key = step["physical_attempt_id"]
        if key is None:
            require(pos == len(row["steps"])-1 and (step["status"] == "not_called" or
                    row["reason"] == "unreserved_call_failure"), "uncalled_stage_contract")
            state.reason = row["reason"]
            state.stopped = True
            break
        require(key in own and key not in seen and len(ids) <= 8192, "physical_attempt_reuse_or_overflow")
        seen.append(key)
        request = json.loads((ledger/(key+".reserved.json")).read_bytes())
        finished = json.loads((ledger/(key+".finished.json")).read_bytes())
        guard = request["actual_base_state"]
        require(request["physical_attempt_id"] == finished["physical_attempt_id"] == key
                and request["slot"] == finished["slot"] == f"{claim_id}-{arm}"
                and request["protocol"] == protocol and request["observation"] == obs and request["schema"] == schema
                and request["prompt_sha256"] == identity(prompt) and request["token_ids_sha256"] == identity(ids)
                and request["max_output_tokens"] == 512
                and request["remaining_seconds"] == step["remaining_seconds"]
                and 0 < step["remaining_seconds"] <= MAX_SECONDS
                and guard["base_model_sha256"] == MODEL_SHA and guard["adapter_loaded"] is False
                and guard["run_identity"] == run_identity and guard["episode_id"] == episode_id
                and guard["frame_sha256"] == identity(frame)
                and guard["training"] is False and guard["lora_parameter_or_module_count"] == 0, "actual_call_binding")
        require(finished["usage_known"] is True and finished["usage"] == step["usage"], "usage_binding")
        ms = finished["elapsed_ms"]
        require(type(ms) in {int, float} and math.isfinite(ms) and ms >= 0
                and request["remaining_seconds"] <= MAX_SECONDS-physical_seconds+1e-6, "shared_deadline")
        physical_seconds += ms/1000
        require(physical_seconds <= elapsed+1e-6, "physical_time_exceeds_elapsed")
        parsed, raw_sha = None, ""
        if finished["status"] == "returned":
            response = finished["response"]
            require(response["usage"] == step["usage"], "wire_usage_binding")
            try:
                complete_wire(response, len(ids), private)
                parsed = state.parse(json.loads(response["raw"], object_pairs_hook=_no_duplicate_keys), doc, available)
                raw_sha = hashlib.sha256(response["raw"].encode()).hexdigest()
            except (ValueError, KeyError, TypeError):
                require(step["status"] == "failed", "invalid_wire_presented_as_valid")
        if step["status"] == "valid":
            require(parsed is not None and parsed == step["decision"], "raw_decision_binding")
        else:
            require(step["status"] == "failed" and (parsed is None or elapsed >= MAX_SECONDS), "false_failed_response")
        state.accept(step, doc, verifier_identity(prompt, schema, raw_sha))
    require(set(seen) == set(own), "unassigned_physical_calls")
    if elapsed >= MAX_SECONDS:
        state.terminal, state.assembled = None, None
        state.reason, state.stopped = "episode_deadline", True
    elif not state.stopped and state.terminal is None:
        require(state.next_call() is None, "premature_episode_end")
    expected = _result(state, frame, row["steps"], corpus, elapsed)
    require(row == expected, "ref_assembly_prediction_or_accounting_changed")
    return dict(row)
