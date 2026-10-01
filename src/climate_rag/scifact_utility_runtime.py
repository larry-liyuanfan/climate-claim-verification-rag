"""Gold-free diagnostic matrix with a durable physical-call ledger."""
from __future__ import annotations

import copy
import json
from pathlib import Path
import time
from typing import Any

from .agent_protocol import ModelResponseValidationError
from .scifact_bounded_runtime import ARMS as BOUNDED_ARMS, run_bounded_slot
from .scifact_read_continuation import ordered_write
from .scifact_terminal import PROTOCOL as TERMINAL, render_answer, render_scifact_prompt, to_original_prediction
from .scifact_utility_contract import ARMS, MAX_GENERATIONS, PROTOCOL, UtilityDiagnostic, identity, read_intervention


class JournalProvider:
    """Only the real underlying generate allocates a physical ID; no warmups."""
    capacity_ceiling = 168

    def render(self, observation: Any, schema: Any) -> str:
        return render_scifact_prompt(self.base.tokenizer, observation, schema)

    def __init__(self, backend: Any, directory: Path, *, max_generations: int = MAX_GENERATIONS,
                 protocol: str = PROTOCOL, physical_guard: Any = None) -> None:
        if type(max_generations) is not int or not 1 <= max_generations <= self.capacity_ceiling:
            raise ValueError("bounded_generation_capacity")
        directory.mkdir(mode=0o700)  # No resume/reset of a partially consumed run.
        self.backend, self.directory = backend, directory
        self.name, self.kind, self.gap = backend.name, backend.kind, backend.gap
        self.base, self.terminal_protocol = backend.base, backend.terminal_protocol
        self.slot = "unassigned"
        self.max_generations, self.protocol, self.physical_guard = max_generations, protocol, physical_guard

    def count_text(self, text: str) -> int:
        return int(self.backend.count_text(text))

    def count_prompt(self, observation: Any, schema: Any) -> int:
        return int(self.backend.count_prompt(observation, schema))

    def generate(self, observation: Any, schema: Any, max_output_tokens: int,
                 remaining_seconds: float) -> dict[str, Any]:
        number = len(list(self.directory.glob("g*.reserved.json")))
        if number >= self.max_generations or self.slot == "unassigned":
            raise RuntimeError("physical_generation_budget_or_slot")
        key = f"g{number:02d}"
        started = time.monotonic()
        prompt = self.render(observation, schema)
        guard = {"actual_base_state": self.physical_guard(self.backend)} if self.physical_guard else {}
        ordered_write(self.directory / (key + ".reserved.json"), {
            "physical_attempt_id": key, "slot": self.slot, "protocol": self.protocol, **guard,
            "observation": observation, "schema": schema, "prompt_sha256": identity(prompt),
            "token_ids_sha256": identity(self.base.tokenizer.encode(prompt, add_special_tokens=False)),
            "max_output_tokens": max_output_tokens, "remaining_seconds": remaining_seconds,
            "usage": None, "status": "reserved_before_actual_generate"})
        try:
            response: dict[str, Any] = self.backend.generate(observation, schema, max_output_tokens, remaining_seconds)
            response = copy.deepcopy(response)
            response.setdefault("diagnostics", {})["physical_attempt_id"] = key
            ordered_write(self.directory / (key + ".finished.json"), {
                "physical_attempt_id": key, "slot": self.slot, "status": "returned",
                "response": response, "usage": response.get("usage"),
                "usage_known": all(type(response.get("usage", {}).get(k)) is int
                    and response["usage"][k] >= 0 for k in ("input_tokens", "output_tokens"))
                    and not response.get("diagnostics", {}).get("output_usage_unknown", False),
                "elapsed_ms": (time.monotonic() - started) * 1000})
            return response
        except BaseException as exc:
            usage = getattr(exc, "usage", {})
            diagnostics = dict(getattr(exc, "diagnostics", {}), physical_attempt_id=key)
            known = (all(type(usage.get(k)) is int for k in ("input_tokens", "output_tokens"))
                     and not diagnostics.get("output_usage_unknown", False))
            # An I/O failure after a real return must not overwrite that receipt.
            finished = self.directory / (key + ".finished.json")
            if not finished.exists():
                ordered_write(finished, {"physical_attempt_id": key, "slot": self.slot,
                    "status": "failed", "error_type": type(exc).__name__, "usage": usage or None,
                    "usage_known": known, "diagnostics": diagnostics,
                    "elapsed_ms": (time.monotonic() - started) * 1000})
            if isinstance(exc, ModelResponseValidationError):
                exc.diagnostics.update(physical_attempt_id=key)
            raise


def ledger_cost(directory: Path, ids: list[str] | None = None) -> dict[str, Any]:
    all_ids = {p.name.removesuffix(".reserved.json") for p in directory.glob("g*.reserved.json")}
    wanted = set(ids) if ids is not None else all_ids
    if not wanted <= all_ids:
        raise ValueError("unreserved_physical_attempt_reference")
    rows = []
    for key in sorted(wanted):
        path = directory / (key + ".finished.json")
        try:
            rows.append(json.loads(path.read_bytes()))
        except (OSError, ValueError):
            rows.append({"usage": None, "usage_known": False})
    known_sum = {k: sum(v for r in rows if type(v := (r.get("usage") or {}).get(k)) is int and v >= 0)
                 for k in ("input_tokens", "output_tokens")}
    unknown = sum(not r.get("usage_known", False) for r in rows)
    return {"unique_physical_calls": len(wanted), "known_token_lower_bound": known_sum,
            "total_tokens": known_sum if not unknown else None, "unknown_usage_attempts": unknown,
            "elapsed_ms": sum(r["elapsed_ms"] for r in rows) if all("elapsed_ms" in r for r in rows) else None,
            "api_currency_cost": None}


def reranker_cost(directory: Path) -> dict[str, Any]:
    requested = completed = tokens = unknown = 0
    elapsed = 0.0
    reservations = list(directory.glob("r*.reserved.json"))
    for path in reservations:
        request = json.loads(path.read_bytes())
        requested += request["requested_pairs"]
        finished = path.with_name(path.name.replace(".reserved.json", ".finished.json"))
        try:
            result = json.loads(finished.read_bytes())
            completed += result["completed_pairs"]
            elapsed += result["elapsed_ms_including_swaps"]
            if result["observed_nonpadding_tokens"] is None or any(r["status"] != "completed" for r in result["batches"]):
                unknown += 1
            else:
                tokens += result["observed_nonpadding_tokens"]
        except (OSError, ValueError, KeyError):
            unknown += 1
    return {"physical_requests": len(reservations), "requested_pairs": requested, "completed_pairs": completed,
            "nonpadding_tokens": tokens if not unknown else None, "known_token_lower_bound": tokens,
            "elapsed_ms_including_swaps": elapsed if not unknown else None,
            "unknown_requests": unknown, "generator_calls": 0, "api_currency_cost": None}


def direct_prediction(prefix: dict[str, Any], claim_id: int, corpus: Any) -> dict[str, Any] | None:
    p = prefix["payload"]
    decision = p["decision"]
    if p["attempt"]["status"] != "valid_decision" or not decision or decision["action"] not in {"answer", "abstain"}:
        return None
    answer = render_answer(decision, p["frame"]["visible"]) if decision["action"] == "answer" else None
    result = {"protocol": TERMINAL, "answer": answer,
              "outcome": "ids_validated_semantics_unmeasured" if answer else "model_abstention:" + decision["reason"]}
    return {"result": result, **to_original_prediction(claim_id, result, corpus)}


def run_matrix(claims: list[dict[str, Any]], backend: Any, retrieve: Any, rerank: Any,
               corpus: Any, output: Path, *, natural_fit: bool = False) -> dict[str, Any]:
    from .scifact_natural_contract import audit_attempt, base_state, ranked_preview_read, valid_physical_prefix
    from .scifact_natural_selection import PROTOCOL as NATURAL, SCOPE
    if (not (1 <= len(claims) <= 24 if natural_fit else len(claims) == 8)
            or len({r["id"] for r in claims}) != len(claims)
            or any(set(r) != {"id", "claim"} for r in claims) or backend.gap):
        raise ValueError("fixed_eight_gold_free_claims_required")
    protocol, limit = (NATURAL, 7 * len(claims)) if natural_fit else (PROTOCOL, MAX_GENERATIONS)
    if natural_fit:
        base_state(backend)
    output.mkdir(mode=0o700)
    slots = [{"claim_id": r["id"], "arm": arm} for r in claims for arm in ARMS]
    ordered_write(output / "planned.json", {"protocol": protocol, "slots": slots,
        "max_generations": limit, "warmup_calls": 0, "restart_or_replacement_allowed": False})
    journal = JournalProvider(backend, output / "ledger", max_generations=limit, protocol=protocol,
                              physical_guard=base_state if natural_fit else None)
    results = []
    for r in claims:
        capture: dict[str, Any] = {}
        direct = None
        prefix_error: str | None = None
        for arm in ARMS:
            key = f"{r['id']}-{arm}"
            journal.slot = key
            slot_dir = output / key
            slot_dir.mkdir(mode=0o700)
            inherited = capture.get("prefix", {}).get("payload", {}).get("elapsed_seconds", 0) if arm != "A" else 0
            clock = {"started_unix": time.time(), "deadline_seconds": max(0, 120 - inherited)} if natural_fit else {}
            ordered_write(slot_dir / "reserved.json", {"claim_id": r["id"], "arm": arm, **clock})
            frames: list[dict[str, Any]] = []
            def observer(kind: str, value: dict[str, Any]) -> None:
                if kind == "prefix":
                    ordered_write(slot_dir / "prefix.json", value)
                    capture["prefix"] = value
                elif kind == "initial_frame":
                    prompt = render_scifact_prompt(backend.base.tokenizer, value["observation"], value["schema"])
                    physical = {"frame": value, "rendered_prompt_sha256": identity(prompt),
                                "token_ids_sha256": identity(backend.base.tokenizer.encode(prompt, add_special_tokens=False))}
                    ordered_write(slot_dir / "initial-frame.json", physical)
                    if arm == "A":
                        capture["initial"] = physical
                    elif identity(physical) != identity(capture["initial"]):
                        raise ValueError("counterfactual_initial_prompt_changed")
                elif kind == "frame":
                    ordered_write(slot_dir / f"frame-{len(frames)}.json", value)
                    frames.append(value)
            prefix = capture.get("prefix")
            intervention, reason = None, None
            if arm != "A":
                if prefix is None:
                    reason = "attempt0_capsule_unavailable"
                elif natural_fit and prefix_error is not None:
                    reason = prefix_error
                elif arm == "B":
                    intervention, reason = (ranked_preview_read(prefix) if natural_fit else read_intervention(prefix))
                else:
                    intervention = {"action": "rerank"}
            if reason:
                row = {"claim_id": r["id"], "arm": arm, "status": "not_run", "reason": reason,
                       "prediction": None, "result": None}
            else:
                try:
                    backend.start_slot(slot_dir / "private-responses")
                    row = run_bounded_slot(r["id"], r["claim"], "adaptive", BOUNDED_ARMS[0], journal,
                        retrieve, rerank, corpus, lambda value: ordered_write(slot_dir / "raw-result.json", value),
                        diagnostic=UtilityDiagnostic(observer, prefix if arm != "A" else None, intervention))
                    row.update(arm=arm, status="finished")
                except Exception as exc:
                    row = {"claim_id": r["id"], "arm": arm, "status": "failed", "reason": type(exc).__name__,
                           "prediction": None, "result": None}
            if arm == "A" and "prefix" in capture:
                if natural_fit:
                    try:
                        valid_physical_prefix(capture["prefix"], journal.directory, slot_dir / "private-responses",
                                              backend.base.tokenizer, key)
                    except Exception as exc:
                        prefix_error = "initial_prefix_unusable:" + type(exc).__name__
                if prefix_error is None:
                    direct = direct_prediction(capture["prefix"], r["id"], corpus)
            if natural_fit:
                row.update(origin="natural_model_trajectory" if arm == "A" else "scripted_tool_single_continuation",
                           audit_status="unresolved", initial_prefix_error=prefix_error)
                result = row.get("result") or {}
                attempts = result.get("generation_attempts", [])
                if attempts and frames:
                    try:
                        decision = audit_attempt(attempts[-1], frames[-1], journal.directory,
                            slot_dir / "private-responses", backend.base.tokenizer, key)
                        if decision["action"] in {"answer", "abstain"} and row.get("prediction") is not None:
                            row["audit_status"] = "valid_terminal"
                    except Exception as exc:
                        row["terminal_audit_error"] = type(exc).__name__
                if row["audit_status"] != "valid_terminal":
                    row["unaudited_prediction"] = row.get("prediction")
                    row["prediction"] = None
            row["direct_attempt0"] = direct
            shared = (capture.get("prefix", {}).get("payload", {}).get("attempt", {}).get("diagnostics", {})
                      .get("physical_attempt_id"))
            own = [p.name.removesuffix(".reserved.json") for p in journal.directory.glob("g*.reserved.json")
                   if json.loads(p.read_bytes())["slot"] == key]
            row["physical_generation_ids"] = own
            row["logical_generation_ids"] = ([shared] if arm != "A" and shared else []) + own
            row["incremental_generation_cost"] = ledger_cost(journal.directory, own)
            row["logical_generation_cost"] = ledger_cost(journal.directory, row["logical_generation_ids"])
            if natural_fit and row["logical_generation_cost"]["unknown_usage_attempts"]:
                row.update(audit_status="unresolved", terminal_audit_error="incomplete_trajectory_cost",
                           unaudited_prediction=row.get("prediction"), prediction=None)
            if frames:
                initial = capture["initial"]["frame"]["visible"]
                newly = set(frames[-1]["visible"]) - set(initial)
                row["newly_visible_sentence_ids"] = sorted(newly)
            ordered_write(slot_dir / "result.json", row)
            results.append(row)
    report = {"protocol": protocol, "planned_slots": len(slots), "runs": results,
              "physical_generation_cost": ledger_cost(journal.directory), "gold_loaded": False,
              "strategy_comparison_not_equal_compute_causal_effect": True,
              "scope": SCOPE if natural_fit else "outcome_selected_exposed_TRAIN_diagnostic_not_evaluation_or_training"}
    ordered_write(output / "run.json", report)
    return report
