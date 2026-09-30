"""Versioned single-attempt component execution. No scoring-target imports.

All text/wire records are PRIVATE. Only the separate scorer exports aggregates.
The operator owns the unique release root, so changing a worker output path is
not a new authorization. Interrupted/unknown attempts must never be replayed.
"""
from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import Any

from .component_decoder import decoder_identity, decoder_schema
from .component_preflight import preflight_cases
from .scifact_component_contract import LIMITS, ContractError, packing, parse, render, require, schema_for
from .scifact_component_runtime import ComponentPersistenceError, known_usage
from .scifact_semantic_contract import checked, encoded, sha

RELEASE = "scifact-component-single-attempt-20261001-r2"
INFRASTRUCTURE_LINEAGE = {
    "logical_release": "scifact-component-single-attempt-20261001-v1", "infra_retry": 1,
    "prior_job": "31729507", "prior_source_git": "79f069d2eb4c8db3b82b152fa8a698b5ae3230ad",
    "prior_source_archive_sha256": "b46a1866f659e193e7078a1ed57f3e805d177a8143be77a818605649194e0b19",
    "prior_inference_log_sha256": "9e1e88b2973a87ad01ee568e5ba97a1972d97113fc3cb78358a2016f61fd73d3",
    "prior_operator_status_sha256": "f72d25cb3e431afc48526fe976dc655791dfa7d0eca5d134a800c9ace51c440d",
    "prior_model_calls": 0, "prior_allocated_gpu_seconds": 81,
}
PREPARATION_GIT = "426ff7343fb30e1ffcde4dfa4a43f1c00c210cfb"
PROTOCOL_SHA = "2ce563ccc34efbd5ee1a21cf12fa47fafd063c853c629996909cb37e7247ca1e"
SLOTS_SHA = "84524e2a837aacab5824e5562a61d827ef02b1832dd9ab9f383ae73caa309ac9"
TARGETS_SHA = "a6d38d3fe9d72e6beb0b16836e650e4519d2016e31b63ed1d79e05999b665d9a"


def verify_infrastructure_predecessor(runs: Path) -> None:
    """Bind this one prepared retry to the preserved, pre-model v1 failure."""
    prior = runs / str(INFRASTRUCTURE_LINEAGE["logical_release"])
    for name, key in (("inference.log", "prior_inference_log_sha256"),
                      ("operator-status.json", "prior_operator_status_sha256")):
        checked(prior / name, str(INFRASTRUCTURE_LINEAGE[key]))
    require(not prior.is_symlink() and all(not (prior / name).exists() and not (prior / name).is_symlink()
        for name in ("worker-identity.json", "provider-load-private", "inference", "inference-exited.json")),
        "predecessor_not_proven_pre_model")


def durable(path: Path, value: Any) -> None:
    with path.open("xb") as stream:
        stream.write(encoded(value))
        stream.flush()
        os.fsync(stream.fileno())
    if os.name == "posix":
        descriptor = os.open(path.parent, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
        try:
            os.fsync(descriptor)
        finally:
            os.close(descriptor)


def frozen_slots(path: Path) -> list[dict[str, Any]]:
    rows = json.loads(checked(path, SLOTS_SHA))
    require(len(rows) == 33 and [r["slot"] for r in rows] == list(range(33)), "frozen_slot_matrix")
    require(all(set(r) == {"slot", "input", "packing"} for r in rows), "inference_fields")
    return list(rows)


def complete_usage(value: Any, diagnostics: Any) -> bool:
    return (isinstance(value, dict) and set(value) == {"input_tokens", "output_tokens"}
            and all(type(v) is int and v >= 0 for v in value.values())
            and isinstance(diagnostics, dict) and not diagnostics.get("output_usage_unknown", False))


def skeleton(slot: dict[str, Any], phase: str) -> dict[str, Any]:
    return {"slot": slot["slot"], "phase": phase, "component": (slot["input"] or {}).get("component"),
            "packing": slot["packing"], "attempted": False, "usage_known": False, "usage": None,
            "status": "preparation_gap" if slot["packing"]["status"] != "prepared" else "not_attempted_after_stop"}


def execute_slot(slot: dict[str, Any], provider: Any, root: Path, phase: str) -> dict[str, Any]:
    require(phase in {"preflight", "diagnostic"}, "phase")
    require(type(slot["slot"]) is int and 0 <= slot["slot"] < (4 if phase == "preflight" else 33), "slot_range")
    output = root / f'{phase}-{slot["slot"]:02d}'
    output.mkdir(mode=0o700)  # exclusive even if the same slot was interrupted
    record = skeleton(slot, phase)
    durable(output / "initialized.json", record)
    if record["status"] == "preparation_gap":
        durable(output / "completed.json", record)
        return record
    spec = slot["input"]
    expected = packing(provider.tokenizer, spec)
    require(expected == slot["packing"], "frozen_packing_mismatch")
    schema = schema_for(spec)
    wire = {"input": spec, "canonical_schema": schema, "decoder_schema": decoder_schema(schema),
            "decoder": decoder_identity(schema), "prompt": render(provider.tokenizer, spec), "packing": expected}
    durable(output / "wire.json", wire)
    record["wire_sha256"] = sha(encoded(wire))
    provider.start_slot(output / "private")
    record.update(attempted=True, status="provider_failed")
    durable(output / "started.json", record)  # call reservation before entering provider
    begin = time.perf_counter()
    last_diagnostics: dict[str, Any] = {}
    try:
        response = provider.generate(spec, schema, LIMITS["output_tokens"], LIMITS["seconds"])
        usage, diagnostics = response.get("usage"), response.get("diagnostics", {})
        last_diagnostics = diagnostics
        record["known_usage_lower_bound"] = known_usage(usage)
        record["usage_known"] = complete_usage(usage, diagnostics)
        record["usage"] = usage if record["usage_known"] else None
        durable(output / "response.json", response)
        record["response_sha256"] = sha(encoded(response))
        require(record["usage_known"], "unknown_usage")
        require(usage["input_tokens"] == expected["input_tokens"] and usage["output_tokens"] <= 512, "actual_token_budget")
        require(diagnostics.get("eos_observed") is True and diagnostics.get("reached_max_new_tokens") is False,
                "incomplete_termination")
        require(time.perf_counter() - begin <= LIMITS["seconds"], "slot_deadline")
        try:
            record["prediction"] = parse(response["raw"], spec)
            record["status"] = "valid"
        except ContractError as exc:
            record.update(status="schema_failed", error_category=str(exc))
    except Exception as exc:
        usage = getattr(exc, "usage", None)
        diagnostics = getattr(exc, "diagnostics", last_diagnostics)
        if isinstance(usage, dict):
            record["known_usage_lower_bound"] = known_usage(usage)
            record["usage_known"] = complete_usage(usage, diagnostics)
            record["usage"] = usage if record["usage_known"] else None
        record["error_category"] = str(exc) if isinstance(exc, ContractError) else type(exc).__name__
        if isinstance(diagnostics, dict):
            record["failure_diagnostics"] = diagnostics
        failure = {"exception_type": type(exc).__name__, "usage": record["usage"] or record.get("known_usage_lower_bound"),
                   "diagnostics": diagnostics | {"output_usage_unknown": not record["usage_known"]}}
        try:
            durable(output / "failure.json", failure)
            record["failure_sha256"] = sha(encoded(failure))
        except Exception as persistence:
            raise ComponentPersistenceError(record) from persistence
    record["elapsed_ms"] = (time.perf_counter() - begin) * 1000
    try:
        durable(output / "completed.json", record)
    except Exception as exc:
        raise ComponentPersistenceError(record) from exc
    return record


def execute_matrix(rows: list[dict[str, Any]], provider: Any, root: Path) -> dict[str, Any]:
    require(len(rows) == 33 and [r["slot"] for r in rows] == list(range(33)), "matrix_shape")
    root.mkdir(mode=0o700)
    report: dict[str, Any] = {"release": RELEASE, "scoring_targets_loaded": False,
        "oracle_relation_explicit_in_rationale_input": True, "preflight": [], "diagnostic": [], "status": "stopped"}
    stopped = False
    for i, (spec, expected) in enumerate(preflight_cases()):
        slot = {"slot": i, "input": spec, "packing": packing(provider.tokenizer, spec)}
        record = skeleton(slot, "preflight") if stopped else execute_slot(slot, provider, root, "preflight")
        record["preflight_pass"] = record.get("status") == "valid" and record.get("prediction") == expected
        report["preflight"].append(record)
        stopped = stopped or not record["preflight_pass"]
    for slot in rows:
        record = skeleton(slot, "diagnostic") if stopped else execute_slot(slot, provider, root, "diagnostic")
        report["diagnostic"].append(record)
        stopped = stopped or record["status"] == "provider_failed"
    report["status"] = "stopped" if stopped else "complete"
    durable(root / "run.json", report)
    return report
