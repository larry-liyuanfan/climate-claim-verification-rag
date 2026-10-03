"""Single-attempt fixture interface; real model execution is deliberately absent."""
from __future__ import annotations

import time
from pathlib import Path
from typing import Any

from .scifact_component_contract import ContractError, LIMITS, packing, parse, require, schema_for
from .scifact_semantic_contract import encoded, sha, write_once


class ComponentPersistenceError(RuntimeError):
    def __init__(self, record: dict[str, Any]) -> None:
        super().__init__("component_cost_not_durable_stop_no_retry")
        self.partial_report = record


def known_usage(value: Any) -> dict[str, int]:
    if not isinstance(value, dict):
        return {}
    return {k: v for k, v in value.items() if k in {"input_tokens", "output_tokens"} and type(v) is int and v >= 0}


def fixture_slot(spec: dict[str, Any], provider: Any, output: Path) -> dict[str, Any]:
    """No retries; exclusive directory prevents accidental replay after any attempt.

    This interface deliberately rejects local_model/API providers. A future real
    backend requires its own exact-hash review/release and new-schema grammar.
    """
    require(getattr(provider, "kind", None) == "fixture", "real_model_not_authorized")
    packed = packing(provider.tokenizer, spec)
    output.mkdir(mode=0o700)
    record: dict[str, Any] = {"component": spec["component"], "execution_kind": "fixture",
        "model_calls": 0, "attempted": False, "usage_known": False, "usage": None,
        "packing": packed, "status": "preparation_gap"}
    write_once(output / "initialized.json", record)
    if packed["status"] != "prepared":
        write_once(output / "completed.json", record)
        return record
    record.update(attempted=True, status="provider_failed")
    # Durable reservation BEFORE generate: interruption is attempted/unknown,
    # never a zero-cost unattempted slot. A failed reservation makes zero calls.
    write_once(output / "started.json", record)
    begin = time.perf_counter()
    try:
        response = provider.generate(spec, schema_for(spec), LIMITS["output_tokens"], LIMITS["seconds"])
        usage = response.get("usage")
        record["known_usage_lower_bound"] = known_usage(usage)
        if (isinstance(usage, dict) and set(usage) == {"input_tokens", "output_tokens"}
                and all(type(v) is int and v >= 0 for v in usage.values())
                and not response.get("diagnostics", {}).get("output_usage_unknown", False)):
            record.update(usage=usage, usage_known=True)
        # Preserve the response and partial cost BEFORE validity/budget checks.
        write_once(output / "response.json", response)
        record["response_sha256"] = sha(encoded(response))
        require(record["usage_known"], "usage_unknown")
        require(usage["input_tokens"] == packed["input_tokens"], "actual_prompt_count_mismatch")
        require(usage["output_tokens"] <= LIMITS["output_tokens"], "output_token_budget")
        raw = response["raw"]
        require(isinstance(raw, str), "raw_response_type")
        diagnostics = response.get("diagnostics", {})
        require(diagnostics.get("eos_observed") is True and diagnostics.get("reached_max_new_tokens") is False,
                "incomplete_termination")
        require(time.perf_counter() - begin <= LIMITS["seconds"], "deadline")
        try:
            record["prediction"] = parse(raw, spec)
            record["status"] = "valid"
        except ContractError as exc:
            record.update(status="schema_failed", error_category=str(exc))
    except Exception as exc:
        # A provider exception may carry known lower-bound usage; unknown output
        # usage must remain unknown, not be replaced with a zero-cost success.
        exception_usage = getattr(exc, "usage", None)
        if isinstance(exception_usage, dict):
            record["known_usage_lower_bound"] = known_usage(exception_usage)
        record["error_category"] = str(exc) if isinstance(exc, ContractError) else type(exc).__name__
    record["elapsed_ms"] = (time.perf_counter() - begin) * 1000
    try:
        write_once(output / "completed.json", record)
    except Exception as exc:
        raise ComponentPersistenceError(record) from exc
    return record
