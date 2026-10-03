"""Four independent synthetic checks per arm; no official examples or gold."""
from __future__ import annotations

import json
from collections.abc import Callable
from typing import Any

from .agent_protocol import ModelResponseValidationError
from .evidence_gap_candidate import GapProviderAdapter
from .scifact_runtime_smoke import smoke_cases
from .scifact_terminal import parse_action


def _smoke_report(records: list[dict[str, Any]]) -> dict[str, Any]:
    return {"schema_version": "scifact-bounded-synthetic-smoke-v1", "records": records,
            "status": "passed" if all(r["passed"] for r in records) else "failed",
            "attempted_calls": sum(r["provider_generate_attempted"] for r in records),
            "official_data_read": False, "quality_evaluation": False,
            "scope": "forced synthetic protocol checks, not autonomous model tool selection"}


def bounded_runtime_smoke(backend: Any, *, persist_case: Callable[[int, str, dict[str, Any]], None] | None = None) -> dict[str, Any]:
    records: list[dict[str, Any]] = []
    for case in smoke_cases():
        provider = GapProviderAdapter(backend) if backend.gap else backend
        record: dict[str, Any] = {"passed": False, "provider_generate_attempted": False}
        cap = case["max_output_tokens"]
        count = None
        writing_journal = False
        try:
            count = provider.count_prompt(case["observation"], case["schema"])
            record.update(input_tokens_expected=count, max_output_tokens=cap)
            if persist_case is not None:
                writing_journal = True
                persist_case(len(records), "started", dict(record))
                writing_journal = False
            record["provider_generate_attempted"] = True
            response = provider.generate(case["observation"], case["schema"], cap, 45)
            record.update(usage=response["usage"], diagnostics=response["diagnostics"])
            if cap == 1:
                try:
                    json.loads(response["raw"])
                    incomplete = False
                except ValueError:
                    incomplete = True
                passed = incomplete and response["usage"]["output_tokens"] == 1
            else:
                decision = parse_action(json.loads(response["raw"]), ["answer"], case["visible"], case["aliases"], 2)
                passed = {d["source_id"]: d["label"] for d in decision["documents"]} == dict(
                    zip(case["aliases"], ["SUPPORTS", "REFUTES"], strict=True))
            diag = response["diagnostics"]
            record["passed"] = bool(passed and response["usage"]["input_tokens"] == count
                                     and diag["eos_observed"] == (cap != 1)
                                     and diag["reached_max_new_tokens"] == (cap == 1))
        except ModelResponseValidationError as exc:
            record.update(usage=exc.usage, diagnostics=exc.diagnostics, failure="provider_response_invalid")
            # G must reject the deliberately truncated envelope, yet retain cost
            # and a complete raw receipt. Do not convert it into a valid action.
            d = exc.diagnostics
            record["passed"] = bool(backend.gap and cap == 1 and d.get("category") == "gap_envelope_invalid"
                                     and exc.usage.get("input_tokens") == count and exc.usage.get("output_tokens") == 1
                                     and d.get("reached_max_new_tokens") is True and d.get("eos_observed") is False
                                     and d.get("private_attachment", {}).get("truncated") is False)
        except Exception as exc:
            if writing_journal:
                raise
            record["failure"] = type(exc).__name__
        records.append(record)
        if persist_case is not None:
            try:
                persist_case(len(records) - 1, "completed", dict(record))
            except Exception as exc:
                # The response/usage already exists. Return it before stopping,
                # allowing the caller one summary write, never another model call.
                return _smoke_report(records) | {"status": "failed",
                    "journal_failure": {"case_index": len(records) - 1, "phase": "completed",
                                        "exception_type": type(exc).__name__},
                    "case_journal_complete": False}
    return _smoke_report(records)
