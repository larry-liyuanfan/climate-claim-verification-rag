"""Synthetic-only runtime smoke; no data loader, gold or benchmark queries."""

from __future__ import annotations

import json
from typing import Any

from .agent_protocol import ModelResponseValidationError
from .agent_v3 import V3Provider
from .scifact_terminal import PROTOCOL, action_schema, parse_action


def smoke_cases() -> list[dict[str, Any]]:
    cases = []
    for aliases, cap in [
        (("c0", "c1"), 512),
        (("c5", "c6"), 512),
        (("c0", "c1"), 512),
        (("c5", "c6"), 1),
    ]:
        visible: dict[str, Any] = {
            f"{alias}:{i}": {} for alias in aliases for i in range(4)
        }
        schema = action_schema(["answer"], list(aliases), list(visible), 2)
        documents = schema["anyOf"][0]["properties"]["documents"]
        documents["minItems"] = 2
        for branch, label in zip(
            documents["items"]["anyOf"], ["SUPPORTS", "REFUTES"], strict=True
        ):
            branch["properties"]["label"]["enum"] = [label]
        observation = {
            "immutable_claim": "Synthetic schema check; not a scientific finding.",
            "allowed_actions": ["answer"],
            "current_citable": [
                {"sentence_id": sid, "text": "Synthetic protocol fixture sentence."}
                for sid in visible
            ],
            "preview_only": [],
            "feedback": None,
            "remaining_calls": 1,
            "remaining_tools": 0,
        }
        cases.append(
            {
                "aliases": list(aliases),
                "visible": visible,
                "schema": schema,
                "observation": observation,
                "max_output_tokens": cap,
            }
        )
    return cases


def synthetic_runtime_smoke(
    provider: V3Provider, *, execution_kind: str
) -> dict[str, Any]:
    if getattr(provider, "terminal_protocol", None) != PROTOCOL:
        raise ValueError("wrong smoke provider contract")
    if execution_kind not in {"mock_hf", "real_model"}:
        raise ValueError("explicit smoke execution kind required")
    records = []
    for case in smoke_cases():
        record: dict[str, Any] = {
            "input_tokens_expected": None,
            "max_output_tokens": case["max_output_tokens"],
            "passed": False,
            "provider_generate_attempted": False,
        }
        try:
            count = provider.count_prompt(case["observation"], case["schema"])
            record["input_tokens_expected"] = count
            record["provider_generate_attempted"] = True
            response = provider.generate(
                case["observation"], case["schema"], case["max_output_tokens"], 45
            )
            # Retain incurred cost even if parsing or a later check fails.
            record.update(usage=response["usage"], diagnostics=response["diagnostics"])
            if case["max_output_tokens"] == 1:
                try:
                    json.loads(response["raw"])
                    incomplete = False
                except json.JSONDecodeError:
                    incomplete = True
                passed = (
                    incomplete
                    and response["usage"]["output_tokens"] == 1
                    and response["diagnostics"]["reached_max_new_tokens"]
                    and not response["diagnostics"]["eos_observed"]
                )
            else:
                value = parse_action(
                    json.loads(response["raw"]),
                    ["answer"],
                    case["visible"],
                    case["aliases"],
                    2,
                )
                expected = dict(
                    zip(case["aliases"], ["SUPPORTS", "REFUTES"], strict=True)
                )
                passed = (
                    {d["source_id"]: d["label"] for d in value["documents"]} == expected
                    and response["diagnostics"]["eos_observed"]
                    and not response["diagnostics"]["reached_max_new_tokens"]
                )
            record["passed"] = bool(
                passed and response["usage"]["input_tokens"] == count
            )
        except ModelResponseValidationError as exc:
            record.update(
                usage=exc.usage,
                diagnostics=exc.diagnostics,
                failure="provider_response_invalid",
            )
        except Exception as exc:
            record["failure"] = type(exc).__name__
            record["failure_stage"] = (
                "before_generation"
                if not record["provider_generate_attempted"]
                else (
                    "model_protocol_validation"
                    if "usage" in record
                    else "provider_call"
                )
            )
        records.append(record)
    return {
        "schema_version": "scifact-synthetic-runtime-smoke-v1",
        "execution_kind": execution_kind,
        "status": "passed" if all(r["passed"] for r in records) else "failed",
        "records": records,
        "attempted_calls": sum(r["provider_generate_attempted"] for r in records),
        "attempted_calls_definition": "provider.generate invocations, not tokenizer-only failures",
        "failure_policy": "Record each failure; continue remaining independent synthetic cases without retrying a failed case.",
        "official_data_read": False,
        "quality_evaluation": False,
        "boundary": "Forced synthetic syntax/identity checks, not autonomous tool selection or model quality.",
    }
