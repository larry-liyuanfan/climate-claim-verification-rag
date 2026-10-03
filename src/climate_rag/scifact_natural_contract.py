"""Fail-closed physical-prefix checks for natural FIT tool-utility diagnostics."""
from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from typing import Any

from .agent_v3 import V3Budget, valid_search_query
from .evidence_gap_candidate import _no_duplicate_keys
from .scifact_semantic_contract import MODEL_SHA
from .scifact_terminal import parse_action, render_scifact_prompt
from .scifact_utility_contract import identity, validate_prefix


def require(ok: bool, reason: str) -> None:
    if not ok:
        raise ValueError(reason)


def base_state(backend: Any) -> dict[str, Any]:
    model = backend.base.model
    parameters = [name for name, _ in model.named_parameters()]
    modules = [name for name, _ in model.named_modules()]
    require(backend.gap is False and not hasattr(model, "peft_config")
            and not any("lora_" in n.lower() for n in parameters + modules)
            and model.training is False, "actual_unwrapped_eval_base_required")
    return {"base_model_sha256": MODEL_SHA, "model_class": type(model).__module__ + "." + type(model).__name__,
            "peft_config_present": False, "lora_parameter_or_module_count": 0,
            "named_parameter_count": len(parameters), "named_module_count": len(modules),
            "training": False, "adapter_loaded": False}


def complete_wire(response: dict[str, Any], count: int, private: Path) -> None:
    usage, d = response["usage"], response["diagnostics"]
    require(type(usage["input_tokens"]) is int and usage["input_tokens"] == count
            and 0 < count <= 8192 and type(usage["output_tokens"]) is int
            and 0 < usage["output_tokens"] <= 512 and not d.get("output_usage_unknown")
            and d.get("output_tokens") == usage["output_tokens"], "physical_usage_mismatch")
    require(d.get("eos_observed") is True, "truncated_no_eos")
    require(d.get("grammar_log_nonempty") is False, "grammar_not_empty")
    raw = response["raw"]
    require(isinstance(raw, str), "raw_text_required")
    for key in ("private_attachment", "grammar_log"):
        receipt = d[key]
        require(receipt.get("truncated") is False and receipt.get("io_failed") is False
                and receipt.get("dropped_bytes") == 0
                and type(receipt.get("stored_bytes")) is int
                and receipt["stored_bytes"] == receipt.get("attempted_bytes")
                and receipt.get("sha256") == receipt.get("stored_prefix_sha256"), "incomplete_wire_receipt")
    payload = raw.encode()
    expected = hashlib.sha256(payload).hexdigest()
    require(d["output_sha256"] == expected == d["private_attachment"]["sha256"]
            and len(payload) == d["private_attachment"]["stored_bytes"], "raw_receipt_mismatch")
    require(d["grammar_log"]["stored_bytes"] == 0
            and d["grammar_log"]["sha256"] == hashlib.sha256(b"").hexdigest(), "empty_grammar_receipt")
    paths = list(private.glob("*-response.txt"))
    require(not any(p.is_symlink() for p in paths)
            and any(p.read_bytes() == payload for p in paths), "physical_response_missing_or_changed")
    elapsed = d.get("generation_elapsed_ms")
    require(type(elapsed) in {int, float} and math.isfinite(elapsed) and 0 <= elapsed <= 120000,
            "physical_generation_deadline")


def audit_attempt(attempt: dict[str, Any], frame: dict[str, Any], ledger: Path,
                  private: Path, tokenizer: Any, slot: str) -> dict[str, Any]:
    key = attempt["diagnostics"]["physical_attempt_id"]
    require(isinstance(key, str) and key.startswith("g") and key[1:].isdigit()
            and 0 <= int(key[1:]) < 168, "physical_id_range")
    request = json.loads((ledger / (key + ".reserved.json")).read_bytes())
    finished = json.loads((ledger / (key + ".finished.json")).read_bytes())
    require(request["physical_attempt_id"] == finished["physical_attempt_id"] == key
            and request["slot"] == finished["slot"] == slot and finished["status"] == "returned"
            and finished["usage_known"] is True and attempt["status"] == "valid_decision"
            and attempt["usage_known"] is True, "physical_call_not_valid")
    prompt = render_scifact_prompt(tokenizer, frame["observation"], frame["schema"])
    count = len(tokenizer.encode(prompt, add_special_tokens=False))
    require(identity(request["observation"]) == identity(frame["observation"])
            and identity(request["schema"]) == identity(frame["schema"])
            and request["prompt_sha256"] == identity(prompt)
            and request["token_ids_sha256"] == identity(tokenizer.encode(prompt, add_special_tokens=False))
            and count == frame["prompt_tokens"] == attempt["input_prompt_tokens"], "physical_initial_frame_changed")
    response = finished["response"]
    require(response["usage"] == attempt["usage"] and finished["usage"] == attempt["usage"], "attempt_usage_changed")
    complete_wire(response, count, private)
    d = parse_action(json.loads(response["raw"], object_pairs_hook=_no_duplicate_keys),
                     frame["observation"]["allowed_actions"], frame["visible"], list(frame["alias_to_source"]), 5)
    require(d["action"] == attempt["action"], "attempt_action_changed")
    return dict(d)


def valid_physical_prefix(prefix: dict[str, Any], ledger: Path, private: Path,
                          tokenizer: Any, slot: str) -> None:
    p = validate_prefix(prefix, prefix["payload"]["claim"], V3Budget())
    require(p["elapsed_seconds"] < 120, "prefix_deadline")
    decision = audit_attempt(p["attempt"], p["frame"], ledger, private, tokenizer, slot)
    require(decision == p["decision"], "prefix_decision_changed")
    if decision["action"] == "read":
        require(decision["source_ids"] != p["selected"], "prefix_read_loop")
    if decision["action"] == "rewrite":
        require(valid_search_query(p["claim"], decision["query"]), "prefix_rewrite_invalid")


def ranked_preview_read(prefix: dict[str, Any]) -> tuple[dict[str, Any] | None, str | None]:
    """First preview-only candidate in original recall rank, independent of response."""
    p = prefix["payload"]
    previews = {r["source_id"] for r in p["frame"]["observation"]["preview_only"] if r["citable"] is False}
    for candidate in p["candidates"]:
        if candidate in previews and candidate not in p["selected"]:
            return {"action": "read", "source_ids": [candidate]}, None
    return None, "initial_ranked_preview_unavailable"
