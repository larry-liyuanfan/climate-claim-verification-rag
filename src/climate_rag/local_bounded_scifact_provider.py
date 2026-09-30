"""Versioned F/F+G backend; frozen v1/source80 provider is not modified.

Loading, model hashes and private-store quotas inherit the verified loader.
Only prompt rendering/grammar and diagnostic metadata differ. No model or
dataset is loaded by importing this module.
"""
from __future__ import annotations

import hashlib
import json
import logging
import time
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from .agent_protocol import ModelResponseValidationError
from .bounded_scifact_grammar import PROTOCOL, build_bounded_scifact_prefix
from .evidence_gap_candidate import CANDIDATE_PROTOCOL, _no_duplicate_keys, render_gap_prompt
from .local_scifact_provider import LocalQwenSciFactProvider, require_clean_grammar_environment
from .model_diagnostics import response_diagnostics
from .private_diagnostics_v3 import PrivateDiagnosticStore, PrivateLogHandler
from .scifact_terminal import PROTOCOL as TERMINAL_PROTOCOL
from .scifact_terminal import render_scifact_prompt


def observation_identity(observation: Mapping[str, Any]) -> dict[str, Any]:
    """Actual final model input: previews are NOT treated as citable evidence."""
    return {
        "visible": [{"sentence_id": r["sentence_id"],
                     "sha256": hashlib.sha256(r["text"].encode()).hexdigest()}
                    for r in observation["current_citable"]],
        "previews": [{"source_id": r["source_id"],
                      "sha256": hashlib.sha256(r["preview"].encode()).hexdigest(),
                      "citable": r["citable"]} for r in observation["preview_only"]],
    }


def raw_action(raw: str, gap: bool) -> str | None:
    """Observational only: this never validates a proposal or executes a tool."""
    try:
        value = json.loads(raw, object_pairs_hook=_no_duplicate_keys)
        value = value.get("decision") if gap and isinstance(value, dict) else value
        action = value.get("action") if isinstance(value, dict) else None
        return action if isinstance(action, str) else None
    except (ValueError, TypeError):
        return None


class LocalQwenBoundedSciFactProvider(LocalQwenSciFactProvider):
    def __init__(self, model_dir: Path, manifest: dict[str, str], *,
                 private_dir: Path, device: str = "cuda", gap: bool = False) -> None:
        super().__init__(model_dir, manifest, private_dir=private_dir, device=device)
        from .bounded_token_traversal import PlainStringPartition
        self.gap = gap
        self.wire_protocol = CANDIDATE_PROTOCOL if gap else TERMINAL_PROTOCOL
        # The envelope backend must go through GapProviderAdapter, never directly
        # into the old action controller.
        self.terminal_protocol = self.wire_protocol
        self.name = self.base.name + ":" + PROTOCOL + (":gap" if gap else ":original")
        self.plain_partition = PlainStringPartition(self.tokenizer_data)
        self.wire_byte_limit = max(32768, 512 * max(
            len(text.encode("utf-8")) for _, text, _ in self.tokenizer_data.regular_tokens) + 16)

    def start_slot(self, path: Path) -> None:
        """Exclusive storage: five calls, wire plus grammar per call.

        Runner uses different directories for each slot and synthetic preflight.
        No prior files are removed; repeated slot paths fail closed.
        """
        path.mkdir(mode=0o700)
        self.private_store = PrivateDiagnosticStore(
            path, max_files=10, max_bytes=5 * (self.wire_byte_limit + 16384))

    def render(self, observation: Mapping[str, Any], schema: Mapping[str, Any]) -> str:
        fn = render_gap_prompt if self.gap else render_scifact_prompt
        return fn(self.base.tokenizer, observation, schema)

    def count_prompt(self, observation: Mapping[str, Any], schema: Mapping[str, Any]) -> int:
        return len(self.base.tokenizer.encode(self.render(observation, schema), add_special_tokens=False))

    def generate(self, observation: dict[str, Any], schema: dict[str, Any],
                 max_output_tokens: int, remaining_seconds: float) -> dict[str, Any]:
        require_clean_grammar_environment()
        begin = time.perf_counter()
        action = schema["properties"]["decision"] if self.gap else schema
        prefix = build_bounded_scifact_prefix(self.tokenizer_data, action,
                                              schema if self.gap else None,
                                              plain_partition=self.plain_partition)
        inputs = self.base.tokenizer(self.render(observation, schema), return_tensors="pt",
                                     add_special_tokens=False).to(self.base.model.device)
        length = int(inputs.input_ids.shape[1])
        remaining = remaining_seconds - (time.perf_counter() - begin)
        if remaining <= 0:
            raise ModelResponseValidationError({}, {"category": "deadline_before_generation"})
        log_sink = self.private_store.sink("grammar", 16384)
        root = logging.getLogger()
        old_handlers, old_level = root.handlers[:], root.level
        handler = PrivateLogHandler(log_sink)
        root.handlers, root.level = [handler], logging.ERROR
        try:
            with self.base._torch.inference_mode():
                output = self.base.model.generate(
                    **inputs, do_sample=False, num_beams=1,
                    max_new_tokens=max_output_tokens, max_time=remaining,
                    prefix_allowed_tokens_fn=prefix)
        except Exception as exc:
            raise ModelResponseValidationError(
                {"input_tokens": length, "output_tokens": 0},
                {"category": "model_or_grammar_failure", "output_usage_unknown": True,
                 "exception_type": type(exc).__name__, "grammar_log": log_sink.receipt()}) from exc
        finally:
            root.handlers, root.level = old_handlers, old_level
            handler.close()
        generated = output[0][length:]
        usage = {"input_tokens": length, "output_tokens": len(generated)}
        try:
            raw = self.base.tokenizer.decode(generated, skip_special_tokens=True)
        except Exception as exc:
            raise ModelResponseValidationError(usage, {"category": "decode_failure"}) from exc
        try:
            eos = self.base.model.generation_config.eos_token_id
            eos_ids = [eos] if isinstance(eos, int) else (eos or [])
            diagnostics = response_diagnostics(
                raw, output_tokens=len(generated), max_new_tokens=max_output_tokens,
                eos_observed=bool(len(generated) and int(generated[-1]) in eos_ids),
                generation_elapsed_ms=(time.perf_counter() - begin) * 1000)
            raw_sink = self.private_store.sink("response", self.wire_byte_limit)
            raw_sink.write(raw)
            diagnostics.update(private_attachment=raw_sink.receipt(), grammar_log=log_sink.receipt())
            config = prefix.token_enforcer.root_parser.config
            diagnostics.update(
                grammar=PROTOCOL, raw_wire_action=raw_action(raw, self.gap),
                raw_action_is_validated=False, actual_observation=observation_identity(observation),
                effective_parser_config={
                    "alphabet_sha256": hashlib.sha256(config.alphabet.encode()).hexdigest(),
                    "max_consecutive_whitespaces": config.max_consecutive_whitespaces,
                    "force_json_field_order": config.force_json_field_order,
                    "max_json_array_length": config.max_json_array_length},
                grammar_log_nonempty=log_sink.attempted > 0)
        except Exception as exc:
            raise ModelResponseValidationError(
                usage, {"category": "diagnostic_failure", "exception_type": type(exc).__name__}) from exc
        if diagnostics["grammar_log_nonempty"] or log_sink.io_failed:
            raise ModelResponseValidationError(usage, {**diagnostics, "category": "grammar_backend_error"})
        if raw_sink.io_failed:
            raise ModelResponseValidationError(usage, {**diagnostics, "category": "diagnostic_write_failure"})
        receipt = diagnostics["private_attachment"]
        if receipt["truncated"] or receipt["stored_bytes"] != receipt["attempted_bytes"]:
            raise ModelResponseValidationError(usage, {**diagnostics, "category": "private_response_incomplete"})
        return {"raw": raw, "usage": usage, "diagnostics": diagnostics}
