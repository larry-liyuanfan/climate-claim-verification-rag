"""Candidate-only Qwen/LMFE provider; real weights are NOT validated this release.

Versioned narrow override of local_scifact_provider: same loader/manifest, greedy
generation, timeout, private quotas and usage accounting; different renderer and
ordered grammar callback. Serial offline only. Never patches the old provider.
"""

from __future__ import annotations

import hashlib
import logging
import time
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from .agent_protocol import ModelResponseValidationError
from .evidence_gap_candidate import CANDIDATE_PROTOCOL, render_gap_prompt
from .evidence_gap_grammar import build_ordered_gap_prefix
from .local_scifact_provider import (
    LocalQwenSciFactProvider,
    require_clean_grammar_environment,
)
from .model_diagnostics import response_diagnostics
from .private_diagnostics_v3 import PrivateLogHandler


class LocalQwenEvidenceGapProvider(LocalQwenSciFactProvider):
    wire_protocol = CANDIDATE_PROTOCOL
    # A bare backend must not pass the frozen controller's action-only contract.
    # Only GapProviderAdapter exposes that compatibility identity after mapping.
    terminal_protocol = CANDIDATE_PROTOCOL

    def __init__(
        self,
        model_dir: Path,
        manifest: dict[str, str],
        *,
        private_dir: Path,
        device: str = "cuda",
    ) -> None:
        super().__init__(model_dir, manifest, private_dir=private_dir, device=device)
        self.name = self.base.name + ":" + CANDIDATE_PROTOCOL + ":ordered-lmfe0.11.3"

    def count_prompt(
        self, observation: Mapping[str, Any], schema: Mapping[str, Any]
    ) -> int:
        prompt = render_gap_prompt(self.base.tokenizer, observation, schema)
        return len(self.base.tokenizer.encode(prompt, add_special_tokens=False))

    def generate(
        self,
        observation: dict[str, Any],
        schema: dict[str, Any],
        max_output_tokens: int,
        remaining_seconds: float,
    ) -> dict[str, Any]:
        require_clean_grammar_environment()
        begin = time.perf_counter()
        prefix_fn = build_ordered_gap_prefix(self.tokenizer_data, schema)
        parser = prefix_fn.token_enforcer.root_parser
        prompt = render_gap_prompt(self.base.tokenizer, observation, schema)
        inputs = self.base.tokenizer(
            prompt, return_tensors="pt", add_special_tokens=False
        ).to(self.base.model.device)
        length = int(inputs.input_ids.shape[1])
        remaining = remaining_seconds - (time.perf_counter() - begin)
        if remaining <= 0:
            raise ModelResponseValidationError(
                {}, {"category": "deadline_before_generation"}
            )
        log_sink = self.private_store.sink("grammar", 16384)
        root = logging.getLogger()
        old_handlers, old_level = root.handlers[:], root.level
        handler = PrivateLogHandler(log_sink)
        root.handlers, root.level = [handler], logging.ERROR
        try:
            with self.base._torch.inference_mode():
                output = self.base.model.generate(
                    **inputs,
                    do_sample=False,
                    num_beams=1,
                    max_new_tokens=max_output_tokens,
                    max_time=remaining,
                    prefix_allowed_tokens_fn=prefix_fn,
                )
        except Exception as exc:
            raise ModelResponseValidationError(
                {"input_tokens": length, "output_tokens": 0},
                {
                    "category": "model_or_grammar_failure",
                    "output_usage_unknown": True,
                    "exception_type": type(exc).__name__,
                    "grammar_log": log_sink.receipt(),
                },
            ) from exc
        finally:
            root.handlers, root.level = old_handlers, old_level
            handler.close()
        generated = output[0][length:]
        usage = {"input_tokens": length, "output_tokens": len(generated)}
        try:
            raw = self.base.tokenizer.decode(generated, skip_special_tokens=True)
        except Exception as exc:
            raise ModelResponseValidationError(
                usage, {"category": "decode_failure"}
            ) from exc
        try:
            eos = self.base.model.generation_config.eos_token_id
            eos_ids = [eos] if isinstance(eos, int) else (eos or [])
            diagnostics = response_diagnostics(
                raw,
                output_tokens=len(generated),
                max_new_tokens=max_output_tokens,
                eos_observed=bool(len(generated) and int(generated[-1]) in eos_ids),
                generation_elapsed_ms=(time.perf_counter() - begin) * 1000,
            )
            raw_sink = self.private_store.sink("response", 32768)
            raw_sink.write(raw)
            diagnostics["private_attachment"] = raw_sink.receipt()
            diagnostics["grammar_log"] = log_sink.receipt()
        except Exception as exc:
            raise ModelResponseValidationError(
                usage,
                {
                    "category": "diagnostic_failure",
                    "exception_type": type(exc).__name__,
                },
            ) from exc
        diagnostics["grammar_log_nonempty"] = log_sink.attempted > 0
        diagnostics["grammar"] = "lm-format-enforcer0.11.3/candidate-ordered-gap"
        diagnostics["effective_parser_config"] = {
            "alphabet_sha256": hashlib.sha256(
                parser.config.alphabet.encode()
            ).hexdigest(),
            "max_consecutive_whitespaces": parser.config.max_consecutive_whitespaces,
            "force_json_field_order": parser.config.force_json_field_order,
            "max_json_array_length": parser.config.max_json_array_length,
        }
        if diagnostics["grammar_log_nonempty"] or log_sink.io_failed:
            raise ModelResponseValidationError(
                usage, {**diagnostics, "category": "grammar_backend_error"}
            )
        if raw_sink.io_failed:
            raise ModelResponseValidationError(
                usage, {**diagnostics, "category": "diagnostic_write_failure"}
            )
        return {"raw": raw, "usage": usage, "diagnostics": diagnostics}
