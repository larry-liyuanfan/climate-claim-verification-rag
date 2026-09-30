"""SciFact-specific Qwen/LMFE provider; serial offline only, never a dev runner.

Inherits verified loading/hash checks/private quotas from frozen v3. Generation
is a narrow versioned copy because its renderer is hard-coded; parity is tested.
"""

from __future__ import annotations

import hashlib
import logging
import os
import time
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from .agent_protocol import ModelResponseValidationError
from .local_agent_v3 import (
    LocalQwenV3Provider,
    fixed_grammar_config,
    grammar_config_identity,
)
from .model_diagnostics import response_diagnostics
from .private_diagnostics_v3 import PrivateLogHandler
from .scifact_terminal import PROTOCOL, render_scifact_prompt


def require_clean_grammar_environment() -> None:
    # LMFE TokenEnforcer replaces parser.config using environment-backed defaults.
    # Frozen v3's child runner removes these variables. This reusable adapter
    # fails closed instead of mutating process-global environment behind callers.
    if any(key.startswith("LMFE_") for key in os.environ):
        raise ValueError("LMFE environment overrides are not allowed for this protocol")


class LocalQwenSciFactProvider(LocalQwenV3Provider):
    terminal_protocol = PROTOCOL

    def __init__(
        self,
        model_dir: Path,
        manifest: dict[str, str],
        *,
        private_dir: Path,
        device: str = "cuda",
    ) -> None:
        require_clean_grammar_environment()
        super().__init__(model_dir, manifest, private_dir=private_dir, device=device)
        self.name = self.base.name + ":" + PROTOCOL + ":lmfe0.11.3"

    def count_prompt(
        self, observation: Mapping[str, Any], schema: Mapping[str, Any]
    ) -> int:
        prompt = render_scifact_prompt(self.base.tokenizer, observation, schema)
        return len(self.base.tokenizer.encode(prompt, add_special_tokens=False))

    def generate(
        self,
        observation: dict[str, Any],
        schema: dict[str, Any],
        max_output_tokens: int,
        remaining_seconds: float,
    ) -> dict[str, Any]:
        require_clean_grammar_environment()
        from lmformatenforcer import JsonSchemaParser
        from lmformatenforcer.integrations.transformers import (
            build_transformers_prefix_allowed_tokens_fn,
        )

        begin = time.perf_counter()
        parser = JsonSchemaParser(schema, config=fixed_grammar_config())
        prefix_fn = build_transformers_prefix_allowed_tokens_fn(
            self.tokenizer_data, parser
        )
        prompt = render_scifact_prompt(self.base.tokenizer, observation, schema)
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
            # No free retry; GPU may have worked even if no token sequence returns.
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
        diagnostics["grammar"] = "lm-format-enforcer0.11.3/fresh-inline-anyOf"
        diagnostics["grammar_config"] = grammar_config_identity()
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
