"""Opt-in v3 local model adapter. No model/sample change or API fallback.

LMFE controls syntax only. Its internal fallback can emit EOS on errors, so
every response still goes through the controller's strict semantic-ID contract.
Logging is isolated for this serial offline provider because LMFE may log the
entire prefix on an internal error. It is NOT a concurrent HTTP provider.
"""

from __future__ import annotations

import importlib.metadata
import json
import logging
import time
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from .agent_protocol import ModelResponseValidationError
from .agent_v3 import system_prompt_v3
from .local_agent_model import LocalQwenDecisionProvider
from .model_diagnostics import response_diagnostics, validate_private_directory
from .private_diagnostics_v3 import PrivateDiagnosticStore, PrivateLogHandler


def render_v3_prompt(
    tokenizer: Any, observation: Mapping[str, Any], schema: Mapping[str, Any]
) -> str:
    return str(
        tokenizer.apply_chat_template(
            [
                {
                    "role": "system",
                    "content": system_prompt_v3()
                    + "\n"
                    + json.dumps(schema, separators=(",", ":")),
                },
                {
                    "role": "user",
                    "content": json.dumps(
                        observation, ensure_ascii=False, separators=(",", ":")
                    ),
                },
            ],
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )
    )


class LocalQwenV3Provider:
    kind = "local_model"

    def __init__(
        self,
        model_dir: Path,
        manifest: dict[str, str],
        *,
        private_dir: Path,
        device: str = "cuda",
    ) -> None:
        if importlib.metadata.version("lm-format-enforcer") != "0.11.3":
            raise ValueError("v3 grammar adapter pins lm-format-enforcer0.11.3")
        self.private_dir = validate_private_directory(private_dir)
        self.private_store = PrivateDiagnosticStore(self.private_dir)
        self.base = LocalQwenDecisionProvider(model_dir, manifest, device=device)
        self.name = self.base.name + ":sentence-id-v3:lmfe0.11.3"
        from lmformatenforcer.integrations.transformers import (
            build_token_enforcer_tokenizer_data,
        )

        # Only tokenizer data may be reused; parser/prefix state is per generation.
        self.tokenizer_data = build_token_enforcer_tokenizer_data(
            self.base.tokenizer, use_bitmask=False
        )

    def count_text(self, text: str) -> int:
        return len(self.base.tokenizer.encode(text, add_special_tokens=False))

    def count_prompt(
        self, observation: Mapping[str, Any], schema: Mapping[str, Any]
    ) -> int:
        prompt = render_v3_prompt(self.base.tokenizer, observation, schema)
        return len(self.base.tokenizer.encode(prompt, add_special_tokens=False))

    def generate(
        self,
        observation: dict[str, Any],
        schema: dict[str, Any],
        max_output_tokens: int,
        remaining_seconds: float,
    ) -> dict[str, Any]:
        from lmformatenforcer import JsonSchemaParser
        from lmformatenforcer.integrations.transformers import (
            build_transformers_prefix_allowed_tokens_fn,
        )

        begin = time.perf_counter()
        parser = JsonSchemaParser(schema)
        prefix_fn = build_transformers_prefix_allowed_tokens_fn(
            self.tokenizer_data, parser
        )
        prompt = render_v3_prompt(self.base.tokenizer, observation, schema)
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
        if diagnostics["grammar_log_nonempty"] or log_sink.io_failed:
            raise ModelResponseValidationError(
                usage, {**diagnostics, "category": "grammar_backend_error"}
            )
        if raw_sink.io_failed:
            raise ModelResponseValidationError(
                usage, {**diagnostics, "category": "diagnostic_write_failure"}
            )
        return {"raw": raw, "usage": usage, "diagnostics": diagnostics}
