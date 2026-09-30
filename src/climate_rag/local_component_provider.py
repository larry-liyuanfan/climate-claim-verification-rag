"""Opt-in component backend; importing this module never loads model weights."""
from __future__ import annotations

import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from .agent_protocol import ModelResponseValidationError
from .component_decoder import decoder_identity, decoder_schema
from .local_scifact_provider import LocalQwenSciFactProvider
from .private_diagnostics_v3 import PrivateDiagnosticStore
from .scifact_component_contract import LIMITS, VERSION, packing, render, require, schema_for
from .scifact_semantic_contract import MODEL_SHA, TOKENIZER_SHA, checked, sha


class LocalQwenComponentProvider(LocalQwenSciFactProvider):
    component_protocol = VERSION

    def __init__(self, model_dir: Path, manifest: dict[str, str], *, private_dir: Path, device: str = "cuda") -> None:
        require(sha(json.dumps(manifest, sort_keys=True).encode()) == MODEL_SHA, "component_model_identity")
        for name, expected in TOKENIZER_SHA.items():
            checked(model_dir / name, expected)
        super().__init__(model_dir, manifest, private_dir=private_dir, device=device)
        self.name = self.base.name + ":" + VERSION
        self.tokenizer = self.base.tokenizer
        self._slot_ready = False

    def start_slot(self, directory: Path) -> None:
        directory.mkdir(mode=0o700)
        self.private_store = PrivateDiagnosticStore(directory, max_files=2, max_bytes=49152)
        self._slot_ready = True

    def render(self, observation: Mapping[str, Any], schema: Mapping[str, Any]) -> str:
        require(schema == schema_for(observation), "component_schema_mutation")
        return render(self.base.tokenizer, observation)

    def decoder_schema(self, schema: Mapping[str, Any]) -> dict[str, Any]:
        return decoder_schema(schema)

    def generate(self, observation: dict[str, Any], schema: dict[str, Any],
                 max_output_tokens: int, remaining_seconds: float) -> dict[str, Any]:
        require(self._slot_ready, "component_single_attempt_only")
        require(max_output_tokens == LIMITS["output_tokens"] and 0 < remaining_seconds <= LIMITS["seconds"], "component_budget")
        expected = packing(self.base.tokenizer, observation)
        require(expected["status"] == "prepared" and schema == schema_for(observation), "component_packing")
        self._slot_ready = False  # never retry after entry into the base generator
        response = super().generate(observation, schema, max_output_tokens, remaining_seconds)
        # Everything after the actual call preserves its usage, including missing
        # receipt fields and prompt-count mismatch. Never turn paid work into zero.
        diagnostics = dict(response.get("diagnostics", {}))
        diagnostics.update(actual_component_packing=expected, decoder=decoder_identity(schema))
        response["diagnostics"] = diagnostics
        try:
            for key in ("private_attachment", "grammar_log"):
                receipt = diagnostics[key]
                require(receipt.get("truncated") is False and receipt.get("io_failed") is False
                        and receipt.get("stored_bytes") == receipt.get("attempted_bytes"),
                        "component_private_response_incomplete")
            require(response["usage"]["input_tokens"] == expected["input_tokens"], "component_actual_prompt_mismatch")
        except Exception as exc:
            raise ModelResponseValidationError(response.get("usage", {}), diagnostics |
                {"category": "component_post_generation_validation", "exception_type": type(exc).__name__}) from exc
        return response
