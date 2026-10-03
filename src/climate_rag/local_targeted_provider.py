"""Renderer specialization only: reuse verified loader, grammar and generation binding."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any

from .agent_protocol import ModelResponseValidationError
from .local_agent_v3 import render_v3_prompt
from .local_scifact_provider import LocalQwenSciFactProvider
from .private_diagnostics_v3 import PrivateDiagnosticStore
from .scifact_natural_contract import complete_wire
from .targeted_query import PROTOCOL
from . import stop_acquire
from . import fair_acquisition


class LocalTargetedProvider(LocalQwenSciFactProvider):
    generation_binding: Any
    gap = False
    terminal_protocol = PROTOCOL

    def __init__(
        self, model_dir: Path, manifest: dict[str, str], *, private_dir: Path, protocol: str = PROTOCOL
    ) -> None:
        super().__init__(model_dir, manifest, private_dir=private_dir)
        self.terminal_protocol = protocol
        self.name = self.base.name + ":" + protocol

    def render(self, observation: Mapping[str, Any], schema: Mapping[str, Any]) -> str:
        if observation.get("protocol") not in {PROTOCOL, stop_acquire.PROTOCOL, fair_acquisition.PROTOCOL, fair_acquisition.COVERAGE_PROTOCOL}:
            raise ValueError("targeted_prompt_protocol")
        return render_v3_prompt(self.base.tokenizer, observation, schema)

    def start_slot(self, path: Path) -> None:
        path.mkdir(mode=0o700)
        self.private_dir = path
        self.private_store = PrivateDiagnosticStore(
            path, max_files=10, max_bytes=5 * (32768 + 16384)
        )

    def generate(
        self,
        observation: dict[str, Any],
        schema: dict[str, Any],
        max_output_tokens: int,
        remaining_seconds: float,
    ) -> dict[str, Any]:
        result = super().generate(
            observation, schema, max_output_tokens, remaining_seconds
        )
        try:
            complete_wire(
                result, self.count_prompt(observation, schema), self.private_dir
            )
        except (ValueError, KeyError, TypeError, OSError) as exc:
            raise ModelResponseValidationError(
                result.get("usage", {}),
                dict(
                    result.get("diagnostics", {}),
                    category="incomplete_physical_response",
                ),
            ) from exc
        return result
