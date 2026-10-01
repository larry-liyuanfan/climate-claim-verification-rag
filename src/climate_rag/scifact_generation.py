"""Opt-in, source-bound decoding receipt for the paired SciFact experiment.

No weights are loaded here. Old protocols keep their historical provider path.
The seed fixes RNG initialization, not CUDA bitwise determinism or answer quality.
"""
from __future__ import annotations

import copy
import hashlib
from dataclasses import asdict, is_dataclass
import importlib.metadata
from functools import lru_cache
import math
from pathlib import Path
import random
from typing import Any

from .local_agent_v3 import grammar_config_identity
from .scifact_read_continuation import ordered_write
from .scifact_semantic_contract import checked
from .scifact_utility_contract import identity

CONFIG_FILE_SHA = "2325da0f15bb848e018c5ae071b7943332e9f871d6b60e2ed22ca97d4cb993d2"
FILE_DEFAULTS = {
    "bos_token_id": 151643, "do_sample": True, "eos_token_id": [151645, 151643],
    "pad_token_id": 151643, "temperature": 0.6, "top_k": 20, "top_p": 0.95,
    "transformers_version": "4.51.0",
}
CONTRACT: dict[str, Any] = {
    "version": "scifact-paired-decoding-v1-20261002",
    "generation_config_file_sha256": CONFIG_FILE_SHA,
    "generation_config_file_defaults": FILE_DEFAULTS,
    "transformers": "4.51.3", "seed": 20261002,
    "seed_scope": "reset_python_numpy_torch_before_each_physical_generation",
    "overrides": {"do_sample": False, "num_beams": 1, "max_new_tokens": 512},
    "max_time": "remaining_episode_seconds_minus_preparation_elapsed",
    "max_length": "prompt_tokens_plus_max_new_tokens",
    "use_model_defaults": False,
    "stop_rules": ["max_length", "max_time", "eos_token_id"],
    "custom_stopping_criteria": False, "stop_strings": None,
    "grammar": grammar_config_identity(), "fresh_parser_per_call": True,
    "tokenizer_eos_token_id": 151645, "lmfe_eos_token_id": 151645, "lmfe_use_bitmask": False,
    "cuda_bitwise_determinism_claimed": False,
}


def snapshot(config: Any) -> dict[str, Any]:
    result = copy.deepcopy(config.to_dict())
    compile_config = getattr(config, "compile_config", None)
    if compile_config is not None and (not is_dataclass(compile_config) or isinstance(compile_config, type)):
        raise ValueError("unsupported_compile_config")
    result["compile_config"] = asdict(compile_config) if compile_config is not None else None
    return dict(result)


def expected_defaults() -> dict[str, Any]:
    # Parent release validation happens before worker/model loading, so the
    # accepted Spartan Torch 2.1 shim must precede this first Transformers import.
    from .torch_compat import ensure_torch_pytree_compat
    ensure_torch_pytree_compat()
    from transformers.generation.configuration_utils import GenerationConfig

    if importlib.metadata.version("transformers") != CONTRACT["transformers"]:
        raise ValueError("paired_transformers_version")
    return snapshot(GenerationConfig.from_dict(copy.deepcopy(FILE_DEFAULTS)))


def frozen_contract() -> dict[str, Any]:
    return dict(CONTRACT, expanded_model_defaults=expected_defaults())


class GenerationBinding:
    def __init__(self, base: Any, model_dir: Path, contract: dict[str, Any]) -> None:
        if contract != frozen_contract():
            raise ValueError("paired_generation_contract")
        checked(model_dir / "generation_config.json", CONFIG_FILE_SHA)
        self.defaults = snapshot(base.model.generation_config)
        if self.defaults != expected_defaults():
            raise ValueError("loaded_generation_defaults_changed")
        self.base = base
        self.receipt = {
            "contract": copy.deepcopy(contract), "contract_sha256": identity(contract),
            "loaded_model_defaults": self.defaults,
            "loaded_model_defaults_sha256": identity(self.defaults),
            "source_file_version": FILE_DEFAULTS["transformers_version"],
        }
        self.path: Path | None = None

    def bind_record(self, path: Path) -> None:
        if self.path is not None:
            raise ValueError("previous_generation_record_not_consumed")
        self.path = path

    def prepare(self, length: int, tokens: int, remaining: float, schema: Any,
                decoder: Any, tokenizer_data: Any) -> tuple[Any, dict[str, Any]]:
        if self.path is None or tokens != 512 or not math.isfinite(remaining) or remaining <= 0:
            raise ValueError("unbound_generation_or_budget")
        if snapshot(self.base.model.generation_config) != self.defaults:
            raise ValueError("generation_defaults_mutated")
        config = copy.deepcopy(self.base.model.generation_config)
        overrides = dict(CONTRACT["overrides"], max_time=remaining, max_length=length + tokens)
        if config.update(**overrides):
            raise ValueError("unconsumed_generation_override")
        # Explicit config alone is insufficient: 4.51.3 can refill do_sample
        # from model defaults. Disable that merge at the actual generate call.
        config._from_model_config = False
        if str(config.get_generation_mode().value) != "greedy_search":
            raise ValueError("not_greedy_generation")
        effective = snapshot(config)
        parser = decoder.parser.config
        if (tokenizer_data.eos_token_id != 151645 or self.base.tokenizer.eos_token_id != 151645
                or tokenizer_data.use_bitmask is not False
                or parser.alphabet != tokenizer_data.tokenizer_alphabet
                or parser.max_consecutive_whitespaces != 12 or parser.force_json_field_order
                or parser.max_json_array_length != 20):
            raise ValueError("actual_lmfe_binding_changed")
        import numpy as np

        random.seed(CONTRACT["seed"])
        np.random.seed(CONTRACT["seed"])
        self.base._torch.manual_seed(CONTRACT["seed"])
        row = {**self.receipt, "overrides": overrides, "effective": effective,
               "effective_sha256": identity(effective), "schema_sha256": identity(schema),
               "input_tokens": length, "use_model_defaults": False,
               "effective_parser_config": {
                   "alphabet_sha256": hashlib.sha256(parser.alphabet.encode()).hexdigest(),
                   "max_consecutive_whitespaces": 12, "force_json_field_order": False,
                   "max_json_array_length": 20},
               "runtime_numerics": {
                   "deterministic_algorithms": self.base._torch.are_deterministic_algorithms_enabled(),
                   "cuda_matmul_allow_tf32": self.base._torch.backends.cuda.matmul.allow_tf32,
                   "cudnn_allow_tf32": self.base._torch.backends.cudnn.allow_tf32,
                   "cudnn_deterministic": self.base._torch.backends.cudnn.deterministic,
                   "cudnn_benchmark": self.base._torch.backends.cudnn.benchmark}}
        ordered_write(self.path, row)  # Durable even when generate is killed.
        self.path = None
        return config, row


@lru_cache(maxsize=2)
def expected_parser_config(tokenizer: Any) -> dict[str, Any]:
    from lmformatenforcer.integrations.transformers import build_token_enforcer_tokenizer_data
    data = build_token_enforcer_tokenizer_data(tokenizer, use_bitmask=False)
    if data.eos_token_id != 151645:
        raise ValueError("audit_tokenizer_eos")
    return {"alphabet_sha256": hashlib.sha256(data.tokenizer_alphabet.encode()).hexdigest(),
            "max_consecutive_whitespaces": 12, "force_json_field_order": False, "max_json_array_length": 20}


def audit_generation(row: Any, request: Any, diagnostics: Any, parser_config: Any) -> None:
    """Verify the recorded config without loading any weights or changing scores."""
    defaults = expected_defaults()
    overrides = dict(CONTRACT["overrides"], max_time=row["overrides"]["max_time"],
                     max_length=row["input_tokens"] + 512)
    expected = dict(defaults, **overrides, _from_model_config=False)
    if (row["contract"] != frozen_contract() or row["contract_sha256"] != identity(frozen_contract())
            or row["loaded_model_defaults"] != defaults
            or row["loaded_model_defaults_sha256"] != identity(defaults)
            or row["source_file_version"] != "4.51.0"
            or row["overrides"] != overrides or row["effective"] != expected
            or row["effective_sha256"] != identity(expected)
            or row["schema_sha256"] != identity(request["schema"])
            or row["use_model_defaults"] is not False
            or row["effective_parser_config"] != parser_config
            or ("effective_parser_config" in diagnostics and diagnostics["effective_parser_config"] != parser_config)
            or not 0 < overrides["max_time"] <= request["remaining_seconds"]
            or diagnostics.get("generation_binding") != row):
        raise ValueError("paired_effective_generation_binding")
