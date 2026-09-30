"""Frozen-adapter binding for the existing bare bounded controller, not a new Agent."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .agent_protocol import ModelResponseValidationError
from .agent_v3 import V3Budget
from .local_bounded_scifact_provider import LocalQwenBoundedSciFactProvider
from .scifact_semantic_contract import (
    BUDGET, MODEL_SHA, RERANKER_SHA, ROUTES, TOKENIZER_SHA, checked, encoded, load_inference, sha,
)
from .scifact_terminal import system_prompt_scifact

PROTOCOL = 'scifact-adapter-bare-four-route-regression-20261001-v1'
PREVIOUS_SOURCE = '99cd9ff707697ea395a9bc067f3ce91395071cdb'
PREVIOUS_PROTOCOL = '5595143cfb6d276184e91f40856efa388ab440dc78527fe7fcffd52c12bb296d'
INFERENCE_SHA = 'def10d4030d402ef5294680e0426ba1ed05ac42cb9b1d7bf4e8daa0a328961c9'
SCORING_SHA = '7e2303364b69be920de017179485bbb6f9d0e5d48367717576b3411c92c8ca8e'
SCORING_MANIFEST = '9bb6b831360130a11b4bb17621d26a0bf1c5b41110a697c74c6ce6b5e2e8aa04'
TRAINING_SHA = 'f5e6a865ec09cb67a520e36ba4646fb297a381383d4d0357208488ef6e9168a0'
ADAPTER_SHA = 'dd1974a26549b3337824252a9c38773873acce72427f92e656835bf9d71935d4'
DATA_SHA = '28de2c5d1d531aabdb757ccb45db0b91232a54ac7089ac7dc5ebf42e65ba3b2f'
FIT_SHA = '70e40db85b0e51292ee1235be6bb708272846a0fcd4be2d123cfb4f2238d15a5'
CONFIG_SHA = '0e63b57b59c5dcf322dcfd620b881edc30b7e71a3ea61bcc9a3def869bcb85f2'
SOURCES = ('scifact_terminal.py', 'scifact_bounded_agent.py', 'scifact_bounded_runtime.py',
           'local_bounded_scifact_provider.py', 'bounded_scifact_grammar.py',
           'bounded_token_traversal.py', 'evidence_gap_candidate.py', 'local_agent_v3.py')


def require(ok: bool, reason: str) -> None:
    if not ok:
        raise ValueError(reason)


def policy_identity(source: Path) -> dict[str, Any]:
    require(V3Budget().model_dump() == BUDGET, 'unchanged_bounded_budget')
    return {'protocol': PROTOCOL, 'scope': 'previously_exposed_TRAIN_regression_not_heldout',
        'base_model_manifest_sha256': MODEL_SHA, 'adapter_model_sha256': ADAPTER_SHA,
        'adapter_training_sha256': TRAINING_SHA, 'adapter_name': 'default', 'adapter_active': True,
        'reranker_manifest_sha256': RERANKER_SHA, 'tokenizer_sha256': TOKENIZER_SHA,
        'prompt_sha256': sha(system_prompt_scifact().encode()),
        'implementation_sha256': {n: sha((source / 'src/climate_rag' / n).read_bytes()) for n in SOURCES},
        'schema': 'unchanged_state_dependent_scifact_action_schema_bound_per_physical_call',
        'decoder': 'greedy_nonthinking_bounded_grammar_no_sampling',
        'packing': 'existing_CommonPacking_max_bare_empty_G_active_not_pure_bare_counting',
        'gap': False, 'routes': list(ROUTES), 'budget': BUDGET,
        'planned_slots': 48, 'max_generator_calls': 168, 'max_rerank_calls': 36,
        'max_rerank_pairs': 720, 'warmup_or_synthetic_preflight_calls': 0,
        'alias_policy': 'unchanged_controller_c0_based_not_SFT_c1_based',
        'reranker_settings': {'max_length': 2048, 'batch_size': 1, 'dtype': 'bfloat16'}}


def load_frozen(directory: Path) -> tuple[list[dict[str, Any]], bytes]:
    # Old protocol validates the exact old bundle. It is NOT the execution identity
    # of this new bare-adapter regression; no semantic G controller is invoked.
    _, claims, corpus = load_inference(directory, PREVIOUS_PROTOCOL, PREVIOUS_SOURCE)
    return claims, corpus


def checkpoint_metadata(adapter: Path) -> dict[str, Any]:
    training: dict[str, Any] = json.loads(checked(adapter / 'complete.json', TRAINING_SHA))
    require(training['data_manifest_sha256'] == DATA_SHA and training['config_sha256'] == CONFIG_SHA,
            'frozen_training_background')
    require(training['adapter_files']['adapter_model.safetensors'] == ADAPTER_SHA, 'fixed_adapter_file')
    for name, digest in training['adapter_files'].items():
        require(Path(name).name == name, 'checkpoint_filename')
        checked(adapter / 'final' / name, digest)
    return training


def active_state(model: Any, expected_layers: int = 72) -> dict[str, Any]:
    require(model.active_adapters == ['default'] and not model.training, 'active_default_eval_adapter')
    layers = [m for _, m in model.named_modules() if hasattr(m, 'lora_A') and hasattr(m, 'lora_B')]
    require(len(layers) == expected_layers, 'adapter_layer_count')
    for layer in layers:
        require(not layer.disable_adapters and not layer.merged
                and layer.active_adapters == ['default']
                and set(layer.lora_A) == set(layer.lora_B) == {'default'}, 'disabled_or_merged_adapter')
    require(all(not p.requires_grad for p in model.parameters()), 'inference_only_frozen_parameters')
    return {'active_adapters': ['default'], 'lora_layers': len(layers),
            'enabled': True, 'merged': False, 'trainable_parameters': 0}


class ActiveAdapterProvider(LocalQwenBoundedSciFactProvider):
    """Only adds identity/active checks; inherits all generation and packing logic."""

    def bind(self, policy: dict[str, Any]) -> dict[str, Any]:
        require(self.gap is False and policy['adapter_active'] is True, 'bare_active_provider')
        state = active_state(self.base.model)
        self.regression_policy_sha = sha(encoded(policy))
        self.name += ':active-adapter:' + ADAPTER_SHA
        return state

    def generate(self, observation: dict[str, Any], schema: dict[str, Any],
                 max_output_tokens: int, remaining_seconds: float) -> dict[str, Any]:
        binding = {'regression_policy_sha256': self.regression_policy_sha,
            'base_model_sha256': MODEL_SHA, 'adapter_sha256': ADAPTER_SHA,
            'adapter_state': active_state(self.base.model),
            'actual_prompt_sha256': sha(self.render(observation, schema).encode()),
            'actual_schema_sha256': sha(encoded(schema))}
        try:
            result = super().generate(observation, schema, max_output_tokens, remaining_seconds)
        except ModelResponseValidationError as exc:
            exc.diagnostics.update(binding)
            raise
        result['diagnostics'].update(binding)
        return result
