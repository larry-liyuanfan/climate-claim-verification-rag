"""Fail-closed LoRA restoration, including PEFT versions without key_mapping.

Only the known CausalLM -> bare transformer prefix change is permitted. Counting
injected parameters is not evidence that a checkpoint was actually restored.
"""
from __future__ import annotations

from collections.abc import Callable, Mapping
from typing import Any

import numpy as np


def align_adapter_state(
    saved: Mapping[str, Any], expected: Mapping[str, Any]
) -> tuple[dict[str, Any], int]:
    if not expected or not any("lora_" in key for key in expected):
        raise ValueError("target has no LoRA checkpoint tensors")
    aligned: dict[str, Any] = {}
    remapped = 0
    for key, value in saved.items():
        target = key
        if target not in expected and key.startswith("base_model.model.model."):
            target = "base_model.model." + key[len("base_model.model.model."):]
        if target not in expected:
            raise ValueError(f"unrecognized adapter tensor: {key}")
        if target in aligned:
            raise ValueError(f"adapter key collision: {target}")
        if tuple(value.shape) != tuple(expected[target].shape):
            raise ValueError(f"adapter shape mismatch: {key}")
        aligned[target] = value
        remapped += int(target != key)
    missing = sorted(set(expected) - set(aligned))
    if missing:
        raise ValueError(f"missing adapter tensors: {missing}")
    return aligned, remapped


def load_verified_lora(auto_model: Any, adapter_path: str) -> tuple[Any, dict[str, Any]]:
    """Restore local tensors explicitly; never rely on ignored loader kwargs."""
    try:
        import peft
        import torch
        from peft import PeftConfig, PeftModel
        from peft.utils.save_and_load import (
            get_peft_model_state_dict,
            load_peft_weights,
            set_peft_model_state_dict,
        )
    except ImportError as exc:
        raise RuntimeError("peft and torch are required to load a dense adapter") from exc
    config = PeftConfig.from_pretrained(adapter_path)
    if str(config.peft_type) not in ("LORA", "PeftType.LORA"):
        raise ValueError("dense adapter integrity supports LoRA only")
    config.inference_mode = True
    # Generic wrapper: the served model is Qwen3Model, not Qwen3ForCausalLM.
    model = PeftModel(auto_model, config)
    # PEFT 0.15.2 leaves this callable unannotated; runtime identity is unchanged.
    get_state: Callable[..., dict[str, Any]] = get_peft_model_state_dict
    expected = get_state(model, save_embedding_layers=False)
    saved = load_peft_weights(adapter_path, device="cpu")
    aligned, remapped = align_adapter_state(saved, expected)
    for key, value in aligned.items():
        if not bool(torch.isfinite(value).all()):
            raise ValueError(f"non-finite checkpoint tensor: {key}")
    loaded = set_peft_model_state_dict(model, aligned, adapter_name="default")
    missing_lora = [key for key in loaded.missing_keys if "lora_" in key]
    if missing_lora or loaded.unexpected_keys:
        raise ValueError(
            f"adapter restoration mismatch: {missing_lora}; {loaded.unexpected_keys}"
        )
    restored = get_state(model, save_embedding_layers=False)
    if set(restored) != set(aligned):
        raise ValueError("restored adapter key set changed")
    for key, value in restored.items():
        reference = aligned[key].to(device=value.device, dtype=value.dtype)
        if not bool(torch.isfinite(value).all()) or not torch.equal(value, reference):
            raise ValueError(f"checkpoint value not restored: {key}")
    model.set_adapter("default")
    model.requires_grad_(False)
    model.eval()
    count = sum(value.numel() for key, value in restored.items() if "lora_" in key)
    if count <= 0:
        raise ValueError("restored adapter contains no LoRA parameters")
    return model, {
        "schema_version": 1,
        "peft_version": peft.__version__,
        "torch_version": torch.__version__,
        "checkpoint_tensor_count": len(saved),
        "restored_tensor_count": len(restored),
        "remapped_tensor_count": remapped,
        "lora_parameter_count": count,
        "all_checkpoint_values_match": True,
        "comparison": "exact equality after cast to destination dtype",
        "active_adapter": "default",
        "output_effect_verified": False,
    }


def embedding_effect_audit(
    enabled: np.ndarray[Any, Any], disabled: np.ndarray[Any, Any],
    repeated: np.ndarray[Any, Any], *, atol: float = 1e-7
) -> dict[str, Any]:
    arrays = [np.asarray(value, dtype=np.float32) for value in (enabled, disabled, repeated)]
    if atol <= 0 or not np.isfinite(atol):
        raise ValueError("probe tolerance must be finite and positive")
    if arrays[0].ndim != 2 or not arrays[0].size:
        raise ValueError("probe requires nonempty embedding matrices")
    if any(value.shape != arrays[0].shape or not np.isfinite(value).all() for value in arrays):
        raise ValueError("probe embeddings have incompatible shapes or non-finite values")
    delta = float(np.max(np.abs(arrays[0] - arrays[1])))
    repeat_delta = float(np.max(np.abs(arrays[0] - arrays[2])))
    return {
        "probe_count": len(arrays[0]),
        "dimension": arrays[0].shape[1],
        "max_absolute_enabled_disabled_delta": delta,
        "max_absolute_repeat_delta": repeat_delta,
        "absolute_tolerance": atol,
        "effect_verified": delta > atol and repeat_delta <= atol,
        "boundary": "fixed-input integrity check; not retrieval quality evidence",
    }
