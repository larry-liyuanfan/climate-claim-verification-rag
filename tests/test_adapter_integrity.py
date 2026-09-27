from __future__ import annotations

import numpy as np
import pytest

from climate_rag.adapter_integrity import (
    align_adapter_state,
    embedding_effect_audit,
    load_verified_lora,
)

PREFIX = "base_model.model.layers.0.q_proj."


def test_exact_and_known_prefix_mapping() -> None:
    expected = {PREFIX + "lora_A.weight": np.zeros((2, 4)),
                PREFIX + "lora_B.weight": np.zeros((4, 2))}
    aligned, count = align_adapter_state(expected, expected)
    assert set(aligned) == set(expected) and count == 0
    saved = {key.replace("base_model.model.", "base_model.model.model.", 1): value + 1
             for key, value in expected.items()}
    aligned, count = align_adapter_state(saved, expected)
    assert count == 2
    assert all(np.array_equal(value, expected[key] + 1) for key, value in aligned.items())


@pytest.mark.parametrize("corruption", ["missing", "unknown", "collision", "shape"])
def test_invalid_checkpoint_fails_before_encoding(corruption: str) -> None:
    expected = {PREFIX + "lora_A.weight": np.zeros((2, 4)),
                PREFIX + "lora_B.weight": np.zeros((4, 2))}
    saved = dict(expected)
    if corruption == "missing":
        saved.pop(PREFIX + "lora_B.weight")
    elif corruption == "unknown":
        saved["unexpected.weight"] = np.ones((2, 4))
    elif corruption == "collision":
        saved["base_model.model.model.layers.0.q_proj.lora_A.weight"] = np.ones((2, 4))
    else:
        saved[PREFIX + "lora_A.weight"] = np.ones((3, 4))
    with pytest.raises(ValueError):
        align_adapter_state(saved, expected)


def test_embedding_probe_rejects_no_effect_and_nonrepeatable_results() -> None:
    base = np.array([[1.0, 0.0]], dtype=np.float32)
    adapted = np.array([[0.99, 0.01]], dtype=np.float32)
    assert embedding_effect_audit(adapted, base, adapted)["effect_verified"]
    assert not embedding_effect_audit(base, base, base)["effect_verified"]
    assert not embedding_effect_audit(adapted, base, base)["effect_verified"]
    with pytest.raises(ValueError, match="non-finite"):
        embedding_effect_audit(adapted, base * np.nan, adapted)


def test_real_peft_checkpoint_values_and_toggle(tmp_path) -> None:
    """No downloads: real tiny torch/PEFT model, including the 0.15.2 runtime."""
    torch = pytest.importorskip("torch")
    from climate_rag.torch_compat import ensure_torch_pytree_compat
    ensure_torch_pytree_compat()
    peft = pytest.importorskip("peft")
    safetensors = pytest.importorskip("safetensors.torch")
    from peft.utils.save_and_load import get_peft_model_state_dict

    class Tiny(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.q_proj = torch.nn.Linear(4, 4, bias=False)

        def forward(self, x):
            return self.q_proj(x)

    torch.manual_seed(7)
    bare = Tiny()
    base_weights = {key: value.clone() for key, value in bare.state_dict().items()}
    source = peft.PeftModel(bare, peft.LoraConfig(r=2, target_modules=["q_proj"]))
    with torch.no_grad():
        for name, value in source.named_parameters():
            if "lora_A" in name:
                value.fill_(0.2)
            if "lora_B" in name:
                value.fill_(0.3)
    source.peft_config["default"].save_pretrained(tmp_path)
    state = get_peft_model_state_dict(source, save_embedding_layers=False)
    saved = {key.replace("base_model.model.", "base_model.model.model.", 1): value
             for key, value in state.items()}
    safetensors.save_file(saved, str(tmp_path / "adapter_model.safetensors"))
    target = Tiny()
    target.load_state_dict(base_weights)
    restored, audit = load_verified_lora(target, str(tmp_path))
    assert audit["all_checkpoint_values_match"]
    assert audit["remapped_tensor_count"] == 2
    assert audit["lora_parameter_count"] == 16
    assert not any(value.requires_grad for value in restored.parameters())
    x = torch.ones((2, 4))
    with torch.no_grad():
        enabled = restored(x).numpy()
        with restored.disable_adapter():
            disabled = restored(x).numpy()
        repeated = restored(x).numpy()
    assert embedding_effect_audit(enabled, disabled, repeated)["effect_verified"]
    for key, value in get_peft_model_state_dict(restored, save_embedding_layers=False).items():
        assert torch.equal(value, state[key])
