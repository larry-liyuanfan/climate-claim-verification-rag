from __future__ import annotations

from types import SimpleNamespace

import pytest

from climate_rag.rerank import _resolve_torch_dtype


def test_resolve_torch_dtype_accepts_explicit_precision_aliases() -> None:
    torch_stub = SimpleNamespace(bfloat16="BF16", float16="FP16", float32="FP32")

    assert _resolve_torch_dtype(torch_stub, "auto") == "auto"
    assert _resolve_torch_dtype(torch_stub, "bf16") == "BF16"
    assert _resolve_torch_dtype(torch_stub, "float16") == "FP16"
    assert _resolve_torch_dtype(torch_stub, "fp32") == "FP32"


def test_resolve_torch_dtype_rejects_unknown_precision() -> None:
    with pytest.raises(ValueError, match="dtype must be"):
        _resolve_torch_dtype(SimpleNamespace(), "int8")
