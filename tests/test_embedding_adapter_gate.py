from __future__ import annotations

import sys
from types import SimpleNamespace

from climate_rag import dense
from climate_rag.embedding_adapter_gate import (
    heldout_query_texts,
    select_heldout_claims,
)
from climate_rag.models import Claim


def test_heldout_query_texts_and_claim_resolution() -> None:
    rows = [
        {
            "messages": [
                {"role": "system", "content": "retrieve"},
                {"role": "user", "content": "Claim A"},
            ]
        },
        {"messages": [{"role": "user", "content": "Claim B"}]},
    ]
    texts = heldout_query_texts(rows)
    selected = select_heldout_claims(
        {
            "a": Claim("a", "Claim A", evidence_ids=("ea",)),
            "b": Claim("b", "Claim B", evidence_ids=("eb",)),
            "c": Claim("c", "Claim C", evidence_ids=("ec",)),
        },
        texts,
    )
    assert set(selected) == {"a", "b"}


def test_sentence_transformer_encoder_loads_adapter(monkeypatch) -> None:
    calls: list[str] = []

    class FakeSentenceTransformer:
        def __init__(self, *args, **kwargs) -> None:
            del args, kwargs

        def load_adapter(self, path: str) -> None:
            calls.append(path)

        def get_sentence_embedding_dimension(self) -> int:
            return 1024

    monkeypatch.setattr(dense, "ensure_torch_pytree_compat", lambda: None)
    monkeypatch.setitem(
        sys.modules,
        "sentence_transformers",
        SimpleNamespace(SentenceTransformer=FakeSentenceTransformer),
    )
    encoder = dense.SentenceTransformerEncoder("base", adapter_path="adapter")
    assert encoder.adapter_path == "adapter"
    assert calls == ["adapter"]
