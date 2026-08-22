from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

from climate_rag import dense
from climate_rag.embedding_adapter_gate import (
    full_corpus_promotion_decision,
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
    calls: list[tuple[object, str, dict[str, str]]] = []

    class FakeParameter:
        def numel(self) -> int:
            return 7

    class FakeAutoModel:
        pass

    class FakePeftModel:
        def named_parameters(self):
            return [("layers.0.q_proj.lora_A.default.weight", FakeParameter())]

    class FakePeftFactory:
        @classmethod
        def from_pretrained(cls, model, path: str, *, key_mapping):
            calls.append((model, path, key_mapping))
            return FakePeftModel()

    class FakeTransformerModule:
        def __init__(self) -> None:
            self.auto_model = FakeAutoModel()

    class FakeSentenceTransformer:
        def __init__(self, *args, **kwargs) -> None:
            del args, kwargs
            self.module = FakeTransformerModule()

        def __getitem__(self, index: int):
            assert index == 0
            return self.module

        def get_sentence_embedding_dimension(self) -> int:
            return 1024

    monkeypatch.setattr(dense, "ensure_torch_pytree_compat", lambda: None)
    monkeypatch.setitem(
        sys.modules,
        "sentence_transformers",
        SimpleNamespace(SentenceTransformer=FakeSentenceTransformer),
    )
    monkeypatch.setitem(sys.modules, "peft", SimpleNamespace(PeftModel=FakePeftFactory))
    encoder = dense.SentenceTransformerEncoder("base", adapter_path="adapter")
    assert encoder.adapter_path == "adapter"
    assert encoder.adapter_parameter_count == 7
    assert len(calls) == 1
    assert calls[0][1:] == ("adapter", {r"^model\.": ""})


def test_full_corpus_promotion_requires_recall_ci_and_secondary_non_regression() -> None:
    comparisons = {
        metric: {"mean_difference": 0.01, "ci_lower": 0.001}
        for metric in ("recall@5", "mrr@10", "ndcg@10", "evidence_f1")
    }
    assert full_corpus_promotion_decision(comparisons)[
        "candidate_passes_full_corpus_gate"
    ]
    comparisons["evidence_f1"]["mean_difference"] = -0.0001
    decision = full_corpus_promotion_decision(comparisons)
    assert not decision["candidate_passes_full_corpus_gate"]
    assert not decision["secondary_mean_non_regression"]


def test_full_corpus_gate_does_not_persist_rebuildable_index_by_default() -> None:
    source = (
        Path(__file__).parents[1] / "scripts" / "evaluate_embedding_adapter_full_gate.py"
    ).read_text(encoding="utf-8")
    assert '"--save-index"' in source
    assert 'if args.save_index else None' in source
    assert "disabled by default" in source
