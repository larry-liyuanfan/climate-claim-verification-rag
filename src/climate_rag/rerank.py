from __future__ import annotations

import json
import os
import urllib.error
import urllib.request
from collections.abc import Sequence
from typing import Protocol

from .fusion import build_candidate_features
from .models import RankedDocument
from .torch_compat import ensure_torch_pytree_compat


class Reranker(Protocol):
    name: str

    def rerank(
        self, query: str, candidates: Sequence[RankedDocument], top_k: int
    ) -> list[RankedDocument]: ...


def weighted_rank_fuse(
    base: Sequence[RankedDocument],
    reranked: Sequence[RankedDocument],
    top_k: int,
    *,
    k: int = 60,
    base_weight: float = 1.0,
    reranker_weight: float = 1.0,
) -> list[RankedDocument]:
    """Fuse first-stage and cross-encoder ranks without mixing score scales."""

    if top_k <= 0:
        return []
    if k < 0:
        raise ValueError("rank-fusion k must be non-negative")
    if base_weight < 0 or reranker_weight < 0:
        raise ValueError("rank-fusion weights must be non-negative")
    if base_weight == 0 and reranker_weight == 0:
        raise ValueError("at least one rank-fusion weight must be positive")

    base_by_id = {row.evidence_id: row for row in base}
    reranked_by_id = {row.evidence_id: row for row in reranked}
    evidence_ids = sorted(base_by_id.keys() | reranked_by_id.keys())
    rows: list[tuple[float, RankedDocument]] = []
    for evidence_id in evidence_ids:
        base_row = base_by_id.get(evidence_id)
        reranked_row = reranked_by_id.get(evidence_id)
        score = 0.0
        if base_row is not None:
            score += base_weight / (k + base_row.rank)
        if reranked_row is not None:
            score += reranker_weight / (k + reranked_row.rank)
        source_row = reranked_row or base_row
        assert source_row is not None
        rows.append((score, source_row))
    rows.sort(key=lambda item: (-item[0], item[1].evidence_id))
    return [
        RankedDocument(
            evidence_id=row.evidence_id,
            text=row.text,
            score=float(score),
            rank=rank,
            source="weighted-rank-fusion",
            features={
                "base_rank": float(base_by_id[row.evidence_id].rank)
                if row.evidence_id in base_by_id
                else 0.0,
                "reranker_rank": float(reranked_by_id[row.evidence_id].rank)
                if row.evidence_id in reranked_by_id
                else 0.0,
            },
        )
        for rank, (score, row) in enumerate(rows[:top_k], start=1)
    ]


class DeterministicFeatureReranker:
    """Local fallback based on auditable lexical/numeric overlap features."""

    name = "deterministic-feature-fallback"

    def rerank(
        self, query: str, candidates: Sequence[RankedDocument], top_k: int
    ) -> list[RankedDocument]:
        rows: list[tuple[float, RankedDocument]] = []
        for row in candidates:
            features = build_candidate_features(query, row.text)
            score = (
                0.65 * features["token_overlap"]
                + 0.20 * features["number_overlap"]
                + 0.15 * features["year_overlap"]
            )
            rows.append((score, row))
        rows.sort(key=lambda item: (-item[0], item[1].evidence_id))
        return [
            RankedDocument(
                evidence_id=row.evidence_id,
                text=row.text,
                score=float(score),
                rank=rank,
                source=self.name,
                features={"input_score": row.score},
            )
            for rank, (score, row) in enumerate(rows[: max(top_k, 0)], start=1)
        ]


class Qwen3CausalLMReranker:
    """Qwen3 reranker adapter following the official yes/no logit recipe."""

    def __init__(
        self,
        model_name: str = "Qwen/Qwen3-Reranker-0.6B",
        device: str | None = None,
        *,
        max_length: int = 8192,
        batch_size: int = 8,
        instruction: str = "Given a climate claim, retrieve evidence that helps verify the claim",
    ) -> None:
        try:
            import torch
            ensure_torch_pytree_compat()
            from transformers import AutoModelForCausalLM, AutoTokenizer
        except ImportError as exc:
            raise RuntimeError("torch and transformers>=4.51 are required for Qwen3 reranking") from exc
        self.name = model_name
        self._torch = torch
        self.max_length = max_length
        self.batch_size = batch_size
        self.instruction = instruction
        self._tokenizer = AutoTokenizer.from_pretrained(model_name, padding_side="left")
        self._model = AutoModelForCausalLM.from_pretrained(model_name).eval()
        if device:
            self._model.to(device)
        self._false_token_id = self._tokenizer.convert_tokens_to_ids("no")
        self._true_token_id = self._tokenizer.convert_tokens_to_ids("yes")
        prefix = (
            '<|im_start|>system\nJudge whether the Document meets the requirements based on '
            'the Query and the Instruct provided. Note that the answer can only be "yes" or '
            '"no".<|im_end|>\n<|im_start|>user\n'
        )
        suffix = "<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"
        self._prefix_tokens = self._tokenizer.encode(prefix, add_special_tokens=False)
        self._suffix_tokens = self._tokenizer.encode(suffix, add_special_tokens=False)

    def _score(self, query: str, documents: Sequence[str]) -> list[float]:
        values: list[float] = []
        content_length = self.max_length - len(self._prefix_tokens) - len(self._suffix_tokens)
        if content_length <= 0:
            raise ValueError("max_length is too small for the Qwen3 reranker template")
        for start in range(0, len(documents), self.batch_size):
            pairs = [
                f"<Instruct>: {self.instruction}\n<Query>: {query}\n<Document>: {document}"
                for document in documents[start : start + self.batch_size]
            ]
            encoded = self._tokenizer(
                pairs,
                padding=False,
                truncation="longest_first",
                return_attention_mask=False,
                max_length=content_length,
            )
            encoded["input_ids"] = [
                self._prefix_tokens + row + self._suffix_tokens for row in encoded["input_ids"]
            ]
            batch = self._tokenizer.pad(
                encoded, padding=True, return_tensors="pt", max_length=self.max_length
            )
            batch = {key: tensor.to(self._model.device) for key, tensor in batch.items()}
            with self._torch.inference_mode():
                logits = self._model(**batch).logits[:, -1, :]
                binary = self._torch.stack(
                    [logits[:, self._false_token_id], logits[:, self._true_token_id]], dim=1
                )
                values.extend(self._torch.softmax(binary, dim=1)[:, 1].detach().cpu().tolist())
        return [float(value) for value in values]

    def rerank(
        self, query: str, candidates: Sequence[RankedDocument], top_k: int
    ) -> list[RankedDocument]:
        if not candidates or top_k <= 0:
            return []
        scores = self._score(query, [row.text for row in candidates])
        paired = sorted(
            zip((float(value) for value in scores), candidates, strict=True),
            key=lambda item: (-item[0], item[1].evidence_id),
        )
        return [
            RankedDocument(row.evidence_id, score, rank, row.text, self.name)
            for rank, (score, row) in enumerate(paired[:top_k], start=1)
        ]


class ModelStudioReranker:
    """Alibaba Cloud Model Studio text rerank API adapter.

    API credentials are read only from ``DASHSCOPE_API_KEY``. They are never
    accepted as CLI arguments or written into run artifacts.
    """

    def __init__(
        self,
        model: str = "qwen3-rerank",
        *,
        endpoint: str = "https://dashscope-intl.aliyuncs.com/api/v1/services/rerank/text-rerank/text-rerank",
        timeout_seconds: float = 30.0,
    ) -> None:
        self.name = f"model-studio:{model}"
        self.model = model
        self.endpoint = endpoint
        self.timeout_seconds = timeout_seconds

    def rerank(
        self, query: str, candidates: Sequence[RankedDocument], top_k: int
    ) -> list[RankedDocument]:
        api_key = os.getenv("DASHSCOPE_API_KEY")
        if not api_key:
            raise RuntimeError("DASHSCOPE_API_KEY is required for Model Studio reranking")
        if not candidates or top_k <= 0:
            return []
        body = json.dumps(
            {
                "model": self.model,
                "input": {"query": query, "documents": [row.text for row in candidates]},
                "parameters": {"top_n": min(top_k, len(candidates)), "return_documents": False},
            }
        ).encode("utf-8")
        request = urllib.request.Request(
            self.endpoint,
            data=body,
            headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=self.timeout_seconds) as response:
                payload = json.load(response)
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")
            raise RuntimeError(f"Model Studio rerank failed ({exc.code}): {detail[:500]}") from exc
        results = payload.get("output", {}).get("results", [])
        ranked: list[RankedDocument] = []
        for rank, result in enumerate(results, start=1):
            index = int(result["index"])
            source = candidates[index]
            ranked.append(
                RankedDocument(
                    evidence_id=source.evidence_id,
                    text=source.text,
                    score=float(result["relevance_score"]),
                    rank=rank,
                    source=self.name,
                )
            )
        return ranked
