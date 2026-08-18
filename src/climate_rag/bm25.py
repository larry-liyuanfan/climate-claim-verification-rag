from __future__ import annotations

import gzip
import math
import pickle
from collections import Counter, defaultdict
from collections.abc import Iterable
from pathlib import Path

from .models import EvidenceDocument, RankedDocument
from .tokenize import climate_tokenize


class BM25Index:
    """Deterministic inverted-index BM25 implementation.

    The persisted pickle is an internal, trusted artifact. Never load an index from
    an untrusted source because pickle is code-executable by design.
    """

    FORMAT_VERSION = 1

    def __init__(self, *, k1: float = 1.5, b: float = 0.75) -> None:
        if k1 <= 0:
            raise ValueError("k1 must be positive")
        if not 0 <= b <= 1:
            raise ValueError("b must be between 0 and 1")
        self.k1 = float(k1)
        self.b = float(b)
        self.doc_ids: list[str] = []
        self.texts: list[str] = []
        self.doc_lengths: list[int] = []
        self.postings: dict[str, list[tuple[int, int]]] = {}
        self.idf: dict[str, float] = {}
        self.avg_doc_length = 0.0

    def fit(self, documents: Iterable[EvidenceDocument]) -> "BM25Index":
        postings: dict[str, list[tuple[int, int]]] = defaultdict(list)
        seen: set[str] = set()
        for doc_index, document in enumerate(documents):
            if document.evidence_id in seen:
                raise ValueError(f"duplicate evidence id: {document.evidence_id}")
            seen.add(document.evidence_id)
            tokens = climate_tokenize(document.text)
            self.doc_ids.append(document.evidence_id)
            self.texts.append(document.text)
            self.doc_lengths.append(len(tokens))
            for token, frequency in Counter(tokens).items():
                postings[token].append((doc_index, frequency))
        if not self.doc_ids:
            raise ValueError("cannot build a BM25 index with no documents")
        count = len(self.doc_ids)
        self.avg_doc_length = sum(self.doc_lengths) / count
        self.postings = dict(postings)
        self.idf = {
            token: math.log(1.0 + (count - len(rows) + 0.5) / (len(rows) + 0.5))
            for token, rows in self.postings.items()
        }
        return self

    def search(self, query: str, top_k: int = 10) -> list[RankedDocument]:
        if top_k <= 0:
            return []
        if not self.doc_ids:
            raise RuntimeError("BM25 index has not been fitted")
        scores: dict[int, float] = defaultdict(float)
        for token in set(climate_tokenize(query)):
            token_idf = self.idf.get(token)
            if token_idf is None:
                continue
            for doc_index, frequency in self.postings[token]:
                length_norm = 1.0 - self.b + self.b * self.doc_lengths[doc_index] / max(
                    self.avg_doc_length, 1e-12
                )
                denominator = frequency + self.k1 * length_norm
                scores[doc_index] += token_idf * frequency * (self.k1 + 1.0) / denominator
        ordered = sorted(scores, key=lambda index: (-scores[index], self.doc_ids[index]))[:top_k]
        return [
            RankedDocument(
                evidence_id=self.doc_ids[index],
                text=self.texts[index],
                score=float(scores[index]),
                rank=rank,
                source="bm25",
            )
            for rank, index in enumerate(ordered, start=1)
        ]

    def save(self, path: str | Path) -> None:
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "format_version": self.FORMAT_VERSION,
            "k1": self.k1,
            "b": self.b,
            "doc_ids": self.doc_ids,
            "texts": self.texts,
            "doc_lengths": self.doc_lengths,
            "postings": self.postings,
            "idf": self.idf,
            "avg_doc_length": self.avg_doc_length,
        }
        with gzip.open(target, "wb", compresslevel=3) as handle:
            pickle.dump(payload, handle, protocol=pickle.HIGHEST_PROTOCOL)

    @classmethod
    def load(cls, path: str | Path) -> "BM25Index":
        with gzip.open(path, "rb") as handle:
            payload = pickle.load(handle)  # noqa: S301 - documented trusted artifact
        if payload.get("format_version") != cls.FORMAT_VERSION:
            raise ValueError("unsupported BM25 index format")
        index = cls(k1=payload["k1"], b=payload["b"])
        index.doc_ids = payload["doc_ids"]
        index.texts = payload["texts"]
        index.doc_lengths = payload["doc_lengths"]
        index.postings = payload["postings"]
        index.idf = payload["idf"]
        index.avg_doc_length = payload["avg_doc_length"]
        return index

