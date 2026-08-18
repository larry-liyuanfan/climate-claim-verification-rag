from __future__ import annotations

import hashlib
import json
from collections.abc import Sequence
from pathlib import Path
from typing import Protocol

import numpy as np

from .io import write_json
from .models import EvidenceDocument, RankedDocument
from .tokenize import climate_tokenize
from .torch_compat import ensure_torch_pytree_compat


def l2_normalize(matrix: np.ndarray) -> np.ndarray:
    values = np.asarray(matrix, dtype=np.float32)
    if values.ndim == 1:
        values = values.reshape(1, -1)
    norms = np.linalg.norm(values, axis=1, keepdims=True)
    return values / np.maximum(norms, np.finfo(np.float32).eps)


class DenseEncoder(Protocol):
    dimension: int
    name: str

    def encode_queries(self, texts: Sequence[str], batch_size: int = 32) -> np.ndarray: ...

    def encode_documents(self, texts: Sequence[str], batch_size: int = 32) -> np.ndarray: ...


class HashDenseEncoder:
    """Dependency-free deterministic encoder for tests and offline fallbacks.

    It is a signed feature-hashing baseline, not a learned semantic embedding model.
    Production experiments should select ``SentenceTransformerEncoder`` explicitly.
    """

    def __init__(self, dimension: int = 256) -> None:
        if dimension <= 0:
            raise ValueError("dimension must be positive")
        self.dimension = dimension
        self.name = f"hash-baseline-{dimension}"

    def _encode(self, texts: Sequence[str]) -> np.ndarray:
        vectors = np.zeros((len(texts), self.dimension), dtype=np.float32)
        for row, text in enumerate(texts):
            tokens = climate_tokenize(text)
            for token in tokens:
                digest = hashlib.blake2b(token.encode("utf-8"), digest_size=16).digest()
                column = int.from_bytes(digest[:8], "little") % self.dimension
                sign = 1.0 if digest[8] & 1 else -1.0
                vectors[row, column] += sign
        return l2_normalize(vectors)

    def encode_queries(self, texts: Sequence[str], batch_size: int = 32) -> np.ndarray:
        del batch_size
        return self._encode(texts)

    def encode_documents(self, texts: Sequence[str], batch_size: int = 32) -> np.ndarray:
        del batch_size
        return self._encode(texts)


class SentenceTransformerEncoder:
    """Lazy adapter for Qwen3/BGE or another Sentence Transformers model."""

    def __init__(
        self,
        model_name: str = "Qwen/Qwen3-Embedding-0.6B",
        *,
        query_prefix: str = "",
        query_prompt_name: str | None = None,
        device: str | None = None,
    ) -> None:
        try:
            ensure_torch_pytree_compat()
            from sentence_transformers import SentenceTransformer
        except ImportError as exc:
            raise RuntimeError(
                "sentence-transformers is required for learned dense encoding; "
                "install the 'dense' extra"
            ) from exc
        self.name = model_name
        self.query_prefix = query_prefix
        self.query_prompt_name = query_prompt_name or (
            "query" if "Qwen3-Embedding" in model_name and not query_prefix else None
        )
        self._model = SentenceTransformer(model_name, device=device)
        self.dimension = int(self._model.get_sentence_embedding_dimension())

    def _encode(self, texts: Sequence[str], batch_size: int) -> np.ndarray:
        values = self._model.encode(
            list(texts),
            batch_size=batch_size,
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=False,
        )
        return np.asarray(values, dtype=np.float32)

    def encode_queries(self, texts: Sequence[str], batch_size: int = 32) -> np.ndarray:
        if self.query_prefix:
            return self._encode([self.query_prefix + text for text in texts], batch_size)
        if self.query_prompt_name:
            values = self._model.encode(
                list(texts),
                batch_size=batch_size,
                prompt_name=self.query_prompt_name,
                convert_to_numpy=True,
                normalize_embeddings=True,
                show_progress_bar=False,
            )
            return np.asarray(values, dtype=np.float32)
        return self._encode(texts, batch_size)

    def encode_documents(self, texts: Sequence[str], batch_size: int = 32) -> np.ndarray:
        return self._encode(texts, batch_size)


class NumpyFlatIndex:
    """Exact inner-product index used when FAISS is unavailable."""

    def __init__(self) -> None:
        self.vectors: np.ndarray | None = None

    def build(self, vectors: np.ndarray) -> None:
        self.vectors = l2_normalize(vectors)

    def search(self, queries: np.ndarray, top_k: int) -> tuple[np.ndarray, np.ndarray]:
        if self.vectors is None:
            raise RuntimeError("index has not been built")
        if top_k <= 0:
            rows = len(np.atleast_2d(queries))
            return np.empty((rows, 0), dtype=np.float32), np.empty((rows, 0), dtype=np.int64)
        normalized = l2_normalize(queries)
        scores = normalized @ self.vectors.T
        top_k = min(top_k, self.vectors.shape[0])
        indices = np.argsort(-scores, axis=1, kind="stable")[:, :top_k]
        selected = np.take_along_axis(scores, indices, axis=1)
        return selected.astype(np.float32), indices.astype(np.int64)

    def save(self, path: str | Path) -> None:
        if self.vectors is None:
            raise RuntimeError("index has not been built")
        np.save(path, self.vectors)

    @classmethod
    def load(cls, path: str | Path) -> "NumpyFlatIndex":
        index = cls()
        index.vectors = np.load(path, mmap_mode="r")
        return index


class FaissANNIndex:
    """FAISS adapter for exact FlatIP, HNSW and IVF-PQ indexes."""

    SUPPORTED_KINDS = frozenset({"flat", "hnsw", "ivfpq"})

    def __init__(
        self,
        dimension: int,
        *,
        kind: str = "flat",
        hnsw_m: int = 32,
        nlist: int = 256,
        pq_m: int = 32,
        nbits: int = 8,
        nprobe: int = 16,
    ) -> None:
        try:
            import faiss
        except ImportError as exc:
            raise RuntimeError("faiss is unavailable; install the 'dense' extra") from exc
        if kind not in self.SUPPORTED_KINDS:
            raise ValueError(f"unsupported FAISS kind: {kind}")
        if kind == "ivfpq" and dimension % pq_m != 0:
            raise ValueError("dimension must be divisible by pq_m for IVF-PQ")
        self.dimension = int(dimension)
        self.kind = kind
        self.parameters = {
            "hnsw_m": int(hnsw_m),
            "nlist": int(nlist),
            "pq_m": int(pq_m),
            "nbits": int(nbits),
            "nprobe": int(nprobe),
        }
        if kind == "flat":
            self.index = faiss.IndexFlatIP(dimension)
        elif kind == "hnsw":
            self.index = faiss.IndexHNSWFlat(dimension, hnsw_m, faiss.METRIC_INNER_PRODUCT)
        else:
            quantizer = faiss.IndexFlatIP(dimension)
            self.index = faiss.IndexIVFPQ(
                quantizer, dimension, nlist, pq_m, nbits, faiss.METRIC_INNER_PRODUCT
            )
            self.index.nprobe = nprobe

    def build(self, vectors: np.ndarray, training_vectors: np.ndarray | None = None) -> None:
        normalized = np.ascontiguousarray(l2_normalize(vectors), dtype=np.float32)
        if self.kind == "ivfpq" and not self.index.is_trained:
            training = normalized if training_vectors is None else np.ascontiguousarray(
                l2_normalize(training_vectors), dtype=np.float32
            )
            if len(training) < self.parameters["nlist"]:
                raise ValueError("IVF-PQ training rows must be at least nlist")
            self.index.train(training)
        self.index.add(normalized)

    def search(self, queries: np.ndarray, top_k: int) -> tuple[np.ndarray, np.ndarray]:
        if top_k <= 0:
            rows = len(np.atleast_2d(queries))
            return np.empty((rows, 0), dtype=np.float32), np.empty((rows, 0), dtype=np.int64)
        query_vectors = np.ascontiguousarray(l2_normalize(queries), dtype=np.float32)
        return self.index.search(query_vectors, top_k)

    def save(self, path: str | Path) -> None:
        import faiss

        faiss.write_index(self.index, str(path))

    @classmethod
    def load(cls, path: str | Path, metadata: dict[str, object]) -> "FaissANNIndex":
        import faiss

        instance = cls.__new__(cls)
        instance.dimension = int(metadata["dimension"])
        instance.kind = str(metadata["kind"])
        instance.parameters = dict(metadata.get("parameters", {}))
        instance.index = faiss.read_index(str(path))
        return instance


class DenseRetriever:
    """Encoder + exact/approximate vector index with stable document mapping."""

    def __init__(self, encoder: DenseEncoder, backend: NumpyFlatIndex | FaissANNIndex) -> None:
        self.encoder = encoder
        self.backend = backend
        self.doc_ids: list[str] = []
        self.texts: list[str] = []

    def fit(self, documents: Sequence[EvidenceDocument], batch_size: int = 32) -> "DenseRetriever":
        texts = [document.text for document in documents]
        vectors = self.encoder.encode_documents(texts, batch_size=batch_size)
        return self.fit_vectors(documents, vectors)

    def fit_vectors(
        self, documents: Sequence[EvidenceDocument], vectors: np.ndarray
    ) -> "DenseRetriever":
        self.doc_ids = [document.evidence_id for document in documents]
        if len(set(self.doc_ids)) != len(self.doc_ids):
            raise ValueError("duplicate evidence ids are not allowed")
        self.texts = [document.text for document in documents]
        if len(vectors) != len(documents):
            raise ValueError("embedding row count does not match document count")
        self.backend.build(vectors)
        return self

    def search(self, query: str, top_k: int = 10) -> list[RankedDocument]:
        query_vector = self.encoder.encode_queries([query])
        scores, indices = self.backend.search(query_vector, min(top_k, len(self.doc_ids)))
        rows: list[tuple[str, float, int]] = []
        for score, index in zip(scores[0], indices[0], strict=True):
            if index < 0:
                continue
            rows.append((self.doc_ids[int(index)], float(score), int(index)))
        rows.sort(key=lambda row: (-row[1], row[0]))
        return [
            RankedDocument(
                evidence_id=evidence_id,
                text=self.texts[index],
                score=score,
                rank=rank,
                source="dense",
            )
            for rank, (evidence_id, score, index) in enumerate(rows, start=1)
        ]

    def save(self, directory: str | Path) -> None:
        target = Path(directory)
        target.mkdir(parents=True, exist_ok=True)
        backend_type = "faiss" if isinstance(self.backend, FaissANNIndex) else "numpy"
        index_name = "index.faiss" if backend_type == "faiss" else "vectors.npy"
        self.backend.save(target / index_name)
        encoder_spec: dict[str, object] = {
            "type": "hash" if isinstance(self.encoder, HashDenseEncoder) else "sentence_transformer",
            "name": self.encoder.name,
            "dimension": self.encoder.dimension,
        }
        if isinstance(self.encoder, SentenceTransformerEncoder):
            encoder_spec["query_prefix"] = self.encoder.query_prefix
            encoder_spec["query_prompt_name"] = self.encoder.query_prompt_name
        backend_spec: dict[str, object] = {"type": backend_type}
        if isinstance(self.backend, FaissANNIndex):
            backend_spec.update(
                {
                    "kind": self.backend.kind,
                    "dimension": self.backend.dimension,
                    "parameters": self.backend.parameters,
                }
            )
        write_json(
            target / "dense_index.json",
            {
                "schema_version": 1,
                "encoder": encoder_spec,
                "backend": backend_spec,
                "index_file": index_name,
                "doc_ids": self.doc_ids,
                "texts": self.texts,
            },
        )

    @classmethod
    def load(cls, directory: str | Path, *, device: str | None = None) -> "DenseRetriever":
        target = Path(directory)
        spec = json.loads((target / "dense_index.json").read_text(encoding="utf-8"))
        encoder_spec = spec["encoder"]
        if encoder_spec["type"] == "hash":
            encoder: DenseEncoder = HashDenseEncoder(int(encoder_spec["dimension"]))
        else:
            encoder = SentenceTransformerEncoder(
                str(encoder_spec["name"]),
                query_prefix=str(encoder_spec.get("query_prefix", "")),
                query_prompt_name=encoder_spec.get("query_prompt_name"),
                device=device,
            )
        backend_spec = spec["backend"]
        index_path = target / spec["index_file"]
        if backend_spec["type"] == "faiss":
            backend: NumpyFlatIndex | FaissANNIndex = FaissANNIndex.load(index_path, backend_spec)
        else:
            backend = NumpyFlatIndex.load(index_path)
        retriever = cls(encoder, backend)
        retriever.doc_ids = [str(item) for item in spec["doc_ids"]]
        retriever.texts = [str(item) for item in spec["texts"]]
        return retriever
