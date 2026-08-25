from __future__ import annotations

import threading
import time
import uuid
from collections import defaultdict
from typing import Any

try:
    from pydantic import BaseModel, Field
except ImportError:  # pragma: no cover - surfaced by create_app with a clearer message
    BaseModel = object  # type: ignore[assignment,misc]
    Field = None  # type: ignore[assignment]

from .fusion import reciprocal_rank_fusion
from .pipeline import HybridRetriever
from .verification import (
    AbstainingVerifier,
    VerificationProvider,
    VerificationResult,
    decompose_claim,
    evidence_coverage,
    extract_constraints,
    normalise_claim,
    validate_verification_result,
)


class RetrievalRequest(BaseModel):  # type: ignore[misc,valid-type]
    claim_text: str = Field(min_length=1, max_length=10_000) if Field else ""  # type: ignore[misc]
    top_k: int = Field(default=5, ge=1, le=100) if Field else 5  # type: ignore[misc]


class _TraceStore:
    def __init__(self, limit: int = 1_000) -> None:
        self.limit = limit
        self._lock = threading.Lock()
        self._rows: dict[str, dict[str, Any]] = {}

    def put(self, trace_id: str, value: dict[str, Any]) -> None:
        with self._lock:
            self._rows[trace_id] = value
            while len(self._rows) > self.limit:
                self._rows.pop(next(iter(self._rows)))

    def get(self, trace_id: str) -> dict[str, Any] | None:
        with self._lock:
            return self._rows.get(trace_id)


class _Metrics:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self.counters: dict[str, int] = defaultdict(int)
        self.latencies_ms: dict[str, list[float]] = defaultdict(list)

    def observe(self, route: str, latency_ms: float, *, status: str = "ok") -> None:
        with self._lock:
            self.counters[f"{route}:{status}"] += 1
            rows = self.latencies_ms[route]
            rows.append(latency_ms)
            if len(rows) > 10_000:
                del rows[: len(rows) - 10_000]

    def render(self) -> str:
        with self._lock:
            lines = ["# TYPE climate_rag_requests_total counter"]
            for key, value in sorted(self.counters.items()):
                route, status = key.split(":", 1)
                lines.append(
                    f'climate_rag_requests_total{{route="{route}",status="{status}"}} {value}'
                )
            lines.append("# TYPE climate_rag_latency_ms gauge")
            for route, values in sorted(self.latencies_ms.items()):
                ordered = sorted(values)
                for label, quantile in (("p50", 0.50), ("p95", 0.95)):
                    index = min(len(ordered) - 1, int(quantile * len(ordered)))
                    lines.append(
                        f'climate_rag_latency_ms{{route="{route}",quantile="{label}"}} '
                        f"{ordered[index]:.6f}"
                    )
            return "\n".join(lines) + "\n"


def create_app(
    retriever: HybridRetriever,
    *,
    verifier: VerificationProvider | None = None,
    default_top_k: int = 5,
    max_queries: int = 2,
    re_retrieval_coverage_threshold: float = 0.45,
    verification_coverage_threshold: float = 0.15,
    trace_limit: int = 1_000,
):
    try:
        from fastapi import FastAPI, HTTPException, Response
    except ImportError as exc:
        raise RuntimeError("FastAPI is unavailable; install the 'serve' extra") from exc

    if max_queries < 1:
        raise ValueError("max_queries must be positive")
    active_verifier = verifier or AbstainingVerifier()
    traces = _TraceStore(trace_limit)
    metrics = _Metrics()

    app = FastAPI(
        title="Climate Evidence Retrieval and Grounded Verification API",
        version="0.2.0",
        description=(
            "Multi-stage evidence retrieval with citation validation and fail-closed "
            "abstention when a verifier or sufficient evidence is unavailable."
        ),
    )

    def _search(
        request: RetrievalRequest, *, trace_id: str
    ) -> tuple[dict[str, Any], list[Any]]:
        started = time.perf_counter()
        claim = normalise_claim(request.claim_text)
        stage_rows: list[dict[str, Any]] = []
        first_started = time.perf_counter()
        rows = retriever.retrieve(
            claim,
            recall_k=max(100, request.top_k),
            rerank_k=max(50, request.top_k),
            final_k=max(request.top_k, 10),
        )
        stage_rows.append(
            {
                "stage": "initial_retrieval",
                "query": claim,
                "candidate_count": len(rows),
                "latency_ms": (time.perf_counter() - first_started) * 1000.0,
            }
        )
        coverage = evidence_coverage(claim, rows)
        queries = [claim]
        re_retrieval_triggered = False
        if coverage < re_retrieval_coverage_threshold and max_queries > 1:
            decomposed = decompose_claim(claim, max_queries=max_queries - 1)
            if decomposed:
                re_retrieval_triggered = True
                rankings = {"original": rows}
                for index, query in enumerate(decomposed, start=1):
                    query_started = time.perf_counter()
                    query_rows = retriever.retrieve(
                        query,
                        recall_k=max(100, request.top_k),
                        rerank_k=max(50, request.top_k),
                        final_k=max(request.top_k, 10),
                    )
                    rankings[f"decomposition_{index}"] = query_rows
                    queries.append(query)
                    stage_rows.append(
                        {
                            "stage": "bounded_re_retrieval",
                            "query": query,
                            "candidate_count": len(query_rows),
                            "latency_ms": (time.perf_counter() - query_started)
                            * 1000.0,
                        }
                    )
                rows = reciprocal_rank_fusion(
                    rankings, k=retriever.rrf_k, top_k=max(request.top_k, 10)
                )
                coverage = evidence_coverage(claim, rows)
        rows = rows[: request.top_k]
        elapsed_ms = (time.perf_counter() - started) * 1000.0
        payload = {
            "trace_id": trace_id,
            "claim_text": claim,
            "constraints": extract_constraints(claim),
            "queries": queries,
            "query_budget": max_queries,
            "re_retrieval_triggered": re_retrieval_triggered,
            "evidence_coverage": coverage,
            "evidence": [row.to_dict() for row in rows],
            "stages": stage_rows,
            "latency_ms": elapsed_ms,
        }
        return payload, rows

    @app.get("/health")
    def health() -> dict[str, object]:
        document_count = (
            len(retriever.bm25.doc_ids)
            if retriever.bm25 is not None
            else len(
                retriever.dense.doc_ids  # type: ignore[union-attr]
            )
        )
        return {
            "status": "ok",
            "document_count": document_count,
            "routes": {
                "bm25": retriever.bm25 is not None,
                "dense": retriever.dense is not None,
                "reranker": getattr(retriever.reranker, "name", None),
                "verifier": active_verifier.name,
            },
        }

    @app.post("/api/search")
    def search(request: RetrievalRequest) -> dict[str, Any]:
        trace_id = uuid.uuid4().hex
        try:
            payload, _ = _search(request, trace_id=trace_id)
        except (RuntimeError, ValueError) as exc:
            metrics.observe("search", 0.0, status="error")
            raise HTTPException(status_code=500, detail=str(exc)) from exc
        traces.put(trace_id, {"operation": "search", **payload})
        metrics.observe("search", float(payload["latency_ms"]))
        return payload

    @app.post("/api/verify")
    def verify(request: RetrievalRequest) -> dict[str, Any]:
        trace_id = uuid.uuid4().hex
        route_started = time.perf_counter()
        try:
            search_payload, rows = _search(request, trace_id=trace_id)
            if search_payload["evidence_coverage"] < verification_coverage_threshold:
                result = VerificationResult(
                    label="NOT_ENOUGH_INFO",
                    confidence=0.0,
                    abstain_reason="insufficient_evidence_coverage",
                    provider=active_verifier.name,
                )
                validation_warnings: list[str] = []
            else:
                result = active_verifier.verify(str(search_payload["claim_text"]), rows)
                result, validation_warnings = validate_verification_result(result, rows)
            status = "ok"
        except (RuntimeError, ValueError, KeyError) as exc:
            result = VerificationResult(
                label="NOT_ENOUGH_INFO",
                confidence=0.0,
                abstain_reason=f"verifier_failure:{type(exc).__name__}",
                provider=active_verifier.name,
            )
            validation_warnings = ["provider_failure"]
            search_payload = {
                "trace_id": trace_id,
                "claim_text": normalise_claim(request.claim_text),
                "evidence": [],
                "stages": [],
            }
            status = "recovered"
        elapsed_ms = (time.perf_counter() - route_started) * 1000.0
        payload = {
            **search_payload,
            "verification": result.to_dict(),
            "validation_warnings": validation_warnings,
            "total_latency_ms": elapsed_ms,
        }
        traces.put(trace_id, {"operation": "verify", **payload})
        metrics.observe("verify", elapsed_ms, status=status)
        return payload

    @app.get("/api/traces/{trace_id}")
    def get_trace(trace_id: str) -> dict[str, Any]:
        row = traces.get(trace_id)
        if row is None:
            raise HTTPException(status_code=404, detail="trace not found")
        return row

    @app.get("/metrics")
    def prometheus_metrics() -> Response:
        return Response(metrics.render(), media_type="text/plain; version=0.0.4")

    # Backwards-compatible endpoint. It retains the old explicit not-configured
    # classification contract instead of implying an unverified verdict.
    @app.post("/retrieve")
    def retrieve(request: RetrievalRequest) -> dict[str, object]:
        trace_id = uuid.uuid4().hex
        try:
            payload, _ = _search(request, trace_id=trace_id)
        except (RuntimeError, ValueError) as exc:
            raise HTTPException(status_code=500, detail=str(exc)) from exc
        return {
            "claim_text": payload["claim_text"],
            "evidence": payload["evidence"],
            "classification": {"status": "not_configured", "label": None},
        }

    return app
