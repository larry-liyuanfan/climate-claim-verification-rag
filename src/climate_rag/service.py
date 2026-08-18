from .pipeline import HybridRetriever


def create_app(retriever: HybridRetriever, *, default_top_k: int = 5):
    try:
        from fastapi import FastAPI, HTTPException
        from pydantic import BaseModel, Field
    except ImportError as exc:
        raise RuntimeError("FastAPI is unavailable; install the 'serve' extra") from exc

    class RetrievalRequest(BaseModel):
        claim_text: str = Field(min_length=1, max_length=10_000)
        top_k: int = Field(default=default_top_k, ge=1, le=100)

    app = FastAPI(
        title="Climate Claim Retrieval API",
        version="0.1.0",
        description=(
            "Evidence retrieval service. Classification is intentionally not invented when "
            "a trained classifier artifact is not configured."
        ),
    )

    @app.get("/health")
    def health() -> dict[str, object]:
        document_count = len(retriever.bm25.doc_ids) if retriever.bm25 is not None else len(
            retriever.dense.doc_ids  # type: ignore[union-attr]
        )
        return {"status": "ok", "document_count": document_count}

    @app.post("/retrieve")
    def retrieve(request: RetrievalRequest) -> dict[str, object]:
        try:
            rows = retriever.retrieve(
                request.claim_text,
                recall_k=max(100, request.top_k),
                rerank_k=max(50, request.top_k),
                final_k=request.top_k,
            )
        except (RuntimeError, ValueError) as exc:
            raise HTTPException(status_code=500, detail=str(exc)) from exc
        return {
            "claim_text": request.claim_text,
            "evidence": [row.to_dict() for row in rows],
            "classification": {"status": "not_configured", "label": None},
        }

    return app
