"""Source-bound retrieval output for callers; not an answer or entailment model."""

from __future__ import annotations

import hashlib
import re
from collections.abc import Mapping
from typing import Any
from urllib.parse import quote

from .models import EvidenceDocument


def build_evidence_packet(
    search: Mapping[str, Any],
    documents: Mapping[str, EvidenceDocument],
    *,
    corpus_sha256: str,
    corpus_reference: str,
    evidence_track: str,
) -> dict[str, Any]:
    """Bind API rows to the caller's trusted, versioned corpus lookup.

    The caller computes the digest of the corpus it loaded. This function checks
    IDs/text against that lookup, not the truth of the text or URL contents.
    Retriever ``source`` means retrieval route, never publisher provenance.
    """
    if not re.fullmatch(r"[0-9a-f]{64}", corpus_sha256):
        raise ValueError("corpus_sha256 must be a lowercase SHA-256 digest")
    if not corpus_reference or not evidence_track:
        raise ValueError("corpus reference and evidence track are required")
    items = []
    seen: set[str] = set()
    for row in search["evidence"]:
        evidence_id = row["evidence_id"]
        document = documents.get(evidence_id)
        if document is None or evidence_id in seen:
            raise ValueError("unknown or duplicate evidence ID")
        if row["text"] != document.text:
            raise ValueError("retrieved text does not match the source document")
        seen.add(evidence_id)
        source_url = document.metadata.get("url")
        if source_url is not None and (
            not isinstance(source_url, str)
            or not source_url.startswith(("https://", "http://"))
        ):
            raise ValueError("source URL must be http(s), or absent")
        items.append(
            {
                "citation_id": f"sha256:{corpus_sha256}:{quote(evidence_id, safe='')}",
                "evidence_id": evidence_id,
                "text": document.text,
                "text_sha256": hashlib.sha256(document.text.encode()).hexdigest(),
                "span": {"start": 0, "end": len(document.text), "unit": "codepoint"},
                "source": {
                    "corpus_reference": corpus_reference,
                    "corpus_sha256": corpus_sha256,
                    "record_id": evidence_id,
                    "url": source_url,
                    "url_status": "provided_not_fetched"
                    if source_url
                    else "not_provided",
                    "article": document.metadata.get("article"),
                    "publisher": document.metadata.get("source"),
                },
                "retrieval": {
                    "route": row["source"],
                    "rank": row["rank"],
                    "score": row["score"],
                },
            }
        )
    return {
        "schema_version": "1.0",
        "status": "candidates_only" if items else "no_evidence",
        "evidence_track": evidence_track,
        "trace_id": search["trace_id"],
        "claim_text": search["claim_text"],
        "items": items,
        "query_processing": {
            "method": "deterministic_regex_and_lexical_coverage",
            "queries": search["queries"],
            "constraints": search["constraints"],
            "query_budget": search["query_budget"],
            "re_retrieval_triggered": search["re_retrieval_triggered"],
        },
        "answer": None,
        "verdict": {"status": "not_evaluated", "label": None},
        "warnings": [
            "Retrieved candidates are not a grounded answer or verified claim.",
            "Source identity and text integrity do not establish entailment.",
            "Lexical coverage is not evidence sufficiency or calibrated confidence.",
        ],
    }
