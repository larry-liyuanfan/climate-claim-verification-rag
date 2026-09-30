"""Four distinct synthetic inputs for a future real preflight, not executed here."""
from __future__ import annotations

from typing import Any


def preflight_cases() -> list[tuple[dict[str, Any], dict[str, Any]]]:
    def doc(identifier: int, *sentences: str) -> dict[str, Any]:
        return {"document_id": identifier, "title": "Explicit synthetic fixture, not real scientific evidence",
                "sentences": [{"sentence_id": i, "text": s} for i, s in enumerate(sentences)]}
    return [
        ({"component": "screening", "claim": "Synthetic compound X improves endpoint Y.",
          "documents": [doc(1, "Synthetic compound X improves endpoint Y."),
                        doc(10, "This synthetic experiment only measures the colour of compound Z.")]},
         {"decision": "select", "document_ids": [1]}),
        ({"component": "relation", "claim": "Synthetic treatment A increases endpoint B.",
          "documents": [doc(2, "Synthetic treatment A decreases endpoint B, and does not increase it.")]},
         {"decision": "classify", "relation": "CONTRADICT"}),
        ({"component": "rationale", "claim": "Synthetic material C absorbs blue light.", "oracle_relation": "SUPPORT",
          "documents": [doc(3, "This synthetic study uses a box.", "Synthetic material C absorbs blue light.")]},
         {"decision": "select", "sentence_ids": [1]}),
        ({"component": "screening", "claim": "Synthetic intervention Q prevents disease R.",
          "documents": [doc(4, "This synthetic report describes mountain heights only and has no intervention or disease results.")]},
         {"decision": "abstain", "document_ids": []}),
    ]
