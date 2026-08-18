from __future__ import annotations

import re


DEFAULT_STOPWORDS = frozenset(
    {
        "a",
        "an",
        "and",
        "are",
        "as",
        "at",
        "be",
        "been",
        "by",
        "for",
        "from",
        "has",
        "have",
        "in",
        "is",
        "it",
        "of",
        "on",
        "or",
        "that",
        "the",
        "this",
        "to",
        "was",
        "were",
        "will",
        "with",
    }
)
NEGATIONS = frozenset({"not", "no", "nor", "without", "against"})
_TOKEN = re.compile(r"\b[a-zA-Z0-9_]+\b")


def climate_tokenize(text: str, stopwords: frozenset[str] = DEFAULT_STOPWORDS) -> list[str]:
    """Tokenize like the course baseline while preserving fact-changing negations."""
    normalized = text.lower()
    normalized = re.sub(r"\bcarbon\s+dioxide\b", "carbon_dioxide", normalized)
    normalized = re.sub(r"\bco\s*2\b", "carbon_dioxide", normalized)
    blocked = stopwords - NEGATIONS
    return [token for token in _TOKEN.findall(normalized) if len(token) > 1 and token not in blocked]

