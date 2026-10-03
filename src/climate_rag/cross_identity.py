"""Label-blind claim/source identity audit, never an evaluation or split selector."""

from __future__ import annotations

import re
import unicodedata
from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any


def normalized_identity(text: str) -> str:
    return " ".join(unicodedata.normalize("NFKC", text).casefold().split())


def identity_tokens(text: str) -> frozenset[str]:
    return frozenset(re.findall(r"\w+", normalized_identity(text)))


@dataclass(frozen=True)
class IdentityClaim:
    track: str
    identifier: str
    text: str
    sources: tuple[str, ...] = ()
    documents: tuple[str, ...] = ()
    source_status: str = "unknown"

    @property
    def key(self) -> str:
        return self.track + ":" + self.identifier


class Components:
    def __init__(self, keys: Sequence[str]) -> None:
        self.parent = dict.fromkeys(keys, "")
        for key in keys:
            self.parent[key] = key

    def root(self, key: str) -> str:
        while self.parent[key] != key:
            self.parent[key] = self.parent[self.parent[key]]
            key = self.parent[key]
        return key

    def join(self, left: str, right: str) -> None:
        left, right = self.root(left), self.root(right)
        self.parent[max(left, right)] = min(left, right)


def text_edges(
    texts: Mapping[str, str], threshold: float
) -> list[tuple[str, str, str]]:
    """Exact NFKC/casefold and fixed token Jaccard; no semantic-equivalence claim."""
    if not 0 < threshold <= 1:
        raise ValueError("invalid identity threshold")
    normalized: dict[str, str] = {}
    seen: dict[str, frozenset[str]] = {}
    index: dict[str, list[str]] = defaultdict(list)
    edges = []
    for key, text in sorted(texts.items()):
        normalized[key] = normalized_identity(text)
        tokens = identity_tokens(text)
        if not tokens:
            raise ValueError("empty normalized identity")
        # Length ratio is a necessary upper bound on Jaccard, not an approximation.
        shared = Counter(
            other
            for token in sorted(tokens)
            for other in index[token]
            if min(len(tokens), len(seen[other]))
            >= threshold * max(len(tokens), len(seen[other]))
        )
        for other, overlap in sorted(shared.items()):
            exact = normalized[key] == normalized[other]
            if (
                exact
                or overlap / (len(tokens) + len(seen[other]) - overlap) >= threshold
            ):
                edges.append(
                    (other, key, "normalized_exact" if exact else "token_jaccard")
                )
        seen[key] = tokens
        for token in sorted(tokens):
            index[token].append(key)
    return edges


def audit_identity(
    claims: Sequence[IdentityClaim],
    documents: Mapping[str, str],
    *,
    target: str = "scifact_dev",
) -> tuple[dict[str, Any], dict[str, Any]]:
    by_key = {row.key: row for row in claims}
    if len(by_key) != len(claims) or not any(r.track == target for r in claims):
        raise ValueError("duplicate claim key or missing target")
    if any(not set(row.documents) <= set(documents) for row in claims):
        raise ValueError("referenced document text missing")
    claim_edges = text_edges({k: r.text for k, r in by_key.items()}, 0.8)
    document_edges = text_edges(documents, 0.9)
    doc_groups = Components(list(documents))
    for left, right, _ in document_edges:
        doc_groups.join(left, right)
    groups = Components(list(by_key))
    for left, right, _ in claim_edges:
        groups.join(left, right)
    owner: dict[str, str] = {}
    source_links = []
    for key, row in sorted(by_key.items()):
        identities = set(row.sources) | {
            "document-family:" + doc_groups.root(d) for d in row.documents
        }
        for identity in sorted(identities):
            if identity in owner:
                other = owner[identity]
                groups.join(other, key)
                source_links.append((other, key))
            else:
                owner[identity] = key
    members: dict[str, list[str]] = defaultdict(list)
    for key in by_key:
        members[groups.root(key)].append(key)
    connected: dict[str, set[str]] = {
        track: set() for track in sorted({r.track for r in claims} - {target})
    }
    affected = set()
    cross_components = []
    for keys in members.values():
        target_keys = [k for k in keys if by_key[k].track == target]
        other_keys = [k for k in keys if by_key[k].track != target]
        if target_keys and other_keys:
            affected.update(target_keys)
            for key in other_keys:
                connected[by_key[key].track].update(target_keys)
            cross_components.append(sorted(keys))
    direct: Counter[str] = Counter()
    for left, right, reason in claim_edges:
        tracks = (by_key[left].track, by_key[right].track)
        if target in tracks and tracks[0] != tracks[1]:
            other_track = tracks[1] if tracks[0] == target else tracks[0]
            direct[other_track + "/" + reason] += 1
    pair_counts = Counter(
        "/".join(sorted((a.split(":", 1)[0], b.split(":", 1)[0]))) + "/" + reason
        for a, b, reason in document_edges
        if a.split(":", 1)[0] != b.split(":", 1)[0]
    )
    summary = {
        "schema_version": "cross-dataset-identity-v1",
        "target": target,
        "claim_counts": dict(sorted(Counter(r.track for r in claims).items())),
        "document_counts": dict(
            sorted(Counter(k.split(":", 1)[0] for k in documents).items())
        ),
        "thresholds": {"claim_token_jaccard": 0.8, "document_token_jaccard": 0.9},
        "normalization": "NFKC/casefold/whitespace exact; Unicode word token sets for Jaccard",
        "direct_target_claim_pairs": dict(sorted(direct.items())),
        "cross_namespace_document_pairs": dict(sorted(pair_counts.items())),
        "claim_text_edges_all": len(claim_edges),
        "document_variant_edges_all": len(document_edges),
        "source_union_links_all": len(source_links),
        "connected_components": len(members),
        "largest_component": max(map(len, members.values())),
        "cross_target_components": len(cross_components),
        "target_claims_connected_to_consumed": len(affected),
        "target_claims_connected_by_consumed_track": {
            k: len(v) for k, v in connected.items()
        },
        "source_status_counts": {
            track: dict(Counter(r.source_status for r in claims if r.track == track))
            for track in sorted({r.track for r in claims})
        },
        "target_unchanged": True,
        "model_evaluation_performed": False,
        "independence_certified": False,
        "limitations": [
            "Lexical matches are identity candidates, not semantic-equivalence judgements; negation may create false positives.",
            "No labels, scores or model outputs select edges or thresholds; target rows are not removed or reselected.",
            "Namespace-qualified IDs never match solely because their bare numbers coincide.",
            "Passages and paper abstracts have different granularity; Jaccard cannot resolve missing bibliographic mappings.",
            "Missing source provenance and foundation-model pretraining exposure remain unknown.",
        ],
    }
    private = {
        "affected_target_ids": sorted(affected),
        "cross_target_components": cross_components,
        "claim_text_edges": claim_edges,
        "document_variant_edges": document_edges,
        "source_union_links": source_links,
        "claim_components": {key: groups.root(key) for key in sorted(by_key)},
    }
    return summary, private
