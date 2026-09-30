"""Original SciFact ingestion/audit; never an inference-time gold provider.

Only corpus/train/dev archive members are read. Original sentence order, labels,
and alternative rationale sets survive unchanged. BEIR retrieval-only imports
and their historical evaluation gates are intentionally not reused or edited.
"""

from __future__ import annotations

import hashlib
import json
import re
import tarfile
import unicodedata
from collections import Counter, defaultdict
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .io import write_json, write_jsonl
from .public_v2 import file_sha256

SOURCE_URL = "https://scifact.s3-us-west-2.amazonaws.com/release/latest/data.tar.gz"
RELEASE_SHA256 = "11c621288d41ac144d29b13b0f8503b3820b7d6e8b1f6ff24dff335c196d76be"
OFFICIAL_CODE_SHA = "68b98a56d93e0f9da0d2aab4e6c3294699a0f72e"
LABEL_TO_PROJECT = {"SUPPORT": "SUPPORTS", "CONTRADICT": "REFUTES"}
MEMBERS = ("data/corpus.jsonl", "data/claims_train.jsonl", "data/claims_dev.jsonl")


@dataclass(frozen=True)
class Abstract:
    doc_id: int
    title: str
    sentences: tuple[str, ...]
    structured: bool


@dataclass(frozen=True)
class Rationale:
    label: str
    sentences: tuple[int, ...]


@dataclass(frozen=True)
class GoldClaim:
    claim_id: int
    claim: str
    evidence: Mapping[int, tuple[Rationale, ...]]
    cited_doc_ids: tuple[int, ...]

    @property
    def label_scope(self) -> str:
        labels = {r.label for rats in self.evidence.values() for r in rats}
        if not labels:
            return "NOT_ENOUGH_INFO"
        return LABEL_TO_PROJECT[next(iter(labels))] if len(labels) == 1 else "MIXED"

    def inference_row(self) -> dict[str, Any]:
        # Do not expose cited_doc_ids, label_scope, grouping or evidence to a model.
        return {"id": self.claim_id, "claim": self.claim}

    def gold_row(self) -> dict[str, Any]:
        return {
            "id": self.claim_id,
            "claim": self.claim,
            "cited_doc_ids": list(self.cited_doc_ids),
            "evidence": {
                str(doc_id): [
                    {"label": r.label, "sentences": list(r.sentences)} for r in rats
                ]
                for doc_id, rats in self.evidence.items()
            },
        }


def _integer(value: Any) -> int:
    if type(value) is not int or value < 0:
        raise ValueError("expected nonnegative integer, not coerced text/bool")
    return value


def parse_abstract(row: Mapping[str, Any]) -> Abstract:
    sentences = row["abstract"]
    if (
        not isinstance(row["title"], str)
        or not isinstance(sentences, list)
        or not sentences
        or any(not isinstance(s, str) or not s.strip() for s in sentences)
        or type(row["structured"]) is not bool
    ):
        raise ValueError("invalid original SciFact abstract")
    return Abstract(
        _integer(row["doc_id"]), row["title"], tuple(sentences), row["structured"]
    )


def parse_gold(row: Mapping[str, Any], corpus: Mapping[int, Abstract]) -> GoldClaim:
    if not isinstance(row["claim"], str) or not row["claim"].strip():
        raise ValueError("empty claim")
    raw_evidence = row["evidence"]
    if not isinstance(raw_evidence, dict):
        raise ValueError("evidence must be a document mapping")
    evidence: dict[int, tuple[Rationale, ...]] = {}
    for key, raw_rationales in raw_evidence.items():
        if not isinstance(key, str) or not key.isdecimal() or str(int(key)) != key:
            raise ValueError("noncanonical document ID")
        doc_id = int(key)
        if doc_id not in corpus or not isinstance(raw_rationales, list):
            raise ValueError("unknown document or malformed rationale list")
        rationales = []
        for raw in raw_rationales:
            if raw["label"] not in LABEL_TO_PROJECT:
                raise ValueError("unsupported original document label")
            if not isinstance(raw["sentences"], list) or not raw["sentences"]:
                raise ValueError("empty rationale set")
            indices = tuple(_integer(i) for i in raw["sentences"])
            if len(set(indices)) != len(indices) or max(indices) >= len(
                corpus[doc_id].sentences
            ):
                raise ValueError("duplicate or out-of-range gold sentence index")
            rationales.append(Rationale(raw["label"], indices))
        if not rationales:
            raise ValueError("evidence document has no rationale sets")
        evidence[doc_id] = tuple(rationales)
    if not isinstance(row["cited_doc_ids"], list):
        raise ValueError("cited_doc_ids must be a list")
    cited = tuple(_integer(i) for i in row["cited_doc_ids"])
    if any(i not in corpus for i in cited):
        raise ValueError("unknown cited document")
    # The official dev release contains one repeated cited ID. It is metadata,
    # not an evidence vote: preserve the original list, use a set only to group.
    return GoldClaim(_integer(row["id"]), row["claim"], evidence, cited)


def read_original_archive(
    path: Path, *, expected_sha256: str = RELEASE_SHA256
) -> tuple[dict[int, Abstract], dict[str, list[GoldClaim]], dict[str, str]]:
    if file_sha256(path) != expected_sha256:
        raise ValueError("original archive SHA mismatch")
    raw: dict[str, bytes] = {}
    with tarfile.open(path, "r:gz") as archive:
        for name in MEMBERS:
            matches = [m for m in archive.getmembers() if m.name == name]
            if len(matches) != 1 or not matches[0].isfile():
                raise ValueError("missing/duplicate/nonregular allowed archive member")
            if matches[0].size > 64 * 1024 * 1024:
                raise ValueError("archive member exceeds preparation budget")
            handle = archive.extractfile(matches[0])
            if handle is None:
                raise ValueError("unreadable archive member")
            with handle:
                raw[name] = handle.read()
    corpus: dict[int, Abstract] = {}
    for line in raw[MEMBERS[0]].splitlines():
        doc = parse_abstract(json.loads(line))
        if doc.doc_id in corpus:
            raise ValueError("duplicate corpus ID")
        corpus[doc.doc_id] = doc
    splits: dict[str, list[GoldClaim]] = {}
    seen: set[int] = set()
    for name, member in zip(("train", "dev"), MEMBERS[1:]):
        rows = [
            parse_gold(json.loads(line), corpus) for line in raw[member].splitlines()
        ]
        for claim in rows:
            if claim.claim_id in seen:
                raise ValueError("duplicate claim ID within/across source splits")
            seen.add(claim.claim_id)
        splits[name] = rows
    return (
        corpus,
        splits,
        {name: hashlib.sha256(data).hexdigest() for name, data in raw.items()},
    )


def _tokens(text: str) -> frozenset[str]:
    return frozenset(re.findall(r"\w+", unicodedata.normalize("NFKC", text).casefold()))


class _Groups:
    def __init__(self, identifiers: Sequence[int]) -> None:
        self.parent = {i: i for i in identifiers}

    def root(self, i: int) -> int:
        while self.parent[i] != i:
            self.parent[i] = self.parent[self.parent[i]]
            i = self.parent[i]
        return i

    def join(self, a: int, b: int) -> None:
        ra, rb = self.root(a), self.root(b)
        self.parent[max(ra, rb)] = min(ra, rb)


def _near_groups(texts: Mapping[int, str], threshold: float) -> _Groups:
    groups = _Groups(list(texts))
    inverted: dict[str, list[int]] = defaultdict(list)
    seen: dict[int, frozenset[str]] = {}
    for identifier, text in sorted(texts.items()):
        tokens = _tokens(text)
        if not tokens:
            raise ValueError("empty normalized text cannot define an identity")
        shared = Counter(other for token in tokens for other in inverted[token])
        for other, intersection in shared.items():
            union = len(tokens) + len(seen[other]) - intersection
            if intersection / union >= threshold:
                groups.join(identifier, other)
        seen[identifier] = tokens
        for token in tokens:
            inverted[token].append(identifier)
    return groups


def audit_groups(
    corpus: Mapping[int, Abstract], splits: Mapping[str, Sequence[GoldClaim]]
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Conservative lexical/source-family audit, NOT a semantic leakage proof.

    Keep original dev intact. Quarantine train connected to dev before any new
    train-side modelling. If too few remain, fail the later release, never tune
    on dev. Cited documents participate only in identity grouping, not scoring.
    """
    rows = list(splits["train"]) + list(splits["dev"])
    groups = _near_groups({r.claim_id: r.claim for r in rows}, 0.8)
    documents = _near_groups(
        {i: d.title + " " + " ".join(d.sentences) for i, d in corpus.items()}, 0.9
    )
    source_owner: dict[int, int] = {}
    for row in rows:
        for doc_id in set(row.cited_doc_ids) | set(row.evidence):
            source = documents.root(doc_id)
            if source in source_owner:
                groups.join(row.claim_id, source_owner[source])
            source_owner[source] = row.claim_id
    dev_groups = {groups.root(r.claim_id) for r in splits["dev"]}
    quarantined = [
        r.claim_id for r in splits["train"] if groups.root(r.claim_id) in dev_groups
    ]
    retained = [
        r.claim_id for r in splits["train"] if groups.root(r.claim_id) not in dev_groups
    ]
    component_sizes = Counter(groups.root(r.claim_id) for r in rows)
    summary = {
        "schema_version": "scifact-original-identity-audit-v1",
        "corpus_documents": len(corpus),
        "source_split_counts": {s: len(v) for s, v in splits.items()},
        "label_scope_counts": {
            s: dict(sorted(Counter(r.label_scope for r in v).items()))
            for s, v in splits.items()
        },
        "claim_token_jaccard_threshold": 0.8,
        "document_token_jaccard_threshold": 0.9,
        "cited_ids_used_for_grouping_not_gold": True,
        "components": len(component_sizes),
        "largest_component_claims": max(component_sizes.values(), default=0),
        "train_quarantined_for_dev_overlap": len(quarantined),
        "eligible_train_count": len(retained),
        "dev_unchanged": True,
        "document_variant_components": len({documents.root(i) for i in corpus}),
        "mixed_label_claims_preserved": True,
        "duplicate_cited_id_rows": {
            s: sum(len(set(r.cited_doc_ids)) != len(r.cited_doc_ids) for r in v)
            for s, v in splits.items()
        },
        "model_evaluation_performed": False,
        "holdout_release_authorized": False,
        "limitation": "lexical/source grouping cannot certify absence of semantic paraphrase or foundation-model pretraining exposure",
    }
    private = {
        "claim_component": {str(r.claim_id): groups.root(r.claim_id) for r in rows},
        "eligible_train_ids": retained,
        "quarantined_train_ids": quarantined,
        "dev_ids": [r.claim_id for r in splits["dev"]],
    }
    return summary, private


def prepare_original_archive(path: Path, output: Path) -> dict[str, Any]:
    if output.exists():
        raise FileExistsError("preparation never overwrites an existing release")
    corpus, splits, member_hashes = read_original_archive(path)
    summary, private = audit_groups(corpus, splits)
    output.mkdir(parents=True, mode=0o700)
    write_jsonl(
        output / "inference/corpus.jsonl",
        [
            {
                "doc_id": d.doc_id,
                "title": d.title,
                "abstract": list(d.sentences),
                "structured": d.structured,
            }
            for d in corpus.values()
        ],
    )
    eligible = set(private["eligible_train_ids"])
    for split, claims in splits.items():
        kept = [r for r in claims if split == "dev" or r.claim_id in eligible]
        name = "dev" if split == "dev" else "train_eligible"
        write_jsonl(
            output / f"inference/claims_{name}.jsonl", [r.inference_row() for r in kept]
        )
        write_jsonl(
            output / f"gold/claims_{split}.jsonl", [r.gold_row() for r in claims]
        )
    write_json(output / "private-group-assignment.json", private)
    summary.update(
        {
            "source_url": SOURCE_URL,
            "archive_sha256": file_sha256(path),
            "member_sha256": member_hashes,
            "official_reference_sha": OFFICIAL_CODE_SHA,
            "read_members": list(MEMBERS),
            "unlabelled_test_member_read": False,
            "inference_claim_fields": ["id", "claim"],
        }
    )
    summary["output_file_sha256"] = {
        p.relative_to(output).as_posix(): file_sha256(p)
        for p in sorted(output.rglob("*"))
        if p.is_file()
    }
    write_json(output / "preparation-manifest.json", summary)
    return summary


def assemble_context(
    ranked: Sequence[Abstract],
    *,
    count_tokens: Callable[[str], int],
    max_tokens: int,
    max_docs: int,
) -> dict[str, Any]:
    """Prefix-only complete-sentence assembly; receives no claim/gold labels.

    The production caller must pass the frozen generator/reranker tokenizer,
    not a whitespace estimate. IDs/index order survive; titles are not evidence.
    """
    if max_tokens <= 0 or max_docs <= 0:
        raise ValueError("context budgets must be positive")
    if len({d.doc_id for d in ranked}) != len(ranked):
        raise ValueError("duplicate ranked document")
    text = ""
    included: list[dict[str, Any]] = []
    overflow = False
    for doc in ranked[:max_docs]:
        indices: list[int] = []
        header = f"\nDocument {doc.doc_id}: {doc.title}\n"
        for index, sentence in enumerate(doc.sentences):
            addition = (
                header if not indices else ""
            ) + f"[{doc.doc_id}:{index}] {sentence}\n"
            if count_tokens(text + addition) > max_tokens:
                overflow = True
                break
            text += addition
            indices.append(index)
        if indices:
            included.append({"doc_id": doc.doc_id, "sentences": indices})
        if overflow:
            break
    return {
        "text": text,
        "tokens": count_tokens(text),
        "included": included,
        "overflow": overflow,
        "input_documents": len(ranked),
        "omitted_documents": len(ranked) - len(included),
        "omitted_sentences": sum(len(d.sentences) for d in ranked)
        - sum(len(r["sentences"]) for r in included),
        "policy": "ranked-document-prefix/original-sentence-prefix-v1",
    }
