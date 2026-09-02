from __future__ import annotations

import hashlib
import random
import re
import time
import urllib.request
from collections import Counter, defaultdict
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .bm25 import BM25Index
from .io import read_json, read_jsonl, write_json, write_jsonl
from .metrics import evaluate_predictions
from .models import Claim, EvidenceDocument, Prediction

CLIMATE_FEVER_URL = (
    "https://raw.githubusercontent.com/tdiggelm/climate-fever-dataset/"
    "main/dataset/climate-fever.jsonl"
)
SPLIT_SEED = 20260825
SPLIT_RATIOS = {"train": 0.70, "validation": 0.15, "test": 0.15}


@dataclass(frozen=True, slots=True)
class ClimateFeverAnnotation:
    evidence_id: str
    evidence_label: str
    article: str
    text: str


@dataclass(frozen=True, slots=True)
class ClimateFeverRecord:
    claim_id: str
    claim: str
    label: str
    evidences: tuple[ClimateFeverAnnotation, ...]

    @property
    def decisive_evidence_ids(self) -> tuple[str, ...]:
        return tuple(
            item.evidence_id
            for item in self.evidences
            if item.evidence_label in {"SUPPORTS", "REFUTES"}
        )


def _sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def download_climate_fever(
    target: str | Path, *, url: str = CLIMATE_FEVER_URL, timeout_seconds: float = 60.0
) -> dict[str, Any]:
    destination = Path(target)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".download")
    request = urllib.request.Request(url, headers={"User-Agent": "climate-rag/0.1"})
    with (
        urllib.request.urlopen(request, timeout=timeout_seconds) as response,
        temporary.open("wb") as handle,
    ):
        while block := response.read(1024 * 1024):
            handle.write(block)
    temporary.replace(destination)
    return {
        "source_url": url,
        "downloaded_at_unix": time.time(),
        "bytes": destination.stat().st_size,
        "sha256": _sha256(destination),
    }


def load_climate_fever(path: str | Path) -> list[ClimateFeverRecord]:
    records: list[ClimateFeverRecord] = []
    seen: set[str] = set()
    for row in read_jsonl(path):
        claim_id = str(row["claim_id"])
        if claim_id in seen:
            raise ValueError(f"duplicate CLIMATE-FEVER claim id: {claim_id}")
        seen.add(claim_id)
        annotations = tuple(
            ClimateFeverAnnotation(
                evidence_id=str(item["evidence_id"]),
                evidence_label=str(item["evidence_label"]).upper(),
                article=str(item.get("article", "")),
                text=str(item["evidence"]),
            )
            for item in row.get("evidences", [])
        )
        if not annotations:
            raise ValueError(f"claim {claim_id} has no annotated evidence candidates")
        records.append(
            ClimateFeverRecord(
                claim_id=claim_id,
                claim=str(row["claim"]),
                label=str(row["claim_label"]).upper(),
                evidences=annotations,
            )
        )
    if not records:
        raise ValueError("CLIMATE-FEVER input is empty")
    return records


def build_public_corpus(
    records: Iterable[ClimateFeverRecord],
) -> list[EvidenceDocument]:
    by_id: dict[str, EvidenceDocument] = {}
    for record in records:
        for annotation in record.evidences:
            candidate = EvidenceDocument(
                evidence_id=annotation.evidence_id,
                text=annotation.text,
                metadata={
                    "article": annotation.article,
                    "source": "English Wikipedia via CLIMATE-FEVER",
                },
            )
            previous = by_id.get(candidate.evidence_id)
            if previous is not None and previous.text != candidate.text:
                raise ValueError(
                    f"evidence id has inconsistent text: {candidate.evidence_id}"
                )
            by_id[candidate.evidence_id] = candidate
    return [by_id[evidence_id] for evidence_id in sorted(by_id)]


def _normalise(text: str) -> str:
    return " ".join(re.findall(r"[a-z0-9]+", text.lower()))


def _token_set(text: str) -> frozenset[str]:
    return frozenset(_normalise(text).split())


class _UnionFind:
    def __init__(self, values: Iterable[str]) -> None:
        self.parent = {value: value for value in values}

    def find(self, value: str) -> str:
        parent = self.parent[value]
        if parent != value:
            self.parent[value] = self.find(parent)
        return self.parent[value]

    def union(self, left: str, right: str) -> None:
        left_root = self.find(left)
        right_root = self.find(right)
        if left_root != right_root:
            if left_root < right_root:
                self.parent[right_root] = left_root
            else:
                self.parent[left_root] = right_root


def grouped_split(
    records: Iterable[ClimateFeverRecord],
    *,
    seed: int = SPLIT_SEED,
    near_duplicate_threshold: float = 0.90,
) -> dict[str, list[str]]:
    rows = list(records)
    if not 0.0 < near_duplicate_threshold <= 1.0:
        raise ValueError("near_duplicate_threshold must be in (0, 1]")
    union = _UnionFind(record.claim_id for record in rows)

    claims_by_normalised: dict[str, list[str]] = defaultdict(list)
    claims_by_evidence: dict[str, list[str]] = defaultdict(list)
    for record in rows:
        claims_by_normalised[_normalise(record.claim)].append(record.claim_id)
        for item in record.evidences:
            claims_by_evidence[item.evidence_id].append(record.claim_id)
    for groups in (claims_by_normalised.values(), claims_by_evidence.values()):
        for claim_ids in groups:
            for claim_id in claim_ids[1:]:
                union.union(claim_ids[0], claim_id)

    # The public benchmark is small (1,535 claims), so an exact pairwise check is
    # preferable to an approximate near-duplicate index whose misses could leak.
    token_sets = {record.claim_id: _token_set(record.claim) for record in rows}
    ordered_ids = sorted(token_sets)
    for index, left_id in enumerate(ordered_ids):
        left = token_sets[left_id]
        if not left:
            continue
        for right_id in ordered_ids[index + 1 :]:
            right = token_sets[right_id]
            union_size = len(left | right)
            if (
                union_size
                and len(left & right) / union_size >= near_duplicate_threshold
            ):
                union.union(left_id, right_id)

    components: dict[str, list[str]] = defaultdict(list)
    for claim_id in ordered_ids:
        components[union.find(claim_id)].append(claim_id)

    rng = random.Random(seed)
    component_rows = list(components.values())
    rng.shuffle(component_rows)
    component_rows.sort(key=len, reverse=True)
    targets = {name: len(rows) * ratio for name, ratio in SPLIT_RATIOS.items()}
    split: dict[str, list[str]] = {name: [] for name in SPLIT_RATIOS}
    for component in component_rows:
        destination = min(
            split,
            key=lambda name: (
                (len(split[name]) + len(component) - targets[name])
                / max(targets[name], 1.0),
                len(split[name]) / max(targets[name], 1.0),
                name,
            ),
        )
        split[destination].extend(component)
    return {
        name: sorted(
            claim_ids, key=lambda value: int(value) if value.isdigit() else value
        )
        for name, claim_ids in split.items()
    }


def validate_split(
    records: Iterable[ClimateFeverRecord], split: dict[str, list[str]]
) -> None:
    rows = {record.claim_id: record for record in records}
    assigned = [claim_id for claim_ids in split.values() for claim_id in claim_ids]
    if len(assigned) != len(set(assigned)) or set(assigned) != set(rows):
        raise ValueError("split must assign every claim exactly once")
    owner = {
        claim_id: split_name
        for split_name, claim_ids in split.items()
        for claim_id in claim_ids
    }
    evidence_owners: dict[str, set[str]] = defaultdict(set)
    normalised_owners: dict[str, set[str]] = defaultdict(set)
    for claim_id, record in rows.items():
        normalised_owners[_normalise(record.claim)].add(owner[claim_id])
        for item in record.evidences:
            evidence_owners[item.evidence_id].add(owner[claim_id])
    if any(len(names) > 1 for names in evidence_owners.values()):
        raise ValueError("shared evidence leaked across splits")
    if any(len(names) > 1 for names in normalised_owners.values()):
        raise ValueError("normalised duplicate claim leaked across splits")


def audit_split_leakage(
    records: Iterable[ClimateFeverRecord],
    split: dict[str, list[str]],
    *,
    claim_similarity_threshold: float = 0.90,
    evidence_similarity_threshold: float = 0.90,
) -> dict[str, Any]:
    """Audit claim groups, shared evidence and document variants across partitions."""

    rows = list(records)
    validate_split(rows, split)
    owner = {
        claim_id: split_name
        for split_name, claim_ids in split.items()
        for claim_id in claim_ids
    }
    evidence_owners: dict[str, set[str]] = defaultdict(set)
    decisive_evidence_owners: dict[str, set[str]] = defaultdict(set)
    evidence_text: dict[str, str] = {}
    normalised_evidence_owners: dict[str, set[str]] = defaultdict(set)
    normalised_decisive_evidence_owners: dict[str, set[str]] = defaultdict(set)
    for record in rows:
        split_name = owner[record.claim_id]
        for item in record.evidences:
            evidence_owners[item.evidence_id].add(split_name)
            if item.evidence_label in {"SUPPORTS", "REFUTES"}:
                decisive_evidence_owners[item.evidence_id].add(split_name)
            existing = evidence_text.get(item.evidence_id)
            if existing is not None and existing != item.text:
                raise ValueError(f"evidence id has inconsistent text: {item.evidence_id}")
            evidence_text[item.evidence_id] = item.text
            normalised_evidence_owners[_normalise(item.text)].add(split_name)
            if item.evidence_label in {"SUPPORTS", "REFUTES"}:
                normalised_decisive_evidence_owners[_normalise(item.text)].add(split_name)

    claim_tokens = {record.claim_id: _token_set(record.claim) for record in rows}
    claim_owners = {claim_id: {split_name} for claim_id, split_name in owner.items()}
    evidence_tokens = {
        evidence_id: _token_set(text) for evidence_id, text in evidence_text.items()
    }
    near_claim_count, near_claim_examples = _cross_split_near_duplicates(
        claim_tokens,
        claim_owners,
        threshold=claim_similarity_threshold,
    )
    near_evidence_count, near_evidence_examples = _cross_split_near_duplicates(
        evidence_tokens,
        evidence_owners,
        threshold=evidence_similarity_threshold,
    )
    near_decisive_evidence_count, _ = _cross_split_near_duplicates(
        evidence_tokens,
        decisive_evidence_owners,
        threshold=evidence_similarity_threshold,
    )
    result: dict[str, Any] = {
        "shared_evidence_id_cross_split": sum(
            len(names) > 1 for names in evidence_owners.values()
        ),
        "normalised_claim_duplicate_cross_split": sum(
            len(names) > 1
            for names in _owners_by_normalised_claim(rows, owner).values()
        ),
        "near_duplicate_claim_pairs_cross_split": near_claim_count,
        "normalised_evidence_text_cross_split": sum(
            len(names) > 1 for names in normalised_evidence_owners.values()
        ),
        "near_duplicate_evidence_pairs_cross_split": near_evidence_count,
        "shared_decisive_evidence_id_cross_split": sum(
            len(names) > 1 for names in decisive_evidence_owners.values()
        ),
        "normalised_decisive_evidence_text_cross_split": sum(
            len(names) > 1 for names in normalised_decisive_evidence_owners.values()
        ),
        "near_duplicate_decisive_evidence_pairs_cross_split": near_decisive_evidence_count,
        "claim_similarity_threshold": claim_similarity_threshold,
        "evidence_similarity_threshold": evidence_similarity_threshold,
        "near_duplicate_claim_examples": near_claim_examples,
        "near_duplicate_evidence_examples": near_evidence_examples,
    }
    count_fields = (
        "shared_evidence_id_cross_split",
        "normalised_claim_duplicate_cross_split",
        "near_duplicate_claim_pairs_cross_split",
        "normalised_evidence_text_cross_split",
        "near_duplicate_evidence_pairs_cross_split",
    )
    result["status"] = (
        "passed" if all(int(result[name]) == 0 for name in count_fields) else "failed"
    )
    decisive_fields = (
        "shared_decisive_evidence_id_cross_split",
        "normalised_decisive_evidence_text_cross_split",
        "near_duplicate_decisive_evidence_pairs_cross_split",
    )
    result["supervised_relevance_status"] = (
        "passed"
        if all(int(result[name]) == 0 for name in decisive_fields)
        else "failed"
    )
    return result


def _owners_by_normalised_claim(
    records: Iterable[ClimateFeverRecord], owner: dict[str, str]
) -> dict[str, set[str]]:
    result: dict[str, set[str]] = defaultdict(set)
    for record in records:
        result[_normalise(record.claim)].add(owner[record.claim_id])
    return result


def _cross_split_near_duplicates(
    token_sets: dict[str, frozenset[str]],
    owners: dict[str, set[str]],
    *,
    threshold: float,
) -> tuple[int, list[dict[str, Any]]]:
    if not 0.0 < threshold <= 1.0:
        raise ValueError("near-duplicate threshold must be in (0, 1]")
    count = 0
    examples: list[dict[str, Any]] = []
    for left_id, right_id in _near_duplicate_pairs(token_sets, threshold=threshold):
        if len(owners[left_id] | owners[right_id]) <= 1:
            continue
        left = token_sets[left_id]
        right = token_sets[right_id]
        similarity = len(left & right) / len(left | right)
        count += 1
        if len(examples) < 10:
            examples.append(
                {
                    "left_id": left_id,
                    "right_id": right_id,
                    "similarity": similarity,
                    "left_splits": sorted(owners[left_id]),
                    "right_splits": sorted(owners[right_id]),
                }
            )
    return count, examples


def _near_duplicate_pairs(
    token_sets: dict[str, frozenset[str]], *, threshold: float
) -> Iterable[tuple[str, str]]:
    ordered_ids = sorted(token_sets)
    for index, left_id in enumerate(ordered_ids):
        left = token_sets[left_id]
        if not left:
            continue
        for right_id in ordered_ids[index + 1 :]:
            right = token_sets[right_id]
            if not right:
                continue
            smaller, larger = sorted((len(left), len(right)))
            if smaller / larger < threshold:
                continue
            union_size = len(left | right)
            if union_size and len(left & right) / union_size >= threshold:
                yield left_id, right_id


def prepare_public_benchmark(
    source: str | Path,
    output_dir: str | Path,
    *,
    seed: int = SPLIT_SEED,
    source_url: str = CLIMATE_FEVER_URL,
) -> dict[str, Any]:
    records = load_climate_fever(source)
    corpus = build_public_corpus(records)
    split = grouped_split(records, seed=seed)
    validate_split(records, split)
    leakage_audit = audit_split_leakage(records, split)
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)

    write_jsonl(
        output / "evidence.jsonl",
        (
            {
                "evidence_id": item.evidence_id,
                "text": item.text,
                "metadata": dict(item.metadata),
            }
            for item in corpus
        ),
    )
    write_json(
        output / "claims.json",
        {
            record.claim_id: {
                "claim_text": record.claim,
                "claim_label": record.label,
                "evidences": list(record.decisive_evidence_ids),
                "annotated_candidate_ids": [
                    item.evidence_id for item in record.evidences
                ],
            }
            for record in records
        },
    )
    label_counts = Counter(record.label for record in records)
    decisive_claims = sum(bool(record.decisive_evidence_ids) for record in records)
    manifest = {
        "schema_version": 1,
        "dataset": "CLIMATE-FEVER",
        "source_url": source_url,
        "source_sha256": _sha256(source),
        "seed": seed,
        "split_protocol": "claim-grouped-v1-with-posthoc-document-variant-audit",
        "ratios": SPLIT_RATIOS,
        "claim_count": len(records),
        "annotated_pair_count": sum(len(record.evidences) for record in records),
        "unique_evidence_count": len(corpus),
        "claims_with_decisive_evidence": decisive_claims,
        "label_counts": dict(sorted(label_counts.items())),
        "split": split,
        "candidate_selection_status": (
            "eligible" if leakage_audit["status"] == "passed" else "blocked"
        ),
        "leakage_checks": leakage_audit,
        "retrieval_relevance_definition": (
            "Only SUPPORTS/REFUTES evidence annotations are decisive retrieval positives; "
            "NOT_ENOUGH_INFO annotations remain searchable corpus candidates."
        ),
    }
    write_json(output / "split_manifest.json", manifest)
    return manifest


def benchmark_public_bm25(
    prepared_dir: str | Path,
    output_dir: str | Path,
    *,
    split_name: str = "test",
    top_k: int = 50,
) -> dict[str, Any]:
    prepared = Path(prepared_dir)
    manifest = read_json(prepared / "split_manifest.json")
    if split_name not in manifest["split"]:
        raise ValueError(f"unknown split: {split_name}")
    claims_payload = read_json(prepared / "claims.json")
    claim_ids = {str(value) for value in manifest["split"][split_name]}
    claims = {
        claim_id: Claim(
            claim_id=claim_id,
            text=str(row["claim_text"]),
            label=str(row["claim_label"]),
            evidence_ids=tuple(str(value) for value in row["evidences"]),
        )
        for claim_id, row in claims_payload.items()
        if claim_id in claim_ids and row["evidences"]
    }
    evidence = [
        EvidenceDocument(
            evidence_id=str(row["evidence_id"]),
            text=str(row["text"]),
            metadata=row.get("metadata", {}) or {},
        )
        for row in read_jsonl(prepared / "evidence.jsonl")
    ]
    build_started = time.perf_counter()
    index = BM25Index().fit(evidence)
    build_seconds = time.perf_counter() - build_started
    predictions: dict[str, Prediction] = {}
    latency_ms: list[float] = []
    for claim_id in sorted(claims):
        started = time.perf_counter()
        rows = index.search(claims[claim_id].text, top_k=top_k)
        latency_ms.append((time.perf_counter() - started) * 1000.0)
        predictions[claim_id] = Prediction(
            claim_id=claim_id,
            evidence_ids=tuple(row.evidence_id for row in rows),
        )
    raw_metrics, per_claim, errors = evaluate_predictions(claims, predictions)
    metrics: dict[str, Any] = dict(raw_metrics)
    latency_ms.sort()
    percentile = lambda q: latency_ms[
        min(len(latency_ms) - 1, int(q * len(latency_ms)))
    ]
    metrics.update(
        {
            "dataset": "CLIMATE-FEVER",
            "split": split_name,
            "source_sha256": manifest["source_sha256"],
            "split_manifest_sha256": _sha256(prepared / "split_manifest.json"),
            "corpus_document_count": len(evidence),
            "bm25_build_seconds": build_seconds,
            "search_p50_ms": percentile(0.50) if latency_ms else 0.0,
            "search_p95_ms": percentile(0.95) if latency_ms else 0.0,
            "top_k": top_k,
            "fact_boundary": (
                "Public external retrieval baseline only; no verdict model or cross-encoder is "
                "included in this run."
            ),
        }
    )
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    write_json(output / "metrics.json", metrics)
    write_jsonl(output / "per_claim.jsonl", per_claim)
    write_jsonl(output / "errors.jsonl", errors)
    return metrics
