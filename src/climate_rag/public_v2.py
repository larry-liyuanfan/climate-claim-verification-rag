from __future__ import annotations

import hashlib
import json
import os
import random
import shutil
import ssl
import urllib.request
import zipfile
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any

from .evaluation_protocol import stable_id_sha256
from .fusion import DEFAULT_FEATURES
from .io import read_json, write_json, write_jsonl

PUBLIC_V2_SPLIT_COUNTS = {"train": 1_075, "validation": 230, "test": 230}
PUBLIC_V2_ALLOWED_SELECTION_SPLITS = ("train", "validation")
REQUIRED_ADAPTER_LEVELS: dict[str, frozenset[int | float]] = {
    "max_steps": frozenset({100, 300}),
    "lora_rank": frozenset({8, 16}),
    "hard_negatives": frozenset({4, 8}),
    "temperature": frozenset({0.03, 0.05}),
}
REQUIRED_DIAGNOSTIC_SLICES = (
    "spelling",
    "year_or_numeric",
    "entity",
    "geographic",
    "semantic_paraphrase",
)
BEIR_SCIFACT_URL = (
    "https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/scifact.zip"
)
BEIR_SCIFACT_MD5 = "5f7d1de60b170fc8027bb7898e2efca1"


def file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def newline_normalized_sha256(path: str | Path, *, newline: str) -> str:
    """Hash UTF-8 text with an explicitly frozen newline serialization."""

    if newline not in {"lf", "crlf"}:
        raise ValueError("newline must be lf or crlf")
    with Path(path).open("r", encoding="utf-8", newline=None) as handle:
        text = handle.read()
    separator = "\n" if newline == "lf" else "\r\n"
    payload = text.replace("\n", separator).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def file_md5(path: str | Path) -> str:
    digest = hashlib.md5(usedforsecurity=False)
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def tree_sha256(path: str | Path) -> str:
    """Hash a file or directory without embedding its absolute location."""

    target = Path(path)
    if target.is_file():
        return file_sha256(target)
    if not target.is_dir():
        raise FileNotFoundError(target)
    digest = hashlib.sha256()
    files = sorted(item for item in target.rglob("*") if item.is_file())
    if not files:
        raise ValueError(f"cannot hash empty directory: {target}")
    for item in files:
        relative = item.relative_to(target).as_posix().encode("utf-8")
        digest.update(len(relative).to_bytes(8, "big"))
        digest.update(relative)
        with item.open("rb") as handle:
            for block in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(block)
    return digest.hexdigest()


def load_public_v2_protocol(path: str | Path) -> dict[str, Any]:
    payload = read_json(path)
    if not isinstance(payload, dict):
        raise TypeError("public-v2 protocol must be a JSON object")
    return payload


def validate_public_v2_protocol(
    protocol: Mapping[str, Any], policy: Mapping[str, Any]
) -> dict[str, Any]:
    """Validate the complete pre-registered search protocol before quality runs."""

    dataset = _mapping(protocol, "dataset")
    split_counts = {
        str(name): int(count)
        for name, count in _mapping(dataset, "split_counts").items()
    }
    if split_counts != PUBLIC_V2_SPLIT_COUNTS:
        raise ValueError(f"CLIMATE-FEVER v2 counts must be {PUBLIC_V2_SPLIT_COUNTS}")
    allowed = tuple(str(value) for value in dataset.get("selection_splits", ()))
    if allowed != PUBLIC_V2_ALLOWED_SELECTION_SPLITS:
        raise ValueError("selection must use train then validation only")
    if str(dataset.get("selection_split")) != "validation":
        raise ValueError("validation must be the only selection/evaluation split")
    if str(dataset.get("prohibited_split")) != "test":
        raise ValueError("the public v2 test must remain prohibited")
    if str(dataset.get("split_manifest_newline")) != "crlf":
        raise ValueError("the verified split manifest hash freezes CRLF serialization")

    future = _mapping(policy, "future_split_protocol")
    if str(future.get("verified_split_manifest_sha256")) != str(
        dataset.get("split_manifest_sha256")
    ):
        raise ValueError("protocol and policy split manifest hashes differ")
    if str(future.get("verified_split_manifest_newline")) != "crlf":
        raise ValueError("policy must record the verified manifest newline")

    frozen = _mapping(policy, "frozen_test")
    if str(frozen.get("status")) != "consumed":
        raise ValueError("historical test must remain consumed")
    if not bool(frozen.get("permanently_sealed", False)):
        raise ValueError("historical consumed test must be permanently sealed")
    if bool(frozen.get("exact_baseline_reproduction_allowed", True)):
        raise ValueError("permanent seal forbids further exact-test executions")
    if bool(frozen.get("additional_candidate_evaluations_allowed", True)):
        raise ValueError("permanent seal forbids candidate test executions")

    adapters = protocol.get("adapter_matrix")
    if not isinstance(adapters, list) or len(adapters) != 6:
        raise ValueError("adapter_matrix must contain exactly six configurations")
    ids: set[str] = set()
    observed: dict[str, set[int | float]] = {
        name: set() for name in REQUIRED_ADAPTER_LEVELS
    }
    tuples: set[tuple[int, int, int, float]] = set()
    for raw in adapters:
        if not isinstance(raw, Mapping):
            raise TypeError("each adapter configuration must be an object")
        identifier = str(raw.get("id", ""))
        if not identifier or identifier in ids:
            raise ValueError(f"duplicate or empty adapter id: {identifier!r}")
        ids.add(identifier)
        max_steps = int(raw["max_steps"])
        lora_rank = int(raw["lora_rank"])
        hard_negatives = int(raw["hard_negatives"])
        temperature = float(raw["temperature"])
        values: dict[str, int | float] = {
            "max_steps": max_steps,
            "lora_rank": lora_rank,
            "hard_negatives": hard_negatives,
            "temperature": temperature,
        }
        for name, value in values.items():
            if value not in REQUIRED_ADAPTER_LEVELS[name]:
                raise ValueError(f"unregistered {name} value: {value}")
            observed[name].add(value)
        signature = (max_steps, lora_rank, hard_negatives, temperature)
        if signature in tuples:
            raise ValueError(f"duplicate adapter configuration: {signature}")
        tuples.add(signature)
    if observed != REQUIRED_ADAPTER_LEVELS:
        raise ValueError(f"adapter matrix does not cover all fixed levels: {observed}")

    evaluation = _mapping(protocol, "evaluation")
    if int(evaluation.get("bootstrap_samples", 0)) != 5_000:
        raise ValueError(
            "public-v2 evaluation requires exactly 5,000 bootstrap samples"
        )
    if int(evaluation.get("max_full_candidates", 0)) > 2:
        raise ValueError("at most two pilot candidates may advance to full evaluation")
    if int(evaluation.get("pilot_query_count", 0)) <= 0:
        raise ValueError("pilot_query_count must be positive")
    slices = tuple(str(value) for value in evaluation.get("diagnostic_slices", ()))
    if slices != REQUIRED_DIAGNOSTIC_SLICES:
        raise ValueError("diagnostic slices must match the frozen ordered contract")

    ranking = _mapping(protocol, "downstream_ranking")
    if int(ranking.get("candidate_width", 0)) != 100:
        raise ValueError("LambdaMART train/serve candidate width must be exactly 100")
    feature_order = tuple(str(value) for value in ranking.get("feature_order", ()))
    if feature_order != DEFAULT_FEATURES:
        raise ValueError("LambdaMART feature order must match DEFAULT_FEATURES exactly")
    if str(ranking.get("positive_policy")) != "candidate-supported-only":
        raise ValueError("LTR positives must be serving-reachable candidates only")

    external = _mapping(protocol, "external_transfer")
    if str(external.get("dataset")) != "SciFact":
        raise ValueError("the sole external transfer dataset must be SciFact")
    if int(external.get("max_completed_evaluations", 0)) != 1:
        raise ValueError("SciFact permits exactly one completed transfer evaluation")
    if str(external.get("beir_md5")) != BEIR_SCIFACT_MD5:
        raise ValueError("SciFact must use the official BEIR archive hash")
    if str(external.get("split")) != "test":
        raise ValueError("official SciFact transfer uses the BEIR test qrels")

    return {
        "status": "passed",
        "protocol_id": str(protocol.get("protocol_id")),
        "adapter_ids": sorted(ids),
        "adapter_count": len(ids),
        "selection_splits": list(allowed),
        "split_counts": split_counts,
        "bootstrap_samples": 5_000,
        "max_full_candidates": int(evaluation["max_full_candidates"]),
        "candidate_width": 100,
        "feature_order": list(feature_order),
        "external_transfer_budget": 1,
    }


def _mapping(value: Mapping[str, Any], key: str) -> Mapping[str, Any]:
    child = value.get(key)
    if not isinstance(child, Mapping):
        raise TypeError(f"protocol is missing object: {key}")
    return child


def export_public_v2_splits(
    prepared_dir: str | Path, output_dir: str | Path
) -> dict[str, Any]:
    """Export train and validation only; never materialise a test claims file."""

    prepared = Path(prepared_dir)
    manifest = read_json(prepared / "split_manifest.json")
    claims = read_json(prepared / "claims.json")
    if not isinstance(manifest, dict) or not isinstance(claims, dict):
        raise TypeError("prepared CLIMATE-FEVER artifacts are malformed")
    split = manifest.get("split")
    if not isinstance(split, dict):
        raise TypeError("split manifest is missing split assignments")
    counts = {name: len(split.get(name, ())) for name in PUBLIC_V2_SPLIT_COUNTS}
    if counts != PUBLIC_V2_SPLIT_COUNTS:
        raise ValueError(f"unexpected public-v2 split counts: {counts}")
    target = Path(output_dir)
    target.mkdir(parents=True, exist_ok=True)
    exported: dict[str, Any] = {}
    for split_name in PUBLIC_V2_ALLOWED_SELECTION_SPLITS:
        ids = {str(value) for value in split[split_name]}
        rows = {
            claim_id: claims[claim_id] for claim_id in sorted(ids) if claim_id in claims
        }
        if len(rows) != len(ids):
            raise ValueError(f"missing {split_name} claims during export")
        write_json(target / f"{split_name}-claims.json", rows)
        exported[split_name] = {
            "claim_count": len(rows),
            "decisive_claim_count": sum(
                bool(row.get("evidences")) for row in rows.values()
            ),
            "query_id_sha256": stable_id_sha256(sorted(rows)),
            "claims_sha256": file_sha256(target / f"{split_name}-claims.json"),
        }
    seal = {
        "schema_version": 1,
        "status": "permanently-sealed-not-materialised",
        "historical_test_reuse": "forbidden",
        "v2_test_query_count": counts["test"],
        "test_query_id_sha256": stable_id_sha256(
            sorted(str(value) for value in split["test"])
        ),
        "boundary": (
            "The v2 test assignment remains only inside the provenance manifest. "
            "No test claims file is exported and no selection command accepts test."
        ),
    }
    write_json(target / "test-seal.json", seal)
    exported["test_seal"] = seal
    write_json(target / "selection-splits.json", exported)
    return exported


def select_pilot_claim_ids(
    validation_claims: Mapping[str, Any], *, size: int, seed: int
) -> list[str]:
    decisive = sorted(
        str(claim_id)
        for claim_id, row in validation_claims.items()
        if isinstance(row, Mapping) and row.get("evidences")
    )
    if size <= 0 or size > len(decisive):
        raise ValueError("pilot size must be within decisive validation claims")
    rng = random.Random(seed)
    rng.shuffle(decisive)
    return sorted(decisive[:size])


def paired_promotion_decision(metrics: Mapping[str, Any]) -> dict[str, Any]:
    comparisons = metrics.get("paired_bootstrap")
    if not isinstance(comparisons, Mapping):
        raise TypeError("metrics are missing paired_bootstrap")
    recall = _mapping(comparisons, "recall@5")
    secondary_names = ("mrr@10", "ndcg@10", "evidence_f1")
    primary_pass = float(recall["lower"]) > 0.0
    secondary = {
        name: float(_mapping(comparisons, name)["mean_difference"]) >= 0.0
        for name in secondary_names
    }
    return {
        "primary_recall_ci_lower_positive": primary_pass,
        "secondary_mean_non_regression": secondary,
        "promotion_pass": primary_pass and all(secondary.values()),
        "rule": (
            "Recall@5 paired 95% CI lower bound > 0 and non-negative mean deltas "
            "for MRR@10, nDCG@10 and Evidence F1."
        ),
    }


def pilot_advancement_decision(metrics: Mapping[str, Any]) -> dict[str, Any]:
    comparisons = metrics.get("paired_bootstrap")
    if not isinstance(comparisons, Mapping):
        raise TypeError("metrics are missing paired_bootstrap")
    names = ("recall@5", "mrr@10", "ndcg@10", "evidence_f1")
    means = {
        name: float(_mapping(comparisons, name)["mean_difference"]) for name in names
    }
    return {
        "mean_deltas": means,
        "advance_eligible": means["recall@5"] > 0.0
        and all(means[name] >= 0.0 for name in names[1:]),
        "boundary": (
            "Pilot advancement uses mean deltas only. It authorises at most two fixed "
            "full-validation runs and is not a promotion claim."
        ),
    }


def select_pilot_candidates(
    adapter_matrix: Sequence[Mapping[str, Any]],
    metrics_by_id: Mapping[str, Mapping[str, Any]],
    *,
    maximum: int = 2,
) -> dict[str, Any]:
    if maximum < 0 or maximum > 2:
        raise ValueError("pilot advancement maximum must be in [0, 2]")
    expected_ids = {str(row["id"]) for row in adapter_matrix}
    if set(metrics_by_id) != expected_ids:
        raise ValueError("pilot results must cover all six pre-registered adapters")
    records: list[dict[str, Any]] = []
    for config in adapter_matrix:
        identifier = str(config["id"])
        decision = pilot_advancement_decision(metrics_by_id[identifier])
        records.append(
            {
                "id": identifier,
                "config": dict(config),
                "pilot": decision,
                "metrics_sha256": str(
                    metrics_by_id[identifier].get("metrics_sha256", "")
                ),
            }
        )
    ranked = sorted(
        (row for row in records if row["pilot"]["advance_eligible"]),
        key=lambda row: (
            -float(row["pilot"]["mean_deltas"]["recall@5"]),
            -float(row["pilot"]["mean_deltas"]["ndcg@10"]),
            str(row["id"]),
        ),
    )
    selected = [str(row["id"]) for row in ranked[:maximum]]
    diagnostic_fallback = False
    if not selected and maximum > 0:
        fallback = sorted(
            records,
            key=lambda row: (
                -float(row["pilot"]["mean_deltas"]["recall@5"]),
                -float(row["pilot"]["mean_deltas"]["ndcg@10"]),
                str(row["id"]),
            ),
        )
        selected = [str(fallback[0]["id"])]
        diagnostic_fallback = True
    return {
        "schema_version": 1,
        "pilot_result_count": len(records),
        "max_full_candidates": maximum,
        "selected_for_full": selected,
        "diagnostic_fallback": diagnostic_fallback,
        "all_pilot_results": records,
        "negative_results_preserved": [
            str(row["id"]) for row in records if str(row["id"]) not in selected
        ],
    }


def select_full_candidate(
    full_metrics_by_id: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    if len(full_metrics_by_id) > 2:
        raise ValueError("no more than two full candidate evaluations are allowed")
    records: list[dict[str, Any]] = []
    for identifier, metrics in sorted(full_metrics_by_id.items()):
        decision = paired_promotion_decision(metrics)
        candidate = _mapping(metrics, "candidate")
        records.append(
            {
                "id": identifier,
                "promotion": decision,
                "candidate_metrics": {
                    name: float(candidate[name])
                    for name in ("recall@5", "mrr@10", "ndcg@10", "evidence_f1")
                },
                "metrics_sha256": str(metrics.get("metrics_sha256", "")),
                "adapter_sha256": str(metrics.get("adapter_sha256", "")),
            }
        )
    promoted = [row for row in records if row["promotion"]["promotion_pass"]]
    ranked = sorted(
        promoted,
        key=lambda row: (
            -float(row["candidate_metrics"]["recall@5"]),
            -float(row["candidate_metrics"]["ndcg@10"]),
            str(row["id"]),
        ),
    )
    fallback = sorted(
        records,
        key=lambda row: (
            -float(row["candidate_metrics"]["recall@5"]),
            -float(row["candidate_metrics"]["ndcg@10"]),
            str(row["id"]),
        ),
    )
    selected = ranked[0] if ranked else fallback[0] if fallback else None
    return {
        "schema_version": 1,
        "full_result_count": len(records),
        "promoted_candidate_ids": [str(row["id"]) for row in ranked],
        "selected_candidate_id": None if selected is None else str(selected["id"]),
        "selected_candidate_promoted": bool(ranked),
        "all_full_results": records,
        "boundary": (
            "If no adapter passes the promotion rule, the best full-run adapter is "
            "retained only for downstream negative analysis and is not promoted."
        ),
    }


def download_file(
    url: str, target: str | Path, *, timeout_seconds: float = 120.0
) -> dict[str, Any]:
    destination = Path(target)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".download")
    request = urllib.request.Request(url, headers={"User-Agent": "climate-rag/0.1"})
    context = ssl.create_default_context()
    try:
        import certifi

        context = ssl.create_default_context(cafile=certifi.where())
    except ImportError:
        pass
    with (
        urllib.request.urlopen(
            request, timeout=timeout_seconds, context=context
        ) as response,
        temporary.open("wb") as handle,
    ):
        shutil.copyfileobj(response, handle)
    temporary.replace(destination)
    return {
        "url": url,
        "bytes": destination.stat().st_size,
        "sha256": file_sha256(destination),
        "md5": file_md5(destination),
    }


def verify_scifact_archive(path: str | Path) -> dict[str, Any]:
    archive = Path(path)
    md5 = file_md5(archive)
    if md5 != BEIR_SCIFACT_MD5:
        raise ValueError(f"SciFact BEIR MD5 mismatch: {md5}")
    return {
        "status": "verified-official-beir-archive",
        "url": BEIR_SCIFACT_URL,
        "bytes": archive.stat().st_size,
        "md5": md5,
        "sha256": file_sha256(archive),
    }


def prepare_scifact_transfer(
    archive_path: str | Path, output_dir: str | Path
) -> dict[str, Any]:
    """Prepare the official BEIR SciFact test split for the one-shot transfer run."""

    provenance = verify_scifact_archive(archive_path)
    target = Path(output_dir)
    if target.exists() and any(target.iterdir()):
        raise FileExistsError(f"SciFact output must be new and empty: {target}")
    target.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(archive_path) as archive:
        names = set(archive.namelist())
        required = {
            "scifact/corpus.jsonl",
            "scifact/queries.jsonl",
            "scifact/qrels/test.tsv",
        }
        if not required <= names:
            raise ValueError(f"SciFact archive is missing: {sorted(required - names)}")
        corpus_rows = _jsonl_from_zip(archive, "scifact/corpus.jsonl")
        query_rows = _jsonl_from_zip(archive, "scifact/queries.jsonl")
        qrel_lines = archive.read("scifact/qrels/test.tsv").decode("utf-8").splitlines()
    qrels: dict[str, list[str]] = {}
    for index, line in enumerate(qrel_lines):
        if not line.strip():
            continue
        parts = line.split("\t")
        if index == 0 and parts[:2] == ["query-id", "corpus-id"]:
            continue
        if len(parts) != 3:
            raise ValueError(f"invalid SciFact qrel line: {line!r}")
        query_id, corpus_id, score = parts
        if int(score) > 0:
            qrels.setdefault(query_id, []).append(corpus_id)
    queries = {str(row["_id"]): str(row["text"]) for row in query_rows}
    missing_queries = sorted(set(qrels) - set(queries))
    if missing_queries:
        raise ValueError(
            f"SciFact qrels reference missing queries: {missing_queries[:3]}"
        )
    corpus_ids = {str(row["_id"]) for row in corpus_rows}
    missing_corpus = sorted(
        {corpus_id for values in qrels.values() for corpus_id in values} - corpus_ids
    )
    if missing_corpus:
        raise ValueError(
            f"SciFact qrels reference missing corpus rows: {missing_corpus[:3]}"
        )
    write_jsonl(
        target / "evidence.jsonl",
        (
            {
                "evidence_id": str(row["_id"]),
                "text": " ".join(
                    value.strip()
                    for value in (str(row.get("title", "")), str(row.get("text", "")))
                    if value.strip()
                ),
                "metadata": {
                    "title": str(row.get("title", "")),
                    "dataset": "BEIR/SciFact",
                },
            }
            for row in corpus_rows
        ),
    )
    write_json(
        target / "test-claims.json",
        {
            query_id: {
                "claim_text": queries[query_id],
                "claim_label": None,
                "evidences": sorted(set(corpus_ids_for_query)),
            }
            for query_id, corpus_ids_for_query in sorted(qrels.items())
        },
    )
    manifest = {
        "schema_version": 1,
        "dataset": "SciFact",
        "source": "official BEIR archive mirrored by MTEB",
        "split": "test",
        "archive": provenance,
        "corpus_count": len(corpus_rows),
        "all_query_count": len(query_rows),
        "test_query_count": len(qrels),
        "test_qrel_count": sum(len(values) for values in qrels.values()),
        "corpus_sha256": file_sha256(target / "evidence.jsonl"),
        "claims_sha256": file_sha256(target / "test-claims.json"),
        "boundary": (
            "SciFact qrels are opened only by the reserved one-shot external transfer job. "
            "They are not used for climate model selection or tuning."
        ),
    }
    write_json(target / "manifest.json", manifest)
    return manifest


def _jsonl_from_zip(archive: zipfile.ZipFile, name: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for line in archive.read(name).decode("utf-8").splitlines():
        if line.strip():
            row = json.loads(line)
            if not isinstance(row, dict):
                raise TypeError(f"{name} contains a non-object JSON row")
            rows.append(row)
    return rows


def reserve_external_transfer(
    ledger_path: str | Path,
    *,
    frozen_config_sha256: str,
    attempt_id: str,
    maximum_infrastructure_retries: int = 2,
) -> dict[str, Any]:
    """Atomically reserve the one external evaluation, allowing same-config infra retry."""

    if len(frozen_config_sha256) != 64:
        raise ValueError("frozen_config_sha256 must be a SHA-256 digest")
    target = Path(ledger_path)
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists():
        ledger = read_json(target)
        if not isinstance(ledger, dict):
            raise TypeError("external transfer ledger must be an object")
        if str(ledger.get("frozen_config_sha256")) != frozen_config_sha256:
            raise ValueError("external transfer is already reserved for another config")
        if str(ledger.get("status")) == "completed":
            raise ValueError(
                "the sole external transfer evaluation is already completed"
            )
        attempts = ledger.get("attempts", [])
        if not isinstance(attempts, list):
            raise TypeError("external transfer attempts must be a list")
        if len(attempts) >= 1 + maximum_infrastructure_retries:
            raise ValueError(
                "external transfer infrastructure retry budget is exhausted"
            )
        attempts.append({"attempt_id": attempt_id, "status": "running"})
        ledger["attempts"] = attempts
        write_json(target, ledger)
        return ledger
    ledger = {
        "schema_version": 1,
        "dataset": "SciFact",
        "status": "running",
        "frozen_config_sha256": frozen_config_sha256,
        "maximum_completed_evaluations": 1,
        "maximum_infrastructure_retries": maximum_infrastructure_retries,
        "attempts": [{"attempt_id": attempt_id, "status": "running"}],
    }
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
    descriptor = os.open(target, flags, 0o600)
    try:
        os.write(
            descriptor, (json.dumps(ledger, indent=2, sort_keys=True) + "\n").encode()
        )
    finally:
        os.close(descriptor)
    return ledger


def complete_external_transfer(
    ledger_path: str | Path, *, attempt_id: str, metrics_sha256: str
) -> dict[str, Any]:
    target = Path(ledger_path)
    ledger = read_json(target)
    if not isinstance(ledger, dict) or str(ledger.get("status")) != "running":
        raise ValueError("external transfer ledger is not running")
    attempts = ledger.get("attempts")
    if not isinstance(attempts, list) or not attempts:
        raise ValueError("external transfer ledger has no attempts")
    current = attempts[-1]
    if not isinstance(current, dict) or str(current.get("attempt_id")) != attempt_id:
        raise ValueError("attempt id does not own the external transfer reservation")
    current["status"] = "completed"
    current["metrics_sha256"] = metrics_sha256
    ledger["status"] = "completed"
    ledger["completed_evaluations"] = 1
    ledger["metrics_sha256"] = metrics_sha256
    write_json(target, ledger)
    return ledger


def load_metrics_files(paths: Iterable[str | Path]) -> dict[str, dict[str, Any]]:
    result: dict[str, dict[str, Any]] = {}
    for path in paths:
        source = Path(path)
        metrics = read_json(source)
        if not isinstance(metrics, dict):
            raise TypeError(f"metrics file is not an object: {source}")
        identifier = str(metrics.get("adapter_config_id", ""))
        if not identifier or identifier in result:
            raise ValueError(f"duplicate or missing adapter_config_id in {source}")
        metrics["metrics_sha256"] = file_sha256(source)
        result[identifier] = metrics
    return result
