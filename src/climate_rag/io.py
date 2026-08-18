from __future__ import annotations

import json
import os
import tempfile
from collections.abc import Iterable, Iterator, Mapping
from pathlib import Path
from typing import Any

from .models import (
    Claim,
    EvidenceDocument,
    Prediction,
    claim_from_mapping,
    prediction_from_mapping,
)


def read_json(path: str | Path) -> Any:
    with Path(path).open("r", encoding="utf-8") as handle:
        return json.load(handle)


def write_json(path: str | Path, value: Any) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=f".{target.name}.", dir=target.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(value, handle, ensure_ascii=False, indent=2, sort_keys=True)
            handle.write("\n")
        os.replace(temporary, target)
    except BaseException:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass
        raise


def read_jsonl(path: str | Path) -> Iterator[dict[str, Any]]:
    with Path(path).open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError(f"{path}:{line_number}: expected a JSON object")
            yield value


def write_jsonl(path: str | Path, rows: Iterable[Mapping[str, Any]]) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=f".{target.name}.", dir=target.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            for row in rows:
                handle.write(json.dumps(dict(row), ensure_ascii=False, sort_keys=True) + "\n")
        os.replace(temporary, target)
    except BaseException:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass
        raise


def iter_evidence(path: str | Path) -> Iterator[EvidenceDocument]:
    source = Path(path)
    if source.suffix.lower() == ".jsonl":
        for row in read_jsonl(source):
            evidence_id = row.get("evidence_id", row.get("id"))
            if evidence_id is None:
                raise ValueError("JSONL evidence row is missing evidence_id")
            yield EvidenceDocument(
                evidence_id=str(evidence_id),
                text=str(row.get("text", row.get("evidence_text", ""))),
                metadata=row.get("metadata", {}) or {},
            )
        return

    try:
        import ijson  # type: ignore[import-not-found]
    except ImportError:
        payload = read_json(source)
        if isinstance(payload, dict):
            for evidence_id, value in payload.items():
                if isinstance(value, str):
                    yield EvidenceDocument(str(evidence_id), value)
                elif isinstance(value, dict):
                    yield EvidenceDocument(
                        str(evidence_id),
                        str(value.get("text", value.get("evidence_text", ""))),
                        value.get("metadata", {}) or {},
                    )
                else:
                    raise ValueError(f"invalid evidence entry for {evidence_id}")
        elif isinstance(payload, list):
            for row in payload:
                if not isinstance(row, dict):
                    raise ValueError("evidence list entries must be JSON objects")
                evidence_id = row.get("evidence_id", row.get("id"))
                if evidence_id is None:
                    raise ValueError("evidence row is missing evidence_id")
                yield EvidenceDocument(
                    str(evidence_id),
                    str(row.get("text", row.get("evidence_text", ""))),
                    row.get("metadata", {}) or {},
                )
        else:
            raise ValueError("evidence JSON must be an object, a list, or JSONL")
        return

    with source.open("rb") as handle:
        for evidence_id, value in ijson.kvitems(handle, ""):
            if isinstance(value, str):
                yield EvidenceDocument(str(evidence_id), value)
            elif isinstance(value, dict):
                yield EvidenceDocument(
                    str(evidence_id),
                    str(value.get("text", value.get("evidence_text", ""))),
                    value.get("metadata", {}) or {},
                )
            else:
                raise ValueError(f"invalid evidence entry for {evidence_id}")


def load_claims(path: str | Path) -> dict[str, Claim]:
    payload = read_json(path)
    if isinstance(payload, dict):
        return {str(key): claim_from_mapping(str(key), value) for key, value in payload.items()}
    if isinstance(payload, list):
        result: dict[str, Claim] = {}
        for row in payload:
            if not isinstance(row, dict):
                raise ValueError("claim list entries must be JSON objects")
            claim_id = row.get("claim_id", row.get("id"))
            if claim_id is None:
                raise ValueError("claim row is missing claim_id")
            result[str(claim_id)] = claim_from_mapping(str(claim_id), row)
        return result
    raise ValueError("claims JSON must be an object or a list")


def load_predictions(path: str | Path) -> dict[str, Prediction]:
    source = Path(path)
    if source.suffix.lower() == ".jsonl":
        result: dict[str, Prediction] = {}
        for row in read_jsonl(source):
            claim_id = row.get("claim_id", row.get("id"))
            if claim_id is None:
                raise ValueError("prediction row is missing claim_id")
            result[str(claim_id)] = prediction_from_mapping(str(claim_id), row)
        return result
    payload = read_json(source)
    if not isinstance(payload, dict):
        raise ValueError("prediction JSON must be keyed by claim id")
    return {str(key): prediction_from_mapping(str(key), value) for key, value in payload.items()}

