from __future__ import annotations

import argparse
import json
from pathlib import Path

from climate_rag.io import write_json

FIELDS = (
    "job_id",
    "job_name",
    "partition",
    "allocated_tres",
    "requested_memory",
    "elapsed",
    "max_rss",
    "state",
    "exit_code",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compact sacct pipe output by job/array task."
    )
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    return parser.parse_args()


def _base_job_id(value: str) -> str:
    return value.removesuffix(".batch").removesuffix(".extern")


def main() -> int:
    args = parse_args()
    rows: dict[str, dict[str, str]] = {}
    for line in Path(args.input).read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        values = line.rstrip("|").split("|")
        if len(values) != len(FIELDS):
            raise ValueError(f"unexpected sacct row: {line!r}")
        raw = dict(zip(FIELDS, values, strict=True))
        identifier = _base_job_id(raw["job_id"])
        current = rows.setdefault(identifier, {name: "" for name in FIELDS})
        current["job_id"] = identifier
        if "." not in raw["job_id"]:
            current.update(raw)
            current["job_id"] = identifier
        elif raw["job_id"].endswith(".batch") and raw["max_rss"]:
            current["max_rss"] = raw["max_rss"]
    output = [rows[identifier] for identifier in sorted(rows)]
    write_json(args.output, output)
    print(json.dumps(output, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
