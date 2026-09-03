from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

from jsonschema import Draft202012Validator

SECRET_PATTERNS = (
    re.compile(r"-----BEGIN (?:RSA |OPENSSH )?PRIVATE KEY-----"),
    re.compile(r"\bAKIA[0-9A-Z]{16}\b"),
    re.compile(r"(?i)\b(?:api[_-]?key|access[_-]?token|secret)\s*[:=]\s*['\"][^'\"]+"),
    re.compile(r"(?i)\bbearer\s+[A-Za-z0-9._~+/=-]{20,}"),
    re.compile(r"(?:[A-Za-z]:[\\/]|/data/gpfs/|/home/)"),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Validate and scan a compact public-v2 artifact."
    )
    parser.add_argument("artifact")
    parser.add_argument("schema")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    artifact_path = Path(args.artifact)
    raw = artifact_path.read_text(encoding="utf-8")
    for pattern in SECRET_PATTERNS:
        if pattern.search(raw):
            raise ValueError(f"secret/PII/path pattern found: {pattern.pattern}")
    artifact = json.loads(raw)
    schema = json.loads(Path(args.schema).read_text(encoding="utf-8"))
    Draft202012Validator(schema).validate(artifact)
    if artifact["external_transfer"]["completed_evaluations"] != 1:
        raise ValueError("compact artifact must record exactly one external evaluation")
    if len(artifact["adapter_matrix"]) != 6:
        raise ValueError("compact artifact must preserve all six adapter results")
    print("public-v2 artifact schema and redaction checks passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
