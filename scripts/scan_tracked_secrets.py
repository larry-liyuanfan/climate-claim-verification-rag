from __future__ import annotations

import argparse
import re
import subprocess
from pathlib import Path

PATTERNS = {
    "private-key": re.compile(r"-----BEGIN (?:RSA |OPENSSH )?PRIVATE KEY-----"),
    "aws-access-key": re.compile(r"\bAKIA[0-9A-Z]{16}\b"),
    "assigned-secret": re.compile(
        r"(?i)\b(?:api[_-]?key|access[_-]?token|client[_-]?secret)\s*[:=]\s*['\"][^'\"]+"
    ),
    "windows-user-path": re.compile(r"(?i)[A-Za-z]:\\Users\\[^\\\s]+"),
    "iris-user-id": re.compile(r"\byzhang3504\b"),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Scan tracked text files for secret/PII patterns."
    )
    parser.add_argument("--repository", default=".")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    repository = Path(args.repository).resolve()
    output = subprocess.check_output(["git", "ls-files", "-z"], cwd=repository).decode(
        "utf-8"
    )
    findings: list[str] = []
    for relative in (value for value in output.split("\0") if value):
        path = repository / relative
        try:
            text = path.read_text(encoding="utf-8")
        except (UnicodeDecodeError, OSError):
            continue
        for name, pattern in PATTERNS.items():
            if pattern.search(text):
                findings.append(f"{relative}: {name}")
    if findings:
        raise ValueError("tracked secret/PII scan failed:\n" + "\n".join(findings))
    print("tracked secret/PII scan passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
