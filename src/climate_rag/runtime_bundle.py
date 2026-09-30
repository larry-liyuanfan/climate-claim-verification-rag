"""Resolve operator bundle arguments strictly inside a node-local input root."""
from __future__ import annotations

import re
from pathlib import Path

PATH_OPTIONS = {"--evidence", "--protocol", "--model-dir", "--model-manifest", "--dense-index",
                "--reranker-dir", "--reranker-manifest", "--execution-manifest"}
HASH_OPTIONS = {"--expected-protocol-sha256", "--expected-execution-sha256"}


def resolve_bundle_args(args: object, root: Path) -> list[str]:
    if not isinstance(args, list) or len(args) % 2 or not all(isinstance(x, str) for x in args):
        raise ValueError("bundle arguments must be option/value pairs")
    values: dict[str, str] = {}
    resolved = []
    for option, value in zip(args[::2], args[1::2], strict=True):
        if option in values:
            raise ValueError("duplicate bundle argument")
        values[option] = value
        if option in PATH_OPTIONS:
            if not value.startswith("{INPUT}/"):
                raise ValueError("all inputs must use node-local bundle paths")
            path = (root / value[len("{INPUT}/"):]).resolve()
            if not path.is_relative_to(root.resolve()) or not path.exists():
                raise ValueError("bundle path absent or outside input root")
            value = str(path)
        elif option in HASH_OPTIONS:
            if not re.fullmatch(r"[0-9a-f]{64}", value):
                raise ValueError("invalid frozen hash")
        elif option == "--provider":
            if value != "local-qwen":
                raise ValueError("only local-qwen in released bundle")
        elif option == "--phase":
            if value not in {"pilot", "validation", "vnext"}:
                raise ValueError("unregistered phase")
        else:
            raise ValueError("unapproved bundle option")
        resolved.extend([option, value])
    required = {"--evidence", "--protocol", "--model-dir", "--model-manifest", "--execution-manifest",
                "--expected-protocol-sha256", "--expected-execution-sha256", "--provider", "--phase"}
    if not required <= values.keys():
        raise ValueError("required frozen input missing")
    if ("--reranker-dir" in values) != ("--reranker-manifest" in values):
        raise ValueError("reranker and its manifest are inseparable")
    return resolved
