"""CPU metadata/full-byte inventory only; never connect, download or load weights."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from cloud_replay_contract import SELECTION, digest, inspect_input
from climate_rag.targeted_replay import GOLD_SHA, INPUT_SHA, SELECTION_SHA


def inventory(archive: Path, gold: Path, source: Path, *, full: bool = False,
              input_sha: str = INPUT_SHA, gold_sha: str = GOLD_SHA) -> dict[str, Any]:
    result: dict[str, Any] = {"scope": "local_cpu_asset_inventory_not_cloud_readiness",
                              "model_execution_authorized": False,
                              "runtime": {"linux": "unobserved", "gpu": "unobserved",
                                          "driver": "unobserved", "instance": "unobserved"},
                              "assets": {}}
    for name, path, expected in (("gold", gold, gold_sha), ("selection", source / SELECTION, SELECTION_SHA)):
        result["assets"][name] = ({"status": "verified", "sha256": expected, "bytes": path.stat().st_size}
                                  if path.is_file() and digest(path) == expected else {"status": "needs_assets"})
    if archive.is_file():
        try:
            result["assets"]["input"] = inspect_input(archive, input_sha, full=full)
        except (ValueError, KeyError, OSError) as exc:
            result["assets"]["input"] = {"status": "needs_assets", "failure_type": type(exc).__name__}
    else:
        result["assets"]["input"] = {"status": "needs_assets"}
    statuses = {v["status"] for v in result["assets"].values()}
    result["asset_status"] = ("needs_assets" if "needs_assets" in statuses else
                              "needs_full_byte_verification" if statuses != {"verified"} else "verified_local_assets")
    result["ready_for_model_execution"] = False
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", type=Path, required=True)
    parser.add_argument("--gold", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--full-hash", action="store_true")
    parser.add_argument("--input-sha", default=INPUT_SHA)
    args = parser.parse_args()
    result = inventory(args.archive, args.gold, Path(__file__).resolve().parents[1],
                       full=args.full_hash, input_sha=args.input_sha)
    with args.output.open("x", encoding="utf-8", newline="\n") as stream:
        stream.write(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
