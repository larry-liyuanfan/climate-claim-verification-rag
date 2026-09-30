"""CPU-only original train/dev preparation. No model, test member or qrels read."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from climate_rag.scifact_grounding import prepare_original_archive


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = prepare_original_archive(args.archive, args.output)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
