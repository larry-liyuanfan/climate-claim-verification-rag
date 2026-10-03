"""Build two deterministic, exact-allowlist bundles; never submits a job."""

import argparse
import io
import json
import tarfile
from pathlib import Path

from climate_rag.public_v2 import file_sha256


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prepared", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError("no bundle overwrite")
    manifest = json.loads((args.prepared / "manifest.json").read_text())
    groups = {
        "inference": {n: "inference/" + n for n in ("claims.jsonl", "corpus.jsonl", "protocol.json")},
        "scoring": {"gold.jsonl": "private/gold.jsonl", "selected-strata.json": "private/selected-strata.json",
                    "manifest.json": "manifest.json"},
    }
    for mapping in groups.values():
        for relative in mapping.values():
            if relative != "manifest.json" and file_sha256(args.prepared / relative) != manifest["output_sha256"][relative]:
                raise ValueError("prepared input changed")
    args.output.mkdir(parents=True)
    receipts = {}
    for group, mapping in groups.items():
        target = args.output / f"{group}.tar"
        with tarfile.open(target, "x") as archive:
            for name, relative in sorted(mapping.items()):
                raw = (args.prepared / relative).read_bytes()
                item = tarfile.TarInfo(name)
                item.size, item.mode, item.mtime = len(raw), 0o400, 0
                archive.addfile(item, io.BytesIO(raw))
        receipts[group] = {"sha256": file_sha256(target), "bytes": target.stat().st_size,
                           "members": sorted(mapping)}
    receipt = {"schema_version": "scifact-train-diagnostic-bundles-v1",
               "preparation_manifest_sha256": file_sha256(args.prepared / "manifest.json"),
               "protocol_sha256": manifest["output_sha256"]["inference/protocol.json"],
               "bundles": receipts, "job_submitted": False}
    with (args.output / "bundle-receipt.json").open("x", encoding="utf-8") as stream:
        json.dump(receipt, stream, indent=2, sort_keys=True)
        stream.write("\n")
    print(json.dumps(receipt, indent=2))


if __name__ == "__main__":
    main()
