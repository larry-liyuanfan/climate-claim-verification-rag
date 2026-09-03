from __future__ import annotations

import argparse
import json
from pathlib import Path

from climate_rag.climate_fever import (
    download_climate_fever,
    prepare_public_benchmark,
)
from climate_rag.io import read_json, write_json
from climate_rag.public_v2 import (
    BEIR_SCIFACT_URL,
    download_file,
    export_public_v2_splits,
    file_sha256,
    load_public_v2_protocol,
    newline_normalized_sha256,
    validate_public_v2_protocol,
    verify_scifact_archive,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Prepare the sealed public-v2 data inputs."
    )
    parser.add_argument("--protocol", required=True)
    parser.add_argument("--policy", required=True)
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--output-dir", required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    protocol = load_public_v2_protocol(args.protocol)
    policy = read_json(args.policy)
    if not isinstance(policy, dict):
        raise TypeError("evaluation policy must be a JSON object")
    validation = validate_public_v2_protocol(protocol, policy)
    data_root = Path(args.data_root)
    output = Path(args.output_dir)
    if output.exists():
        raise FileExistsError(f"output must be unique: {output}")
    output.mkdir(parents=True)
    climate_root = data_root / "climate-fever-v2"
    climate_source = climate_root / "source" / "climate-fever.jsonl"
    climate_download: dict[str, object] = {}
    if not climate_source.exists():
        climate_download = download_climate_fever(climate_source)
    expected_source_hash = str(protocol["dataset"]["source_sha256"])
    if file_sha256(climate_source) != expected_source_hash:
        raise ValueError("CLIMATE-FEVER source hash differs from the frozen protocol")
    prepared = climate_root / "prepared"
    if not (prepared / "split_manifest.json").exists():
        manifest = prepare_public_benchmark(
            climate_source,
            prepared,
            seed=20260825,
        )
    else:
        manifest = read_json(prepared / "split_manifest.json")
    if not isinstance(manifest, dict):
        raise TypeError("CLIMATE-FEVER split manifest must be an object")
    expected_split_hash = str(protocol["dataset"]["split_manifest_sha256"])
    split_newline = str(protocol["dataset"]["split_manifest_newline"])
    native_split_hash = file_sha256(prepared / "split_manifest.json")
    protocol_split_hash = newline_normalized_sha256(
        prepared / "split_manifest.json", newline=split_newline
    )
    if protocol_split_hash != expected_split_hash:
        raise ValueError(
            "CLIMATE-FEVER split manifest differs from the frozen v2 protocol: "
            f"native={native_split_hash}, protocol={protocol_split_hash}"
        )
    selection_dir = climate_root / "selection-only"
    if not (selection_dir / "selection-splits.json").exists():
        exported = export_public_v2_splits(prepared, selection_dir)
    else:
        exported = read_json(selection_dir / "selection-splits.json")

    scifact_archive = data_root / "scifact" / "source" / "scifact.zip"
    scifact_download: dict[str, object] = {}
    if not scifact_archive.exists():
        scifact_download = download_file(BEIR_SCIFACT_URL, scifact_archive)
    scifact_provenance = verify_scifact_archive(scifact_archive)
    metrics = {
        "schema_version": 1,
        "protocol_validation": validation,
        "climate_fever": {
            "source_sha256": expected_source_hash,
            "split_manifest_sha256": protocol_split_hash,
            "split_manifest_native_sha256": native_split_hash,
            "split_manifest_newline": split_newline,
            "split_counts": {
                name: len(manifest["split"][name])
                for name in ("train", "validation", "test")
            },
            "selection_exports": exported,
            "download": climate_download,
        },
        "scifact": {
            **scifact_provenance,
            "download": scifact_download,
            "qrels_opened": False,
            "boundary": (
                "The archive is hash-verified but its qrels remain unopened until the "
                "one-shot post-freeze transfer job."
            ),
        },
    }
    write_json(output / "metrics.json", metrics)
    print(json.dumps(metrics, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
