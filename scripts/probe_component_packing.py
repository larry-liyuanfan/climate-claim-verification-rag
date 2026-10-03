"""CPU-only frozen-slot JSON roundtrip; exports hashes/counts, never prompt text."""
from __future__ import annotations

import argparse
import importlib.metadata
import json
import os
import sys
import tarfile
import time
from pathlib import Path
from typing import Any

from climate_rag.component_execution import SLOTS_SHA, frozen_slots
from climate_rag.scifact_component_contract import LIMITS, PROMPTS, packing, require, schema_for
from climate_rag.scifact_semantic_contract import MODEL_SHA, TOKENIZER_SHA, checked, encoded, sha, write_once

ASSETS = frozenset(TOKENIZER_SHA) | {"added_tokens.json", "special_tokens_map.json",
                                  "config.json", "generation_config.json"}


def tokenizer_assets(archive: Path, target: Path) -> dict[str, str]:
    """Seek uncompressed tar headers; read only small allowlisted assets.

    The frozen manifest authenticates the bytes read. Deliberately neither
    hash nor extract the weight payloads or the entire multi-GB archive.
    """
    require(not target.exists(), "exclusive_asset_directory")
    with tarfile.open(archive, "r:") as bundle:
        members = bundle.getmembers()
        names = [m.name for m in members]
        require(len(names) == len(set(names)), "duplicate_archive_member")
        manifest_member = bundle.getmember("models/generator/model_manifest.json")
        require(manifest_member.isfile() and manifest_member.size < 16384, "manifest_member")
        stream = bundle.extractfile(manifest_member)
        require(stream is not None, "manifest_stream")
        assert stream is not None
        manifest: dict[str, str] = json.loads(stream.read())
        require(sha(json.dumps(manifest, sort_keys=True).encode()) == MODEL_SHA, "manifest_identity")
        prefix = "models/generator/model/"
        actual = {m.name[len(prefix):] for m in members if m.name.startswith(prefix)}
        permitted = ASSETS | {"README.md", "model.safetensors.index.json"} | {
            f"model-{i:05d}-of-00003.safetensors" for i in (1, 2, 3)}
        require(actual == permitted, "unexpected_runtime_assets")
        selected = [bundle.getmember(prefix + name) for name in sorted(ASSETS)]
        require(all(m.isfile() and 0 < m.size < 20_000_000 for m in selected)
                and sum(m.size for m in selected) < 20_000_000, "tokenizer_asset_size")
        target.mkdir(mode=0o700)
        identities = {}
        for member in selected:
            file = bundle.extractfile(member)
            require(file is not None, "asset_stream")
            assert file is not None
            payload = file.read()
            name = member.name[len(prefix):]
            require(sha(payload) == manifest[name], "asset_hash")
            with (target / name).open("xb") as output:
                output.write(payload)
            identities[name] = manifest[name]
    return identities


def legacy_packing(tokenizer: Any, spec: dict[str, Any]) -> dict[str, Any]:
    """Exact pre-fix serializer, applied to the persisted (sorted-key) input."""
    schema = schema_for(spec)
    messages = [{"role": "system", "content": PROMPTS[spec["component"]] + "\n" +
                 json.dumps(schema, ensure_ascii=False, separators=(",", ":"))},
                {"role": "user", "content": json.dumps(spec, ensure_ascii=False, separators=(",", ":"))}]
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True, enable_thinking=False)
    count = len(tokenizer.encode(text, add_special_tokens=False))
    return {"input_tokens": count, "prompt_sha256": sha(text.encode()), "schema_sha256": sha(encoded(schema)),
            "input_sha256": sha(encoded(spec)),
            "status": "prepared" if count <= LIMITS["input_tokens"] else "full_abstract_over_budget"}


def differences(expected: dict[str, Any], actual: dict[str, Any]) -> dict[str, Any]:
    return {key: {"expected": expected[key], "actual": actual[key]}
            for key in expected if expected[key] != actual[key]}


def probe(slots: Path, stage: Path, archive: Path, work: Path) -> dict[str, Any]:
    require(all(os.environ.get(k) == v for k, v in {"USE_TORCH": "0", "USE_TF": "0",
        "HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1"}.items()), "offline_tokenizer_only")
    require("torch" not in sys.modules, "torch_imported")
    begin = time.perf_counter()
    rows = frozen_slots(slots)
    for name, expected in TOKENIZER_SHA.items():
        checked(stage / name, expected)
    runtime = work / "runtime-tokenizer"
    assets = tokenizer_assets(archive, runtime)
    from transformers.models.auto.tokenization_auto import AutoTokenizer
    stage_tokenizer = AutoTokenizer.from_pretrained(str(stage), local_files_only=True, trust_remote_code=False)
    runtime_tokenizer = AutoTokenizer.from_pretrained(str(runtime), local_files_only=True, trust_remote_code=False)
    records = []
    for row in rows:
        require(row["packing"]["status"] == "prepared", "unexpected_preparation_gap")
        spec, expected = row["input"], row["packing"]
        variants = {"stage_legacy": legacy_packing(stage_tokenizer, spec),
                    "runtime_legacy": legacy_packing(runtime_tokenizer, spec),
                    "stage_restored": packing(stage_tokenizer, spec),
                    "runtime_restored": packing(runtime_tokenizer, spec)}
        records.append({"slot": row["slot"], "component": spec["component"], "expected": expected,
            "diffs": {key: differences(expected, value) for key, value in variants.items()},
            "legacy_tokenizer_paths_equal": variants["stage_legacy"] == variants["runtime_legacy"]})
    # Include a real tokenizer prepare -> write -> load -> pack seam on synthetic data.
    from climate_rag.component_preflight import preflight_cases
    synthetic = [{"input": spec, "packing": packing(stage_tokenizer, spec)} for spec, _ in preflight_cases()]
    write_once(work / "synthetic-roundtrip.json", synthetic)
    loaded = json.loads((work / "synthetic-roundtrip.json").read_bytes())
    roundtrip = [packing(runtime_tokenizer, r["input"]) == r["packing"] for r in loaded]
    require("torch" not in sys.modules, "torch_imported")
    checked(slots, SLOTS_SHA)  # input remained unchanged; scoring targets never opened
    return {"schema_version": "component-packing-cpu-diagnostic-v1", "records": records,
        "all_33_restored_exact": all(not r["diffs"][key] for r in records
                                    for key in ("stage_restored", "runtime_restored")),
        "legacy_mismatch_count": sum(bool(r["diffs"]["stage_legacy"]) for r in records),
        "legacy_tokenizer_paths_equal": all(r["legacy_tokenizer_paths_equal"] for r in records),
        "synthetic_real_tokenizer_roundtrip": roundtrip, "slots_sha256": SLOTS_SHA,
        "stage_assets": TOKENIZER_SHA, "runtime_assets": assets, "model_manifest_sha256": MODEL_SHA,
        "chat_template_sha256": {"stage": sha(stage_tokenizer.chat_template.encode()),
                                  "runtime": sha(runtime_tokenizer.chat_template.encode())},
        "versions": {name: importlib.metadata.version(name) for name in
                     ("transformers", "tokenizers", "Jinja2", "huggingface-hub")},
        "python": sys.version.split()[0], "elapsed_seconds": time.perf_counter() - begin,
        "model_calls": 0, "weights_read_or_loaded": False, "whole_model_archive_hashed": False,
        "targets_read": False, "torch_imported": False, "new_data_selected": False,
        "source_sha256": {name: sha(Path(name).read_bytes()) for name in
                          ("src/climate_rag/scifact_component_contract.py", "scripts/probe_component_packing.py")}}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for option in ("slots", "stage-tokenizer", "runtime-archive", "work", "output"):
        parser.add_argument("--" + option, type=Path, required=True)
    args = parser.parse_args()
    result = probe(args.slots, args.stage_tokenizer, args.runtime_archive, args.work)
    write_once(args.output, result)
    print(json.dumps({key: result[key] for key in ("all_33_restored_exact", "legacy_mismatch_count",
          "legacy_tokenizer_paths_equal", "synthetic_real_tokenizer_roundtrip", "elapsed_seconds", "model_calls")}))
