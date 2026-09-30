"""Released GPU worker: frozen inputs, four synthetic preflights, 33 slots once."""
from __future__ import annotations

import argparse
import os
import time
from pathlib import Path
from typing import Any

from climate_rag.component_execution import (
    INFRASTRUCTURE_LINEAGE, PREPARATION_GIT, RELEASE, SLOTS_SHA, durable, execute_matrix, frozen_slots,
)
from climate_rag.component_preflight import preflight_cases
from climate_rag.local_component_provider import LocalQwenComponentProvider
from climate_rag.scifact_component_contract import packing, require
from climate_rag.scifact_semantic_contract import MODEL_SHA
from run_budget_agent_full_operator import ROOT
from run_scifact_semantic_arm import verify_manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("slots", "model-dir", "model-manifest"):
        parser.add_argument("--" + name, type=Path, required=True)
    args = parser.parse_args()
    require(os.name == "posix" and bool(os.environ.get("SLURM_JOB_ID")) and
            bool(os.environ.get("CUDA_VISIBLE_DEVICES")) and os.environ.get("CLIMATE_COMPONENT_RELEASE") == RELEASE,
            "separate_allocated_release_required")
    source = Path(__file__).resolve().parents[1]
    require((source / "SOURCE_REVISION").read_text().strip() == os.environ["CLIMATE_SOURCE_GIT"], "execution_source_identity")
    require(all(os.environ.get(k) == "1" for k in ("HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE"))
            and os.environ.get("PYTHONHASHSEED") == "0", "offline_deterministic_runtime")
    rows = frozen_slots(args.slots)
    manifest = verify_manifest(args.model_manifest, args.model_dir, MODEL_SHA)
    from transformers.models.auto.tokenization_auto import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(str(args.model_dir), local_files_only=True, trust_remote_code=False)
    for row in rows:
        if row["packing"]["status"] == "prepared":
            require(packing(tokenizer, row["input"]) == row["packing"], "frozen_input_packing")
    preflight: list[dict[str, Any]] = [{"slot": i, "input": spec, "packing": packing(tokenizer, spec)}
                 for i, (spec, _) in enumerate(preflight_cases())]
    require(len({s["packing"]["input_sha256"] for s in preflight}) == 4, "distinct_preflight_inputs")
    result = ROOT / "runs" / RELEASE
    durable(result / "worker-identity.json", {"release": RELEASE, "source_git": os.environ["CLIMATE_SOURCE_GIT"],
        "source_archive_sha256": os.environ["CLIMATE_SOURCE_SHA256"], "preparation_git": PREPARATION_GIT,
        "slots_sha256": SLOTS_SHA, "model_manifest_sha256": MODEL_SHA, "preflight": preflight,
        "infrastructure_lineage": INFRASTRUCTURE_LINEAGE,
        "scoring_targets_loaded": False, "started_unix": time.time()})
    private = result / "provider-load-private"
    private.mkdir(mode=0o700)
    begin = time.perf_counter()
    provider = LocalQwenComponentProvider(args.model_dir, manifest, private_dir=private)
    model_load_ms = (time.perf_counter() - begin) * 1000
    report = execute_matrix(rows, provider, result / "inference")
    durable(result / "worker-finished.json", {"status": report["status"], "model_load_ms": model_load_ms,
        "gpu_peak_allocated_bytes": provider.base._torch.cuda.max_memory_allocated(),
        "model_calls_upper_bound": 37, "scoring_targets_loaded": False})


if __name__ == "__main__":
    main()
