"""Read-only real carry/packing audit plus tiny POSIX checks; no model loading."""
from __future__ import annotations

import argparse
import os
import signal
import stat
import sys
import threading
from pathlib import Path
from typing import Any

from climate_rag.component_continuation import PREVIOUS_RELEASE, load_carried, measurement, policy_identity
from climate_rag.component_execution import durable, frozen_slots
from climate_rag.component_preflight import preflight_cases
from climate_rag.private_diagnostics_v3 import PrivateDiagnosticStore
from climate_rag.scifact_component_contract import packing, require
from climate_rag.scifact_semantic_contract import TOKENIZER_SHA, checked, sha
from run_budget_agent_full_operator import ROOT
from run_scifact_component_operator import inference_exit


def posix_probe(work: Path) -> dict[str, Any]:
    require(os.name == "posix", "requires_real_posix_not_emulation")
    work.mkdir(mode=0o700)
    private = work / "private"
    private.mkdir(mode=0o700)
    sink = PrivateDiagnosticStore(private).sink("response", 64)
    sink.write("explicit synthetic probe")
    require(stat.S_IMODE(sink.path.stat().st_mode) == 0o600, "owner_only")
    (private / "link").symlink_to(work / "nonexistent")
    refused = PrivateDiagnosticStore(private).sink("response", 64)
    refused.write("must fail closed")
    require(refused.io_failed and refused.stored == 0, "symlink_refused")
    durable(work / "directory-fsync.json", {"synthetic": True})
    normal = inference_exit([sys.executable, "-c", "pass"], cwd=work, env=dict(os.environ), log=work / "normal.log")
    require(normal["child_reaped"] and normal["returncode"] == 0, "normal_reap")
    original = signal.signal(signal.SIGINT, signal.default_int_handler)
    timer = threading.Timer(0.3, lambda: os.kill(os.getpid(), signal.SIGINT))
    timer.start()
    try:
        interrupted = inference_exit([sys.executable, "-c", "import time; time.sleep(30)"],
            cwd=work, env=dict(os.environ), log=work / "interrupted.log")
    finally:
        timer.cancel()
        timer.join()
        signal.signal(signal.SIGINT, original)
    require(interrupted["child_reaped"] and interrupted["returncode"] == -signal.SIGTERM
        and interrupted["worker_failure"] == "KeyboardInterrupt", "interrupted_reap")
    return {"owner_only": True, "symlink_refused": True, "directory_fsync": True,
            "normal_exit": normal, "interrupted_exit": interrupted}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    for key in ("tokenizer", "work", "output"):
        parser.add_argument("--" + key, type=Path, required=True)
    args = parser.parse_args()
    require(os.environ.get("USE_TORCH") == "0" and os.environ.get("HF_HUB_OFFLINE") == "1", "tokenizer_only_offline")
    for name, digest in TOKENIZER_SHA.items():
        checked(args.tokenizer / name, digest)
    from transformers.models.auto.tokenization_auto import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(str(args.tokenizer), local_files_only=True, trust_remote_code=False)
    prior = ROOT / "runs" / PREVIOUS_RELEASE
    carried = load_carried(prior, tokenizer)
    slots = frozen_slots(ROOT / "posthoc/scifact-component-preparation-426ff7343fb3/inference/slots.json")
    require(all(packing(tokenizer, row["input"]) == row["packing"] for row in slots), "all_frozen_packings")
    preflight = [packing(tokenizer, spec) for spec, _ in preflight_cases()]
    first = measurement(carried["record"], 0, carried=True)
    report = {"schema": "component-continuation-cpu-probe-v1", "python": sys.version.split()[0],
        "policy": policy_identity(), "source_git": (Path(__file__).resolve().parents[1] / "SOURCE_REVISION").read_text().strip(),
        "carried_reference": carried["reference"], "carried_usage": first["usage"],
        "carried_technical_ready": first["technical_ready"], "carried_semantic_match": first["semantic_match"],
        "matched_real_packing": len(slots), "preflight_packing": preflight, "posix": posix_probe(args.work),
        "scoring_targets_read": False, "new_model_calls": 0, "model_weights_loaded": False,
        "old_artifacts_written": False, "raw_exported": False}
    durable(args.output, report)
    print({"report_sha256": sha(args.output.read_bytes()), "matched_real_packing": len(slots), "new_model_calls": 0})


if __name__ == "__main__":
    main()
