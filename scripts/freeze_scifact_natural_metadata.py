"""Freeze small existing metadata only; no dataset, tokenizer or model imports."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

from climate_rag.scifact_natural_selection import PROTOCOL, digest, select_metadata

ROOT = Path("/data/gpfs/projects/punim2936/portfolio_20260903/climate-public-retrieval-v2")
PINS = {
    "roster": ("posthoc/scifact-mixed-program-cb1e76462537/roster.json",
               "4fdae46c4c41d51f4e1c120b95881316577092bc6004b6d03b836da41d44d349"),
    "old12": ("posthoc/scifact-semantic-preparation-779e49883570/private/selected-before-probe.json",
              "118d57233b1205f294d7090b6820d658f447ba09263fe8179bdf8e66497338a8"),
    "cost": ("runs/scifact-utility8-20261001-v1/cost-before-gold.json",
             "d3228e4c8cb0df7423678d5287f5fa8d5966e5213484216e6a483d29e4840ffc"),
    "exit": ("runs/scifact-utility8-20261001-v1/worker-exit.json",
             "6d102994409104b3f60387b90968dce8e79065775dc5203989e739a17a08115a"),
    "preparation": ("runs/scifact-utility8-20261001-v1/prepared/preparation.json",
                    "7b5a5626f55f4634b64049ceb7a6dbe1ddcd289b08fbe6ed04354bcb2007527c"),
    "utility8": ("runs/scifact-utility8-20261001-v1/prepared/selection.json",
                 "b1746e7931e02e5767e955966f423ee70ab799d7e34b2be4e99f88a5010e810c"),
}


def read_metadata(root: Path) -> dict[str, Any]:
    inputs = {}
    for name, (path, expected) in PINS.items():
        raw = (root / path).read_bytes()
        if hashlib.sha256(raw).hexdigest() != expected:
            raise ValueError("metadata_hash_mismatch:" + name)
        inputs[name] = json.loads(raw)
    if (inputs["cost"]["exit_proof_sha256"] != PINS["exit"][1]
            or inputs["exit"]["preparation_sha256"] != PINS["preparation"][1]
            or inputs["preparation"]["selection_sha256"] != PINS["utility8"][1]):
        raise ValueError("utility_metadata_binding_chain")
    sealed = inputs["roster"]
    if set(sealed) != {"payload", "sha256"} or digest(sealed["payload"]) != sealed["sha256"]:
        raise ValueError("FIT_roster_seal")
    payload = sealed["payload"]
    pool = [{"id": i, "component": payload["components"][str(i)]} for i in payload["claim_ids"]]
    old = [{"id": r["id"], "component": r["component"]} for r in inputs["old12"]["selection"]]
    utility = [{"id": r["id"], "component": r["component"]} for r in inputs["utility8"]["selected"]]
    if (len(pool) != 144 or len(old) != 12 or len(utility) != 8
            or len({r["component"] for r in old}) != 12 or len({r["component"] for r in utility}) != 8
            or inputs["preparation"]["ordered_ids_sha256"] != digest([r["id"] for r in utility])):
        raise ValueError("frozen_roster_counts_or_order")
    selection = select_metadata(pool, old, utility, payload["excluded_components"])
    selection["source_files_sha256"] = {k: v[1] for k, v in PINS.items()}
    return selection


def freeze(root: Path, output: Path) -> dict[str, Any]:
    # Restrict the real command to this project's metadata namespace; no overwrite/resume.
    if output.parent.resolve() != (root / "posthoc").resolve() or output.name != PROTOCOL:
        raise ValueError("fixed_metadata_output_required")
    selection = read_metadata(root)
    output.mkdir(mode=0o700)
    raw = (json.dumps(selection, ensure_ascii=False, separators=(",", ":")) + "\n").encode()
    with (output / "selection.json").open("xb") as stream:
        stream.write(raw)
    receipt = {"protocol": PROTOCOL, "selection_sha256": hashlib.sha256(raw).hexdigest(),
               "counts": selection["counts"], "caps": selection["caps"],
               "source_files_sha256": selection["source_files_sha256"],
               "gold_decoded": False, "protected_split_read": False, "gpu_authorized": False,
               "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    with (output / "receipt.json").open("x", encoding="utf-8") as stream:
        json.dump(receipt, stream, ensure_ascii=False, indent=2)
        stream.write("\n")
    return receipt


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--freeze", action="store_true")
    args = parser.parse_args()
    if args.freeze:
        print(json.dumps(freeze(ROOT, ROOT / "posthoc" / PROTOCOL)))
    else:
        value = read_metadata(ROOT)
        print(json.dumps({k: value[k] for k in ("protocol", "counts", "caps", "gpu_authorized")}))


if __name__ == "__main__":
    main()
