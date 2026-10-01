"""Tokenizer-only stress probes on frozen actual views; no labels/model calls."""
from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path
from typing import Any

from climate_rag.scifact_document_verifier import PROTOCOL, call_input, feedback_row, render_prompt
from climate_rag.scifact_read_continuation import ordered_write
from climate_rag.scifact_semantic_contract import TOKENIZER_SHA, checked
from climate_rag.scifact_utility_contract import identity
from run_scifact_grounding_train_operator import require


def probe(frame: Any, tokenizer: Any) -> dict[str, Any]:
    counts: list[dict[str, Any]] = []
    def measure(stage: str, doc: str | None, feedback: list[Any], calls: int, tools: int, available: list[str]) -> None:
        obs, schema = call_input(frame, stage, doc, feedback, calls, tools, available)
        counts.append({"stage": stage, "feedback_count": len(feedback), "source_id": doc,
            "tokens": len(tokenizer.encode(render_prompt(tokenizer, obs, schema), add_special_tokens=False))})
    order = frame["document_order"]
    measure("plan", None, [], 0, 1, order)
    for doc in order:
        measure("verify", doc, [], 0, 1, [])
    # Max-sized same-document citations and failure feedback, without assigning
    # scientific labels to claims. These are CPU-only synthetic prompt envelopes.
    for status in ("valid", "failed"):
        feedback = []
        for index, doc in enumerate(order[:4]):
            ids = [s for s in frame["visible"] if s.rsplit(":", 1)[0] == doc][:8]
            step = {"status": status, "physical_attempt_id": "g239", "usage": {"input_tokens":8192,"output_tokens":512},
                "decision": {"source_id":doc,"label":"REFUTES","sentence_ids":list(reversed(ids))},
                "failure": "ModelResponseValidationError" if status == "failed" else None}
            feedback.append(feedback_row(frame, doc, step, False))
            measure("terminal", None, feedback, index+1, index+2, [])
            if index < 2:
                measure("plan", None, feedback, 2*(index+1), index+2,
                        order[index+1:] if index == 0 else [])
    maximum = max(r["tokens"] for r in counts)
    return {"input_sha256": identity(frame), "visible_sentences": len(frame["visible"]),
            "probes": counts, "maximum_tokens": maximum, "overflow": maximum > 8192}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("inputs", "inventory", "tokenizer", "output"):
        parser.add_argument("--"+name, type=Path, required=True)
    args = parser.parse_args()
    inventory = json.loads(args.inventory.read_bytes())
    for name, digest in TOKENIZER_SHA.items():
        checked(args.tokenizer/name, digest)
    import transformers
    tokenizer = getattr(transformers, "AutoTokenizer").from_pretrained(args.tokenizer, local_files_only=True)
    rows = []
    for i in inventory["ordered_claim_ids"]:
        initial = f"inference/{i}-A/initial-frame.json"
        prefix_path = f"inference/{i}-A/prefix.json"
        frame = copy.deepcopy(json.loads(checked(args.inputs/initial, inventory["files"][initial]))["frame"])
        prefix = json.loads(checked(args.inputs/prefix_path, inventory["files"][prefix_path]))
        require(frame == prefix["payload"]["frame"], "actual_frozen_frame")
        aliases = {s.rsplit(":",1)[0] for s in frame["visible"]}
        frame["document_order"] = [d for d in prefix["payload"]["candidates"] if d in aliases]
        rows.append({"claim_id": i, **probe(frame, tokenizer)})
    require(len(rows) == 24, "all_24_prompt_probes")
    report = {"protocol": PROTOCOL, "initial_inventory_sha256": identity(inventory),
        "claim_count": len(rows), "maximum_tokens": max(r["maximum_tokens"] for r in rows),
        "overflow_claims": [r["claim_id"] for r in rows if r["overflow"]], "rows": rows,
        "model_calls": 0, "gold_read": False, "repacking": False,
        "scope": "synthetic maximal-citation/failure feedback envelopes on actual inputs; runtime retains overflow guard"}
    ordered_write(args.output, report)
    print(json.dumps({k:v for k,v in report.items() if k != "rows"}))


if __name__ == "__main__":
    main()
