"""Bound actual frozen views with synthetic verdict envelopes; tokenizer only."""
from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path
from typing import Any

from climate_rag.scifact_evidence_commit import CommitState, PROTOCOL, render_prompt
from climate_rag.scifact_natural_contract import require
from climate_rag.scifact_read_continuation import ordered_write
from climate_rag.scifact_semantic_contract import TOKENIZER_SHA, checked
from climate_rag.scifact_utility_contract import identity
from run_scifact_evidence_commit_operator import FROZEN_FIELDS


def probe(frame: Any, tokenizer: Any) -> dict[str, Any]:
    counts: list[dict[str, Any]] = []
    def measure(state: CommitState, stage: str, doc: str | None, available: list[str]) -> None:
        obs, schema = state.inputs(stage, doc, available)
        counts.append({"stage":stage, "feedback_count":len(state.feedback),
            "ref_count":len(state.registry.records()), "selection_count":len(state.registry.catalog()),
            "tokens":len(tokenizer.encode(render_prompt(tokenizer, obs, schema), add_special_tokens=False))})
    state = CommitState(1, "adaptive", frame, "synthetic-prompt-probe:"+"a"*64)
    measure(state, "plan", None, state.order)
    for doc in state.order:
        measure(state, "verify", doc, [])
    # No scientific labels are inferred here. Stress maximal citation strings
    # and failed-call feedback across the maximum two adaptive verifications.
    for status in ("valid", "failed"):
        state = CommitState(1, "adaptive", frame, "synthetic-prompt-probe:"+"a"*64)
        for index, doc in enumerate(state.order[:2]):
            ids = [s for s in frame["visible"] if s.rsplit(":",1)[0] == doc][:8]
            state.accept({"status":status, "physical_attempt_id":f"g{index}",
                "decision":{"source_id":doc,"label":"REFUTES","sentence_ids":list(reversed(ids))},
                "usage":{"input_tokens":8192,"output_tokens":512},
                "failure":"ModelResponseValidationError" if status == "failed" else None}, doc, "b"*64)
            state.calls = 2*(index+1)
            measure(state, "plan", None, state.order[index+1:] if index == 0 else [])
    maximum = max(r["tokens"] for r in counts)
    return {"input_sha256":identity(frame), "probes":counts, "maximum_tokens":maximum, "overflow":maximum > 8192}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("inputs", "inventory", "tokenizer", "output"):
        parser.add_argument("--"+name, type=Path, required=True)
    args = parser.parse_args()
    inventory = json.loads(args.inventory.read_bytes())
    require(identity(inventory) == FROZEN_FIELDS["initial_inventory_sha256"], "frozen_inventory_required")
    for name, digest in TOKENIZER_SHA.items():
        checked(args.tokenizer/name, digest)
    import transformers
    tokenizer = getattr(transformers, "AutoTokenizer").from_pretrained(args.tokenizer, local_files_only=True)
    rows = []
    for claim_id in inventory["ordered_claim_ids"]:
        initial = f"inference/{claim_id}-A/initial-frame.json"
        prefix_path = f"inference/{claim_id}-A/prefix.json"
        frame = copy.deepcopy(json.loads(checked(args.inputs/initial, inventory["files"][initial]))["frame"])
        prefix = json.loads(checked(args.inputs/prefix_path, inventory["files"][prefix_path]))
        require(frame == prefix["payload"]["frame"], "actual_frozen_frame")
        aliases = {s.rsplit(":",1)[0] for s in frame["visible"]}
        frame["document_order"] = [d for d in prefix["payload"]["candidates"] if d in aliases]
        rows.append({"claim_id":claim_id, **probe(frame, tokenizer)})
    require(len(rows) == 24, "all24_required")
    report = {"protocol":PROTOCOL, "inventory_sha256":identity(inventory), "claims":len(rows),
        "maximum_tokens":max(r["maximum_tokens"] for r in rows),
        "overflow_claims":[r["claim_id"] for r in rows if r["overflow"]], "rows":rows,
        "model_calls":0, "gold_read":False, "synthetic_feedback_not_predictions":True}
    ordered_write(args.output, report)
    print(json.dumps({k:v for k,v in report.items() if k != "rows"}))


if __name__ == "__main__":
    main()
