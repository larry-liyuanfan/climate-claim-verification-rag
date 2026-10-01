"""Bound actual frozen views with synthetic verdict envelopes; tokenizer only."""
from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path
from typing import Any

from climate_rag.scifact_evidence_commit import CommitState, PROTOCOL, PROTOCOLS, render_prompt
from climate_rag.scifact_natural_contract import require
from climate_rag.scifact_read_continuation import ordered_write
from climate_rag.scifact_semantic_contract import TOKENIZER_SHA, checked
from climate_rag.scifact_utility_contract import identity
from run_scifact_evidence_commit_operator import FROZEN_FIELDS
import scifact_evidence_input as prospective_inputs


def probe(frame: Any, tokenizer: Any, *, protocol: str = PROTOCOL) -> dict[str, Any]:
    require(protocol in PROTOCOLS, 'unknown_commit_protocol')
    counts: list[dict[str, Any]] = []
    def measure(state: CommitState, stage: str, doc: str | None, available: list[str]) -> None:
        obs, schema = state.inputs(stage, doc, available)
        counts.append({"stage":stage, "feedback_count":len(state.feedback),
            "ref_count":len(state.registry.records()), "selection_count":len(state.registry.catalog()),
            "tokens":len(tokenizer.encode(render_prompt(tokenizer, obs, schema), add_special_tokens=False))})
    state = CommitState(1, "adaptive", frame, "synthetic-prompt-probe:"+"a"*64, protocol=protocol)
    measure(state, "plan", None, state.order)
    for doc in state.order:
        measure(state, "verify", doc, [])
    # No scientific labels are inferred here. Stress maximal citation strings
    # and failed-call feedback across the maximum two adaptive verifications.
    for status in ("valid", "failed"):
        state = CommitState(1, "adaptive", frame, "synthetic-prompt-probe:"+"a"*64, protocol=protocol)
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
    for name in ("tokenizer", "output"):
        parser.add_argument("--"+name, type=Path, required=True)
    for name in ("inputs", "inventory", "prepared", "release"):
        parser.add_argument("--"+name, type=Path)
    args = parser.parse_args()
    for name, digest in TOKENIZER_SHA.items():
        checked(args.tokenizer/name, digest)
    import transformers
    tokenizer = getattr(transformers, "AutoTokenizer").from_pretrained(args.tokenizer, local_files_only=True)
    if args.prepared is not None:
        require(args.release is not None and args.inputs is None and args.inventory is None, 'one_input_protocol_only')
        release = json.loads(args.release.read_bytes())
        require(prospective_inputs.prospective(release), 'prospective_preflight_release')
        receipt, claims = prospective_inputs.check_prepared(args.prepared, release)
        from climate_rag.scifact_grounding import parse_abstract
        from climate_rag.scifact_semantic_contract import CORPUS_SHA
        corpus = {d.doc_id:d for d in (parse_abstract(json.loads(line)) for line in
            checked(args.prepared/'inference/corpus.jsonl', CORPUS_SHA).splitlines())}
        frames = prospective_inputs.load_frames(args.prepared, claims, corpus, tokenizer, release)
        protocol = release['protocol']
        rows = [probe(frame, tokenizer, protocol=protocol) for frame in frames]
        report = {'protocol':protocol, 'input_protocol':release['input_protocol'], 'claims':len(rows),
            'preparation_sha256':release['preparation_sha256'], 'rows':rows,
            'maximum_tokens':max(r['maximum_tokens'] for r in rows),
            'overflow_count':sum(r['overflow'] for r in rows), 'model_calls':0, 'gold_read':False,
            'shared_preparation_cost':receipt['shared_preparation_cost'],
            'probe_scope':'first_two_refs_fixed_synthetic_feedback_not_all_reachable_states'}
        ordered_write(args.output, report)
        print(json.dumps({k:v for k,v in report.items() if k != 'rows'}))
        return
    require(args.inputs is not None and args.inventory is not None and args.release is None, 'old_inventory_required')
    inventory = json.loads(args.inventory.read_bytes())
    require(identity(inventory) == FROZEN_FIELDS["initial_inventory_sha256"], "frozen_inventory_required")
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
