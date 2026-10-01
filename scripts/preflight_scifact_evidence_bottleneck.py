"""One allocated tokenizer-only check on consumed TRAIN24, no weights/gold."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
from typing import Any

from climate_rag.scifact_evidence_bottleneck import PROTOCOL, inputs, parse_selection, render_prompt, source_rows
from climate_rag.scifact_grounding import parse_abstract
from climate_rag.scifact_natural_contract import require
from climate_rag.scifact_read_continuation import ordered_write
from climate_rag.scifact_relation_verifier import RELATION_PROTOCOL
from climate_rag.scifact_semantic_contract import CORPUS_SHA, TOKENIZER_SHA, checked
from climate_rag.scifact_utility_contract import identity
from preflight_scifact_relation_verifier import synthetic_assessment
from run_scifact_evidence_commit_operator import release_fields
from run_scifact_grounding_train_operator import ROOT
import scifact_evidence_input as adapter


def probe(frame: Any, tokenizer: Any) -> dict[str, Any]:
    counts: dict[str, list[int]] = {s: [] for s in ("A", "selector", "B", "C")}
    doc = frame["document_order"][0]
    rows = source_rows(frame, doc)
    for s in ("A", "selector"):
        obs, schema = inputs(frame, s)
        counts[s].append(len(tokenizer.encode(render_prompt(tokenizer, obs, schema), add_special_tokens=False)))
    # Gold-free prompt stress, not predicted selections: all singleton IDs,
    # maximal original-order selection and longest text/ID subsets, no inference.
    choices = [[r["sentence_id"]] for r in rows]
    for key in (lambda r: len(tokenizer.encode(r["text"])), lambda r: len(r["sentence_id"])):
        choices.append([r["sentence_id"] for r in sorted(rows, key=key, reverse=True)[:8]])
    max_selector_response = 0
    for ids in choices:
        selection = parse_selection({"source_id": doc, "sentence_ids": ids}, frame, doc)
        max_selector_response = max(max_selector_response, len(tokenizer.encode(json.dumps(selection, separators=(",", ":"))))+1)
        b, bs = inputs(frame, "B", selection)
        c, cs = inputs(frame, "C", selection)
        require(bs == cs and {k for k in b if b[k] != c[k]} == {"full_document"}, "sole_visibility_intervention")
        for s, obs, schema in (("B", b, bs), ("C", c, cs)):
            counts[s].append(len(tokenizer.encode(render_prompt(tokenizer, obs, schema), add_special_tokens=False)))
    max_a_response = max(len(tokenizer.encode(json.dumps(synthetic_assessment(frame, doc, label), separators=(",", ":"))))+1
                         for label in ("SUPPORTS", "REFUTES", "INSUFFICIENT"))
    max_label_response = max(len(tokenizer.encode(json.dumps({"source_id": doc, "label": label}, separators=(",", ":"))))+1
                             for label in ("SUPPORTS", "REFUTES", "INSUFFICIENT"))
    # Conservative byte-level JSON envelope: replacing B's selected portion by
    # all original sentences is an upper stress case (not a legal selection).
    full_selection = parse_selection({"source_id": doc, "sentence_ids": choices[-1]}, frame, doc)
    envelope, schema = inputs(frame, "B", full_selection)
    envelope["selected_sentences"] = rows
    counts["B"].append(len(tokenizer.encode(render_prompt(tokenizer, envelope, schema), add_special_tokens=False)))
    maxima = {s: max(v) for s, v in counts.items()}
    output_max = max(max_selector_response, max_a_response, max_label_response)
    return {"maximum_prompt_tokens": maxima, "probe_counts": {s: len(v) for s, v in counts.items()},
        "maximum_synthetic_output_tokens_with_eos": output_max,
        "overflow": max(maxima.values()) > 8192 or output_max > 512,
        "scope": "singleton/longest8/max-cardinality plus all-sentences B stress; not every BPE combination"}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    require(os.name == "posix" and bool(os.environ.get("SLURM_JOB_ID"))
            and not os.environ.get("CUDA_VISIBLE_DEVICES"), "allocated_cpu_only")
    fields = release_fields({"protocol": RELATION_PROTOCOL, "input_protocol": adapter.PROTOCOL})
    prepared = ROOT / "runs/scifact-evidence-commit-prospective24-v1-20261002-confirmation-v1/prepared"
    _, claims = adapter.check_prepared(prepared, fields)
    directory = ROOT / "posthoc/scifact-read-continuation-fc4ffd61a761/tokenizer"
    for name, digest in TOKENIZER_SHA.items():
        checked(directory / name, digest)
    import transformers
    tokenizer = getattr(transformers, "AutoTokenizer").from_pretrained(directory, local_files_only=True)
    corpus = {d.doc_id: d for d in (parse_abstract(json.loads(r)) for r in
        checked(prepared / "inference/corpus.jsonl", CORPUS_SHA).splitlines())}
    frames = adapter.load_frames(prepared, claims, corpus, tokenizer, fields)
    require(len(frames) == 24, "same_consumed_train24")
    rows = [probe(f, tokenizer) for f in frames]
    source = Path(__file__).resolve().parents[1]
    git = (source / "SOURCE_REVISION").read_text().strip()
    result = {"protocol": PROTOCOL, "source_git": git, "job_id": os.environ["SLURM_JOB_ID"],
        "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(), "tokenizer_sha256": TOKENIZER_SHA,
        "frames_sha256": fields["frames_sha256"], "frames_identity": identity(frames),
        "selection_sha256": fields["selection_sha256"], "claims": 24, "rows": rows,
        "max_prompt_tokens": {s: max(r["maximum_prompt_tokens"][s] for r in rows) for s in ("A", "selector", "B", "C")},
        "max_output_tokens_with_eos": max(r["maximum_synthetic_output_tokens_with_eos"] for r in rows),
        "overflow_count": sum(r["overflow"] for r in rows), "model_calls": 0, "model_weights_loaded": False,
        "gold_read": False, "new_sampling": False, "protected_split_read": False, "training": False}
    ordered_write(args.output, result)
    print(json.dumps({k: v for k, v in result.items() if k != "rows"}))
    require(result["overflow_count"] == 0, "overflow_no_truncation_no_retry")


if __name__ == "__main__":
    main()
