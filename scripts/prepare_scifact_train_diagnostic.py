"""CPU-only eligible TRAIN preparation; no dev files, weights or scheduling."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
from collections import Counter
from pathlib import Path

from climate_rag.agent_v3 import V3Budget
from climate_rag.io import write_json, write_jsonl
from climate_rag.public_v2 import file_sha256
from climate_rag.scifact_grounding import parse_abstract, parse_gold
from climate_rag.scifact_retrieval import SciFactBM25
from climate_rag.scifact_train_diagnostic import (
    ROUTES, SALT, classify_opportunity, packing_snapshot, rank_key,
    select_component_distinct,
)

from smoke_agent_v3_tokenizer import TOKENIZER_HASHES

PREPARATION_SHA = "3cd6bc1e1c299ece9195098a3853ebb05bb3401507d8b467c1596ba6afb6e138"
RELEASE = "climate-scifact-train-diagnostic-20260930-r1"
ALLOWLIST = ("inference/corpus.jsonl", "inference/claims_train_eligible.jsonl",
             "private-group-assignment.json", "gold/claims_train.jsonl")


def require_clean_source(root):
    if subprocess.check_output(["git", "status", "--porcelain", "--untracked-files=all"], cwd=root):
        raise ValueError("freeze tracked and untracked source before reading train gold")
    for name in ("scripts/prepare_scifact_train_diagnostic.py",
                 "src/climate_rag/scifact_train_diagnostic.py",
                 "src/climate_rag/scifact_retrieval.py"):
        subprocess.run(["git", "ls-files", "--error-unmatch", name], cwd=root,
                       check=True, capture_output=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prepared", type=Path, required=True)
    parser.add_argument("--tokenizer", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if os.environ.get("PYTHONHASHSEED") != "0":
        raise ValueError("PYTHONHASHSEED=0 required before Python starts")
    root = Path(__file__).resolve().parents[1]
    require_clean_source(root)
    if args.output.exists():
        raise FileExistsError("diagnostic output already exists; no overwrite")
    if file_sha256(args.prepared / "preparation-manifest.json") != PREPARATION_SHA:
        raise ValueError("prior preparation manifest mismatch")
    prep = json.loads((args.prepared / "preparation-manifest.json").read_text())
    hashes = {}
    for relative in ALLOWLIST:
        hashes[relative] = file_sha256(args.prepared / relative)
        if hashes[relative] != prep["output_file_sha256"][relative]:
            raise ValueError("preparation input hash mismatch")
    if {p.name: file_sha256(p) for p in args.tokenizer.iterdir() if p.is_file()} != TOKENIZER_HASHES:
        raise ValueError("frozen tokenizer mismatch")

    def rows(relative):
        with (args.prepared / relative).open(encoding="utf-8") as stream:
            return [json.loads(line) for line in stream if line.strip()]

    corpus_list = [parse_abstract(r) for r in rows("inference/corpus.jsonl")]
    corpus = {d.doc_id: d for d in corpus_list}
    if len(corpus) != 5183 or len(corpus_list) != len(corpus):
        raise ValueError("unexpected original corpus")
    assignments = json.loads((args.prepared / "private-group-assignment.json").read_text())
    eligible = set(assignments["eligible_train_ids"])
    inference = rows("inference/claims_train_eligible.jsonl")
    if (len(eligible) != 531 or len(inference) != 531
            or {r["id"] for r in inference} != eligible
            or any(set(r) != {"id", "claim"} for r in inference)):
        raise ValueError("eligible train contract mismatch")
    # JSON lines must be decoded to inspect IDs; quarantined gold is never parsed,
    # scored or used for sampling. No dev/test path is opened by this entry.
    gold = [parse_gold(r, corpus) for r in rows("gold/claims_train.jsonl") if r["id"] in eligible]
    if len(gold) != 531 or {g.claim_id: g.claim for g in gold} != {r["id"]: r["claim"] for r in inference}:
        raise ValueError("eligible gold identity mismatch")
    components = {i: str(assignments["claim_component"][str(i)]) for i in eligible}
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(str(args.tokenizer), local_files_only=True,
                                             trust_remote_code=False)
    retrieve = SciFactBM25(corpus)
    audit = []
    for i, g in enumerate(sorted(gold, key=lambda g: (rank_key(g.claim_id), g.claim_id))):
        first = packing_snapshot(g.claim, retrieve, tokenizer)
        audit.append(classify_opportunity(
            g, first, lambda ids, g=g: packing_snapshot(g.claim, retrieve, tokenizer, ids)))
        if (i + 1) % 50 == 0:
            print(json.dumps({"cpu_claims_audited": i + 1, "model_calls": 0}), flush=True)
    selected, selection = select_component_distinct(audit, components)
    lookup = {g.claim_id: g for g in gold}
    ids = [r["id"] for r in selected]
    args.output.mkdir(parents=True)
    (args.output / "inference").mkdir()
    (args.output / "private").mkdir()
    # Corpus is copied mechanically; no selected gold sentence injection.
    import shutil
    shutil.copyfile(args.prepared / "inference/corpus.jsonl", args.output / "inference/corpus.jsonl")
    write_jsonl(args.output / "inference/claims.jsonl", [lookup[i].inference_row() for i in ids])
    write_jsonl(args.output / "private/gold.jsonl", [lookup[i].gold_row() for i in ids])
    write_json(args.output / "private/sampling-audit.json", audit)
    write_json(args.output / "private/selected-strata.json", selected)
    protocol = {"release_id": RELEASE, "scope": "biased_eligible_train_diagnostic_not_dev_test",
                "routes": list(ROUTES), "budget": V3Budget().model_dump(),
                "corpus_documents": 5183, "claims": len(ids), "seed_salt": SALT,
                "slots": [{"claim_id": i, "route": r} for i in ids for r in ROUTES],
                "inference_file_sha256": {p.name: file_sha256(p)
                                          for p in sorted((args.output / "inference").iterdir())},
                "gpu_submission_authorized": False, "official_dev_read": False,
                "model_calls": 0, "pythonhashseed": "0"}
    write_json(args.output / "inference/protocol.json", protocol)
    manifest = {"schema_version": "scifact-train-diagnostic-preparation-v1",
                "release_id": RELEASE, "source_git": subprocess.check_output(
                    ["git", "rev-parse", "HEAD"], cwd=root, text=True).strip(),
                "source_worktree_dirty": False, "preparation_sha256": PREPARATION_SHA,
                "input_sha256": hashes, "tokenizer_sha256": TOKENIZER_HASHES,
                "audited_eligible_train": len(audit), "selected_queries": len(ids),
                "slots": 4 * len(ids), "model_generations": 0,
                "official_dev_read": False, "test_read": False,
                "inference_claim_fields": ["id", "claim"],
                "candidate_counts": dict(Counter(r["stratum"] or r["excluded_reason"] for r in audit)),
                **selection,
                "output_sha256": {p.relative_to(args.output).as_posix(): file_sha256(p)
                                  for p in sorted(args.output.rglob("*")) if p.is_file()},
                "boundary": "Gold-stratified train opportunity sampling; scripted CPU packing only, no benchmark quality."}
    write_json(args.output / "manifest.json", manifest)
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
