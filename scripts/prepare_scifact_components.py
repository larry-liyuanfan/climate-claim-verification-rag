"""Bounded private Stage A preparation; tokenizer only, no model or scheduler.

Only frozen selected gold, corpus and the two completed run identities are read.
No remaining TRAIN population, official dev or test selection is implemented.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import tarfile
from pathlib import Path
from typing import Any

from climate_rag.scifact_component_contract import protocol, require
from climate_rag.scifact_component_preparation import coverage, prepare_claim
from climate_rag.scifact_semantic_contract import (
    CORPUS_SHA, TOKENIZER_SHA, checked, encoded, sha, write_once,
)

ROOT = Path("/data/gpfs/projects/punim2936/portfolio_20260903/climate-public-retrieval-v2")
RELEASE = "climate-scifact-semantic-pair-20260930-v1"
OLD_SOURCE = "99cd9ff707697ea395a9bc067f3ce91395071cdb"
OLD_PROTOCOL_SHA = "5595143cfb6d276184e91f40856efa388ab440dc78527fe7fcffd52c12bb296d"
RUNS = {
    "scifact-gap-original-v1": "557b7b8ddc7f4e40c1ac1abbe2abd68af37536427f3b106e9e0063e3f6ae746a",
    "scifact-gap-semantic-policy-v1": "486f1c5866011e926b47fe2848189ce4c121213ab7fc86fc22a2c18ea68403f8",
}
SCORE_SHA = "5cee393c8d5c671def659885d12d3f663465cc633e4f6d0eab83c06fde07fa73"
BUNDLES = {"inference": "def10d4030d402ef5294680e0426ba1ed05ac42cb9b1d7bf4e8daa0a328961c9",
           "scoring": "7e2303364b69be920de017179485bbb6f9d0e5d48367717576b3411c92c8ca8e"}


def member(bundle: tarfile.TarFile, name: str) -> bytes:
    found = [m for m in bundle.getmembers() if m.name == name]
    require(len(found) == 1 and found[0].isfile() and found[0].size < 40_000_000, "archive_member")
    handle = bundle.extractfile(found[0])
    require(handle is not None, "archive_stream")
    assert handle is not None
    return handle.read()


def prepare(output: Path, tokenizer_dir: Path, source_git: str) -> dict[str, Any]:
    require(output.parent.resolve() == (ROOT / "posthoc").resolve() and
            output.name == "scifact-component-preparation-" + source_git[:12], "output_scope")
    require(len(source_git) == 40 and all(c in "0123456789abcdef" for c in source_git), "source_git")
    source = Path(__file__).resolve().parents[1]
    require((source / "SOURCE_REVISION").read_text().strip() == source_git, "source_revision")
    require(not output.exists(), "exclusive_preparation")
    bundle_dir = ROOT / "envs/scifact-semantic-bundles-99cd9ff"
    for key, expected in BUNDLES.items():
        checked(bundle_dir / (key + ".tar"), expected)
    with tarfile.open(bundle_dir / "inference.tar") as bundle:
        corpus_raw = member(bundle, "corpus.jsonl")
        claims_raw = member(bundle, "claims.jsonl")
        old_protocol_raw = member(bundle, "protocol.json")
    require(sha(corpus_raw) == CORPUS_SHA, "corpus_hash")
    require(sha(old_protocol_raw) == OLD_PROTOCOL_SHA, "old_protocol_hash")
    old_protocol = json.loads(old_protocol_raw)
    claims = [json.loads(line) for line in claims_raw.splitlines()]
    ids = [c["id"] for c in claims]
    require(len(ids) == len(set(ids)) == 12 and ids == old_protocol["ordered_claim_ids"], "selected_claims")
    corpus = {r["doc_id"]: r for r in (json.loads(line) for line in corpus_raw.splitlines())}
    require(len(corpus) == 5183, "corpus_count")
    with tarfile.open(bundle_dir / "scoring.tar") as bundle:
        gold_raw = member(bundle, "gold.jsonl")
    gold_rows = [json.loads(line) for line in gold_raw.splitlines()]
    require([g["id"] for g in gold_rows] == ids and len(gold_rows) == 12, "selected_gold_order")
    require(sum(bool(g["evidence"]) for g in gold_rows) == 9, "nine_evidence_three_nei")
    require(all(c["claim"] == g["claim"] for c, g in zip(claims, gold_rows, strict=True)), "selected_claim_text")
    runs = {}
    for policy, expected in RUNS.items():
        raw = checked(ROOT / "runs" / (RELEASE + "-" + policy) / "inference/run.json", expected)
        run = json.loads(raw)
        require(run["source_git"] == OLD_SOURCE, "old_run_git")
        rows = [r for r in run["runs"] if r["route"] == "fixed_rerank"]
        require(len(rows) == 12 and [r["claim_id"] for r in rows] == ids, "fixed_route_identity")
        require(all(r["arm"] == policy and r["source_git"] == OLD_SOURCE and
                    r["result"]["arm"] == policy and r["result"]["source_git"] == OLD_SOURCE and
                    r["result"]["route"] == "fixed_rerank" and r["result"]["claim_id"] == r["claim_id"]
                    for r in rows), "frozen_row_source_binding")
        runs[policy] = rows
    checked(ROOT / "runs" / (RELEASE + "-scifact-gap-semantic-policy-v1") / "score.json", SCORE_SHA)
    for name, expected in TOKENIZER_SHA.items():
        checked(tokenizer_dir / name, expected)
    os.environ.update(USE_TORCH="0", USE_TF="0", HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1",
                      HF_HUB_DISABLE_TELEMETRY="1", TOKENIZERS_PARALLELISM="false")
    require("torch" not in sys.modules, "weights_runtime_must_not_be_imported")
    from transformers.models.auto.tokenization_auto import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(str(tokenizer_dir), local_files_only=True, trust_remote_code=False)
    rows = []
    for gold, run_row in zip(gold_rows, runs["scifact-gap-original-v1"], strict=True):
        rows.extend(prepare_claim(gold, run_row["result"], corpus, tokenizer))
    require(len(rows) == 33, "planned_matrix")
    require("torch" not in sys.modules, "unexpected_torch_import")
    output.mkdir(mode=0o700)
    inference, scoring = output / "inference", output / "scoring"
    inference.mkdir(mode=0o700)
    scoring.mkdir(mode=0o700)
    # Directory separation is dataflow isolation, not a same-user OS sandbox.
    input_rows = [{"slot": i, "input": r["input"], "packing": r["packing"]} for i, r in enumerate(rows)]
    targets = [{"slot": i, "claim_id": r["claim_id"], "component": r["component"], "target": r["target"],
                "preparation_status": r["packing"]["status"]} for i, r in enumerate(rows)]
    write_once(inference / "slots.json", input_rows)
    write_once(scoring / "targets.json", targets)
    frozen = protocol() | {"source_git": source_git, "old_source_git": OLD_SOURCE,
        "run_sha256": RUNS, "score_sha256": SCORE_SHA, "bundle_sha256": BUNDLES,
        "corpus_sha256": CORPUS_SHA, "selected_gold_sha256": sha(gold_raw),
        "selected_claims_sha256": sha(claims_raw), "inference_slots_sha256": sha(encoded(input_rows)),
        "scoring_targets_sha256": sha(encoded(targets)), "coverage": coverage(rows)}
    write_once(output / "protocol.json", frozen)
    compact = {"status": "CPU_prepared_not_model_evaluated", "source_git": source_git,
               "protocol_sha256": sha(encoded(frozen)), "coverage": frozen["coverage"],
               "input_sha256": frozen["inference_slots_sha256"], "scoring_sha256": frozen["scoring_targets_sha256"],
               "run_sha256": RUNS, "score_sha256": SCORE_SHA, "model_calls": 0,
               "official_dev_read": False, "remaining_train_read": False, "gpu_authorized": False}
    write_once(output / "compact.json", compact)
    return {"status": compact["status"], "compact_sha256": sha(encoded(compact)), "coverage": frozen["coverage"]}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--tokenizer-dir", type=Path, required=True)
    parser.add_argument("--source-git", required=True)
    args = parser.parse_args()
    print(json.dumps(prepare(args.output, args.tokenizer_dir, args.source_git), sort_keys=True))


if __name__ == "__main__":
    main()
