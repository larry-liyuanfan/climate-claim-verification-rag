"""Mechanical selected-array conversion, not new sampling or gold analysis.

Run only in the Climate private project. Stdlib only; no model, index or probe.
Public receipt is hashes/counts only. Full private source directory is never packed.
"""
from __future__ import annotations

import argparse
import io
import json
import tarfile
from pathlib import Path
from typing import Any

from climate_rag.scifact_semantic_contract import (
    BUDGET, COMPACT_SHA, CORPUS_MEMBER, CORPUS_SHA, INPUT_SHA, MODEL_SHA,
    PAIR, POLICIES, PREPARATION_GIT, PROMPTS, RELEASE, RERANKER_SHA, ROUTES,
    checked, encoded, jsonl, sha, validate_protocol, write_once,
)


def package(prepared: Path, input_archive: Path, output: Path, git: str) -> dict[str, Any]:
    compact = json.loads(checked(prepared / "compact.json", COMPACT_SHA))
    raw = {n: checked(prepared / "private" / n, s) for n, s in compact["private_output_sha256"].items()}
    private = {n: json.loads(b) for n, b in raw.items()}
    selection = private["selected-before-probe.json"]["selection"]
    claims = private["selected-inference.json"]
    gold = private["selected-gold.json"]
    reviews = private["current-opportunity.json"]
    draft = private["future-pair-protocol-draft.json"]
    ledger = private["consumption-ledger.json"]
    ids = [r["id"] for r in selection]
    if (compact["source_git"] != PREPARATION_GIT or compact["prompt_sha256"] != PROMPTS
            or any([r["id"] for r in rows] != ids for rows in (claims, gold, reviews))
            or draft["ordered_claim_ids"] != ids or draft["prompt_sha256"] != PROMPTS
            or draft["budget"] != BUDGET or len({r["component"] for r in selection}) != 12
            or set(ids) & set(ledger["component_excluded_eligible_ids"])
            or {r["component"] for r in selection} & set(ledger["component_excluded"])):
        raise ValueError("prepared_selection_join")
    # The source archive bytes are checked once and used directly; no unchecked reread.
    archive_bytes = checked(input_archive, INPUT_SHA)
    with tarfile.open(fileobj=io.BytesIO(archive_bytes)) as archive:
        members = [m for m in archive.getmembers() if m.name == CORPUS_MEMBER]
        if len(members) != 1 or not members[0].isfile():
            raise ValueError("corpus_archive_member")
        stream = archive.extractfile(members[0])
        if stream is None:
            raise ValueError("missing_corpus_stream")
        corpus = stream.read()
    if sha(corpus) != CORPUS_SHA:
        raise ValueError("corpus_identity")
    claim_bytes = jsonl(claims)
    protocol = {
        "release_id": RELEASE, "comparison_protocol": PAIR,
        "preparation_source_git": PREPARATION_GIT, "execution_source_git": git,
        "preparation_compact_sha256": COMPACT_SHA,
        "scope": draft["scope"], "ordered_claim_ids": ids, "policies": list(POLICIES),
        "routes": list(ROUTES), "budget": BUDGET, "policy_prompt_sha256": PROMPTS,
        "tokenizer_sha256": compact["tokenizer_sha256"],
        "model_manifest_sha256": MODEL_SHA, "reranker_manifest_sha256": RERANKER_SHA,
        "slots_per_arm": [{"claim_id": i, "route": r} for i in ids for r in ROUTES],
        "paired_slots": [{"claim_id": i, "route": r, "policy": a} for a in POLICIES for i in ids for r in ROUTES],
        "inference_file_sha256": {"claims.jsonl": sha(claim_bytes), "corpus.jsonl": CORPUS_SHA},
        "corpus_documents": 5183, "model_calls": 0, "official_dev_read": False,
        "fresh_baseline_required": True, "old_G_results_reusable": False,
        "gpu_authorized": False, "release_requires_external_coordinator_authorization": True,
        "preflight_calls_per_arm": 4, "decoding": draft["decoding"], "packing": draft["packing"],
    }
    validate_protocol(protocol, claims, git)
    inference = {"protocol.json": encoded(protocol), "corpus.jsonl": corpus, "claims.jsonl": claim_bytes}
    scoring = {
        "gold.jsonl": jsonl(gold), "selected-strata.json": encoded(selection),
        "current-opportunity.json": raw["current-opportunity.json"],
        "consumption-ledger.json": raw["consumption-ledger.json"],
        "selection-freeze.json": raw["selected-before-probe.json"],
        "preparation-draft.json": raw["future-pair-protocol-draft.json"],
    }
    manifest = {"execution_source_git": git, "preparation_source_git": PREPARATION_GIT,
                "protocol_sha256": sha(inference["protocol.json"]),
                "inference_file_sha256": protocol["inference_file_sha256"],
                "scoring_file_sha256": {n: sha(b) for n, b in scoring.items()},
                "preparation_private_sha256": compact["private_output_sha256"]}
    scoring["manifest.json"] = encoded(manifest)
    output.mkdir(mode=0o700)  # exclusive: no overwrite, rerun, reselection or append
    receipts = {}
    for name, payloads in (("inference", inference), ("scoring", scoring)):
        path = output / f"{name}.tar"
        with tarfile.open(path, "x") as archive:
            for member, payload in sorted(payloads.items()):
                info = tarfile.TarInfo(member)
                info.size, info.mode, info.mtime = len(payload), 0o400, 0
                archive.addfile(info, io.BytesIO(payload))
        receipts[name] = {"sha256": sha(path.read_bytes()), "bytes": path.stat().st_size,
                          "members": sorted(payloads)}
    receipt = {"schema_version": "scifact-semantic-bundles-v1", "release_id": RELEASE,
        "preparation_source_git": PREPARATION_GIT, "execution_source_git": git,
        "preparation_compact_sha256": COMPACT_SHA, "protocol_sha256": manifest["protocol_sha256"],
        "scoring_manifest_sha256": sha(scoring["manifest.json"]), "bundles": receipts,
        "policy_prompt_sha256": PROMPTS, "selected_queries": len(ids), "slots_per_arm": 48,
        "model_calls": 0, "gpu_authorized": False, "reselected": False}
    write_once(output / "bundle-receipt.json", receipt)
    return receipt


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    for name in ("prepared", "input-archive", "output"):
        p.add_argument("--" + name, type=Path, required=True)
    p.add_argument("--execution-git", required=True)
    args = p.parse_args()
    source = Path(__file__).resolve().parents[1]
    if (source / "SOURCE_REVISION").read_text().strip() != args.execution_git:
        raise ValueError("frozen_packaging_source_required")
    print(json.dumps(package(args.prepared, args.input_archive, args.output, args.execution_git)))


if __name__ == "__main__":
    main()
