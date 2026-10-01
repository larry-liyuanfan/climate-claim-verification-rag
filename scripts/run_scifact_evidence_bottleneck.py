"""Frozen CPU-testable runner. Real weights require a NEW exact-source release."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

from climate_rag.scifact_evidence_bottleneck import BottleneckJournal, BottleneckProvider, PROTOCOL, ROUTES, run_case
from climate_rag.scifact_generation import GenerationBinding, frozen_contract
from climate_rag.scifact_grounding import parse_abstract
from climate_rag.scifact_natural_contract import require
from climate_rag.scifact_read_continuation import ordered_write
from climate_rag.scifact_semantic_contract import CORPUS_SHA, MODEL_SHA, TOKENIZER_SHA, checked
from climate_rag.scifact_utility_contract import identity
from run_scifact_bounded_arm import verify_manifest
from prepare_scifact_semantic_pair import verify_source_tree
import scifact_evidence_input as adapter


def run_suite(claims: Any, frames: Any, corpus: Any, backend: Any, output: Path, release: Any,
              release_sha: str) -> list[Any]:
    require(release["protocol"] == PROTOCOL and release["max_generations"] == 96
            and release["routes"] == list(ROUTES), "bottleneck_release_contract")
    require(len(claims) == len(frames) == (24 if release["scope"] != "synthetic_fixture" else len(claims))
            and 1 <= len(claims) <= 24 and len({c["id"] for c in claims}) == len(claims)
            and all(set(c) == {"id", "claim"} and c["claim"] == f["observation"]["immutable_claim"]
                    for c, f in zip(claims, frames, strict=True)), "complete_bound_claim_matrix")
    require(release["frames_identity"] == identity(frames), "frozen_frames_identity")
    require(release["ordered_ids_sha256"] == identity([c["id"] for c in claims])
            and release["model_sha256"] == MODEL_SHA, "frozen_roster_and_model")
    output.mkdir(mode=0o700)
    ordered_write(output / "release.json", release)
    ordered_write(output / "planned.json", {"protocol": PROTOCOL, "release_sha256": release_sha,
        "slots": [{"claim_id": c["id"], "route": s} for c in claims for s in ROUTES],
        "scope": release["scope"], "max_generations": 96, "model_sha256": MODEL_SHA,
        "source_git": release["source_git"], "frames_identity": identity(frames)})
    ordered_write(output / "initial-frames.json", frames)
    journal = BottleneckJournal(backend, output / "ledger", release_sha=release_sha)
    return [run_case(c["id"], f, journal, corpus, output / str(c["id"]))
            for c, f in zip(claims, frames, strict=True)]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--release", type=Path, required=True)
    parser.add_argument("--release-sha", required=True)
    args = parser.parse_args()
    release = json.loads(checked(args.release, args.release_sha))
    require(release["protocol"] == PROTOCOL and release["status"] == "authorized_model_run"
            and release["model_execution_authorized"] is True
            and release["scope"] == adapter.SCOPE and release["generation_contract"] == frozen_contract()
            and release["model_sha256"] == MODEL_SHA, "new_exact_source_authorization_required")
    require(os.name == "posix" and bool(os.environ.get("SLURM_JOB_ID"))
            and bool(os.environ.get("CUDA_VISIBLE_DEVICES")), "allocated_gpu_required")
    source = Path(__file__).resolve().parents[1]
    verify_source_tree(source, Path(release["source_archive"]), release["source_archive_sha256"], release["source_git"])
    prepared = Path(release["prepared"])
    _, claims = adapter.check_prepared(prepared, release)
    corpus = {d.doc_id: d for d in (parse_abstract(json.loads(r)) for r in
        checked(prepared / "inference/corpus.jsonl", CORPUS_SHA).splitlines())}
    model = Path(release["model_directory"])
    manifest = verify_manifest(model.parent / "model_manifest.json", model, MODEL_SHA)
    for name, digest in TOKENIZER_SHA.items():
        checked(model / name, digest)
    output = Path(release["output"])
    require(not output.exists(), "no_reuse_or_resume")
    private = output.parent / (output.name + "-private-loader")
    private.mkdir(mode=0o700)
    provider = BottleneckProvider(model, manifest, private_dir=private)
    provider.generation_binding = GenerationBinding(provider.base, model, release["generation_contract"])
    frames = adapter.load_frames(prepared, claims, corpus, provider.base.tokenizer, release)
    run_suite(claims, frames, corpus, provider, output, release, args.release_sha)


if __name__ == "__main__":
    main()
