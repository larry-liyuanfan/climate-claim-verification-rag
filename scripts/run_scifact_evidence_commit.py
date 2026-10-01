"""One base-model worker, all frozen FIT24 x3, no training or gold loading."""
from __future__ import annotations

import json
import os
from pathlib import Path
import time
from typing import Callable, cast

from climate_rag.scifact_evidence_commit import ARMS, PROTOCOL, CALL_CAPS, EvidenceCommitProvider
from climate_rag.scifact_evidence_commit_runtime import CommitJournal, run_episode
from climate_rag.scifact_grounding import parse_abstract
from climate_rag.scifact_natural_contract import base_state
from climate_rag.scifact_read_continuation import ordered_write
from climate_rag.scifact_semantic_contract import CORPUS_SHA, MODEL_SHA, TOKENIZER_SHA, checked
from prepare_scifact_document_verifier import load_frames
from run_scifact_bounded_arm import verify_manifest
from run_scifact_evidence_commit_operator import validate_release
from run_scifact_grounding_train_operator import require, sha
from run_scifact_utility8 import arguments


def main() -> None:
    args = arguments()
    release = json.loads(checked(args.release, args.release_sha))
    validate_release(release)
    source = Path(__file__).resolve().parents[1]
    require(os.name == "posix" and bool(os.environ.get("SLURM_JOB_ID")) and bool(os.environ.get("CUDA_VISIBLE_DEVICES"))
            and (source / "SOURCE_REVISION").read_text().strip() == release["source_git"]
            and str(args.output) == release["output"] + "/inference", "allocated_bound_source_required")
    preparation = json.loads((args.inference_dir.parent / "preparation.json").read_bytes())
    claims = json.loads(checked(args.inference_dir / "claims.json", preparation["claims_sha256"]))
    docs = [parse_abstract(json.loads(r)) for r in checked(args.inference_dir / "corpus.jsonl", CORPUS_SHA).splitlines()]
    corpus = {d.doc_id: d for d in docs}
    require(len(docs) == len(corpus) == 5183, "complete_public_corpus")
    generator = args.model_root / "generator"
    verifier = cast(Callable[[Path, Path, str], dict[str, str]], verify_manifest)
    manifest = verifier(generator / "model_manifest.json", generator / "model", MODEL_SHA)
    for name, digest in TOKENIZER_SHA.items():
        checked(generator / "model" / name, digest)
    started = time.monotonic()
    (args.output.parent / "private-loader").mkdir(mode=0o700)
    provider = EvidenceCommitProvider(generator / "model", manifest, private_dir=args.output.parent / "private-loader")
    base_state(provider)
    frames = load_frames(claims, corpus, provider.base.tokenizer, release["initial_inventory_sha256"])
    args.output.mkdir(mode=0o700)
    ordered_write(args.output / "initial-frames.json", frames)
    ordered_write(args.output.parent / "model-load.json", {"base_model_sha256": MODEL_SHA,
        "adapter_loaded": False, "warmup_calls": 0, "reranker_loaded": False,
        "load_and_input_validation_seconds": time.monotonic() - started,
        "initial_frames_sha256": sha(args.output / "initial-frames.json"), "release_sha256": args.release_sha})
    ordered_write(args.output / "planned.json", {"protocol": PROTOCOL,
        "slots": [{"claim_id": c["id"], "arm": a} for c in claims for a in ARMS],
        "max_generations": 240, "call_caps": CALL_CAPS, "max_tools_per_episode": 5, "max_seconds_per_episode": 120})
    journal = CommitJournal(provider, args.output / "ledger", max_generations=240, protocol=PROTOCOL,
                            physical_guard=base_state, run_identity=args.release_sha)
    for claim, frame in zip(claims, frames, strict=True):
        for arm in ARMS:
            run_episode(claim["id"], arm, frame, journal, corpus, args.output / f"{claim['id']}-{arm}", run_identity=args.release_sha)


if __name__ == "__main__":
    main()
