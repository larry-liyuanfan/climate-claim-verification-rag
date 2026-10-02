"""Allocated worker or post-exit scorer. Draft releases cannot start either."""

from __future__ import annotations

import argparse
import json
import os
import re
from pathlib import Path
import time
from typing import Any

from climate_rag.agent_v3 import Source
from climate_rag.bm25 import BM25Index
from climate_rag.io import iter_evidence
from climate_rag.local_agent_model import verify_model_files
from climate_rag.local_targeted_provider import LocalTargetedProvider
from climate_rag.rerank import Qwen3CausalLMReranker
from climate_rag.scifact_generation import GenerationBinding
from climate_rag.scifact_natural_contract import base_state
from climate_rag.scifact_read_continuation import ordered_write
from climate_rag.scifact_utility_runtime import ledger_cost, reranker_cost
from climate_rag.targeted_replay import (
    CORPUS_SHA,
    GOLD_SHA,
    INPUT_NAME,
    MODEL_SHA,
    OLD_PROTOCOL_SHA,
    ROOT,
    RERANKER_SHA,
    MANIFEST_BYTES,
    SELECTION_SHA,
    frozen_tasks,
    run_matrix,
    sha,
)
from climate_rag.targeted_score import audit, score
from run_scifact_utility8 import SerialRerank
from run_budget_agent_full_operator import digest
from score_budget_agent import load_selection_manifest, validate_gold
from climate_rag import fair_acquisition
from climate_rag.fair_replay import validate_binding, validate_tasks

SOURCE = Path(__file__).resolve().parents[1]
SELECTION = (
    SOURCE / "docs/verified-runs/budget-agent-validation-selection-20260929.json"
)


class TargetedRerank(SerialRerank):
    capacity_ceiling = 384

    def start_slot(self, slot: str) -> None:
        self.request_context = {"slot": slot}

    def cost_for_slot(self, slot: str) -> dict[str, Any]:
        ids = [
            p.name.removesuffix(".reserved.json")
            for p in self.directory.glob("r*.reserved.json")
            if json.loads(p.read_bytes())["slot"] == slot
        ]
        return dict(reranker_cost(self.directory, ids))


def load_inputs(
    input_dir: Path, release: dict[str, Any] | None = None,
) -> tuple[list[dict[str, str]], list[Any], dict[str, Source]]:
    fair = release is not None and release.get("purpose") == fair_acquisition.PROTOCOL
    if fair:
        assert release is not None
        binding = release["fair_binding"]
        validate_binding(binding)
        cohort = Path(release["cohort_path"])
        exposure = Path(release["exposure_audit_path"])
        if cohort.resolve() != cohort or exposure.resolve() != exposure:
            raise ValueError("fair_asset_symlink")
        if digest(cohort) != binding["cohort_sha256"] or digest(exposure) != binding["exposure_audit_sha256"]:
            raise ValueError("fair_asset_sha")
        tasks = json.loads(cohort.read_bytes())
        validate_tasks(tasks, binding)
        audit_data = json.loads(exposure.read_bytes())
        canonical_raw = SELECTION.read_bytes()
        if sha(canonical_raw) != SELECTION_SHA:
            raise ValueError("canonical_consumption_receipt_hash")
        canonical_consumed = json.loads(canonical_raw)["selected_ids"]
        selected = [t["id"] for t in tasks]
        if (audit_data["selected_ids"] != selected or audit_data["sealed_test_read"] is not False
                or audit_data["model_results_read"] is not False or audit_data["labels_used_for_selection"] is not False
                or audit_data["consumption_receipt_sha256"] != SELECTION_SHA
                or audit_data["consumed_ids"] != sorted(canonical_consumed)
                or any(set(c) & set(audit_data["consumed_ids"]) for c in audit_data["selected_component_ids"])):
            raise ValueError("fair_exposure_eligibility")
    else:
        tasks = frozen_tasks(
        (input_dir / "validation-protocol.json").read_bytes(), SELECTION.read_bytes()
    )
    if digest(input_dir / "evidence.jsonl") != CORPUS_SHA:
        raise ValueError("corpus_hash")
    documents = list(iter_evidence(input_dir / "evidence.jsonl"))
    sources = {
        d.evidence_id: Source(
            d.evidence_id, str(d.metadata.get("title", "")), (d.text,)
        )
        for d in documents
    }
    if len(documents) != len(sources) or len(sources) != 5240:
        raise ValueError("corpus_identity_count")
    return tasks, documents, sources


def worker(release: dict[str, Any], input_dir: Path) -> None:
    if (
        os.name != "posix"
        or not os.environ.get("SLURM_JOB_ID")
        or not os.environ.get("CUDA_VISIBLE_DEVICES")
    ):
        raise ValueError("allocated_gpu_required")
    worker_body(release, input_dir)


def worker_body(release: dict[str, Any], input_dir: Path) -> None:
    """Shared implementation; entrypoints must validate their own backend first."""
    if release.get("purpose") == fair_acquisition.PROTOCOL:
        if (release.get("authorization") != "validated_worker_projection"
                or not re.fullmatch(r"[0-9a-f]{40}", str(release.get("source_git")))
                or not re.fullmatch(r"[0-9a-f]{64}", str(release.get("source_archive_sha256")))
                or not re.fullmatch(r"[a-z0-9][a-z0-9-]{5,79}", str(release.get("run_id")))):
            raise ValueError("fair_worker_unbound_run_identity")
    tasks, documents, sources = (load_inputs(input_dir, release) if release["purpose"] == fair_acquisition.PROTOCOL else load_inputs(input_dir))
    manifests = {}
    for key, identity in (("generator", MODEL_SHA), ("reranker", RERANKER_SHA)):
        root = input_dir / "models" / key
        raw = (root / "model_manifest.json").read_bytes()
        if sha(raw) != MANIFEST_BYTES[key]:
            raise ValueError("manifest_file_hash")
        manifests[key] = json.loads(raw)
        if verify_model_files(root / "model", manifests[key]) != identity:
            raise ValueError("model_identity")
    output = Path(release["output"])
    private = output / "private-loader"
    private.mkdir(mode=0o700)
    started = time.monotonic()
    model_dir = input_dir / "models/generator/model"
    backend = LocalTargetedProvider(
        model_dir, manifests["generator"], private_dir=private, protocol=release["purpose"]
    )
    backend.generation_binding = GenerationBinding(
        backend.base, model_dir, release["generation_contract"]
    )
    base_state(backend)
    # Loader/forwards share existing serial-residency, pair-token and swap accounting.
    ranker = Qwen3CausalLMReranker(
        str(input_dir / "models/reranker/model"),
        device="cpu",
        dtype="bfloat16",
        max_length=2048,
        batch_size=1,
    )
    rerank = TargetedRerank(
        backend, ranker, output / "reranker-ledger", max_requests=(len(tasks) * 12 if release["purpose"] == fair_acquisition.PROTOCOL else 128)
    )
    index = BM25Index().fit(documents)

    def retrieve(query: str, width: int) -> list[Source]:
        return [sources[r.evidence_id] for r in index.search(query, width)]

    ordered_write(
        output / "model-load.json",
        {
            "model_sha256": MODEL_SHA,
            "reranker_sha256": RERANKER_SHA,
            "warmup_calls": 0,
            "load_index_elapsed_ms": (time.monotonic() - started) * 1000,
            "input_archive": release.get("input_transport_sha256", INPUT_NAME),
        },
    )
    run_matrix(
        tasks,
        backend,
        retrieve,
        rerank,
        output / "inference",
        physical_guard=base_state,
        protocol=release["purpose"],
        binding=release.get("fair_binding"),
    )


def score_after_exit(release: dict[str, Any], input_dir: Path, *, gold_path: Path | None = None) -> None:
    output = Path(release["output"])
    proof = json.loads((output / "allocation/worker-exit.json").read_bytes())
    if not (
        proof["child_started"]
        and proof["child_reaped"]
        and proof["returncode"] == 0
        and proof["interrupted"] is None
    ):
        raise ValueError("worker_not_cleanly_reaped")
    if not (output / "cost-before-quality.json").is_file():
        raise ValueError("missing_cost_before_gold")
    tasks, _, sources = (load_inputs(input_dir, release) if release["purpose"] == fair_acquisition.PROTOCOL else load_inputs(input_dir))
    run = json.loads((output / "inference/run.json").read_bytes())
    if run.get("protocol") != release["purpose"]:
        raise ValueError("run_release_protocol_mismatch")
    if release["purpose"] == fair_acquisition.PROTOCOL and run.get("fair_binding") != release["fair_binding"]:
        raise ValueError("fair_run_release_binding_mismatch")
    # Tokenizer-only, not weights: bind physical prompt IDs and decoding receipts.
    from transformers.models.auto.tokenization_auto import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(
        str(input_dir / "models/generator/model"), local_files_only=True
    )
    audit(
        run,
        tasks,
        sources,
        output / "inference/ledger",
        tokenizer=tokenizer,
        generation_contract=release["generation_contract"],
        reranker_directory=output / "reranker-ledger",
    )
    cost = json.loads((output / "cost-before-quality.json").read_bytes())
    if (
        cost["generation"] != ledger_cost(output / "inference/ledger")
        or cost["reranker"] != reranker_cost(output / "reranker-ledger")
        or cost["planned_slots"] != cost["completed_slots"]
        or cost["completed_slots"] != release.get("policy", {}).get("planned_slots", 160)
    ):
        raise ValueError("cost_before_quality_drift")
    if release["purpose"] == fair_acquisition.PROTOCOL:
        from climate_rag.fair_replay import score as fair_score
        if gold_path is None or digest(gold_path) != release["gold_sha256"]:
            raise ValueError("fair_gold_hash_after_exit")
        compact = fair_score(run, json.loads(gold_path.read_bytes()))
        compact.update(source_git=release["source_git"], policy=release["policy"],
            raw_run_sha256=digest(output / "inference/run.json"), cost_sha256=digest(output / "cost-before-quality.json"))
        ordered_write(output / "compact.json", compact)
        return
    claim_hashes = {r["task_id"]: r["immutable_claim_sha256"] for r in run["runs"]}
    protocol = json.loads((input_dir / "validation-protocol.json").read_bytes())
    # First and only real-gold read in this program, after worker death + audit.
    gold = validate_gold(
        protocol,
        "validation",
        OLD_PROTOCOL_SHA,
        claim_hashes,
        (
            gold_path if gold_path is not None else
            ROOT / "envs" / ("budget-agent-validation-gold-" + GOLD_SHA + ".json")
        ).read_bytes(),
        load_selection_manifest(),
    )
    if gold is None:
        raise ValueError("missing_official_gold")
    compact = score(run, gold)
    compact.update(
        source_git=release["source_git"],
        policy=release["policy"],
        raw_run_sha256=digest(output / "inference/run.json"),
        cost_sha256=digest(output / "cost-before-quality.json"),
    )
    ordered_write(output / "compact.json", compact)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=["worker", "score"])
    for option in ("release", "input"):
        parser.add_argument("--" + option, type=Path, required=True)
    parser.add_argument("--release-sha", required=True)
    args = parser.parse_args()
    from run_targeted_replay_operator import load_release

    release = load_release(args.release, args.release_sha, SOURCE)
    if (
        args.input.resolve()
        != Path(os.environ["CLIMATE_TARGETED_WORK"]).resolve() / "input"
    ):
        raise ValueError("private_input_path")
    (worker if args.stage == "worker" else score_after_exit)(release, args.input)


if __name__ == "__main__":
    main()
