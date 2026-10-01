"""One frozen base-only inference process. No gold loader or training entry."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import time
from typing import Any, Callable, cast

from climate_rag.local_bounded_scifact_provider import LocalQwenBoundedSciFactProvider
from climate_rag.rerank import Qwen3CausalLMReranker
from climate_rag.scifact_grounding import parse_abstract
from climate_rag.scifact_read_continuation import ordered_write
from climate_rag.scifact_retrieval import SciFactBM25, rerank_sources
from climate_rag.scifact_semantic_contract import CORPUS_SHA, MODEL_SHA, RERANKER_SHA, TOKENIZER_SHA, checked
from climate_rag.scifact_utility_contract import identity
from climate_rag.scifact_utility_runtime import run_matrix
from run_scifact_bounded_arm import verify_manifest
from run_scifact_grounding_train_operator import require, sha


class SerialRerank:
    """Exclusive GPU residency; real forward tokens, not an estimate from text."""
    def __init__(self, provider: Any, model: Any, directory: Path, *, max_requests: int = 16) -> None:
        require(type(max_requests) is int and 1 <= max_requests <= 48, "bounded_rerank_capacity")
        directory.mkdir(mode=0o700)
        self.provider, self.model, self.directory = provider, model, directory
        self.max_requests = max_requests

    def __call__(self, query: str, candidates: Any) -> Any:
        number = len(list(self.directory.glob("r*.reserved.json")))
        require(number < self.max_requests and 0 < len(candidates) <= 20, "rerank_physical_limit")
        key = f"r{number:02d}"
        began = time.monotonic()
        ordered_write(self.directory / (key + ".reserved.json"), {
            "requested_pairs": len(candidates), "generator_calls": 0,
            "query_sha256": identity(query), "candidate_source_sha256": [r.text_sha256 for r in candidates],
            "status": "reserved_before_gpu_swap", "actual_tokens": None})
        torch = self.provider.base._torch
        batches: list[dict[str, Any]] = []
        def before(module: Any, args: Any, kwargs: Any) -> None:
            torch.cuda.synchronize()
            ids = kwargs["input_ids"].detach().cpu().tolist()
            mask = kwargs.get("attention_mask")
            tokens = int(mask.sum().item()) if mask is not None else None
            row = {"batch": len(batches), "pairs": len(ids), "token_ids_sha256": identity(ids),
                   "nonpadding_tokens": tokens, "padded_tokens": sum(len(r) for r in ids),
                   "at_length_cap_not_proof_of_truncation": any(len(r) == 2048 for r in ids),
                   "status": "forward_started", "started": time.monotonic()}
            ordered_write(self.directory / f"{key}-batch-{len(batches):02d}.started.json", row)
            batches.append(row)
        def after(module: Any, args: Any, kwargs: Any, result: Any) -> None:
            torch.cuda.synchronize()
            row = dict(batches[-1], status="completed")
            row["elapsed_ms"] = (time.monotonic() - row.pop("started")) * 1000
            ordered_write(self.directory / f"{key}-batch-{row['batch']:02d}.finished.json", row)
            batches[-1] = row
        pre = post = None
        status, output = "failed", None
        try:
            self.provider.base.model.to("cpu")
            torch.cuda.empty_cache()
            self.model._model.to("cuda")
            require(all(p.device.type == "cpu" for p in self.provider.base.model.parameters()), "generator_not_evacuated")
            pre = self.model._model.register_forward_pre_hook(before, with_kwargs=True)
            post = self.model._model.register_forward_hook(after, with_kwargs=True)
            output = rerank_sources(self.model, query, candidates)
            status = "completed"
        finally:
            if pre is not None:
                pre.remove()
            if post is not None:
                post.remove()
            try:
                self.model._model.to("cpu")
                torch.cuda.empty_cache()
                self.provider.base.model.to("cuda")
                torch.cuda.synchronize()
            finally:
                ordered_write(self.directory / (key + ".finished.json"), {
                    "status": status, "requested_pairs": len(candidates),
                    "completed_pairs": sum(r["pairs"] for r in batches if r["status"] == "completed"),
                    "observed_nonpadding_tokens": sum(r["nonpadding_tokens"] for r in batches)
                        if batches and all(r["nonpadding_tokens"] is not None for r in batches) else None,
                    "generator_calls": 0, "batches": batches,
                    "elapsed_ms_including_swaps": (time.monotonic() - began) * 1000})
        return output


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("release", "inference-dir", "model-root", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--release-sha", required=True)
    return parser.parse_args()


def run_inference(args: argparse.Namespace, validate_release: Any, *, natural_fit: bool = False) -> None:
    source = Path(__file__).resolve().parents[1]
    release = json.loads(checked(args.release, args.release_sha))
    validate_release(release)
    require(os.name == "posix" and bool(os.environ.get("SLURM_JOB_ID"))
            and bool(os.environ.get("CUDA_VISIBLE_DEVICES")), "allocated_gpu_only")
    require((source / "SOURCE_REVISION").read_text().strip() == release["source_git"], "source_revision")
    require(str(args.output) == release["output"] + "/inference", "fixed_inference_output")
    preparation = json.loads((args.inference_dir.parent / "preparation.json").read_bytes())
    claims = json.loads(checked(args.inference_dir / "claims.json", preparation["claims_sha256"]))
    if natural_fit:
        from prepare_scifact_natural import SELECTION_SHA
        selection = json.loads(checked(args.inference_dir.parent / "selection.json", SELECTION_SHA))
        require([r["id"] for r in claims] == [r["id"] for r in selection["selected"]]
                and len(claims) == 24 and preparation["selection_sha256"] == SELECTION_SHA,
                "frozen_natural_claim_order")
    docs = [parse_abstract(json.loads(r)) for r in checked(args.inference_dir / "corpus.jsonl", CORPUS_SHA).splitlines()]
    corpus = {d.doc_id: d for d in docs}
    require(len(corpus) == len(docs) == 5183, "complete_public_corpus")
    verifier = cast(Callable[[Path, Path, str], dict[str, str]], verify_manifest)
    generator, ranker = args.model_root / "generator", args.model_root / "reranker"
    model_manifest = verifier(generator / "model_manifest.json", generator / "model", MODEL_SHA)
    verifier(ranker / "model_manifest.json", ranker / "model", RERANKER_SHA)
    for name, digest in TOKENIZER_SHA.items():
        checked(generator / "model" / name, digest)
    started = time.monotonic()
    (args.output.parent / "private-loader").mkdir(mode=0o700)
    provider = LocalQwenBoundedSciFactProvider(generator / "model", model_manifest,
        private_dir=args.output.parent / "private-loader", gap=False)
    require(not hasattr(provider.base.model, "peft_config")
            and not any("lora_" in n for n, _ in provider.base.model.named_parameters()), "base_without_adapter")
    # CPU load: only the generator occupies GPU memory until a real tool call.
    reranker = Qwen3CausalLMReranker(str(ranker / "model"), device="cpu", dtype="bfloat16", max_length=2048, batch_size=1)
    tool = SerialRerank(provider, reranker, args.output.parent / "reranker-ledger", max_requests=48 if natural_fit else 16)
    ordered_write(args.output.parent / "model-load.json", {"model_sha256": MODEL_SHA,
        "reranker_sha256": RERANKER_SHA, "adapter_loaded": False, "warmup_generation_calls": 0,
        "load_elapsed_ms": (time.monotonic() - started) * 1000,
        "serial_gpu_residency": True, "release_sha256": args.release_sha,
        "preparation_sha256": sha(args.inference_dir.parent / "preparation.json")})
    run_matrix(claims, provider, SciFactBM25(corpus), tool, corpus, args.output, natural_fit=natural_fit)


def main() -> None:
    from run_scifact_utility8_operator import validate_release
    run_inference(arguments(), validate_release)


if __name__ == "__main__":
    main()
