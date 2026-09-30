"""Bounded grounding train/evaluate/score entrypoints; CPU package is NOT a release.

train/evaluate require a separately reviewed hash-bound release and a GPU Slurm
allocation. This file is implemented/tested on fixtures only in the CPU package.
score runs after inference, in a separate invocation with access to gold.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import random
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

from climate_rag.scifact_grounding import parse_abstract, parse_gold
from climate_rag.scifact_grounding_eval import audit_arm, evaluate_arm, score_arm
from climate_rag.scifact_grounding_sft import (
    CONFIG, advancement, canonical_context, config_sha, require, token_record, training_steps,
)
from climate_rag.scifact_semantic_contract import MODEL_SHA, checked, encoded, sha
from climate_rag.component_execution import durable as write_once
from climate_rag.torch_compat import ensure_torch_pytree_compat


def load_bundle(bundle: Path, digest: str) -> tuple[dict[str, Any], dict[int, Any]]:
    manifest = json.loads(checked(bundle / "manifest.json", digest))
    require(manifest["config_sha256"] == config_sha() and manifest["status"] == "cpu_prepared",
            "unprepared_or_config_changed")
    payload = checked(bundle / "inference/corpus.jsonl", manifest["files"]["inference/corpus.jsonl"])
    corpus = {a.doc_id: a for a in (parse_abstract(json.loads(line)) for line in payload.splitlines())}
    return manifest, corpus


def release_check(args: Any, purpose: str) -> dict[str, Any]:
    require(bool(args.release and args.release_sha), "separate_execution_release_required")
    release = json.loads(checked(args.release, args.release_sha))
    require(release.get("authorization") == "coordinator_exact_hash_release", "draft_not_authorized")
    require(release["purpose"] == purpose and release["config_sha256"] == config_sha()
            and release["data_manifest_sha256"] == args.data_sha
            and release["output"] == str(args.output.resolve()), "release_identity")
    source = Path(__file__).resolve().parents[1]
    require((source / "SOURCE_REVISION").read_text().strip() == release["source_git"], "execution_source")
    require(bool(os.environ.get("SLURM_JOB_ID")) and bool(os.environ.get("CUDA_VISIBLE_DEVICES")),
            "gpu_allocation_required_not_login_node")
    require(not args.output.exists(), "no_retry_or_checkpoint_overwrite")
    return dict(release)


def model_manifest(path: Path) -> dict[str, str]:
    value = json.loads(path.read_bytes())
    require(sha(json.dumps(value, sort_keys=True).encode()) == MODEL_SHA, "base_model_manifest")
    return dict(value)


def restore_causal_adapter(base: Any, path: Path) -> tuple[Any, dict[str, Any]]:
    """CausalLM-specific exact tensor restoration; not the dense bare-model loader."""
    import torch
    ensure_torch_pytree_compat()
    from peft import PeftConfig, PeftModel
    from peft.utils.save_and_load import get_peft_model_state_dict, load_peft_weights
    settings = PeftConfig.from_pretrained(str(path))
    require(str(settings.task_type) in {"CAUSAL_LM", "TaskType.CAUSAL_LM"}
            and settings.r == CONFIG["lora_r"] and settings.lora_alpha == CONFIG["lora_alpha"]
            and settings.lora_dropout == CONFIG["lora_dropout"]
            and set(settings.target_modules) == set(CONFIG["target_modules"]), "causal_adapter_config")
    loader: Any = PeftModel.from_pretrained
    model = loader(base, str(path), is_trainable=False, local_files_only=True).eval()
    state_reader: Any = get_peft_model_state_dict
    actual = state_reader(model, save_embedding_layers=False)
    expected = load_peft_weights(str(path), device="cpu")
    require(bool(actual) and set(actual) == set(expected), "causal_adapter_tensor_keys")
    for name, tensor in actual.items():
        require(bool(torch.isfinite(tensor).all()) and torch.equal(tensor,
            expected[name].to(device=tensor.device, dtype=tensor.dtype)), "causal_adapter_tensor_value")
    return model, {"tensor_count": len(actual), "all_checkpoint_values_equal": True,
                   "generative_quality_verified": False}


def train(args: Any) -> None:
    release_check(args, "train")
    manifest, corpus = load_bundle(args.bundle, args.data_sha)
    rows = json.loads(checked(args.bundle / "fit/records.json", manifest["files"]["fit/records.json"]))
    expected_steps = training_steps(len(rows))
    args.output.mkdir(mode=0o700)
    write_once(args.output / "started.json", {"config": CONFIG, "data_manifest_sha256": args.data_sha,
                                             "records": len(rows), "expected_steps": expected_steps})
    import torch
    ensure_torch_pytree_compat()
    from peft import LoraConfig, get_peft_model
    from climate_rag.local_agent_model import LocalQwenDecisionProvider
    torch.manual_seed(CONFIG["seed"])
    provider = LocalQwenDecisionProvider(args.model, model_manifest(args.model_manifest))
    config = LoraConfig(r=CONFIG["lora_r"], lora_alpha=CONFIG["lora_alpha"],
        lora_dropout=CONFIG["lora_dropout"], target_modules=CONFIG["target_modules"],
        task_type="CAUSAL_LM", bias="none")
    # PEFT is intentionally imported only at the released execution boundary.
    builder: Any = get_peft_model
    model = builder(provider.model, config)
    model.config.use_cache = False
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    model.enable_input_require_grads()
    model.train()
    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],
        lr=CONFIG["learning_rate"], weight_decay=CONFIG["weight_decay"])
    order = list(range(len(rows)))
    random.Random(CONFIG["seed"]).shuffle(order)
    losses: list[float] = []
    steps = 0
    width = CONFIG["gradient_accumulation"]
    torch.cuda.synchronize()
    fit_started = time.perf_counter()
    first_started = fit_started
    for offset in range(0, len(order), width):
        if steps == CONFIG["max_optimizer_steps"]:
            break
        chunk = order[offset:offset + width]
        write_once(args.output / f"step-{steps + 1:02d}-started.json", {
            "planned_record_indices_sha256": sha(encoded(chunk)), "planned_records": len(chunk),
            "completed_prior_steps": steps, "started_unix": time.time()})
        optimizer.zero_grad(set_to_none=True)
        for index in chunk:
            row = rows[index]
            require(row["provenance"] == "official_annotated_document_alternative"
                    and row["weight"] == 1.0, "fit_label_provenance")
            ctx = canonical_context(row["context"], corpus)
            tokens = token_record(provider.tokenizer, ctx, row["target"])
            require({k: v for k, v in tokens.items() if k not in {"input_ids", "labels", "attention_mask"}}
                    == row["packing"], "training_tokens_changed")
            tensors = {k: torch.tensor([tokens[k]], device=model.device)
                       for k in ("input_ids", "labels", "attention_mask")}
            loss = model(**tensors).loss
            require(bool(torch.isfinite(loss)), "nonfinite_training_loss")
            losses.append(float(loss.detach().cpu()))
            (loss / len(chunk)).backward()
        torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad],
                                      1.0, error_if_nonfinite=True)
        optimizer.step()
        steps += 1
        if steps == 1:
            torch.cuda.synchronize()
            import resource
            # POSIX-only runtime; Windows type stubs omit these attributes.
            posix_resource: Any = resource
            device = torch.cuda.get_device_properties(model.device)
            write_once(args.output / "first-step-pilot.json", {
                "optimizer_steps": steps, "records_seen": len(chunk),
                "counted_in_full_epoch": True, "optimizer_reinitialized": False,
                "elapsed_seconds": time.perf_counter() - first_started,
                "finite_losses": all(math.isfinite(x) for x in losses),
                "gpu_name": device.name, "gpu_total_memory_bytes": device.total_memory,
                "gpu_peak_allocated_bytes": torch.cuda.max_memory_allocated(),
                "host_maxrss_kib_linux": posix_resource.getrusage(posix_resource.RUSAGE_SELF).ru_maxrss,
                "first_group_record_indices_sha256": sha(encoded(chunk)),
                "remaining_optimizer_steps": expected_steps - steps,
                "projection_is_not_measured_full_runtime": True,
            })
    require(steps == expected_steps and len(losses) == len(rows), "one_epoch_step_ceiling")
    check_finite_adapter(model)
    final = args.output / "final"
    model.save_pretrained(final, safe_serialization=True)
    hashes = {p.name: sha(p.read_bytes()) for p in final.iterdir() if p.is_file()}
    write_once(args.output / "complete.json", {"config_sha256": config_sha(),
        "data_manifest_sha256": args.data_sha, "optimizer_steps": steps,
        "records_seen_once": len(losses), "losses": losses,
        "adapter_files": hashes, "checkpoint_policy": "final_only",
        "fit_elapsed_seconds": time.perf_counter() - fit_started,
        "max_memory_allocated_bytes": torch.cuda.max_memory_allocated(),
        "training_is_not_quality_evidence": True})


def check_finite_adapter(model: Any) -> None:
    import torch
    require(all(bool(torch.isfinite(p).all()) for p in model.parameters() if p.requires_grad),
            "nonfinite_final_adapter")


def evaluate(args: Any) -> None:
    release = release_check(args, "evaluate_" + args.partition)
    manifest, corpus = load_bundle(args.bundle, args.data_sha)
    require(args.partition in {"tune", "validation"}, "partition")
    if args.partition == "validation":
        gate = json.loads(checked(args.gate, release["tune_gate_sha256"]))
        require(gate["passed"] is True and gate["data_manifest_sha256"] == args.data_sha
                and gate["adapter_training_sha256"] == release["adapter_training_sha256"], "closed_validation_gate")
    training = json.loads(checked(args.adapter / "complete.json", release["adapter_training_sha256"]))
    require(training["data_manifest_sha256"] == args.data_sha
            and training["config_sha256"] == config_sha(), "trained_adapter_identity")
    for name, digest in training["adapter_files"].items():
        require(Path(name).name == name, "adapter_path")
        checked(args.adapter / "final" / name, digest)
    name = f"inference/{args.partition}.json"
    rows = json.loads(checked(args.bundle / name, manifest["files"][name]))
    require(len(rows) == 12, "evaluation_24_call_cap")
    args.output.mkdir(mode=0o700)
    from climate_rag.local_bounded_scifact_provider import LocalQwenBoundedSciFactProvider
    provider = LocalQwenBoundedSciFactProvider(args.model, model_manifest(args.model_manifest),
        private_dir=args.output, gap=False)
    provider.base.model, integrity = restore_causal_adapter(provider.base.model, args.adapter / "final")
    write_once(args.output / "adapter-integrity.json", integrity)
    # Same loaded CausalLM and same full input. Only adapter state differs.
    try:
        with provider.base.model.disable_adapter():
            base = evaluate_arm(rows, corpus, provider, args.output / "base", "base")
        if len(base["records"]) != 12 or any(not r["usage_known"] or r.get("stop_required") for r in base["records"]):
            return  # no second arm after uncertain/infrastructure failure
        adapted = evaluate_arm(rows, corpus, provider, args.output / "adapted", "adapted")
        require(base["input_identity"] == adapted["input_identity"], "paired_inputs")
    finally:
        terminate_inference(args.output, args.partition, args.data_sha,
                            release["adapter_training_sha256"])


def terminate_inference(output: Path, partition: str, data_sha: str, adapter_sha: str) -> None:
    """Preserve partial runs and unattempted arm; no fabricated zero-cost calls."""
    arms: dict[str, dict[str, Any]] = {}
    for arm in ("base", "adapted"):
        path = output / arm / "complete.json"
        root = output / arm
        count = len(list(root.glob("slot-*/started.json")))
        arms[arm] = {"attempted": count, "not_attempted": 12 - count,
                     "summary_sha256": sha(path.read_bytes()) if path.exists() else None}
        require(0 <= count <= 12, "actual_call_ceiling")
    write_once(output / "inference-terminated.json", {
        "partition": partition, "data_manifest_sha256": data_sha,
        "adapter_training_sha256": adapter_sha, "arms": arms,
        "attempted_calls": sum(a["attempted"] for a in arms.values()), "gold_loaded": False})


def score(args: Any) -> None:
    manifest, corpus = load_bundle(args.bundle, args.data_sha)
    exit_file = args.output.with_name(args.output.name + "-execution") / "worker-exit.json"
    exit_proof = json.loads(exit_file.read_bytes())
    require(exit_proof["child_reaped"] is True, "inference_exit_required")
    end = json.loads(checked(args.output / "inference-terminated.json", exit_proof["terminated_sha256"]))
    require(end["data_manifest_sha256"] == args.data_sha and end["partition"] == args.partition
            and 0 <= end["attempted_calls"] <= 24 and end["gold_loaded"] is False, "inference_termination_identity")
    name = f"scoring/{args.partition}.json"
    gold = [parse_gold(r, corpus) for r in json.loads(checked(args.bundle / name, manifest["files"][name]))]
    input_name = f"inference/{args.partition}.json"
    rows = json.loads(checked(args.bundle / input_name, manifest["files"][input_name]))
    results = {}
    for arm in ("base", "adapted"):
        info = end["arms"][arm]
        directory = args.output / arm
        if info["summary_sha256"] is not None:
            checked(directory / "complete.json", info["summary_sha256"])
            run = audit_arm(directory, rows, corpus)
        else:
            require(info["attempted"] == 0 and not directory.exists(), "interrupted_summary_requires_cost_reconciliation")
            run = {"input_identity": sha(encoded(rows)), "attempts": 0, "records": []}
        require(run["attempts"] == info["attempted"], "physical_attempt_denominator")
        results[arm] = score_arm(run, gold, corpus)
    write_once(args.output / "score.json", results)
    if args.partition == "tune":
        write_once(args.output / "gate.json", {
            "passed": end["attempted_calls"] == 24 and exit_proof["returncode"] == 0
                      and advancement(results["base"], results["adapted"]),
            "data_manifest_sha256": args.data_sha,
            "adapter_training_sha256": end["adapter_training_sha256"],
            "score_sha256": sha(encoded(results)), "next_validation_calls": 24})


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("operation", choices=("train", "evaluate", "score"))
    for name in ("bundle", "output", "release", "model", "model-manifest", "adapter", "gate"):
        parser.add_argument("--" + name, type=Path, required=name in {"bundle", "output"})
    parser.add_argument("--data-sha", required=True)
    parser.add_argument("--release-sha")
    parser.add_argument("--partition", choices=("tune", "validation"))
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.operation in {"train", "evaluate"} and not args.worker:
        purpose = "train" if args.operation == "train" else "evaluate_" + args.partition
        release = release_check(args, purpose)
        bound = release["max_runtime_seconds"]
        require(type(bound) is int and 0 < bound <= 86400, "reviewed_runtime_bound")
        execution = args.output.with_name(args.output.name + "-execution")
        execution.mkdir(mode=0o700)
        with (execution / "worker.log").open("xb") as log:
            worker_started = time.time()
            child = subprocess.Popen([sys.executable, *sys.argv, "--worker"], stdout=log, stderr=log)
            try:
                code = child.wait(timeout=bound)
            except subprocess.TimeoutExpired:
                child.kill()
                code = child.wait()
        terminated = args.output / "inference-terminated.json"
        write_once(execution / "worker-exit.json", {"child_reaped": True, "returncode": code,
            "started_unix": worker_started, "ended_unix": time.time(),
            "job_id": os.environ.get("SLURM_JOB_ID"),
            "terminated_sha256": sha(terminated.read_bytes()) if terminated.exists() else None,
            "release_sha256": args.release_sha, "data_manifest_sha256": args.data_sha})
        if code:
            raise SystemExit(code)
        return
    {"train": train, "evaluate": evaluate, "score": score}[args.operation](args)


if __name__ == "__main__":
    main()
