"""Frozen CPU-only preparation. Private inputs/output remain on Spartan.

Reuse legacy 531-row strata; exclude consumed components before selection. Only
selected TRAIN gold is parsed for new packing probes. No dev/test or model load.
"""
from __future__ import annotations

import argparse
import json
import os
import tarfile
from pathlib import Path
from typing import Any

from climate_rag.agent_v3 import V3Budget
from climate_rag.scifact_consumption import (
    consumption_ledger, review_selected, select_unconsumed, verify_bytes,
)
from climate_rag.scifact_grounding import parse_abstract, parse_gold
from climate_rag.scifact_retrieval import SciFactBM25
from climate_rag.scifact_semantic_policy import PAIR, POLICIES, policy_sha
from climate_rag.scifact_train_diagnostic import ROUTES, SALT
from climate_rag.public_v2 import file_sha256
from smoke_agent_v3_tokenizer import TOKENIZER_HASHES

PREP = "data/scifact-original-20260930/prepared-r1"
LEGACY = "data/scifact-train-diagnostic-20260930-r1"
TOKENIZER = "data/qwen3-4b-tokenizer-v3"
PREP_SHA = "3cd6bc1e1c299ece9195098a3853ebb05bb3401507d8b467c1596ba6afb6e138"
LEGACY_SHA = "dceda481dba4966b32cfb0234449977cf224d5bceeaeb55f3eb5790aba364b8e"
AUDIT_SHA = "ce38a8d705296a89f3079c672b535dd05c3d64bd22cf2ffdae60013e138768e9"
ASSIGNMENT_SHA = "6d79861daef5d385878913fe10681dff95df444e6f7b1b76ab7c07f4d4d70a12"
RUNS = (
    ("climate-scifact-train-diagnostic-20260930-r2", "23566957ed986dde054b61a74529350984b1f018a27f69ec771bc4223dcd1c2b", "80fcd07faed28890b70096386c5897f0343d3104"),
    ("climate-scifact-bounded-gap-pair-20260930-v1-format_repaired", "473c0d153f73115cacf03e10875993aa947d6687c317f15ee6bf285675cd77b3", "b248fe7f43b175b15b90ec6143538d5dab8e2f35"),
    ("climate-scifact-bounded-gap-pair-20260930-v1-format_repaired_gap", "e9842c9bc5b8169a8a3b5cfcabfa164155da2cd74f8fdd9ac738b84c4e9809da", "b248fe7f43b175b15b90ec6143538d5dab8e2f35"),
)
PREP_FILES = ("inference/corpus.jsonl", "inference/claims_train_eligible.jsonl",
              "private-group-assignment.json", "gold/claims_train.jsonl")
LEGACY_FILES = ("private/sampling-audit.json", "private/selected-strata.json", "inference/protocol.json")
INPUT_FILES = (tuple(f"{PREP}/{f}" for f in ("preparation-manifest.json", *PREP_FILES))
               + tuple(f"{LEGACY}/{f}" for f in ("manifest.json", *LEGACY_FILES))
               + tuple(f"{TOKENIZER}/{f}" for f in TOKENIZER_HASHES))


def checked(path: Path, sha: str) -> bytes:
    if path.is_symlink():
        raise ValueError("symlink_input")
    payload = path.read_bytes()
    verify_bytes(payload, sha)
    return payload


def write_once(path: Path, value: Any) -> None:
    with path.open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, sort_keys=True)
        stream.write("\n")


def verify_source_tree(source: Path, archive: Path, sha: str, git: str) -> None:
    verify_bytes(archive.read_bytes(), sha)
    expected = set()
    with tarfile.open(archive) as bundle:
        for member in bundle.getmembers():
            target = source / member.name
            if (not target.resolve().is_relative_to(source.resolve()) or target.is_symlink()
                    or not (member.isdir() or member.isfile())):
                raise ValueError("invalid_source_archive_member")
            if member.isfile():
                if member.name in expected:
                    raise ValueError("duplicate_source_archive_file")
                expected.add(member.name)
                stream = bundle.extractfile(member)
                if stream is None or target.read_bytes() != stream.read():
                    raise ValueError("executing_source_differs_from_frozen_archive")
    if ({p.relative_to(source).as_posix() for p in source.rglob("*") if p.is_file()} != expected
            or (source / "SOURCE_REVISION").read_text().strip() != git):
        raise ValueError("extra_source_or_revision_mismatch")


def prepare(inputs: Path, root: Path, output: Path, source_git: str,
            source_archive: Path, source_sha: str) -> dict[str, Any]:
    source = Path(__file__).resolve().parents[1]
    verify_source_tree(source, source_archive, source_sha, source_git)
    if (len(source_git) != 40 or (source / "SOURCE_REVISION").read_text().strip() != source_git
            or os.environ.get("PYTHONHASHSEED") != "0" or not os.environ.get("SLURM_JOB_ID")):
        raise ValueError("frozen_source_cpu_allocation_and_hashseed_required")
    if output.exists() or not output.resolve().is_relative_to(root.resolve() / "posthoc"):
        raise ValueError("new_scoped_output_required")
    if {p.relative_to(inputs).as_posix() for p in inputs.rglob("*") if p.is_file()} != set(INPUT_FILES):
        raise ValueError("exact_private_input_allowlist_required")
    prep = json.loads(checked(inputs / PREP / "preparation-manifest.json", PREP_SHA))
    legacy = json.loads(checked(inputs / LEGACY / "manifest.json", LEGACY_SHA))
    if legacy["preparation_sha256"] != PREP_SHA or legacy["audited_eligible_train"] != 531:
        raise ValueError("legacy_preparation_identity")
    hashes = {}
    payloads = {}
    for name in PREP_FILES:
        sha = prep["output_file_sha256"][name]
        payloads[f"{PREP}/{name}"] = checked(inputs / PREP / name, sha)
        if legacy["input_sha256"][name] != sha:
            raise ValueError("legacy_input_identity")
        hashes[f"{PREP}/{name}"] = sha
    for name in LEGACY_FILES:
        sha = legacy["output_sha256"][name]
        payloads[f"{LEGACY}/{name}"] = checked(inputs / LEGACY / name, sha)
        hashes[f"{LEGACY}/{name}"] = sha
    verify_bytes(payloads[f"{PREP}/private-group-assignment.json"], ASSIGNMENT_SHA)
    verify_bytes(payloads[f"{LEGACY}/private/sampling-audit.json"], AUDIT_SHA)
    assignment = json.loads(payloads[f"{PREP}/private-group-assignment.json"])
    audit = json.loads(payloads[f"{LEGACY}/private/sampling-audit.json"])
    selection = json.loads(payloads[f"{LEGACY}/private/selected-strata.json"])
    old_protocol = json.loads(payloads[f"{LEGACY}/inference/protocol.json"])
    old_ids = [r["id"] for r in selection]
    if len(assignment["eligible_train_ids"]) != 531 or old_protocol["slots"] != [
            {"claim_id": i, "route": route} for i in old_ids for route in ROUTES]:
        raise ValueError("legacy_selection_matrix")
    planned = [{"claim_id": i, "kind": "historical_selected", "source_path": f"{LEGACY}/private/selected-strata.json",
                "source_sha256": legacy["output_sha256"]["private/selected-strata.json"]} for i in old_ids]
    actual = []
    for name, sha, git in RUNS:
        relative = f"runs/{name}/inference/run.json"
        run = json.loads(checked(root / relative, sha))
        if run["source_git"] != git or [(r["claim_id"], r["route"]) for r in run["runs"]] != [
                (i, route) for i in old_ids for route in ROUTES]:
            raise ValueError("consumed_run_identity_or_matrix")
        for r in run["runs"]:
            actual.append({"claim_id": r["claim_id"], "route": r["route"], "result": r["result"],
                           "source_path": relative, "source_sha256": sha})
        hashes[relative] = sha
    ledger = consumption_ledger(assignment, audit, planned, actual)
    selected, matching = select_unconsumed(audit, assignment, ledger)
    ids = [r["id"] for r in selected]
    if not ids:
        raise ValueError("no_eligible_unconsumed_components")
    output.mkdir(mode=0o700)
    private = output / "private"
    private.mkdir(mode=0o700)
    write_once(private / "consumption-ledger.json", ledger)
    write_once(private / "selected-before-probe.json", {
        "source_git": source_git, "source_archive_sha256": source_sha,
        "source_inputs_sha256": hashes, "selection": selected,
        "matching": matching, "salt": SALT, "selection_scope": "legacy_V1_strata_only",
        "probe_not_started": True, "reselection_allowed": False})
    # Full train JSONL is hash-checked/decoded for IDs; only the selected rows
    # enter parse_gold. No official dev or unlabelled/retired test file is opened.
    corpus = {d.doc_id: d for d in (parse_abstract(json.loads(line)) for line in
              payloads[f"{PREP}/inference/corpus.jsonl"].splitlines())}
    inference = [json.loads(line) for line in
                 payloads[f"{PREP}/inference/claims_train_eligible.jsonl"].splitlines()]
    if (len(corpus) != 5183 or len(inference) != 531
            or {r["id"] for r in inference} != set(assignment["eligible_train_ids"])):
        raise ValueError("original_corpus_or_train_contract")
    gold = {}
    for line in payloads[f"{PREP}/gold/claims_train.jsonl"].splitlines():
        row = json.loads(line)
        if row["id"] in ids:
            if row["id"] in gold:
                raise ValueError("duplicate_selected_gold")
            gold[row["id"]] = parse_gold(row, corpus)
    by_id = {r["id"]: r for r in inference}
    if set(gold) != set(ids) or any(gold[i].claim != by_id[i]["claim"] for i in ids):
        raise ValueError("selected_gold_claim_identity")
    for name, sha in TOKENIZER_HASHES.items():
        checked(inputs / TOKENIZER / name, sha)
    from transformers.models.auto.tokenization_auto import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(str(inputs / TOKENIZER), local_files_only=True, trust_remote_code=False)
    retrieve = SciFactBM25(corpus)
    reviews, summary = review_selected(selected, gold, retrieve, tokenizer)
    write_once(private / "current-opportunity.json", reviews)
    write_once(private / "selected-gold.json", [gold[i].gold_row() for i in ids])
    write_once(private / "selected-inference.json", [by_id[i] for i in ids])
    protocol = {
        "comparison_protocol": PAIR, "status": "cpu_prepared_gpu_not_released", "source_git": source_git,
        "scope": "previously_gold_preparation_seen_eligible_train_not_independent_test",
        "ordered_claim_ids": ids, "arms": list(POLICIES), "routes": list(ROUTES),
        "prompt_sha256": {p: policy_sha(p) for p in POLICIES}, "budget": V3Budget().model_dump(),
        "slots_per_arm": len(ids) * 4, "total_slots": len(ids) * 8,
        "fresh_baseline_required": True, "old_G_results_reusable": False,
        "initial_context_gate": "ordered visible sentence and preview IDs/hashes equal; missing fails closed",
        "packing": "max(oldG empty,newG empty,both versions of active actual history,active actual prompt)",
        "decoding": "unchanged greedy nonthinking bounded grammar", "model_calls": 0,
        "generator_manifest_sha256": "d1dd9783afdf4e0fbd21eee824834d71b86982f5a5d5f6f371fe07f2f76f3cf6",
        "reranker_manifest_sha256": "de1d4ac39101816774439e68881e2308c5e5f1bd94d0b0dc4c492a56c2681052",
        "proposed_resources_per_arm_not_submitted": {"a100": 1, "cpus": 8, "mem_gib": 32,
            "wall_seconds": len(ids) * 4 * 120 + 1440},
        "resource_basis": "120s per question +1440s staging/load/4x45s preflight margin; prior F/G MaxRSS<18GiB",
        "total_question_input_token_cap": len(ids) * 8 * 5 * 8192,
        "total_question_output_token_cap": len(ids) * 8 * 5 * 512,
        "preflight_calls_per_arm_separate": 4, "gpu_authorized": False,
        "future_release_needs": ["frozen runner/operator/source archive", "synthetic real-provider preflight", "separate GPU release"],
    }
    write_once(private / "future-pair-protocol-draft.json", protocol)
    compact = {"schema_version": "scifact-semantic-cpu-preparation-v1", "source_git": source_git,
        "source_archive_sha256": source_sha, "exact_source_tree_verified": True,
        "job_id": os.environ["SLURM_JOB_ID"], "original_preparation_sha256": PREP_SHA,
        "legacy_manifest_sha256": LEGACY_SHA, "legacy_audit_sha256": AUDIT_SHA,
        "assignment_sha256": ASSIGNMENT_SHA, "tokenizer_sha256": TOKENIZER_HASHES,
        "model_consumed_unique_ids": len(ledger["model_consumed_ids"]), "uncertain_unique_ids": len(ledger["uncertain_ids"]),
        "excluded_components": len(ledger["component_excluded"]),
        "excluded_eligible_ids": len(ledger["component_excluded_eligible_ids"]), "gold_preparation_seen": 531,
        "selected_queries": len(ids), "selected_gold_rows_parsed": len(gold), "model_calls": 0,
        "official_dev_read": False, "retired_or_unlabelled_test_read": False,
        "gpu_authorized": False, "matching": matching, "packing": summary,
        "prompt_sha256": protocol["prompt_sha256"], "slots_per_arm_draft": len(ids) * 4,
        "private_output_sha256": {p.name: file_sha256(p) for p in private.iterdir()},
        "boundary": "Contract/preparation only; fixture history is not real policy success; selected train is not independent test."}
    write_once(output / "compact.json", compact)
    return {"status": "cpu_prepared_not_model_evaluated", "compact_sha256": file_sha256(output / "compact.json")}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inputs", type=Path, required=True)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--source-git", required=True)
    parser.add_argument("--source-archive", type=Path, required=True)
    parser.add_argument("--source-sha", required=True)
    args = parser.parse_args()
    print(json.dumps(prepare(args.inputs, args.root, args.output, args.source_git, args.source_archive, args.source_sha)))


if __name__ == "__main__":
    main()
