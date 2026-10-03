"""TRAIN-only CPU preparation; no weights, generation, scheduler or dev reader.

Frozen claim components are never recomputed. The missing corpus-only document
family map is restored with the original 0.9 token-Jaccard implementation.
All private records stay in the scoped Spartan posthoc directory.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import tarfile
import time
from collections import Counter
from pathlib import Path
from typing import Any

from climate_rag.component_execution import PROTOCOL_SHA, TARGETS_SHA
from climate_rag.component_execution import verify_infrastructure_predecessor
from climate_rag.component_continuation import FILES, audit_physical_layout, policy_identity
from climate_rag.scifact_consumption import verify_bytes
from climate_rag.scifact_grounding import _near_groups, parse_abstract, parse_gold
from climate_rag.scifact_grounding_ledger import cumulative_ledger
from climate_rag.scifact_grounding_sft import (
    CONFIG, VERSION, config_sha, context, fit_records, require, select_components, token_record,
)
from climate_rag.scifact_retrieval import SciFactBM25
from climate_rag.scifact_semantic_contract import CORPUS_SHA, TOKENIZER_SHA, checked, encoded, sha, write_once
from prepare_scifact_components import ROOT, member
from prepare_scifact_semantic_pair import ASSIGNMENT_SHA, PREP, PREP_SHA, RUNS

HISTORICAL_SHA = "65eac7e26e5603ee8672b44eb6b0dce0d6d8544c8759cfa36184daaf0a518a73"
PARENT_SHA = "306e8a61eef6fb279c56ccfc2b496281df9cc8710e194d2272e6978a23dbe802"
SEMANTIC_SHA = "5595143cfb6d276184e91f40856efa388ab440dc78527fe7fcffd52c12bb296d"
SEMANTIC_RUNS = (
    ("climate-scifact-semantic-pair-20260930-v1-scifact-gap-original-v1", "557b7b8ddc7f4e40c1ac1abbe2abd68af37536427f3b106e9e0063e3f6ae746a", "99cd9ff707697ea395a9bc067f3ce91395071cdb"),
    ("climate-scifact-semantic-pair-20260930-v1-scifact-gap-semantic-policy-v1", "486f1c5866011e926b47fe2848189ce4c121213ab7fc86fc22a2c18ea68403f8", "99cd9ff707697ea395a9bc067f3ce91395071cdb"),
)
COMPONENT_RUNS = (
    ("scifact-component-single-attempt-20261001-r2", "6f809e673e43ec5ffa7d1b0c72cac8f368203f9d4d97990e98f015515fe3971b"),
    ("scifact-component-technical-continuation-20261001-v1", "183a2059f987e1289b35094ee3a3861747eba0b2d2ed905ba18085f29e025c7f"),
)


def collect_registry(root: Path) -> list[dict[str, Any]]:
    protocols = {}
    for path, digest in (
        ("envs/scifact-train-inference-92846f0904992e7b7137e550c46cd0c76b2f77b4f808a6d6be25482f90bf1cca.tar", PARENT_SHA),
        ("envs/scifact-semantic-bundles-99cd9ff/inference.tar", SEMANTIC_SHA),
    ):
        with tarfile.open(root / path) as bundle:
            raw = member(bundle, "protocol.json")
        verify_bytes(raw, digest)
        protocols[digest] = json.loads(raw)
    registry = []
    for name, digest, source in (*RUNS, *SEMANTIC_RUNS):
        directory = root / "runs" / name / "inference"
        run = json.loads(checked(directory / "run.json", digest))
        protocol = protocols[run["protocol_sha256"]]
        expected = protocol.get("slots", protocol.get("slots_per_arm"))
        require(run["source_git"] == source and run["gold_loaded"] is False, "historical_source")
        execution_sha = run["protocol_sha256"]
        if "pair_protocol_sha256" in run:
            execution_sha = "4cc5033928a0833ee99ad64ecd5a7b10c7a1fde7ac4f219d6334d2349d9daaf6"
            require(run["pair_protocol_sha256"] == execution_sha
                    and run["arm"] in {"format_repaired", "format_repaired_gap"}, "historical_pair_policy")
        require([(r["claim_id"], r["route"]) for r in run["runs"]]
                == [(r["claim_id"], r["route"]) for r in expected], "historical_protocol_slots")
        attempts = []
        for index, row in enumerate(run["runs"]):
            require(row["result"]["claim_id"] == row["claim_id"], "nested_claim_identity")
            for j, attempt in enumerate(row["result"]["generation_attempts"]):
                attempts.append({"physical_id": f"{name}/slot-{index+1:02d}/attempt-{j}",
                    "claim_id": row["claim_id"], "receipt_sha256": sha(encoded(attempt)),
                    "status": "unknown" if not attempt.get("usage_known") else
                              ("completed" if attempt.get("status") == "valid_decision" else "failed"),
                    "original_status": attempt.get("status"), "usage_known": attempt.get("usage_known"), "carried": False})
        flight = json.loads(checked(directory / "runtime-preflight.json", run["preflight_sha256"]))
        for j, record in enumerate(flight["records"]):
            attempts.append({"physical_id": f"{name}/synthetic-{j}", "claim_id": None,
                "receipt_sha256": sha(encoded(record)), "status": "completed" if record["passed"] else "failed", "carried": False})
        registry.append({"run_id": name, "protocol_sha256": run["protocol_sha256"],
            "receipt_sha256": digest, "planned_claim_ids": sorted({r["claim_id"] for r in expected}),
            "state": "completed", "attempts": attempts, "proven_unattempted_claim_ids": [],
            "execution_protocol_sha256": execution_sha, "arm": run.get("arm")})
    # The initial startup failed before durable inference. Retain it conservatively
    # as unknown exposure; its planned components are already consumed by r2.
    old_ids = registry[0]["planned_claim_ids"]
    registry.insert(0, {"run_id": "climate-scifact-train-diagnostic-20260930-r1-31620529",
        "protocol_sha256": PARENT_SHA, "receipt_sha256": sha(checked(root / "runs/climate-scifact-train-diagnostic-20260930-r1-31620529.log", sha(b""))),
        "state": "failed", "planned_claim_ids": old_ids, "attempts": [],
        "proven_unattempted_claim_ids": [], "exposure_resolution": "conservative_unknown_empty_log_not_zero_call_proof"})
    prep = root / "posthoc/scifact-component-preparation-426ff7343fb3"
    protocol = json.loads(checked(prep / "protocol.json", PROTOCOL_SHA))
    targets = json.loads(checked(prep / "scoring/targets.json", TARGETS_SHA))
    require(protocol["scoring_targets_sha256"] == TARGETS_SHA and len(targets) == 33,
            "component_target_binding")
    target_ids = sorted({r["claim_id"] for r in targets})
    failed = root / "runs/scifact-component-single-attempt-20261001-v1"
    failure_sha = "f72d25cb3e431afc48526fe976dc655791dfa7d0eca5d134a800c9ace51c440d"
    failure = json.loads(checked(failed / "operator-status.json", failure_sha))
    verify_infrastructure_predecessor(root / "runs")
    require(failure["status"] == "failed_no_automatic_retry" and not (failed / "inference").exists(), "component_failure_evidence")
    registry.append({"run_id": failed.name, "protocol_sha256": PROTOCOL_SHA,
        "receipt_sha256": failure_sha, "planned_claim_ids": target_ids,
        "state": "failed", "attempts": [], "proven_unattempted_claim_ids": target_ids})
    for name, digest in COMPONENT_RUNS:
        directory = root / "runs" / name / "inference"
        run = json.loads(checked(directory / "run.json", digest))
        if name == COMPONENT_RUNS[0][0]:
            for relative, expected_sha in FILES.items():
                checked(root / "runs" / name / relative, expected_sha)
            audit_physical_layout(directory, {"preflight-00"}, {"run.json"})
        else:
            require(run["policy"] == policy_identity(), "continuation_execution_policy")
            manifest = json.loads(checked(root / "posthoc/component-continuation-closeout-31743877/physical-file-hashes.json",
                "011078dc2849557d45d400ea419f04c3c3ef3bc12364bd677fa32e7a59f0f696"))
            for relative, expected_sha in manifest.items():
                checked(directory / relative, expected_sha)
            require({p.relative_to(directory).as_posix() for p in directory.rglob("*") if p.is_file()}
                    == set(manifest), "unregistered_component_physical_file")
        require(len(run["diagnostic"]) == 33 and [r["slot"] for r in run["diagnostic"]] == list(range(33)),
                "component_slot_matrix")
        attempts = []
        called = set()
        for phase in ("preflight", "diagnostic"):
            for row in run[phase]:
                if not row["attempted"]:
                    require(row["status"] == "not_attempted_after_stop", "unattempted_status")
                    require(not (directory / f'{phase}-{row["slot"]:02d}' / "started.json").exists(),
                            "unattempted_has_reservation")
                    continue
                carry = row.get("execution_origin") == "carried"
                origin = COMPONENT_RUNS[0][0] if carry else name
                slot = f'{phase}-{row["slot"]:02d}'
                reservation = root / "runs" / origin / "inference" / slot / "started.json"
                require(reservation.is_file() and not reservation.is_symlink(), "physical_reservation")
                physical = json.loads(reservation.read_bytes())
                require(physical["slot"] == row["slot"] and physical["phase"] == phase
                        and physical["attempted"] is True and physical["packing"] == row["packing"], "physical_slot_join")
                claim = targets[row["slot"]]["claim_id"] if phase == "diagnostic" else None
                if claim is not None:
                    require(targets[row["slot"]]["component"] == row["component"], "component_target_slot")
                    called.add(claim)
                attempts.append({"physical_id": origin + "/" + slot,
                    "claim_id": claim, "receipt_sha256": sha(reservation.read_bytes()),
                    "status": "unknown" if not row["usage_known"] else ("completed" if row["status"] == "valid" else "failed"),
                    "original_status": row["status"], "usage_known": row["usage_known"], "carried": carry})
        registry.append({"run_id": name, "protocol_sha256": PROTOCOL_SHA,
            "receipt_sha256": digest, "planned_claim_ids": target_ids,
            "state": "completed", "attempts": attempts,
            "proven_unattempted_claim_ids": sorted(set(target_ids) - called),
            "execution_policy": run.get("policy", "semantic-exact-v1")})
    return registry


def prepare(output: Path, tokenizer_dir: Path, source_git: str) -> dict[str, Any]:
    require(output.parent.resolve() == ROOT / "posthoc" and not output.exists(), "scoped_exclusive_output")
    source = Path(__file__).resolve().parents[1]
    require((source / "SOURCE_REVISION").read_text().strip() == source_git and len(source_git) == 40,
            "frozen_source_required")
    started = time.perf_counter()
    archive = ROOT / "envs/scifact-semantic-inputs-c4907db6.tar"
    checked(archive, "c4907db623b644c9c883a807f07597f87746595da7fc6435714452304e50a44d")
    names = ("preparation-manifest.json", "private-group-assignment.json", "inference/corpus.jsonl", "gold/claims_train.jsonl")
    with tarfile.open(archive) as bundle:
        raw = {n: member(bundle, PREP + "/" + n) for n in names}
    verify_bytes(raw[names[0]], PREP_SHA)
    original = json.loads(raw[names[0]])
    for n in names[1:]:
        verify_bytes(raw[n], original["output_file_sha256"][n])
    verify_bytes(raw[names[1]], ASSIGNMENT_SHA)
    verify_bytes(raw[names[2]], CORPUS_SHA)
    assignment = json.loads(raw[names[1]])
    eligible = set(assignment["eligible_train_ids"])
    corpus = {a.doc_id: a for a in (parse_abstract(json.loads(line)) for line in raw[names[2]].splitlines())}
    gold = {}
    for line in raw[names[3]].splitlines():
        row = json.loads(line)
        if row["id"] in eligible:
            gold[row["id"]] = parse_gold(row, corpus)
    require(set(gold) == eligible, "eligible_gold_identity")
    historical = json.loads(checked(ROOT / "posthoc/scifact-semantic-preparation-779e49883570/private/consumption-ledger.json", HISTORICAL_SHA))
    ledger = cumulative_ledger(assignment, ASSIGNMENT_SHA, historical, HISTORICAL_SHA, collect_registry(ROOT))
    split = select_components(assignment, {i: g.label_scope for i, g in gold.items()}, ledger["component_excluded"])
    output.mkdir(mode=0o700)
    for name in ("private", "fit", "inference", "scoring"):
        (output / name).mkdir(mode=0o700)
    write_once(output / "private/ledger.json", ledger)
    write_once(output / "private/selection-before-packing.json", split)
    # Restore ONLY corpus document families, not train/dev claim grouping.
    families = _near_groups({i: a.title + " " + " ".join(a.sentences) for i, a in corpus.items()}, 0.9)
    family = {i: families.root(i) for i in corpus}
    require(len(corpus) == original["corpus_documents"]
            and len(set(family.values())) == original["document_variant_components"], "document_family_audit_mismatch")
    owner: dict[int, str] = {}
    component = {i: str(assignment["claim_component"][str(i)]) for i in eligible}
    for i, g in gold.items():
        for doc in set(g.cited_doc_ids) | set(g.evidence):
            f = family[doc]
            require(f not in owner or owner[f] == component[i], "frozen_claim_family_inconsistent")
            owner[f] = component[i]
    partition = {component[i]: part for part, ids in split.items() for i in ids}
    # Unselected eligible TRAIN components get a deterministic background pool.
    # Consumed families cannot enter tune/validation. Unowned families are excluded.
    for c in sorted(set(component.values()) - set(partition)):
        bucket = int(sha(f"{VERSION}:background:{c}".encode())[:8], 16) % 20
        partition[c] = "fit" if c in ledger["component_excluded"] or bucket < 14 else ("tune" if bucket < 17 else "validation")
    pools = {part: {i: a for i, a in corpus.items() if family[i] in owner
                    and partition[owner[family[i]]] == part} for part in split}
    write_once(output / "private/families.json", {"corpus_sha256": CORPUS_SHA, "threshold": 0.9,
        "document_family": family, "family_component": owner, "component_partition": partition,
        "claim_component_sha256": ASSIGNMENT_SHA, "claim_grouping_recomputed": False})
    for n, digest in TOKENIZER_SHA.items():
        checked(tokenizer_dir / n, digest)
    os.environ.update(USE_TORCH="0", USE_TF="0", HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1",
                      TOKENIZERS_PARALLELISM="false")
    require("torch" not in sys.modules, "no_weights_runtime")
    from transformers.models.auto.tokenization_auto import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(str(tokenizer_dir), local_files_only=True, trust_remote_code=False)
    records = []
    failures = []
    for i in split["fit"]:
        try:
            for row in fit_records(gold[i], corpus, component[i]):
                pack = token_record(tokenizer, row["context"], row["target"])
                row["packing"] = {k: v for k, v in pack.items() if k not in {"input_ids", "labels", "attention_mask"}}
                records.append(row)
        except ValueError as exc:
            failures.append({"partition": "fit", "claim_id": i, "category": str(exc)})
    require(len(records) <= CONFIG["max_fit_records"], "fit_record_ceiling")
    write_once(output / "fit/records.json", records)
    evaluation_stats = {}
    for part in ("tune", "validation"):
        retrieve = SciFactBM25(pools[part])
        inputs: list[dict[str, Any]] = []
        for i in split[part]:
            candidates = retrieve(gold[i].claim, CONFIG["candidate_k"])
            try:
                ctx = context(gold[i].claim, [corpus[int(d.source_id)] for d in candidates])
                inputs.append({"claim_id": i, "context": ctx, "packing": token_record(tokenizer, ctx)})
            except ValueError as exc:
                failures.append({"partition": part, "claim_id": i, "category": str(exc)})
        write_once(output / f"inference/{part}.json", inputs)
        write_once(output / f"scoring/{part}.json", [gold[i].gold_row() for i in split[part]])
        availability: Counter[str] = Counter()
        for row in inputs:
            g = gold[row["claim_id"]]
            candidate_ids = set(row["context"]["document_ids"])
            availability["candidate_occurrences"] += len(candidate_ids)
            if not g.evidence:
                availability["official_nei_queries"] += 1
                continue
            covered = set(g.evidence) & candidate_ids
            key = "zero" if not covered else ("all" if covered == set(g.evidence) else "partial")
            availability[key + "_gold_document_coverage_queries"] += 1
            availability["annotated_gold_document_occurrences"] += len(g.evidence)
            availability["gold_document_occurrences_in_candidates"] += len(covered)
            availability["at_least_one_complete_alternative_visible_queries"] += bool(covered)
            availability["at_least_one_first3_eligible_alternative_visible_queries"] += any(
                len(r.sentences) <= 3 for doc in covered for r in g.evidence[doc])
        evaluation_stats[part] = {"queries_prepared": len(inputs), "pool_documents": len(pools[part]),
            "source_owned_other_partition_documents_excluded": sum(len(pools[p]) for p in pools if p != part),
            "input_token_counts": [r["packing"]["input_tokens"] for r in inputs],
            "frozen_candidate_availability_no_backfill": dict(availability)}
    # Original public corpus retained privately here; inputs only use partitioned pools.
    with (output / "inference/corpus.jsonl").open("xb") as handle:
        handle.write(raw[names[2]])
    write_once(output / "private/gaps.json", failures)
    files = {p.relative_to(output).as_posix(): sha(p.read_bytes()) for p in output.rglob("*") if p.is_file()}
    manifest = {"version": VERSION, "status": "blocked_preparation_gap" if failures else "cpu_prepared",
        "source_git": source_git, "config_sha256": config_sha(), "config": CONFIG, "files": files,
        "original_manifest_sha256": PREP_SHA, "assignment_sha256": ASSIGNMENT_SHA,
        "training_or_generation_authorized": False, "official_dev_read": False,
        "claim_grouping_recomputed": False, "selection_after_packing_changed": False}
    write_once(output / "manifest.json", manifest)
    summary = {"version": VERSION, "status": manifest["status"], "source_git": source_git,
        "manifest_sha256": sha(encoded(manifest)), "config_sha256": config_sha(),
        "ledger_sha256": files["private/ledger.json"], "split_sha256": files["private/selection-before-packing.json"],
        "family_sha256": files["private/families.json"], "fit_sha256": files["fit/records.json"],
        "selected_components": {k: len(v) for k, v in split.items()},
        "consumed_claims": len(ledger["model_consumed_ids"]), "excluded_components": len(ledger["component_excluded"]),
        "uncertain_claims": len(ledger["uncertain_ids"]), "registry_runs": len(ledger["runs"]),
        "physical_calls_deduplicated": ledger["new_physical_calls"], "carried_references": ledger["carried_references_deduplicated"],
        "fit_records": len(records), "fit_labels": dict(Counter(r["target"]["documents"][0]["label"] for r in records)),
        "fit_input_token_counts": [r["packing"]["input_tokens"] for r in records],
        "fit_output_token_counts": [r["packing"]["target_tokens"] for r in records],
        "evaluation_packing": evaluation_stats, "gap_counts": dict(Counter(r["category"] for r in failures)),
        "unowned_documents_excluded": sum(f not in owner for f in family.values()),
        "document_families": len(set(family.values())), "preparation_seconds": time.perf_counter() - started,
        "model_calls": 0, "official_dev_read": False, "new_slurm_submissions": 0,
        "weak_negatives_used": False, "boundary": "CPU_only_TRAIN_internal_gold_preparation_seen_no_quality_result"}
    write_once(output / "compact.json", summary)
    return {"status": manifest["status"], "compact_sha256": sha(encoded(summary)), "gaps": summary["gap_counts"]}


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--tokenizer-dir", type=Path, required=True)
    p.add_argument("--source-git", required=True)
    args = p.parse_args()
    print(json.dumps(prepare(args.output, args.tokenizer_dir, args.source_git)))


if __name__ == "__main__":
    main()
