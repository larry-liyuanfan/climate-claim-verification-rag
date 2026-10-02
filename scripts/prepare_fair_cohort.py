"""Validation-only label-blind component projection and single deterministic freeze.

Reads only explicitly named archive members; no sealed test, model or result
file. Gold export occurs AFTER selection and is never inside the worker cohort.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
import hashlib
import json
from pathlib import Path
import tarfile
from typing import Any

from climate_rag.climate_fever import _UnionFind, _near_duplicate_pairs, _normalise, _token_set
from climate_rag.fair_acquisition import PROTOCOL
from climate_rag.fair_replay import STUDY, identity, validate_binding, validate_tasks
from climate_rag.scifact_read_continuation import ordered_write
from climate_rag.targeted_replay import CORPUS_SHA, MODEL_SHA, RERANKER_SHA, SELECTION_SHA
from climate_rag.verification import normalise_claim
from prepare_agent_validation import MEMBER

VALIDATION_SHA = "1154448777e72e2b14e7e57e71937a5f123e36f63b8b7b2c3ab7ddfe9a449ad5"
SALT = "fair-three-arm-components-20261003-v1"


def member_bytes(archive: Path, name: str, expected: str) -> bytes:
    with tarfile.open(archive) as bundle:
        matches = [m for m in bundle.getmembers() if m.name == name]
        if len(matches) != 1 or not matches[0].isfile():
            raise ValueError("named_member_missing_or_duplicate")
        stream = bundle.extractfile(matches[0])
        if stream is None:
            raise ValueError("named_member_unreadable")
        raw = stream.read()
    if hashlib.sha256(raw).hexdigest() != expected:
        raise ValueError("named_member_hash")
    return raw


def components(metadata: dict[str, Any], evidence: dict[str, str]) -> list[list[str]]:
    """Same exact/0.90 Jaccard graph rules as the registered grouped split.

    Only claim text and ALL annotated candidate IDs are permitted here; no
    positive IDs, labels, retrieval coverage or model outcomes drive selection.
    """
    union = _UnionFind(metadata)
    evidence_claims: dict[str, list[str]] = defaultdict(list)
    exact_claims: dict[str, str] = {}
    claims = {}
    for key, row in metadata.items():
        if set(row) != {"claim_text", "annotated_candidate_ids"}:
            raise ValueError("selection_metadata_contains_gold_or_results")
        claims[key] = _token_set(row["claim_text"])
        normal = _normalise(row["claim_text"])
        if normal in exact_claims:
            union.union(exact_claims[normal], key)
        else:
            exact_claims[normal] = key
        for eid in row["annotated_candidate_ids"]:
            if eid not in evidence:
                raise ValueError("missing_annotated_evidence_variant")
            evidence_claims[eid].append(key)
    for ids in evidence_claims.values():
        for key in ids[1:]:
            union.union(ids[0], key)
    for left, right in _near_duplicate_pairs(claims, threshold=0.90):
        union.union(left, right)
    evidence_tokens = {eid: _token_set(evidence[eid]) for eid in evidence_claims}
    exact_evidence: dict[str, str] = {}
    for eid in evidence_claims:
        normal = _normalise(evidence[eid])
        if normal in exact_evidence:
            ids = evidence_claims[exact_evidence[normal]] + evidence_claims[eid]
            for key in ids[1:]:
                union.union(ids[0], key)
        else:
            exact_evidence[normal] = eid
    for left, right in _near_duplicate_pairs(evidence_tokens, threshold=0.90):
        ids = evidence_claims[left] + evidence_claims[right]
        for key in ids[1:]:
            union.union(ids[0], key)
    grouped: dict[str, list[str]] = defaultdict(list)
    for key in sorted(metadata):
        grouped[union.find(key)].append(key)
    return sorted(grouped.values())


def freeze(metadata: dict[str, Any], evidence: dict[str, str], consumed: set[str], *, cap: int = 32) -> tuple[list[dict[str, str]], dict[str, Any]]:
    if not 1 <= cap <= 32 or not consumed <= set(metadata):
        raise ValueError("freeze_cap_or_consumption_roster")
    groups = components(metadata, evidence)
    eligible = [c for c in groups if not set(c) & consumed]
    def order(c: list[str]) -> str:
        return identity({"domain": SALT, "component_ids": c})
    selected_components = sorted(eligible, key=order)[:cap]
    selected = [min(c, key=lambda k: identity({"domain": SALT + ":claim", "id": k})) for c in selected_components]
    tasks = [{"id": key, "claim_text": metadata[key]["claim_text"]} for key in selected]
    audit = {"protocol": PROTOCOL, "salt": SALT, "cap": cap, "components": groups,
        "consumed_ids": sorted(consumed), "selected_component_ids": selected_components,
        "selected_ids": selected, "input_count": len(metadata), "component_count": len(groups),
        "excluded_component_count": len(groups) - len(eligible), "eligible_component_count": len(eligible),
        "eligibility": "component-disjoint from consumed32 under frozen exact/0.90-Jaccard graph",
        "exposure": "ALL validation previously exposed to retrieval selection; development-only, NOT independent test",
        "selection_rule": "one claim per component; first domain-separated SHA order; no label quotas, opportunity filters, replacement or alternate salt",
        "sealed_test_read": False, "model_results_read": False, "labels_used_for_selection": False}
    return tasks, audit


def prepare(validation_archive: Path, inference_archive: Path, consumed_file: Path,
            contract: Path, output: Path) -> dict[str, Any]:
    raw = member_bytes(validation_archive, MEMBER, VALIDATION_SHA)
    # Project before graph selection. Labels are scorer-only; not passed to freeze.
    rows = json.loads(raw)
    metadata = {str(key): {k: row[k] for k in ("claim_text", "annotated_candidate_ids")} for key, row in rows.items()}
    if len(metadata) != 230:
        raise ValueError("registered_validation_count")
    corpus = member_bytes(inference_archive, "evidence.jsonl", CORPUS_SHA)
    documents = [json.loads(line) for line in corpus.splitlines() if line.strip()]
    evidence = {d["evidence_id"]: d["text"] for d in documents}
    if len(evidence) != len(documents) or len(evidence) != 5240:
        raise ValueError("registered_corpus_count")
    consumed_raw = consumed_file.read_bytes()
    consumed = json.loads(consumed_raw)
    if (hashlib.sha256(consumed_raw).hexdigest() != SELECTION_SHA
            or len(consumed["selected_ids"]) != 32 or len(set(consumed["selected_ids"])) != 32):
        raise ValueError("consumed32_roster_required")
    tasks, audit = freeze(metadata, evidence, set(consumed["selected_ids"]))
    audit.update(validation_member_sha256=VALIDATION_SHA, corpus_sha256=CORPUS_SHA,
                 consumption_receipt_sha256=hashlib.sha256(consumed_file.read_bytes()).hexdigest(),
                 graph_provenance="climate_fever.grouped_split registered validation partition; no outside-partition claim read")
    output.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    output.mkdir(mode=0o700)  # exclusive reservation; no repeat/new salt over same outcomes
    ordered_write(output / "cohort.json", tasks)
    ordered_write(output / "exposure-audit.json", audit)
    contract_sha = hashlib.sha256(contract.read_bytes()).hexdigest()
    binding = {"protocol": PROTOCOL, "study_kind": STUDY, "task_count": len(tasks),
        "cohort_sha256": hashlib.sha256((output / "cohort.json").read_bytes()).hexdigest(),
        "tasks_sha256": identity(tasks), "exposure_audit_sha256": hashlib.sha256((output / "exposure-audit.json").read_bytes()).hexdigest(),
        "initial_contract_sha256": contract_sha, "scoring_contract_sha256": contract_sha,
        "corpus_sha256": CORPUS_SHA, "model_sha256": MODEL_SHA, "reranker_sha256": RERANKER_SHA,
        "eligible": True, "selection_rule": audit["selection_rule"]}
    validate_binding(binding)
    validate_tasks(tasks, binding)
    ordered_write(output / "binding.json", binding)
    # AFTER roster frozen. Official annotation proxy, not new semantic judgments.
    gold = {"tasks_sha256": identity(tasks), "provenance": "official CLIMATE-FEVER labels/decisive IDs; no stance/entailment blind annotation",
        "claims": {t["id"]: {"claim_sha256": hashlib.sha256(normalise_claim(t["claim_text"]).encode()).hexdigest(),
            "label": rows[t["id"]]["claim_label"], "evidence_ids": rows[t["id"]]["evidences"]} for t in tasks}}
    scoring = output / "scorer-only"
    scoring.mkdir(mode=0o700)
    ordered_write(scoring / "fair-gold.json", gold)
    return {"task_count": len(tasks), "component_count": audit["component_count"],
        "eligible_component_count": audit["eligible_component_count"], "binding": binding,
        "gold_sha256": hashlib.sha256((scoring / "fair-gold.json").read_bytes()).hexdigest(),
        "model_execution_authorized": False, "test_read": False}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("validation-archive", "inference-archive", "consumed-file", "contract", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(prepare(args.validation_archive, args.inference_archive, args.consumed_file, args.contract, args.output)))


if __name__ == "__main__":
    main()
