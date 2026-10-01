"""Freeze eight outcome-selected exposed TRAIN IDs; no new gold deserialization."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import tarfile
from typing import Any, BinaryIO

from climate_rag.scifact_fit_selection import IdScanner, MAX_LINE, no_duplicate_keys
from climate_rag.scifact_read_continuation import ordered_write
from climate_rag.scifact_semantic_contract import CORPUS_SHA, checked
from climate_rag.scifact_utility_contract import ARMS, PROTOCOL, identity
from prepare_scifact_state_supervision import member
from run_scifact_grounding_train_operator import ROOT, require, sha

ARCHIVE = "scifact-semantic-inputs-c4907db6.tar"
ARCHIVE_SHA = "c4907db623b644c9c883a807f07597f87746595da7fc6435714452304e50a44d"
PREP = "data/scifact-original-20260930/prepared-r1"
MANIFEST_SHA = "3cd6bc1e1c299ece9195098a3853ebb05bb3401507d8b467c1596ba6afb6e138"
TRAIN_SHA = "9e07864a797dd515391fa1c3f3573249b70399f95ff6c740bf536ad11c8aa6ca"
OLD = "posthoc/scifact-shared-corpus-fit-6fd92eb0ac91"
SUPPLEMENTAL = "posthoc/scifact-supplemental-fit-09f3d9e72216"
OLD_COMPACT_SHA = "27ad7cec3bb9aa1d9c12cd8618d9f9546956ed8b171e0483950333a3959c686d"
OLD_REPORT_SHA = "4120b3f65793f5807d26ce256e9b154f11cf133655b46a25a7c5bcf06da0faea"
SUPPLEMENTAL_COMPACT_SHA = "4ef3d782ecde3efd36ca9dfef1482b9117f3f0ed695cca020bacdec8c047284e"
CORPUS = "posthoc/scifact-grounding-candidate-40d84a377bd1/inference/corpus.jsonl"


def select_eight(old: list[dict[str, Any]], supplemental: list[dict[str, Any]]) -> list[dict[str, Any]]:
    require(len(old) == 48 and len(supplemental) == 96, "fixed_report_cohorts")
    reads = [r for r in old if r["program_teacher_reads"] == 1]
    require(len(reads) == 1, "unique_original_natural_read")
    groups = [("original_natural_read", reads, 1),
              ("original_initial_sufficient", [r for r in old if r["initial_complete"]], 2),
              ("supplemental_initial_sufficient", [r for r in supplemental if r["initial_complete"]], 2),
              ("supplemental_context_abstain", [r for r in supplemental if r["training_ready"]
                and not r["initial_complete"] and not r["post_read_complete"]], 1),
              ("supplemental_official_nei", [r for r in supplemental
                if r["official_label_scope"] == "NOT_ENOUGH_INFO"], 2)]
    result: list[dict[str, Any]] = []
    for role, rows, count in groups:
        require(len(rows) >= count, "fixed_selection_unavailable_no_replacement")
        result.extend({"id": r["claim_id"], "component": r["component"], "role": role} for r in rows[:count])
    require(len(result) == len({r["id"] for r in result}) == len({r["component"] for r in result}) == 8,
            "eight_unique_claim_components")
    return result


def select_queries(stream: BinaryIO, expected_sha: str, ids: list[int]) -> list[dict[str, Any]]:
    digest, selected, seen = hashlib.sha256(), {}, set()
    while raw := stream.readline(MAX_LINE + 1):
        digest.update(raw)
        i = IdScanner(raw).identifier()
        require(i not in seen, "duplicate_query_id")
        seen.add(i)
        if i in ids:
            selected[i] = raw
    require(digest.hexdigest() == expected_sha and set(selected) == set(ids), "query_member_identity_or_coverage")
    rows = [json.loads(selected[i], object_pairs_hook=no_duplicate_keys) for i in ids]
    require(all(set(r) == {"id", "claim"} for r in rows), "gold_free_query_schema")
    return rows


def prepare(out: Path, root: Path = ROOT) -> dict[str, Any]:
    old = json.loads(checked(root / OLD / "compact.json", OLD_COMPACT_SHA))
    new = json.loads(checked(root / SUPPLEMENTAL / "compact.json", SUPPLEMENTAL_COMPACT_SHA))
    require(old["private_file_sha256"]["claim-reports.json"] == OLD_REPORT_SHA, "original_report_binding")
    report_sha = new["private_file_sha256"]["claim-reports.json"]
    selected = select_eight(json.loads(checked(root / OLD / "claim-reports.json", OLD_REPORT_SHA)),
                            json.loads(checked(root / SUPPLEMENTAL / "claim-reports.json", report_sha)))
    out.mkdir(mode=0o700)
    # Freeze before decoding any new query string. Failures never replace an ID.
    selection = {"protocol": PROTOCOL, "selected": selected, "branch_order": list(ARMS),
        "outcome_selected_diagnostic_not_new_evaluation": True,
        "old_report_sha256": OLD_REPORT_SHA, "supplemental_report_sha256": report_sha,
        "old_compact_sha256": OLD_COMPACT_SHA, "supplemental_compact_sha256": SUPPLEMENTAL_COMPACT_SHA}
    ordered_write(out / "selection.json", selection)
    ids = [r["id"] for r in selected]
    bundle = root / "envs" / ARCHIVE
    require(sha(bundle) == ARCHIVE_SHA, "original_archive_identity")
    with tarfile.open(bundle) as archive:
        with member(archive, PREP + "/preparation-manifest.json") as stream:
            raw = stream.read()
        require(hashlib.sha256(raw).hexdigest() == MANIFEST_SHA, "preparation_manifest_identity")
        manifest = json.loads(raw)
        query_sha = manifest["output_file_sha256"]["inference/claims_train_eligible.jsonl"]
        require(manifest["output_file_sha256"]["gold/claims_train.jsonl"] == TRAIN_SHA, "later_scoring_member_identity")
        with member(archive, PREP + "/inference/claims_train_eligible.jsonl") as stream:
            claims = select_queries(stream, query_sha, ids)
    inference = out / "inference"
    inference.mkdir(mode=0o700)
    ordered_write(inference / "claims.json", claims)
    corpus = checked(root / CORPUS, CORPUS_SHA)
    with (inference / "corpus.jsonl").open("xb") as stream:
        stream.write(corpus)
    receipt = {"protocol": PROTOCOL, "claims_sha256": sha(inference / "claims.json"),
        "corpus_sha256": CORPUS_SHA, "selection_sha256": sha(out / "selection.json"),
        "query_member_sha256": query_sha, "ordered_ids_sha256": identity(ids),
        "official_gold_decoded": False, "protected_split_read": False}
    ordered_write(out / "preparation.json", receipt)
    return receipt
