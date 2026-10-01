"""Allocated CPU-only diagnosis of existing TRAIN24; no model or score rewrite."""
from __future__ import annotations

import argparse
from collections import Counter
import json
import os
from pathlib import Path
import tarfile
from typing import Any

from audit_scifact_bounded_closeout import strict_whole_answer
from climate_rag.scifact_fit_selection import select_complete_fit
from climate_rag.scifact_grounding import GoldClaim, LABEL_TO_PROJECT, Rationale, parse_abstract
from climate_rag.scifact_read_continuation import ordered_write
from climate_rag.scifact_semantic_contract import CORPUS_SHA, checked
from climate_rag.scifact_scoring import ClaimPrediction, PredictedAbstract
from prepare_scifact_natural import SELECTION_SHA
from prepare_scifact_state_supervision import member
from prepare_scifact_utility8 import ARCHIVE, ARCHIVE_SHA, PREP, TRAIN_SHA
from run_scifact_grounding_train_operator import ROOT, require, sha

RUN = ROOT / "runs/scifact-document-verifier-v1-20261001-r2"
QUALITY_SHA = "95c43ad2e30cd65b0193a6a8bc42a9c97e2dd5796169c6a97c95743ce08a58e9"
COMPLETE_SHA = "5faae91c1041623243c32f2b5cd30a8064ea7962d8f890c0c2a610491c1117d4"
FRAMES_SHA = "b45dce8612962ddcb1a832a86186dcbf440254a50eb65451c8d56c88aad9216a"


def match(label: str | None, indices: list[int], rats: tuple[Rationale, ...]) -> tuple[bool, bool]:
    """Original alternative-rationale OR and first-three ordering, not any-hit."""
    require(bool(rats) and len({r.label for r in rats}) == 1, "single_gold_document_label")
    return label == LABEL_TO_PROJECT[rats[0].label], any(set(r.sentences) <= set(indices[:3]) for r in rats)


def diagnose_claim(gold: GoldClaim, frame: Any, row: Any) -> dict[str, Any]:
    require(row["claim_id"] == gold.claim_id and row["arm"] == "fixed", "paired_fixed_identity")
    aliases: dict[str, int] = {}
    visible: dict[int, set[int]] = {}
    for sid, value in frame["visible"].items():
        alias, index = sid.rsplit(":", 1)
        doc = int(value["source_id"])
        require(alias not in aliases or aliases[alias] == doc, "one_alias_one_document")
        aliases[alias] = doc
        visible.setdefault(doc, set()).add(int(index))
    feedback = row["verification_feedback"]
    require(len(feedback) == 4 and all(f["status"] == "valid" and not f["model_selected"] for f in feedback),
            "four_frozen_scripted_verifiers")
    judgments = {}
    for f in feedback:
        j = f["judgment"]
        doc = aliases[j["source_id"]]
        require(int(f["provenance"]["original_source_id"]) == doc and doc not in judgments, "verifier_doc_identity")
        require(all(s.rsplit(":", 1)[0] == j["source_id"] for s in j["sentence_ids"]), "same_document_ids")
        judgments[doc] = (j["label"], [int(s.rsplit(":", 1)[1]) for s in j["sentence_ids"]])
    final = row["prediction"]["evidence"] if row["state"] == "valid_terminal" else {}
    counters: Counter[str] = Counter()
    pairs: Counter[str] = Counter()
    all_gold_joint = bool(gold.evidence)
    for doc, rats in gold.evidence.items():
        counters["gold_docs"] += 1
        reachable = any(len(r.sentences) <= 3 and set(r.sentences) <= visible.get(doc, set()) for r in rats)
        counters["initial_first3_reachable_gold_docs"] += reachable
        if doc not in judgments:
            all_gold_joint = False
            continue
        counters["verified_gold_docs"] += 1
        counters["verified_first3_reachable_gold_docs"] += reachable
        label, indices = judgments[doc]
        label_ok, evidence_ok = match(label, indices, rats)
        joint = label_ok and evidence_ok
        counters["verifier_label_correct_gold_docs"] += label_ok
        counters["verifier_first3_evidence_correct_gold_docs"] += evidence_ok
        counters["verifier_label_and_first3_correct_gold_docs"] += joint
        counters["verifier_full_selection_covers_rationale_gold_docs"] += any(
            set(r.sentences) <= set(indices) for r in rats)
        counters["verifier_selection_has_unannotated_sentences_gold_docs"] += not set(indices) <= {
            i for r in rats for i in r.sentences}
        predicted = final.get(str(doc), {})
        final_label, final_evidence = match(LABEL_TO_PROJECT.get(predicted.get("label")), predicted.get("sentences", []), rats)
        final_joint = final_label and final_evidence
        pairs[f"verifier_{int(joint)}_final_{int(final_joint)}"] += 1
        counters["verifier_first3_joint_lost_to_invalid_terminal"] += joint and row["state"] != "valid_terminal"
        counters["verifier_first3_joint_lost_to_valid_terminal"] += joint and row["state"] == "valid_terminal" and not final_joint
        all_gold_joint = all_gold_joint and joint
    non_gold_positive = sum(doc not in gold.evidence and j[0] != "INSUFFICIENT" for doc, j in judgments.items())
    inverse_labels = {v: k for k, v in LABEL_TO_PROJECT.items()}
    projected = ClaimPrediction(gold.claim_id, {
        doc: PredictedAbstract(inverse_labels[label], tuple(indices))
        for doc, (label, indices) in judgments.items() if label != "INSUFFICIENT"})
    # Annotation agreement only, not a generated final answer or a changed score.
    # In particular, four negative document judgments cannot establish claim NEI.
    strict_annotation_match = strict_whole_answer(projected, gold) if gold.evidence else None
    return {"claim_id": gold.claim_id, "positive": bool(gold.evidence), "counts": dict(counters),
        "verified_gold_doc_label_first3_pairs": dict(pairs), "non_gold_positive_verifier_judgments": non_gold_positive,
        "all_gold_docs_label_and_first3_verified": all_gold_joint,
        "positive_verdict_projection_matches_strict_annotations": strict_annotation_match,
        "terminal_state": row["state"],
        "note": "Local first3 and full-selection diagnostics are distinct. Unannotated citations are not proven semantic errors. Projection is not a generated final answer; no old score is changed."}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    require(os.name == "posix" and bool(os.environ.get("SLURM_JOB_ID")), "allocated_CPU_only")
    analysis_source = (Path(__file__).resolve().parents[1] / "SOURCE_REVISION").read_text().strip()
    require(len(analysis_source) == 40 and all(c in "0123456789abcdef" for c in analysis_source), "exact_analysis_source")
    require(not args.output.exists() and args.output.parent.is_dir(), "exclusive_new_compact_output")
    complete = json.loads(checked(RUN / "complete.json", COMPLETE_SHA))
    require(complete["status"] == "scored" and complete["source_git"] == "86414e39d4bd9516c9f8cc2d20c10b184c56f2b7", "completed_frozen_r2")
    quality = json.loads(checked(RUN / "quality.json", QUALITY_SHA))
    require(json.loads((RUN / "worker-exit.json").read_bytes())["child_reaped"] is True, "only_after_exit")
    selection = json.loads(checked(RUN / "prepared/selection.json", SELECTION_SHA))
    ids = [r["id"] for r in selection["selected"]]
    require(len(ids) == len(set(ids)) == 24, "same_24_no_reselection")
    corpus = {d.doc_id: d for d in (parse_abstract(json.loads(r)) for r in
        checked(RUN / "prepared/inference/corpus.jsonl", CORPUS_SHA).splitlines())}
    require(sha(ROOT / "envs" / ARCHIVE) == ARCHIVE_SHA, "original_archive")
    with tarfile.open(ROOT / "envs" / ARCHIVE) as bundle:
        with member(bundle, PREP + "/gold/claims_train.jsonl") as stream:
            gold = select_complete_fit(stream, TRAIN_SHA, ids, corpus)
    frames = json.loads(checked(RUN / "inference/initial-frames.json", FRAMES_SHA))
    counts: Counter[str] = Counter()
    pairs: Counter[str] = Counter()
    cases = []
    input_hashes = {}
    for i, frame in zip(ids, frames, strict=True):
        path = RUN / "inference" / f"{i}-fixed/result.json"
        input_hashes[f"{i}-fixed/result.json"] = sha(path)
        row = diagnose_claim(gold[i], frame, json.loads(path.read_bytes()))
        row["actual_strict_correct"] = next(c["strict_whole_answer"] for c in quality["arms"]["fixed"]["cases"] if c["claim_id"] == i)
        cases.append(row)
        counts.update(row["counts"])
        pairs.update(row["verified_gold_doc_label_first3_pairs"])
    report = {"audit": "document_verifier_gold_decomposition_v1", "script_sha256": sha(Path(__file__)),
        "analysis_source_git": analysis_source,
        "execution_source": complete["source_git"], "job_id": os.environ["SLURM_JOB_ID"],
        "quality_sha256": QUALITY_SHA, "complete_sha256": COMPLETE_SHA, "train_member_sha256": TRAIN_SHA,
        "claims": 24, "positive_claims": sum(c["positive"] for c in cases), "verifier_calls": 96,
        "counts": dict(counts), "verified_gold_doc_label_first3_pairs": dict(pairs),
        "non_gold_positive_verifier_judgments": sum(c["non_gold_positive_verifier_judgments"] for c in cases),
        "all_gold_docs_label_and_first3_verified_claims": sum(c["all_gold_docs_label_and_first3_verified"] for c in cases),
        "positive_verdict_projection_matches_strict_annotations_claims": sum(
            c["positive_verdict_projection_matches_strict_annotations"] is True for c in cases),
        "cases": cases, "input_result_sha256": input_hashes, "original_scores_changed": False,
        "model_calls": 0, "protected_split_read": False, "raw_gold_exported": False}
    require(sha(RUN / "quality.json") == QUALITY_SHA and sha(RUN / "complete.json") == COMPLETE_SHA, "old_outputs_unchanged")
    ordered_write(args.output, report)
    print(json.dumps({k: v for k, v in report.items() if k not in {"cases", "input_result_sha256"}}))


if __name__ == "__main__":
    main()
