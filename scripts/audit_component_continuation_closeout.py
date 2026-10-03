"""One read-only physical closeout; preserve the already-written independent score.

Run with the a104115 source on PYTHONPATH. Only compact aggregates leave Spartan.
No provider, model, rescoring entrypoint, scheduler mutation or new dataset use.
"""
from __future__ import annotations

import json
import subprocess
from collections import Counter
from pathlib import Path

import climate_rag.component_continuation as continuation
from climate_rag.component_audit import reconcile
from climate_rag.component_continuation import audit_physical_layout, cost_partition, load_carried, measurement, policy_identity
from climate_rag.component_execution import SLOTS_SHA, TARGETS_SHA, durable, frozen_slots
from climate_rag.scifact_component_contract import require
from climate_rag.scifact_semantic_contract import MODEL_SHA, checked, encoded, sha
from run_budget_agent_full_operator import ROOT

SOURCE = "a104115ecc370223617ee3238fb7e5b24af526ea"
ARCHIVE = "ad9ebf177ca03cf5bf8e44c38ee4fa3aa2e28c92af0b0178672bdc8d160be0e1"
COMPACT = "96bbd5a9f754023894c5f19ef340fb8da4737533fce937639bcd9472b583f7dc"
JOB = "31743877"


def main() -> None:
    source = Path(continuation.__file__).resolve().parents[2]
    require((source / "SOURCE_REVISION").read_text().strip() == SOURCE, "frozen_audit_dependencies")
    run = ROOT / "runs" / continuation.RELEASE
    compact = json.loads(checked(run / "compact.json", COMPACT))
    marker = json.loads(checked(run / "inference-exited.json", compact["inference_exit_sha256"]))
    require(marker["child_reaped"] and marker["returncode"] == 0 and marker["worker_failure"] is None, "terminal_exit")
    identity = json.loads(checked(run / "worker-identity.json", marker["worker_identity_sha256"]))
    state = json.loads((run / "operator-status.json").read_bytes())
    finished = json.loads((run / "worker-finished.json").read_bytes())
    for record in (compact, identity, state):
        require(record["source_git"] == SOURCE and record["source_archive_sha256"] == ARCHIVE
            and record["release"] == continuation.RELEASE, "execution_identity")
    require(compact["model_manifest_sha256"] == identity["model_manifest_sha256"] == MODEL_SHA
        and identity["scoring_targets_loaded"] is False and finished["scoring_targets_loaded"] is False,
        "model_and_target_boundary")
    require(state["job_id"] == JOB and state["compact_sha256"] == COMPACT
        and state["status"] == finished["status"] == compact["status"] == "complete"
        and state["stage"] == "scoring_after_exit" and state["targets_read"] is True, "terminal_state")
    require(identity["preflight_policy"] == state["preflight_policy"] == compact["preflight_policy"] == policy_identity(), "policy")
    require((run / "worker-finished.json").stat().st_mtime_ns <= (run / "inference-exited.json").stat().st_mtime_ns
        <= (run / "private-score.json").stat().st_mtime_ns <= (run / "compact.json").stat().st_mtime_ns,
        "completion_exit_score_order")
    carried = load_carried(ROOT / "runs" / continuation.PREVIOUS_RELEASE)
    require(identity["carried_reference"] == compact["carried_reference"] == carried["reference"], "carry_identity")
    inference = run / "inference"
    audit_physical_layout(inference, {f"preflight-{i:02d}" for i in (1, 2, 3)} |
        {f"diagnostic-{i:02d}" for i in range(33)}, {"carried-reference.json", "run.json"})
    require(json.loads((inference / "carried-reference.json").read_bytes()) == carried["reference"], "carry_reference")
    preflight = [measurement(carried["record"], 0, carried=True)] + [
        measurement(reconcile(slot, inference, "preflight"), slot["slot"], carried=False)
        for slot in identity["preflight"][1:]]
    prep = ROOT / "posthoc/scifact-component-preparation-426ff7343fb3"
    slots = frozen_slots(prep / "inference/slots.json")
    records = [reconcile(slot, inference, "diagnostic") | {"execution_origin": "new"} for slot in slots]
    execution = json.loads((inference / "run.json").read_bytes())
    require(execution["preflight"] == preflight and execution["diagnostic"] == records, "executor_physical_equality")
    partition = cost_partition(preflight, records)
    require(partition == execution["cost_partition"] == compact["cost_partition"], "paid_cost_equality")
    require(partition["new"]["attempted"] == 36 and partition["cumulative"]["attempted"] == 37
        and partition["cumulative"]["unknown_cost_attempts"] == 0, "all_attempts")
    require(len(list(inference.glob("*/started.json"))) == 36 and not (inference / "preflight-00").exists(), "no_duplicate_call")
    require([r["semantic_match"] for r in preflight] == [False, True, True, True], "negative_carry_retained")

    # Join already-written private scores to freshly verified physical records.
    # Do not invoke score_run, aggregate, or write over any original score.
    scored = json.loads(checked(run / "private-score.json", compact["private_score_sha256"]))
    targets = json.loads(checked(prep / "scoring/targets.json", TARGETS_SHA))
    require(compact["targets_sha256"] == TARGETS_SHA and compact["slots_sha256"] == SLOTS_SHA, "frozen_input_hashes")
    require(len(scored) == len(targets) == 33 and [r["slot"] for r in scored] == list(range(33)), "score_denominators")
    for physical, saved, target in zip(records, scored, targets, strict=True):
        require(all(saved[k] == v for k, v in physical.items()) and saved["claim_id"] == target["claim_id"]
            and saved["component"] == target["component"], "physical_score_join")
    confusion: dict[str, Counter[str]] = {"all": Counter(), "gold_document": Counter(), "nei_control": Counter()}
    screen_counts: Counter[str] = Counter()
    for slot, saved, target in zip(slots, scored, targets, strict=True):
        if saved["component"] == "relation":
            expected = target["target"]["relation"]
            predicted = saved["prediction"].get("relation") or "ABSTAIN"
            key = expected + "->" + predicted
            confusion["all"][key] += 1
            confusion["nei_control" if target["target"]["nei_control"] else "gold_document"][key] += 1
        elif saved["component"] == "screening":
            pool, chosen = set(target["target"]["pool_doc_ids"]), set(saved["prediction"]["document_ids"])
            require(pool == {d["document_id"] for d in slot["input"]["documents"]}, "screening_pool_identity")
            screen_counts["candidate_occurrences"] += len(pool)
            screen_counts["selected_all_candidates_cases"] += chosen == pool
            screen_counts["cases_with_no_annotated_gold_in_pool"] += not bool(set(target["target"]["relevant_doc_ids"]) & pool)
            if not target["target"]["relevant_doc_ids"]:
                screen_counts["official_nei_claims"] += 1
                screen_counts["selected_on_official_nei_claims"] += len(chosen)
    rationale = [r["score"] for r in scored if r["component"] == "rationale"]
    diagnostic = {"relation_confusion": {k: dict(v) for k, v in confusion.items()},
        "screening_pool_observations": dict(screen_counts),
        "rationale_failures": {
            "no_complete_alternative": sum(not r["alternative_complete"] for r in rationale),
            "complete_but_not_in_first3": sum(r["alternative_complete"] and not r["first3_complete"] for r in rationale),
            "complete_but_not_exact": sum(r["alternative_complete"] and not r["exact_any_alternative"] for r in rationale)},
        "formal_status_counts": dict(Counter(r["status"] for r in records)),
        "formal_error_categories": dict(Counter(r["error_category"] for r in records if "error_category" in r))}
    resource = subprocess.run(["sacct", "-n", "-j", JOB, "-P", "--units=K",
        "--format=JobIDRaw,State,ExitCode,Start,End,ElapsedRaw,TotalCPU,MaxRSS,AllocTRES"],
        capture_output=True, text=True, timeout=30, check=True)
    fields = ["job_id", "state", "exit_code", "start", "end", "elapsed_seconds", "total_cpu", "max_rss", "allocated_tres"]
    resources = [dict(zip(fields, line.split("|"), strict=True)) for line in resource.stdout.splitlines() if line.strip()]
    require({r["job_id"] for r in resources} == {JOB, JOB + ".batch", JOB + ".extern"}
        and all(r["state"] == "COMPLETED" and r["exit_code"] == "0:0" for r in resources), "slurm_terminal")
    names = ["compact.json", "private-score.json", "worker-identity.json", "worker-finished.json",
             "inference-exited.json", "operator-status.json"]
    before = {name: sha((run / name).read_bytes()) for name in names}
    physical_hashes = {p.relative_to(inference).as_posix(): sha(p.read_bytes()) for p in inference.rglob("*") if p.is_file()}
    report = {"schema": "component-continuation-physical-closeout-v1", "job_id": JOB,
        "analysis_script_sha256": sha(Path(__file__).read_bytes()), "execution_source": SOURCE,
        "execution_archive_sha256": ARCHIVE, "original_artifact_hashes": before,
        "old_artifact_reference": carried["reference"], "new_physical_files": len(physical_hashes),
        "new_physical_manifest_sha256": sha(encoded(physical_hashes)), "reconciled_preflight": 4,
        "reconciled_formal": 33, "new_reservations": 36, "original_scorer_not_rerun": True,
        "exit_before_score_proved": True, "private_score_matches_physical_records": True,
        "cost_partition": partition, "worker_resources": finished, "operator_elapsed_seconds": state["elapsed_seconds"],
        "slurm_resources": resources, "additional_diagnostics": diagnostic,
        "boundary": "already_consumed_TRAIN_component_diagnostic_not_Agent_or_generalization",
        "new_model_calls_in_audit": 0, "raw_or_gold_exported": False}
    output = ROOT / "posthoc/component-continuation-closeout-31743877"
    output.mkdir(mode=0o700)
    durable(output / "physical-file-hashes.json", physical_hashes)
    durable(output / "physical-closeout.json", report)
    require(before == {name: sha((run / name).read_bytes()) for name in names}, "original_outputs_unchanged")
    print(json.dumps({"report_sha256": sha((output / "physical-closeout.json").read_bytes()), "diagnostics": diagnostic}))


if __name__ == "__main__":
    main()
