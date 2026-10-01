"""Frozen roster/cost/comparison helpers, reusing the original scientific scorer."""
from __future__ import annotations

import copy
from typing import Any

from climate_rag.scifact_evidence_commit import ARMS, ISOLATED_PROTOCOL
from climate_rag.scifact_relation_verifier import RELATION_PROTOCOL
from climate_rag.scifact_generation import frozen_contract
from climate_rag.scifact_natural_contract import require
from climate_rag.scifact_utility_contract import identity
from run_scifact_grounding_train_operator import ROOT

PAIR = "scifact-evidence-commit-paired-v2-v3-20261002"
VERSIONS = (ISOLATED_PROTOCOL, RELATION_PROTOCOL)
OUTPUT = ROOT / "runs" / PAIR
WRAPPER = "hpc/scifact_evidence_commit_paired.sbatch"
RESOURCE = {"gpu": "A100:1", "cpus": 8, "host_ram_gib": 32, "scratch_gib": 30, "slurm_seconds": 5400}
CPU_PREFLIGHT_SHA = "7e21b603c225374723122763f700f73dacc04f4005cde7784e23e837814cb388"


def child_extensions(protocol: str) -> dict[str, Any]:
    require(protocol in VERSIONS, "paired_protocol")
    tag = "v2" if protocol == ISOLATED_PROTOCOL else "v3"
    return {"paired_comparison": PAIR, "attempt_id": PAIR + "-" + tag,
            "output": (OUTPUT / tag).as_posix(), "generation_contract": frozen_contract(),
            "comparison_status": "prospective_paired_same_consumed24_not_yet_evaluated"}


def pair_fields(children: list[dict[str, Any]]) -> dict[str, Any]:
    require([c["protocol"] for c in children] == list(VERSIONS), "paired_order")
    common = ("source_git", "source_archive_sha256", "wrapper_sha256", "frames_sha256",
              "selection_sha256", "ordered_ids_sha256", "model_archive_sha256", "generation_contract",
              "runtime_files_sha256", "claims_sha256", "assembler")
    require(all(children[0][key] == children[1][key] for key in common), "paired_identity_mismatch")
    return {"protocol": PAIR, "output": OUTPUT.as_posix(), "resource_cap": RESOURCE,
            "phase_seconds": 2400, "max_worker_seconds": 1980, "scoring_seconds_per_phase": 180,
            "scoring_reserve_seconds": 420, "max_operator_seconds": 5340,
            "planned_episodes": 144, "max_generator_calls": 480,
            "phase_order": list(VERSIONS), "children": copy.deepcopy(children),
            "cpu_preflight_job": "31969505", "cpu_preflight_sha256": CPU_PREFLIGHT_SHA,
            "continue_after_infrastructure_failure": False, "score_after_all_workers_reaped": True,
            "model_calls_before_authorization": 0, "automatic_retry": False,
            **{k: copy.deepcopy(children[0][k]) for k in common}}


def validate_pair(release: dict[str, Any]) -> None:
    from run_scifact_evidence_commit_operator import validate_release
    require(release.get("authorization") == "coordinator_exact_hash_release", "draft_is_not_executable")
    children = release.get("children", [])
    require(len(children) == 2, "two_independent_releases")
    for child in children:
        validate_release(child)
        require(child.get("paired_comparison") == PAIR, "paired_child_binding")
    expected = dict(pair_fields(children), authorization="coordinator_exact_hash_release")
    require(release == expected, "paired_frozen_contract")


def sum_costs(costs: list[dict[str, Any]]) -> dict[str, Any]:
    # g00 in v2 and g00 in v3 are different physical calls: never deduplicate IDs.
    unknown = sum(c["unknown_usage_attempts"] for c in costs)
    calls = sum(c["unique_physical_calls"] for c in costs)
    require(len(costs) == 2 and all(c["unique_physical_calls"] <= 240 for c in costs)
            and calls <= 480, "paired_call_cap")
    tokens = {k: sum(c["known_token_lower_bound"][k] for c in costs)
              for k in ("input_tokens", "output_tokens")}
    return {"unique_physical_calls": calls, "unknown_usage_attempts": unknown,
            "known_token_lower_bound": tokens, "total_tokens": None if unknown else tokens,
            "planned_slots": 144, "namespace": "protocol_and_child_release_sha_not_bare_g_id"}


def compare_quality(reports: list[dict[str, Any]]) -> dict[str, Any]:
    """Only subtract already original-scored aggregates; no new scientific metric."""
    if len(reports) != 2 or any(r.get("status") != "scored" for r in reports):
        return {"status": "no_paired_quality", "planned_slots": 144,
                "reason": "both_complete_audited_protocols_required_no_complete_case_subset"}
    require([r["protocol"] for r in reports] == list(VERSIONS), "quality_protocol_order")
    ids = [c["claim_id"] for c in reports[0]["arms"][ARMS[0]]["cases"]]
    require(len(ids) == len(set(ids)) == 24, "quality_24_cases")
    for r in reports:
        for arm in ARMS:
            require(r["arms"][arm]["planned"] == 24
                    and [c["claim_id"] for c in r["arms"][arm]["cases"]] == ids,
                    "quality_roster_mismatch")
    def difference(before: Any, after: Any) -> dict[str, Any]:
        require([c["positive"] for c in before["cases"]] == [c["positive"] for c in after["cases"]], "gold_pairing")
        require(all(before["official_micro"][k] == after["official_micro"][k]
                    for k in ("schema_version","official_reference_sha","claim_count")), "original_scorer_identity")
        def delta(a: Any, b: Any) -> Any:
            return {key: delta(a[key], b[key]) for key in a} if isinstance(a, dict) else b - a
        return {"positive_correct_delta": after["positive_correct"] - before["positive_correct"],
                "nei_correct_delta": after["nei_correct"] - before["nei_correct"],
                "unresolved_delta": after["unresolved"] - before["unresolved"],
                "original_official_micro_delta": delta(before["official_micro"]["metrics"], after["official_micro"]["metrics"]),
                "paired_positive_wins": sum(a["positive"] and not a["strict_whole_answer"] and b["strict_whole_answer"]
                                             for a,b in zip(before["cases"],after["cases"],strict=True)),
                "paired_positive_losses": sum(a["positive"] and a["strict_whole_answer"] and not b["strict_whole_answer"]
                                               for a,b in zip(before["cases"],after["cases"],strict=True))}
    return {"status": "paired_scored", "planned_slots": 144, "ordered_ids_sha256": identity(ids),
            "fixed_policy_v3_minus_v2": {a: difference(reports[0]["arms"][a],reports[1]["arms"][a]) for a in ARMS[:2]},
            "adaptive_minus_fixed_within_version": {r["protocol"]: {
                a: difference(r["arms"][a],r["arms"]["adaptive"]) for a in ARMS[:2]} for r in reports},
            "adaptive_cross_version_joint_system_delta": difference(reports[0]["arms"]["adaptive"],reports[1]["arms"]["adaptive"]),
            "limits": "Consumed TRAIN diagnostic. Across-version adaptive changes verifier plus feedback; no isolated feedback causal claim. Fixed empty positives remain unresolved, not correct NEI."}
