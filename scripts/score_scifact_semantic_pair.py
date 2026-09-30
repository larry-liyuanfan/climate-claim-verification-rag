"""Post-inference semantic-pair scorer; no old-r2 baseline or model invocation."""
from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any

from climate_rag.scifact_diagnostic_scoring import score_diagnostic
from climate_rag.scifact_grounding import parse_abstract, parse_gold
from climate_rag.scifact_semantic_contract import (
    POLICIES, PRIVATE_SHA, PROMPTS, ROUTES, SCORING_NAMES, checked, encoded, load_inference, sha, write_once,
)
from climate_rag.scifact_semantic_receipts import read_completed_run
from climate_rag.scifact_semantic_runtime import validate_pair_rows
from audit_scifact_bounded_closeout import audit_slot, physical, tariff


def terminal_counts(gold: Any, rows: list[dict[str, Any]]) -> dict[str, Any]:
    by_id = {g.claim_id: g for g in gold}
    count: Counter[str] = Counter()
    for row in rows:
        result = row["result"]
        last = result["generation_attempts"][-1] if result["generation_attempts"] else {}
        answer = (last.get("status") == "valid_decision" and last.get("action") == "answer"
                  and result["outcome"] == "ids_validated_semantics_unmeasured")
        abstain = (last.get("status") == "valid_decision" and last.get("action") == "abstain"
                   and result["outcome"].startswith("model_abstention:"))
        emitted = bool(row["prediction"]["evidence"])
        count["selected_claims"] += 1
        count["validated_nonempty_answer"] += answer and emitted
        count["valid_model_abstention"] += abstain
        count["failure_or_budget_termination"] += not (answer or abstain)
        if not by_id[row["claim_id"]].evidence:
            count["nei_claims"] += 1
            count["nei_false_evidence"] += emitted
            count["nei_valid_model_abstention"] += abstain
            count["nei_failure_empty_prediction"] += not emitted and not (answer or abstain)
    return dict(count) | {
        "nonempty_answer_coverage": count["validated_nonempty_answer"] / len(rows),
        "claim_verdict_accuracy": None, "boundary": "Empty fallback is not a correct NEI decision; document F1 is separate."}


def validate_prepared_scoring(payload: dict[str, bytes], manifest: dict[str, Any]) -> None:
    if manifest.get("preparation_private_sha256") != PRIVATE_SHA:
        raise ValueError("semantic_preparation_private_manifest")
    for member, original in (("selection-freeze.json", "selected-before-probe.json"),
            ("current-opportunity.json", "current-opportunity.json"),
            ("consumption-ledger.json", "consumption-ledger.json"),
            ("preparation-draft.json", "future-pair-protocol-draft.json")):
        if sha(payload[member]) != PRIVATE_SHA[original]:
            raise ValueError("semantic_prepared_scoring_bytes")
    gold_rows = [json.loads(line) for line in payload["gold.jsonl"].splitlines()]
    if (sha(encoded(gold_rows)) != PRIVATE_SHA["selected-gold.json"]
            or json.loads(payload["selected-strata.json"]) != json.loads(payload["selection-freeze.json"])["selection"]):
        raise ValueError("semantic_prepared_gold_or_strata_replaced")


def score(args: Any, costs: dict[str, Any]) -> dict[str, Any]:
    protocol, claims, corpus_bytes = load_inference(args.inference_dir, args.protocol_sha, args.source_git)
    if {p.name for p in args.scoring_dir.iterdir()} != SCORING_NAMES:
        raise ValueError("semantic_scoring_allowlist")
    manifest = json.loads(checked(args.scoring_dir / "manifest.json", args.manifest_sha))
    if (manifest["execution_source_git"] != args.source_git or manifest["protocol_sha256"] != args.protocol_sha
            or manifest["inference_file_sha256"] != protocol["inference_file_sha256"]
            or set(manifest["scoring_file_sha256"]) != SCORING_NAMES - {"manifest.json"}):
        raise ValueError("semantic_scoring_manifest_join")
    payload = {n: checked(args.scoring_dir / n, h) for n, h in manifest["scoring_file_sha256"].items()}
    validate_prepared_scoring(payload, manifest)
    corpus = {d.doc_id: d for d in (parse_abstract(json.loads(line)) for line in corpus_bytes.splitlines())}
    gold = [parse_gold(json.loads(line), corpus) for line in payload["gold.jsonl"].splitlines()]
    selection, reviews = (json.loads(payload[n]) for n in ("selected-strata.json", "current-opportunity.json"))
    if ([g.claim_id for g in gold] != protocol["ordered_claim_ids"]
            or [r["id"] for r in selection] != protocol["ordered_claim_ids"]
            or [r["id"] for r in reviews] != protocol["ordered_claim_ids"]
            or any(g.claim != c["claim"] for g, c in zip(gold, claims, strict=True))):
        raise ValueError("semantic_scoring_order_or_text_join")
    runs = [read_completed_run(path, protocol, args.protocol_sha, args.source_sha, args.inference_sha, policy)
            for path, policy in zip((args.first_run, args.second_run), POLICIES, strict=True)]
    rows = runs[0]["runs"] + runs[1]["runs"]
    validate_pair_rows(rows, args.source_git, protocol["ordered_claim_ids"], ROUTES)
    gold_map = {g.claim_id: g for g in gold}
    result: dict[str, Any] = {"scope": protocol["scope"], "status": "paired_train_diagnostic_only",
        "independent_test": False, "bootstrap_replicates": 0, "confidence_interval": None,
        "claim_verdict_accuracy": None, "agent_benefit_established": False, "costs": costs,
        "source_git": args.source_git, "preparation_source_git": protocol["preparation_source_git"],
        "protocol_sha256": args.protocol_sha, "policy_prompt_sha256": PROMPTS,
        "initial_contexts_comparable": True, "later_context_equality_asserted": False, "policies": {}}
    for path, policy, run in zip((args.first_run, args.second_run), POLICIES, runs, strict=True):
        flight = json.loads((path.parent / "runtime-preflight.json").read_bytes())
        for i, record in enumerate(flight["records"], 1):
            complete = json.loads((path.parent / f"preflight-{i:02d}-completed.json").read_bytes())
            if complete != {k: flight[k] for k in ("policy", "prompt_sha256", "gap", "source_git", "protocol_sha256", "execution_kind")} | {"record": record}:
                raise ValueError("semantic_preflight_durable_case")
        preflight_wire, _ = physical(path.parent / "private-responses/preflight", flight["records"], True)
        audits = []
        for index, row in enumerate(run["runs"], 1):
            wire, raw = physical(path.parent / f"private-responses/slot-{index:02d}", row["result"]["generation_attempts"], True)
            audits.append({"wire": wire, **audit_slot(row, gold_map[row["claim_id"]], corpus, raw, gap=True)})
        legacy = [{"id": r["id"], "component": r["component"], "stratum": r["legacy_stratum"]} for r in selection]
        quality = score_diagnostic(gold, corpus, run["runs"], legacy)
        for route in ROUTES:
            rr = [r for r in run["runs"] if r["route"] == route]
            quality["routes"][route]["by_legacy_stratum"] = quality["routes"][route].pop("by_stratum")
            quality["routes"][route]["terminal_decisions"] = terminal_counts(gold, rr)
            selected_audits = [a for a, r in zip(audits, run["runs"], strict=True) if r["route"] == route]
            for name in ("chain", "raw_action_proposals", "attempt_statuses"):
                counts: Counter[str] = Counter()
                for a in selected_audits:
                    counts.update(a[name])
                quality["routes"][route][name] = dict(counts)
        result["policies"][policy] = {"quality": quality, "preflight_wire_audit": preflight_wire,
            "slot_wire_audits": [a["wire"] for a in audits], "run_sha256": sha(path.read_bytes()),
            "current_opportunity_by_claim": [{"claim_id": r["id"], "legacy_stratum": r["legacy_stratum"],
                "current_stratum": r["current_opportunity"][policy]["stratum"]} for r in reviews]}
    result["interpretation"] = ("No causal/Agent gain from fixed and adaptive improving together, extra abstention, "
        "fixture read opportunity or format validity alone. Inspect complete-gold-visible fixed routes and actual "
        "new-evidence chains before further hypotheses. No independent test or online SLA.")
    return result


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    for n in ("first-run", "second-run", "inference-dir", "scoring-dir", "output"):
        p.add_argument("--" + n, type=Path, required=True)
    for n in ("protocol-sha", "manifest-sha", "source-sha", "inference-sha", "source-git"):
        p.add_argument("--" + n, required=True)
    args = p.parse_args()
    costs = {}
    for path, policy in zip((args.first_run, args.second_run), POLICIES, strict=True):
        run = json.loads(path.read_bytes())
        flight = json.loads((path.parent / "runtime-preflight.json").read_bytes())
        costs[policy] = {"train": tariff([a for r in run["runs"] for a in r["result"]["generation_attempts"]]),
                         "preflight": tariff(flight["records"])}
    write_once(args.output.with_name("cost-before-quality.json"), costs)
    try:
        report = score(args, costs)
    except Exception as exc:
        write_once(args.output, {"status": "quality_blocked_costs_retained", "costs": costs, "exception_type": type(exc).__name__})
        raise
    write_once(args.output, report)


if __name__ == "__main__":
    main()
