"""Score all saved fair outputs using pinned local NLI; no gold or new agent run."""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import importlib.metadata
import json
from pathlib import Path
from time import perf_counter
from typing import Any

from climate_rag.semantic_proxy import (
    CONTRACT, MODEL, PROTOCOL, REVISION, LocalNLI, aggregate, cited_pair, classify, matrix,
)
from climate_rag import coverage_acquisition
from climate_rag.verification import normalise_claim

# Reuse the accepted physical/delivery audit for exactly this immutable run.
# This is not a generic permission to score arbitrary logs or a smaller subset.
ACCEPTED_RUN_SHA = "3c9eafaa159321bcc202d5f5f18ddcdd8548917dde7e894f3b19def44e1cb7c3"
COVERAGE_RUN_SHA = "48d597d3abc82b8309cf2952895c787ff388c589371c37c1cecd6609cca778a6"
COVERAGE_SOURCE = "f792b9e0c09f95ddc46c7e613541b90eeded3edc"
COVERAGE_RELEASE = "19d36757a8a749aa02cb557a63ae931e0b0f4fcc06b3769f790dd8c1ef1a8901"
COVERAGE_OPERATOR_SHA = "113424f1a4bc904073eed3d83dc8d0d9f3682101246d39d946f6a0a40fb110e7"
COVERAGE_COMPACT_SHA = "f2d5c9a1b483ed97f668299fbb7d6674ff72908058bd9bbe33a661c36ed02968"
COVERAGE_COST_SHA = "d41e4fc210a755580af00289a491f078b513d5f552e4ddb75b6f2dee38c42bcd"


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def validate_coverage_saved(run: dict[str, Any], root: Path) -> None:
    """Exact accepted post-exit run only; no gold or arbitrary new scoring run."""
    matrix(run, expected_tasks=32, expected_protocol=coverage_acquisition.PROTOCOL)
    binding = run["fair_binding"]
    coverage_acquisition.validate_frozen_cohort(binding)
    if (binding["protocol"] != coverage_acquisition.PROTOCOL
            or run["study_kind"] != "consumed_fair32_coverage_regression_not_independent_quality_validation"):
        raise ValueError("coverage_consumed_regression_identity")
    cohort = root.parents[1] / "assets/fair-cohort.json"
    if digest(cohort) != coverage_acquisition.COHORT_SHA:
        raise ValueError("coverage_original_cohort_mismatch")
    tasks = json.loads(cohort.read_bytes())  # input only, never scorer gold
    fingerprint = hashlib.sha256(json.dumps(tasks, ensure_ascii=False, sort_keys=True,
                                separators=(",", ":")).encode()).hexdigest()
    if fingerprint != coverage_acquisition.TASKS_SHA:
        raise ValueError("coverage_task_bytes_mismatch")
    rows = [r for r in run["runs"] if r["route"] == "fixed_multiquery"]
    if (len(tasks) != 32 or any(t["id"] != r["task_id"]
            or normalise_claim(t["claim_text"]) != r["initial_frame"]["immutable_claim"]
            for t, r in zip(tasks, rows, strict=True))):
        raise ValueError("coverage_original_to_normalized_input_mismatch")
    for name, sha in (("operator-status.json", COVERAGE_OPERATOR_SHA),
                      ("compact.json", COVERAGE_COMPACT_SHA), ("cost-before-quality.json", COVERAGE_COST_SHA)):
        if digest(root / name) != sha:
            raise ValueError("coverage_accepted_exit_audit_mismatch")
    operator = json.loads((root / "operator-status.json").read_bytes())
    if (operator["status"] != "completed" or operator["source_git"] != COVERAGE_SOURCE
            or operator["release_sha256"] != COVERAGE_RELEASE):
        raise ValueError("coverage_execution_identity")


def score_saved(run: dict[str, Any], judge: Any, *, expected_protocol: str = PROTOCOL
                ) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    rows = matrix(run, expected_protocol=expected_protocol)
    positions: Counter[str] = Counter()
    records = []
    started = perf_counter()
    calls, tokens = 0, 0
    for row in rows:
        arm = row["route"]
        record: dict[str, Any] = {"route": arm, "slot_position": positions[arm]}
        positions[arm] += 1
        pair = cited_pair(row)  # invalid provenance is fatal, not a neutral label
        if pair is None:
            record["status"] = "abstained" if row["outcome"].startswith("model_abstention:") else "execution_failure"
        else:
            begin = perf_counter()
            try:
                scores, used = judge.score(pair["premise"], pair["hypothesis"])
                calls += 1
                tokens += used
                record["joint_citations"] = classify(scores, pair["target"])
                record["individual_citations"] = []
                for sentence in pair["sentences"]:
                    if len(pair["sentences"]) == 1:
                        item = dict(record["joint_citations"])
                    else:
                        scores, used = judge.score(sentence, pair["hypothesis"])
                        calls += 1
                        tokens += used
                        item = classify(scores, pair["target"])
                    record["individual_citations"].append(item)
                record["status"] = "scored"
            except OverflowError:
                record["status"] = "overlength"
                record.pop("joint_citations", None)
                record.pop("individual_citations", None)
            # Other runtime errors fail the package: do not publish partial success.
            record["elapsed_ms"] = (perf_counter() - begin) * 1000
        records.append(record)
    result = aggregate(records, tasks=len(rows) // 3)
    result["measurement_cost"] = {"nli_calls": calls, "input_tokens": tokens,
        "cpu_wall_seconds": perf_counter() - started, "paid_api_calls": 0,
        "agent_cost_reused_not_recomputed": True}
    return result, records


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--run", type=Path, required=True)
    p.add_argument("--run-sha256", required=True)
    p.add_argument("--model-dir", type=Path, required=True)
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--study", choices=("accepted-original", "accepted-coverage"), default="accepted-original")
    args = p.parse_args()
    accepted_sha = COVERAGE_RUN_SHA if args.study == "accepted-coverage" else ACCEPTED_RUN_SHA
    expected_protocol = coverage_acquisition.PROTOCOL if args.study == "accepted-coverage" else PROTOCOL
    if args.run_sha256 != accepted_sha or digest(args.run) != accepted_sha:
        raise ValueError("saved_run_hash_mismatch")
    run = json.loads(args.run.read_bytes())
    matrix(run, expected_tasks=32, expected_protocol=expected_protocol)
    if args.study == "accepted-coverage":
        validate_coverage_saved(run, args.run.parent.parent)
    model_manifest = json.loads((args.model_dir / "verified-model.json").read_bytes())
    if model_manifest["model"] != MODEL or model_manifest["revision"] != REVISION:
        raise ValueError("pinned_nli_model_required")
    for name, sha in model_manifest["files"].items():
        if digest(args.model_dir / name) != sha:
            raise ValueError("model_file_hash_mismatch")
    output = args.output_dir.resolve()
    repository = Path(__file__).resolve().parents[1]
    if (not args.output_dir.is_absolute() or output.is_relative_to(repository)
            or any(v.casefold() in {"onedrive", "求职"} for v in output.parts)):
        raise ValueError("private_local_output_outside_git_and_onedrive_required")
    if output.exists():
        raise ValueError("unique_output_required")
    begin = perf_counter()
    judge = LocalNLI(str(args.model_dir))
    loaded = perf_counter() - begin
    result, records = score_saved(run, judge, expected_protocol=expected_protocol)
    result.update(run_sha256=args.run_sha256, model_manifest_sha256=digest(args.model_dir / "verified-model.json"),
                  source_sha256=digest(Path(__file__)), contract=CONTRACT,
                  reused_accepted_physical_delivery_audit_run_sha256=accepted_sha,
                  execution_source_git=COVERAGE_SOURCE if args.study == "accepted-coverage" else "e826a4d2178983365c87184e259f7ad43ba7b17a",
                  semantic_proxy_source_sha256=digest(repository / "src/climate_rag/semantic_proxy.py"),
                  protocol=expected_protocol,
                  model_load_seconds=loaded,
                  versions={v: importlib.metadata.version(v) for v in ("torch", "transformers", "numpy")})
    output.mkdir(parents=True, mode=0o700)
    for name, value in (("compact.json", result), ("scores.private.json", records)):
        with (output / name).open("x", encoding="utf-8", newline="\n") as stream:
            json.dump(value, stream, indent=2)
            stream.write("\n")
    print(json.dumps({"compact_sha256": digest(output / "compact.json"), "slots": len(records),
                      "gold_read": False, "agent_model_calls": 0}))


if __name__ == "__main__":
    main()
