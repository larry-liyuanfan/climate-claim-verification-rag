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
    CONTRACT, MODEL, REVISION, LocalNLI, aggregate, cited_pair, classify, matrix,
)

# Reuse the accepted physical/delivery audit for exactly this immutable run.
# This is not a generic permission to score arbitrary logs or a smaller subset.
ACCEPTED_RUN_SHA = "3c9eafaa159321bcc202d5f5f18ddcdd8548917dde7e894f3b19def44e1cb7c3"


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def score_saved(run: dict[str, Any], judge: Any) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    rows = matrix(run)
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
    args = p.parse_args()
    if args.run_sha256 != ACCEPTED_RUN_SHA or digest(args.run) != ACCEPTED_RUN_SHA:
        raise ValueError("saved_run_hash_mismatch")
    run = json.loads(args.run.read_bytes())
    matrix(run, expected_tasks=32)
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
    result, records = score_saved(run, judge)
    result.update(run_sha256=args.run_sha256, model_manifest_sha256=digest(args.model_dir / "verified-model.json"),
                  source_sha256=digest(Path(__file__)), contract=CONTRACT,
                  reused_accepted_physical_delivery_audit_run_sha256=ACCEPTED_RUN_SHA,
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
