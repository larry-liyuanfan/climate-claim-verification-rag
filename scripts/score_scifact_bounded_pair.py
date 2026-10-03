"""Separate post-inference scorer; data and raw responses remain on Spartan."""
import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path

from climate_rag.local_bounded_scifact_provider import raw_action
from climate_rag.public_v2 import file_sha256
from climate_rag.scifact_bounded_comparison import score_bounded_pair
from climate_rag.scifact_bounded_runtime import ARMS
from climate_rag.scifact_grounding import parse_abstract, parse_gold
from run_scifact_bounded_arm import MODEL_SHA, PAIR_SHA, RELEASE, RERANKER_SHA


def audit_private_wire(directory, attempts, gap):
    physical = Counter()
    actions = {}
    for path in directory.iterdir():
        if path.is_symlink() or not path.is_file():
            raise ValueError("private response directory entry invalid")
        if path.name.endswith("-response.txt"):
            raw = path.read_bytes()
            sha = hashlib.sha256(raw).hexdigest()
            physical[sha, len(raw)] += 1
            actions[sha] = raw_action(raw.decode("utf-8"), gap)
    receipts = Counter()
    for attempt in attempts:
        d = attempt["diagnostics"]
        r = d["private_attachment"]
        if (r["truncated"] is not False or r["io_failed"] is not False or r["dropped_bytes"] != 0
                or r["stored_bytes"] != r["attempted_bytes"] or r["sha256"] != r["stored_prefix_sha256"]):
            raise ValueError("incomplete wire receipt")
        receipts[r["sha256"], r["stored_bytes"]] += 1
        if actions.get(r["sha256"], "missing") != d["raw_wire_action"]:
            raise ValueError("physical raw action diagnostic mismatch")
    if physical != receipts:
        raise ValueError("physical wire/receipt multiset mismatch")
    return {"complete_responses": sum(physical.values()), "bytes": sum(n * count for (_, n), count in physical.items())}


def r2_visible_comparison(rows, old_run):
    prior = {(r["claim_id"], r["route"]): r for r in old_run["runs"]}
    report = []
    for row in rows:
        if row["arm"] != ARMS[0]:
            continue
        new = row["result"]
        old = prior[row["claim_id"], row["route"]]["result"]
        old_views, new_views = old["visible_attempts"], new["visible_attempts"]
        same = bool(old_views and new_views and old_views[0]["visible"] == new_views[0]["visible"])
        report.append({"claim_id": row["claim_id"], "route": row["route"], "same_ordered_visible_as_r2": same,
                       "r2_first_actual_tokens": old["generation_attempts"][0]["input_prompt_tokens"] if old_views else None,
                       "F_first_actual_tokens": new["generation_attempts"][0]["input_prompt_tokens"] if new_views else None,
                       "r2_preview_identity": "not_recorded_not_comparable"})
    return report


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ("first-run", "second-run", "corpus", "gold", "selection", "manifest", "r2-run", "output"):
        p.add_argument("--" + name, type=Path, required=True)
    args = p.parse_args()
    if args.output.exists():
        raise FileExistsError("no scoring overwrite")
    manifest = json.loads(args.manifest.read_text())
    for path, member in ((args.corpus, "inference/corpus.jsonl"), (args.gold, "private/gold.jsonl"),
                         (args.selection, "private/selected-strata.json")):
        if file_sha256(path) != manifest["output_sha256"][member]:
            raise ValueError("scoring input identity mismatch")
    runs, wire_audits = [], []
    for path, arm in zip((args.first_run, args.second_run), ARMS, strict=True):
        run = json.loads(path.read_text())
        preflight_path = path.parent / "runtime-preflight.json"
        preflight = json.loads(preflight_path.read_text())
        if (run["release_id"] != RELEASE or run["arm"] != arm or run["pair_protocol_sha256"] != PAIR_SHA
                or run["protocol_sha256"] != manifest["output_sha256"]["inference/protocol.json"]
                or run["preflight_sha256"] != file_sha256(preflight_path) or preflight["status"] != "passed"
                or run["model_sha256"] != MODEL_SHA or run["reranker_sha256"] != RERANKER_SHA
                or run["gold_loaded"] is not False or (runs and run["source_git"] != runs[0]["source_git"])):
            raise ValueError("source/protocol/model/preflight pair join mismatch")
        private = path.parent / "private-responses"
        audits = [audit_private_wire(private / "preflight", preflight["records"], arm == ARMS[1])]
        for index, row in enumerate(run["runs"], 1):
            audits.append(audit_private_wire(private / f"slot-{index:02d}", row["result"]["generation_attempts"], arm == ARMS[1]))
        wire_audits.append({"arm": arm, "directories": audits})
        runs.append(run)
    corpus = {d.doc_id: d for d in (parse_abstract(json.loads(line)) for line in args.corpus.read_text().splitlines())}
    gold = [parse_gold(json.loads(line), corpus) for line in args.gold.read_text().splitlines()]
    selection = json.loads(args.selection.read_text())
    rows = runs[0]["runs"] + runs[1]["runs"]
    report = score_bounded_pair(gold, corpus, rows, selection)
    report.update(source_git=runs[0]["source_git"], pair_protocol_sha256=PAIR_SHA,
                  run_sha256={arm: file_sha256(path) for arm, path in zip(ARMS, (args.first_run, args.second_run), strict=True)},
                  raw_wire_audits=wire_audits,
                  comparison_to_r2=r2_visible_comparison(rows, json.loads(args.r2_run.read_text())),
                  r2_attribution_boundary="F changed grammar AND common packing; r2 exact previews unavailable")
    with args.output.open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(report, stream, indent=2, sort_keys=True)
        stream.write("\n")


if __name__ == "__main__":
    main()
