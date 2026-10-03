"""Post-inference CPU scoring. Gold is never an inference runner argument."""

import argparse
import json
from pathlib import Path

from climate_rag.public_v2 import file_sha256
from climate_rag.scifact_diagnostic_scoring import score_diagnostic
from climate_rag.scifact_grounding import parse_abstract, parse_gold


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("run", "preflight", "corpus", "gold", "selection", "manifest", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError("no scoring overwrite")
    manifest = json.loads(args.manifest.read_text())
    for path, member in ((args.corpus, "inference/corpus.jsonl"),
                         (args.gold, "private/gold.jsonl"),
                         (args.selection, "private/selected-strata.json")):
        if file_sha256(path) != manifest["output_sha256"][member]:
            raise ValueError("scoring input hash mismatch")
    run = json.loads(args.run.read_text())
    if (run["protocol_sha256"] != manifest["output_sha256"]["inference/protocol.json"]
            or run["preflight_sha256"] != file_sha256(args.preflight)):
        raise ValueError("protocol/preflight join mismatch")
    corpus = {d.doc_id: d for d in [parse_abstract(json.loads(line))
                                   for line in args.corpus.read_text().splitlines()]}
    gold = [parse_gold(json.loads(line), corpus) for line in args.gold.read_text().splitlines()]
    selection = json.loads(args.selection.read_text())
    report = score_diagnostic(gold, corpus, run["runs"], selection)
    preflight = json.loads(args.preflight.read_text())
    report.update(run_sha256=file_sha256(args.run), source_git=run["source_git"],
                  separate_model_load_ms=run["model_load_ms"],
                  separate_reranker_load_ms=run["reranker_load_ms"],
                  separate_bm25_build_ms=run["bm25_build_ms"],
                  separate_synthetic_preflight={
                      "status": preflight["status"], "elapsed_ms": preflight["elapsed_ms"],
                      "attempted_calls": preflight["attempted_calls"],
                      "known_token_subtotal": {k: sum(r.get("usage", {}).get(k, 0)
                                                       for r in preflight["records"])
                                               for k in ("input_tokens", "output_tokens")},
                      "records": preflight["records"]})
    with args.output.open("x", encoding="utf-8") as stream:
        json.dump(report, stream, indent=2, sort_keys=True)
        stream.write("\n")


if __name__ == "__main__":
    main()
