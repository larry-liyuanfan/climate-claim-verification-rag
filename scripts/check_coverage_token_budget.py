"""Real pinned tokenizer + SYNTHETIC stop provider; no model inference or gold."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from climate_rag import coverage_acquisition as cov, fair_acquisition as fair
from climate_rag.agent_v3 import SentenceAgentV3, Source, V3Budget
from climate_rag.fair_replay import identity
from climate_rag.local_agent_v3 import render_v3_prompt
from score_saved_citation_nli import digest


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--tokenizer-dir", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    target = args.output.resolve()
    if (not args.output.is_absolute() or target.is_relative_to(Path(__file__).resolve().parents[1])
            or any(v.casefold() in {"onedrive", "求职"} for v in target.parts) or target.exists()):
        raise ValueError("new_private_receipt_required")
    from transformers.models.auto.tokenization_auto import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(str(args.tokenizer_dir), local_files_only=True, trust_remote_code=False)

    class Counter:
        name, kind = "SYNTHETIC-stop-real-tokenizer", "fixture"

        def render(self, observation, schema):
            return render_v3_prompt(tokenizer, observation, schema)

        def count_prompt(self, observation, schema):
            return len(tokenizer.encode(self.render(observation, schema), add_special_tokens=False))

        def count_text(self, text):
            return len(tokenizer.encode(text, add_special_tokens=False))

        def generate(self, observation, schema, max_output_tokens, remaining_seconds):
            if observation["phase"] == "gate":
                value = {"coverage": [{"claim_span": observation["immutable_claim"], "kind": "relation",
                    "status": "uncertain", "sentence_ids": []}], "decision": {"action": "stop"}, "stop_reason": "uncertain_scope"}
            elif observation["phase"] == "plan":
                value = {"action": "plan_queries", "queries": [{"query": "synthetic glacier study", "purpose": "counter_evidence"}],
                         "read_source_ids": observation["readable_source_ids"][:5]}
            else:
                value = {"action": "abstain", "reason": "insufficient_evidence"}
            return {"raw": json.dumps(value), "usage": {"input_tokens": self.count_prompt(observation, schema), "output_tokens": 32}}

    docs = [Source(f"synthetic-{i}", "SYNTHETIC", ("Synthetic °C glacier — observation. " * (i + 1),)) for i in range(20)]
    records = []
    for cap in (1024, 2048, 4096, 8192):
        frames = []
        for route in fair.ROUTES:
            row = SentenceAgentV3(Counter(), lambda q, k: docs[:k], rerank=lambda q, values: values,
                budget=V3Budget(max_input_tokens=cap), protocol=cov.PROTOCOL).run("Synthetic °C glaciers — advance.", route)
            frames.append(row["initial_frame"])
            tokens = [v["usage"]["input_tokens"] for v in row["generation_attempts"]]
            if any(v > cap for v in tokens):
                raise ValueError("actual_prompt_exceeds_bound")
            records.append({"cap": cap, "route": route, "initial_frame_sha256": identity(row["initial_frame"]),
                "model_boundary_prompt_tokens": tokens, "outcome": row["outcome"]})
        if any(f != frames[0] for f in frames):
            raise ValueError("unequal_real_tokenizer_initial_frame")
    result = {"kind": "synthetic_implementation_real_tokenizer_no_model_gain", "protocol": cov.PROTOCOL,
        "generator_tokenizer_revision": "350135a4de9a3407be836fa238cccc1d61503a85",
        "tokenizer_files": {v.name: digest(v) for v in args.tokenizer_dir.iterdir() if v.is_file()},
        "source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(), "records": records,
        "gold_read": False, "test_read": False, "model_inference": False}
    with target.open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(result, stream, indent=2)
        stream.write("\n")
    print(json.dumps({"receipt_sha256": digest(target), "cases": len(records), "actual_tokenizer_bound": True}))


if __name__ == "__main__":
    main()
