"""Real tokenizer/LMFE with synthetic scripted actions. Never loads model weights."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

from climate_rag.agent_v3 import SentenceAgentV3, Source
from climate_rag.local_agent_v3 import render_v3_prompt
from climate_rag.stop_acquire import PROTOCOL
from climate_rag.scifact_read_continuation import ordered_write
from smoke_agent_v3_tokenizer import TOKENIZER_HASHES
from smoke_scifact_provider import cpu_tokenizer_data, prefix_accepts


def smoke(root: Path) -> dict[str, Any]:
    from transformers.models.auto.tokenization_auto import AutoTokenizer

    actual = {
        p.name: hashlib.sha256(p.read_bytes()).hexdigest()
        for p in root.iterdir()
        if p.is_file()
    }
    if actual != TOKENIZER_HASHES:
        raise ValueError("tokenizer_file_set_or_hash")
    tokenizer = AutoTokenizer.from_pretrained(
        str(root), local_files_only=True, trust_remote_code=False
    )
    tokenizer_data_factory: Any = cpu_tokenizer_data
    prefix_check: Any = prefix_accepts
    data = tokenizer_data_factory(tokenizer)
    stop, abstain = (
        {"action": "stop"},
        {"action": "abstain", "reason": "insufficient_evidence"},
    )
    query = {
        "action": "acquire",
        "tool": "query",
        "purpose": "counter_evidence",
        "query": "synthetic independently observed counter evidence",
    }
    records: list[dict[str, Any]] = []

    class Backend:
        name, kind = "scripted-real-tokenizer-not-a-model", "fixture"

        def __init__(self, actions: list[dict[str, Any]]) -> None:
            self.actions = iter(actions)
            self.rows: list[dict[str, Any]] = []

        def count_prompt(self, obs: Any, schema: Any) -> int:
            return len(
                tokenizer.encode(
                    render_v3_prompt(tokenizer, obs, schema), add_special_tokens=False
                )
            )

        def count_text(self, text: str) -> int:
            return len(tokenizer.encode(text, add_special_tokens=False))

        def generate(
            self, obs: Any, schema: Any, cap: int, seconds: float
        ) -> dict[str, Any]:
            action = next(self.actions)
            prompt = render_v3_prompt(tokenizer, obs, schema)
            accepted, output = prefix_check(data, tokenizer, prompt, schema, action)
            length = self.count_prompt(obs, schema)
            assert accepted and output + 1 <= cap and length <= 8192
            assert tokenizer(prompt, add_special_tokens=False)[
                "input_ids"
            ] == tokenizer.encode(prompt, add_special_tokens=False)
            if obs["phase"] == "gate":
                invalid = {"action": "acquire", "tool": "read", "source_ids": ["c0"]}
                rejected, _ = prefix_check(data, tokenizer, prompt, schema, invalid)
                assert not rejected
            else:
                invalid = {
                    "action": "answer",
                    "label": "SUPPORTS",
                    "sentence_ids": ["c999:0"],
                }
                rejected, _ = prefix_check(data, tokenizer, prompt, schema, invalid)
                assert not rejected
            self.rows.append(
                {
                    "phase": obs["phase"],
                    "input_tokens": length,
                    "scripted_output_tokens_including_eos": output + 1,
                    "valid_path_accepted": accepted,
                    "invalid_path_rejected": not rejected,
                }
            )
            return {
                "raw": json.dumps(action),
                "usage": {"input_tokens": length, "output_tokens": output + 1},
            }

    for words in (8, 64, 180):
        sources = [
            Source(
                str(i), "SYNTHETIC SOURCE", (f"Source {i}: " + "synthetic " * words,)
            )
            for i in range(20)
        ]
        new = Source("new", "Synthetic new source", ("SYNTHETIC counter-evidence.",))
        scenarios: tuple[tuple[str, list[dict[str, Any]]], ...] = (
            ("stop", [stop, abstain]),
            (
                "read",
                [
                    {"action": "acquire", "tool": "read", "source_ids": ["c5"]},
                    stop,
                    abstain,
                ],
            ),
            ("query", [query, stop, abstain]),
        )
        for name, actions in scenarios:
            backend = Backend(actions)
            run = SentenceAgentV3(
                backend,
                lambda q, k: [new] if q == query["query"] else sources,
                protocol=PROTOCOL,
            ).run("This is a synthetic claim, not a real evaluation example.")
            assert run["outcome"] == "model_abstention:insufficient_evidence", run[
                "events"
            ]
            assert len(run["generation_attempts"]) == len(actions)
            records.append(
                {
                    "scenario": name,
                    "repeated_synthetic_words_per_document": words,
                    "calls": len(actions),
                    "total_scripted_generator_tokens": sum(run["usage"].values()),
                    "attempts": backend.rows,
                }
            )
    return {
        "protocol": PROTOCOL,
        "scope": "synthetic_tokenizer_contract_not_model_quality_or_runtime",
        "weights_loaded": False,
        "tokenizer_sha256": actual,
        "cases": records,
        "historical_consumed32_baselines_not_paired_with_synthetic": {
            "fixed_rerank_mean_generator_tokens": 48441 / 32,
            "fixed_multiquery_mean_generator_tokens": 66757 / 32,
        },
        "interpretation": "gate prefill plus verdict increases tokens; no efficiency claim or real-task estimate",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tokenizer", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    result = smoke(args.tokenizer)
    if args.output:
        ordered_write(args.output, result)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
