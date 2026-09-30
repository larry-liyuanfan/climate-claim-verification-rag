"""Real pinned tokenizer + synthetic CPU prompts; no weights/model generation."""

from __future__ import annotations

import argparse
import importlib.metadata
import json
from pathlib import Path

from climate_rag.agent_v3 import SentenceAgentV3, Source, V3Budget, action_schema
from climate_rag.io import write_json
from climate_rag.local_agent_v3 import render_v3_prompt
from climate_rag.public_v2 import file_sha256

TOKENIZER_HASHES = {
    "merges.txt": "8831e4f1a044471340f7c0a83d7bd71306a5b867e95fd870f74d0c5308a904d5",
    "tokenizer.json": "aeb13307a71acd8fe81861d94ad54ab689df773318809eed3cbe794b4492dae4",
    "tokenizer_config.json": "5a7303fcb1a27ede63134a2cbd61d5282c247ca6d769ce4746d4ffa124aedd63",
    "vocab.json": "ca10d7e9fb3ed18575dd1e277a2579c16d108e32f27439684afa0e10b1440910",
}


def grammar_accepts(schema, value):
    from lmformatenforcer import JsonSchemaParser

    parser = JsonSchemaParser(schema)
    for char in json.dumps(value, separators=(",", ":")):
        if char not in parser.get_allowed_characters():
            return False
        parser = parser.add_character(char)
    return parser.can_end()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tokenizer", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError("smoke output exists")
    actual_hashes = {
        p.name: file_sha256(p) for p in args.tokenizer.iterdir() if p.is_file()
    }
    if actual_hashes != TOKENIZER_HASHES:
        raise ValueError("frozen tokenizer file set/SHA mismatch")
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(
        str(args.tokenizer), local_files_only=True, trust_remote_code=False
    )

    class TokenizerFixtureProvider:
        name, kind = "real-tokenizer/fixture-decision", "fixture"

        def __init__(self):
            self.prompts = []

        def count_text(self, text):
            return len(tokenizer.encode(text, add_special_tokens=False))

        def count_prompt(self, observation, schema):
            return len(
                tokenizer.encode(
                    render_v3_prompt(tokenizer, observation, schema),
                    add_special_tokens=False,
                )
            )

        def generate(self, observation, schema, max_output_tokens, remaining_seconds):
            count = self.count_prompt(observation, schema)
            self.prompts.append(
                {
                    "tokens": count,
                    "visible_sentences": len(observation["current_citable"]),
                    "preview_sources": len(observation["preview_only"]),
                }
            )
            raw = json.dumps({"action": "abstain", "reason": "insufficient_evidence"})
            assert grammar_accepts(schema, json.loads(raw))
            return {
                "raw": raw,
                "usage": {
                    "input_tokens": count,
                    "output_tokens": len(
                        tokenizer.encode(raw, add_special_tokens=False)
                    ),
                },
            }

    reports = []
    for limit in (1024, 2048, 8192):
        provider = TokenizerFixtureProvider()
        corpus = [
            Source(
                str(i),
                "Synthetic tokenizer fixture only",
                (
                    "The reported variable increased during the observed period. "
                    * 100,
                    "The second complete sentence mentions a different observation.",
                ),
            )
            for i in range(20)
        ]
        result = SentenceAgentV3(
            provider,
            lambda q, k: corpus,
            rerank=lambda q, rows: rows,
            budget=V3Budget(max_input_tokens=limit),
        ).run("The variable increased.")
        assert provider.prompts and provider.prompts[0]["tokens"] <= limit
        reports.append(
            {
                "input_limit": limit,
                "prompts": provider.prompts,
                "outcome": result["outcome"],
                "model_calls": 0,
            }
        )
    schema = action_schema(
        ["abstain", "answer", "read", "rewrite", "rerank"],
        [f"c{i}" for i in range(20)],
        [f"c{i}:999" for i in range(5)],
        5,
    )
    payloads = [
        {
            "action": "answer",
            "label": "SUPPORTS",
            "sentence_ids": ["c0:999", "c1:999", "c2:999"],
        },
        {"action": "read", "source_ids": [f"c{i}" for i in range(15, 20)]},
        {"action": "abstain", "reason": "conflicting_evidence"},
        {"action": "rerank"},
        {
            "action": "rewrite",
            "query": "A representative scientific claim with a year 2010 and numerical value 800.",
        },
    ]
    output_counts = []
    for value in payloads:
        assert grammar_accepts(schema, value)
        count = len(tokenizer.encode(json.dumps(value), add_special_tokens=False))
        assert count < 512
        output_counts.append({"action": value["action"], "tokens": count})
    # Grammar rejects unknown IDs/actions instead of permitting arbitrary strings.
    assert not grammar_accepts(schema, {"action": "read", "source_ids": ["unknown"]})
    assert not grammar_accepts(
        schema, {"action": "answer", "label": "SUPPORTS", "sentence_ids": ["c99:0"]}
    )
    result = {
        "schema_version": "agent-v3-tokenizer-cpu-smoke-v1",
        "status": "passed",
        "versions": {
            p: importlib.metadata.version(p)
            for p in (
                "transformers",
                "tokenizers",
                "lm-format-enforcer",
                "interegular",
                "jinja2",
            )
        },
        "tokenizer_revision": "350135a4de9a3407be836fa238cccc1d61503a85",
        "tokenizer_files_sha256": {
            p.name: file_sha256(p)
            for p in sorted(args.tokenizer.iterdir())
            if p.is_file()
        },
        "full_prompt_budget_cases": reports,
        "legal_output_token_examples": output_counts,
        "model_weights_loaded": False,
        "model_generation_calls": 0,
        "torch_hf_prefix_integration_executed": False,
        "character_grammar_smoke": "passed",
        "boundary": "Synthetic examples with real tokenizer, not exhaustive unicode rewrite bound or model quality. Natural-language rewrite can still exhaust512; preserve failures. Torch2.1.2/HF GPU prefix adapter needs separately authorized allocated preflight.",
    }
    write_json(args.output, result)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
