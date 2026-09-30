"""Synthetic-only smoke: cached tokenizer by default; real HF requires opt-in.

No dataset arguments or Slurm submission. Tokenizer mode loads no weights/Torch.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import importlib.metadata
import json
import logging
import subprocess
import time
from pathlib import Path

from climate_rag.local_agent_v3 import fixed_grammar_config, grammar_config_identity
from climate_rag.local_scifact_provider import (
    LocalQwenSciFactProvider,
    require_clean_grammar_environment,
)
from climate_rag.public_v2 import file_sha256
from climate_rag.scifact_runtime_smoke import smoke_cases, synthetic_runtime_smoke
from climate_rag.scifact_terminal import render_scifact_prompt

from smoke_agent_v3_tokenizer import TOKENIZER_HASHES

MODEL_SHA = "d1dd9783afdf4e0fbd21eee824834d71b86982f5a5d5f6f371fe07f2f76f3cf6"


def cpu_tokenizer_data(tokenizer):
    """Equivalent regular-token construction to LMFE0.11.3 HF integration.

    Use its public TokenEnforcerTokenizerData API without importing the Torch-
    dependent HF bridge. This checks core token-prefix filtering, NOT generate().
    """
    from lmformatenforcer import TokenEnforcerTokenizerData

    zero = tokenizer.encode("0")[-1]
    special = set(tokenizer.all_special_ids)
    regular = []
    for token in range(len(tokenizer)):
        if token in special:
            continue
        after_zero = tokenizer.decode([zero, token])[1:]
        alone = tokenizer.decode([token])
        regular.append((token, after_zero, len(after_zero) > len(alone)))
    return TokenEnforcerTokenizerData(
        regular,
        lambda ids: tokenizer.decode(ids).rstrip("�"),
        tokenizer.eos_token_id,
        use_bitmask=False,
        vocab_size=len(tokenizer),
    )


def prefix_accepts(data, tokenizer, prompt, schema, value):
    from lmformatenforcer import JsonSchemaParser, TokenEnforcer

    require_clean_grammar_environment()
    parser = JsonSchemaParser(schema, config=fixed_grammar_config())
    enforcer = TokenEnforcer(data, parser)
    ids = tokenizer.encode(prompt, add_special_tokens=False)
    output = tokenizer.encode(
        json.dumps(value, separators=(",", ":")), add_special_tokens=False
    )

    class ErrorCounter(logging.Handler):
        count = 0

        def emit(self, record):
            self.count += 1  # No prompt/prefix text emitted or retained.

    root, counter = logging.getLogger(), ErrorCounter()
    handlers, level = root.handlers[:], root.level
    root.handlers, root.level = [counter], logging.ERROR
    try:
        accepted = True
        for token in output:
            if not enforcer.get_allowed_tokens(ids).is_token_allowed(token):
                accepted = False
                break
            ids.append(token)
        accepted = accepted and enforcer.get_allowed_tokens(ids).is_token_allowed(
            tokenizer.eos_token_id
        )
    finally:
        root.handlers, root.level = handlers, level
        counter.close()
    if counter.count:
        raise ValueError("synthetic prefix smoke encountered grammar backend errors")
    return accepted, len(output)


def tokenizer_smoke(root):
    actual = {p.name: file_sha256(p) for p in root.iterdir() if p.is_file()}
    if actual != TOKENIZER_HASHES:
        raise ValueError("cached frozen tokenizer file set/hash mismatch")
    if importlib.metadata.version("lm-format-enforcer") != "0.11.3":
        raise ValueError("wrong LMFE version")
    require_clean_grammar_environment()
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(
        str(root), local_files_only=True, trust_remote_code=False
    )
    began = time.perf_counter()
    data = cpu_tokenizer_data(tokenizer)
    setup_seconds = time.perf_counter() - began
    cases, records = smoke_cases()[:3], []
    for case in cases:
        prompt = render_scifact_prompt(tokenizer, case["observation"], case["schema"])
        encoded = tokenizer.encode(prompt, add_special_tokens=False)
        called = tokenizer(prompt, add_special_tokens=False)["input_ids"]
        if encoded != called or len(encoded) > 8192:
            raise ValueError("prompt tokenization parity/budget failed")
        value = {
            "action": "answer",
            "documents": [
                {
                    "source_id": alias,
                    "label": label,
                    "sentence_ids": [f"{alias}:{i}" for i in [3, 1, 2, 0]],
                }
                for alias, label in zip(
                    case["aliases"], ["SUPPORTS", "REFUTES"], strict=True
                )
            ],
        }
        accepted, output_tokens = prefix_accepts(
            data, tokenizer, prompt, case["schema"], value
        )
        # Same valid two-document structure; vary only one reference binding.
        invalid = copy.deepcopy(value)
        invalid["documents"][0]["sentence_ids"][0] = f"{case['aliases'][1]}:3"
        cross_doc_accepted, _ = prefix_accepts(
            data, tokenizer, prompt, case["schema"], invalid
        )
        stale = copy.deepcopy(value)
        stale_alias = "c5" if case["aliases"][0] == "c0" else "c0"
        stale["documents"][0]["source_id"] = stale_alias
        stale["documents"][0]["sentence_ids"] = [
            f"{stale_alias}:{i}" for i in [3, 1, 2, 0]
        ]
        stale_accepted, _ = prefix_accepts(
            data, tokenizer, prompt, case["schema"], stale
        )
        if not accepted or cross_doc_accepted or stale_accepted or output_tokens >= 512:
            raise ValueError("synthetic token-prefix contract failed")
        records.append(
            {
                "input_tokens": len(encoded),
                "output_tokens": output_tokens,
                "valid_prefix_accepted": accepted,
                "cross_doc_prefix_rejected": not cross_doc_accepted,
                "stale_document_prefix_rejected": not stale_accepted,
            }
        )
    return {
        "schema_version": "scifact-cached-tokenizer-prefix-smoke-v1",
        "status": "passed",
        "tokenizer_revision": "350135a4de9a3407be836fa238cccc1d61503a85",
        "tokenizer_files_sha256": actual,
        "vocab_size": len(tokenizer),
        "grammar_config": grammar_config_identity(),
        "records": records,
        "tokenizer_prefix_setup_seconds": setup_seconds,
        "versions": {
            name: importlib.metadata.version(name)
            for name in [
                "transformers",
                "tokenizers",
                "lm-format-enforcer",
                "interegular",
                "jinja2",
            ]
        },
        "weights_loaded": False,
        "model_generation_calls": 0,
        "lmfe_core_token_prefix_executed": True,
        "hf_torch_generate_or_prefix_bridge_executed": False,
        "official_data_read": False,
        "quality_evaluation": False,
        "boundary": "Real frozen tokenizer/core LMFE only; selected synthetic A/B/A paths, not GPU generation or arbitrary-input certification.",
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--tokenizer", type=Path)
    parser.add_argument("--model-dir", type=Path)
    parser.add_argument("--model-manifest", type=Path)
    parser.add_argument("--private-dir", type=Path)
    parser.add_argument("--allow-real-model-generation", action="store_true")
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError("unique smoke output required")
    if args.allow_real_model_generation:
        if args.tokenizer or not all(
            [args.model_dir, args.model_manifest, args.private_dir]
        ):
            raise ValueError(
                "real synthetic runtime requires isolated model/manifest/private paths"
            )
        manifest = json.loads(args.model_manifest.read_text(encoding="utf-8"))
        if (
            hashlib.sha256(json.dumps(manifest, sort_keys=True).encode()).hexdigest()
            != MODEL_SHA
        ):
            raise ValueError("not the frozen Qwen3-4B manifest")
        provider = LocalQwenSciFactProvider(
            args.model_dir, manifest, private_dir=args.private_dir
        )
        report = synthetic_runtime_smoke(provider, execution_kind="real_model")
        report["model_manifest_sha256"] = MODEL_SHA
    else:
        if not args.tokenizer or any(
            [args.model_dir, args.model_manifest, args.private_dir]
        ):
            raise ValueError(
                "default mode is cached tokenizer only; no model arguments"
            )
        report = tokenizer_smoke(args.tokenizer)
    repo = Path(__file__).resolve().parents[1]
    report["source_git"] = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=repo, text=True
    ).strip()
    report["source_worktree_dirty"] = bool(
        subprocess.check_output(
            ["git", "status", "--porcelain"], cwd=repo, text=True
        ).strip()
    )
    with args.output.open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(report, stream, indent=2, sort_keys=True)
        stream.write("\n")
    print(json.dumps(report, sort_keys=True))
    return 0 if report["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
