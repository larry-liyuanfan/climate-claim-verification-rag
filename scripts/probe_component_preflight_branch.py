"""Tokenizer/grammar-only audit of two public synthetic screening responses.

No model weights, dataset, frozen targets or model calls. Reachability does not
recover generation logits or identify the causal reason for model abstention.
"""
from __future__ import annotations

import argparse
import importlib.metadata
import inspect
import json
import os
import sys
from collections.abc import Callable
from pathlib import Path
from typing import Any

from climate_rag.component_decoder import decoder_schema
from climate_rag.component_preflight import preflight_cases
from climate_rag.local_agent_v3 import fixed_grammar_config
from climate_rag.scifact_component_contract import packing, parse, render, require, schema_for
from climate_rag.scifact_semantic_contract import TOKENIZER_SHA, checked, sha, write_once
from probe_component_tokenizer import token_path
from smoke_scifact_provider import cpu_tokenizer_data

OBSERVED_RAW = '{"decision": "abstain", "document_ids": []}'
EXPECTED_RAW = '{"decision": "select", "document_ids": [1]}'
EXPECTED_PROMPT_SHA = "b49e0108e331ac0b9478fffcb5faaa270a35e253aab1a0b0e235b8215fedba4f"


def shared_prefix(left: list[int], right: list[int]) -> list[int]:
    result = []
    for a, b in zip(left, right):
        if a != b:
            break
        result.append(a)
    return result


def probe(tokenizer_dir: Path) -> dict[str, Any]:
    require(all(os.environ.get(k) == v for k, v in {"USE_TORCH": "0", "USE_TF": "0",
            "HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1"}.items()), "tokenizer_only_environment")
    require(importlib.metadata.version("lm-format-enforcer") == "0.11.3", "frozen_lmfe")
    for name, expected in TOKENIZER_SHA.items():
        checked(tokenizer_dir / name, expected)
    from transformers.models.auto.tokenization_auto import AutoTokenizer
    from transformers.generation.logits_process import PrefixConstrainedLogitsProcessor
    from lmformatenforcer import JsonSchemaParser, TokenEnforcer
    tokenizer = AutoTokenizer.from_pretrained(str(tokenizer_dir), local_files_only=True, trust_remote_code=False)
    spec, expected = preflight_cases()[0]
    packed = packing(tokenizer, spec)
    require(packed["prompt_sha256"] == EXPECTED_PROMPT_SHA and packed["input_tokens"] == 277,
            "actual_frozen_synthetic_prompt_identity")
    require(parse(EXPECTED_RAW, spec) == expected and parse(OBSERVED_RAW, spec) != expected, "synthetic_identity")
    builder: Callable[[Any], Any] = cpu_tokenizer_data
    data = builder(tokenizer)
    variants = (EXPECTED_RAW, OBSERVED_RAW,
                '{"document_ids": [1], "decision": "select"}',
                '{"document_ids": [], "decision": "abstain"}')
    paths = [{"synthetic_raw": raw, **token_path(data, tokenizer, spec, raw)} for raw in variants]
    active = tokenizer.encode(EXPECTED_RAW, add_special_tokens=False)
    abstain = tokenizer.encode(OBSERVED_RAW, add_special_tokens=False)
    common = shared_prefix(active, abstain)
    require(len(common) < min(len(active), len(abstain)), "branch_point")
    enforcer = TokenEnforcer(data, JsonSchemaParser(decoder_schema(schema_for(spec)), config=fixed_grammar_config()))
    ids = tokenizer.encode(render(tokenizer, spec), add_special_tokens=False)
    # Walk states sequentially just like generation; never query an unseen prefix
    # without its parents, which would initialize an incorrect root parser state.
    for token in common:
        require(enforcer.get_allowed_tokens(ids).is_token_allowed(token), "common_prefix_rejected")
        ids.append(token)
    allowed = enforcer.get_allowed_tokens(ids)
    next_active, next_abstain = active[len(common)], abstain[len(common)]
    require(allowed.is_token_allowed(next_active) and allowed.is_token_allowed(next_abstain), "branch_unreachable")
    candidates = sorted(set(allowed.allowed_tokens))
    implementation = inspect.getsource(PrefixConstrainedLogitsProcessor.__call__)
    return {"schema_version": "component-synthetic-branch-cpu-v1", "source_job": "31734906",
        "frozen_prompt": packed, "paths": paths,
        "first_decision_divergence": {
            "prefix_text": tokenizer.decode(common), "prefix_tokens": common,
            "active_next": {"id": next_active, "text": tokenizer.decode([next_active]), "allowed": True},
            "abstain_next": {"id": next_abstain, "text": tokenizer.decode([next_abstain]), "allowed": True},
            "allowed_token_count": len(candidates),
            "allowed_candidates_first_64": [{"id": i, "text": tokenizer.decode([i])} for i in candidates[:64]],
            "candidate_listing_truncated": len(candidates) > 64},
        "observed_raw_sha256": sha(OBSERVED_RAW.encode()),
        "observed_canonical_prediction": parse(OBSERVED_RAW, spec), "expected": expected,
        "observed_retokenized_length_excluding_eos": len(abstain),
        "reported_runtime_generated_tokens_including_eos": 15,
        "actual_runtime_per_token_ids_saved": False, "runtime_logits_saved": False,
        "hf_prefix_processor_source_sha256": sha(implementation.encode()),
        "hf_prefix_processor_operation": "scores + mask: allowed 0, disallowed -inf; no branch-specific positive bias",
        "versions": {name: importlib.metadata.version(name) for name in
                     ("transformers", "tokenizers", "lm-format-enforcer", "interegular")},
        "model_calls": 0, "weights_loaded": False, "dataset_read": False,
        "optional_existing_cpu_torch_imported": "torch" in sys.modules,
        "probe_source_sha256": sha(Path(__file__).read_bytes()),
        "conclusion": "tested active/abstain paths reachable; no causal title or probability conclusion"}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tokenizer-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = probe(args.tokenizer_dir)
    write_once(args.output, result)
    print(json.dumps({"accepted_paths": len(result["paths"]), "model_calls": 0,
                      "branch": result["first_decision_divergence"],
                      "raw_tokens_excluding_eos": result["observed_retokenized_length_excluding_eos"]}))
