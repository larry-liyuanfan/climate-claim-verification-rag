"""Offline frozen-tokenizer reachability, zero weights/model/API/data calls."""
from __future__ import annotations

import argparse
import importlib.metadata
import json
import logging
import sys
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

from climate_rag.component_decoder import decoder_identity, decoder_schema
from climate_rag.component_preflight import preflight_cases
from climate_rag.local_agent_v3 import fixed_grammar_config
from climate_rag.local_scifact_provider import require_clean_grammar_environment
from climate_rag.scifact_component_contract import packing, parse, render, require, schema_for
from climate_rag.scifact_semantic_contract import TOKENIZER_SHA, checked, write_once
from smoke_scifact_provider import cpu_tokenizer_data


def paths() -> list[tuple[dict[str, Any], str]]:
    a = preflight_cases()[0][0]
    a["documents"].append(dict(a["documents"][0], document_id=2))
    b = preflight_cases()[2][0]
    b["documents"][0]["sentences"] = [{"sentence_id": i, "text": f"Synthetic sentence {i}."} for i in range(10)]
    relation = preflight_cases()[1][0]
    return [(a, raw) for raw in ('{"decision":"select","document_ids":[1,10,2]}',
             '{"decision":"select","document_ids":[2,10,1]}',
             '{"decision":"select","document_ids":[1 , 10]}',
             '{"document_ids":[10,2],"decision":"select"}', '{"decision":"abstain","document_ids":[]}')] + [
        (b, '{"decision":"select","sentence_ids":[7,6,5,4,3,2,1,0]}'),
        (b, '{"sentence_ids":[9,1,0],"decision":"select"}'),
        (b, '{"decision":"abstain","sentence_ids":[]}')] + [
        (relation, json.dumps({"relation": label, "decision": "classify"}))
        for label in ("SUPPORT", "CONTRADICT", "NOT_ENOUGH_INFO")] + [
        (relation, '{"decision":"abstain","relation":null}'),
        (a, '{"decision":"select","document_ids":[2,10,1]}')]


def token_path(data: Any, tokenizer: Any, spec: dict[str, Any], raw: str) -> dict[str, Any]:
    from lmformatenforcer import JsonSchemaParser, TokenEnforcer
    from lmformatenforcer.characterlevelparser import ForceStopParser
    parse(raw, spec)  # canonical contract first, never change a path to fit decoder
    schema = schema_for(spec)
    parser = JsonSchemaParser(decoder_schema(schema), config=fixed_grammar_config())
    enforcer = TokenEnforcer(data, parser)
    ids = tokenizer.encode(render(tokenizer, spec), add_special_tokens=False)
    output = tokenizer.encode(raw, add_special_tokens=False)

    class Errors(logging.Handler):
        count = 0
        def emit(self, record: Any) -> None:
            self.count += 1
    root, handler = logging.getLogger(), Errors()
    old_handlers, old_level = root.handlers[:], root.level
    root.handlers, root.level = [handler], logging.ERROR
    try:
        for token in output:
            allowed = enforcer.get_allowed_tokens(ids)
            state = enforcer.prefix_states[tuple(ids)].parser
            require(not isinstance(state, ForceStopParser), "token_force_stop")
            require(not allowed.is_token_allowed(tokenizer.eos_token_id), "premature_eos")
            require(allowed.is_token_allowed(token), "legal_token_rejected")
            ids.append(token)
        allowed = enforcer.get_allowed_tokens(ids)
        require(not isinstance(enforcer.prefix_states[tuple(ids)].parser, ForceStopParser), "terminal_force_stop")
        require(allowed.is_token_allowed(tokenizer.eos_token_id), "complete_eos_unreachable")
        require(handler.count == 0, "token_grammar_error")
    finally:
        root.handlers, root.level = old_handlers, old_level
        handler.close()
    return {"component": spec["component"], "packing": packing(tokenizer, spec),
            "decoder": decoder_identity(schema), "output_tokens": len(output), "accepted": True}


def probe(tokenizer_dir: Path) -> dict[str, Any]:
    require_clean_grammar_environment()
    require("torch" not in sys.modules, "torch_must_not_be_imported")
    require(importlib.metadata.version("lm-format-enforcer") == "0.11.3", "lmfe_version")
    for name, expected in TOKENIZER_SHA.items():
        checked(tokenizer_dir / name, expected)
    from transformers.models.auto.tokenization_auto import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(str(tokenizer_dir), local_files_only=True, trust_remote_code=False)
    begin = time.perf_counter()
    builder: Callable[[Any], Any] = cpu_tokenizer_data
    data = builder(tokenizer)
    records = [token_path(data, tokenizer, spec, raw) for spec, raw in paths()]
    # LMFE tokenlist optionally imports an installed CPU Torch even when using
    # list mode. Report it honestly; no tensor, model loader or HF bridge is used.
    return {"status": "tested_synthetic_token_paths_reachable_not_model_quality", "records": records,
            "tokenizer_files_sha256": TOKENIZER_SHA, "vocab_size": len(tokenizer),
            "elapsed_seconds": time.perf_counter()-begin, "model_calls": 0, "weights_loaded": False,
            "hf_generate_executed": False, "real_data_read": False,
            "optional_cpu_torch_imported_by_lmfe": "torch" in sys.modules}


if __name__ == "__main__":
    args = argparse.ArgumentParser(description=__doc__)
    args.add_argument("--tokenizer-dir", type=Path, required=True)
    args.add_argument("--output", type=Path, required=True)
    options = args.parse_args()
    result = probe(options.tokenizer_dir)
    write_once(options.output, result)
    print(json.dumps({"status": result["status"], "paths": len(result["records"]), "model_calls": 0}))
