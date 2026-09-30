"""Four synthetic component fixtures, zero model calls and no real data reads."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from climate_rag.scifact_component_contract import packing, protocol
from climate_rag.scifact_component_runtime import fixture_slot
from climate_rag.scifact_semantic_contract import encoded, sha, write_once


class SyntheticTokenizer:
    """Character counter ONLY; never reported as Qwen token packing."""
    def apply_chat_template(self, messages: Any, **kwargs: Any) -> str:
        if kwargs != {"tokenize": False, "add_generation_prompt": True, "enable_thinking": False}:
            raise ValueError("fixture_template_contract")
        return json.dumps(messages, ensure_ascii=False)

    def encode(self, text: str, **kwargs: Any) -> list[int]:
        if kwargs != {"add_special_tokens": False}:
            raise ValueError("fixture_encoding_contract")
        return [1] * len(text)


class ScriptedProvider:
    kind = "fixture"
    tokenizer = SyntheticTokenizer()

    def __init__(self, wire: dict[str, Any]) -> None:
        self.wire = wire
        self.calls = 0

    def generate(self, spec: dict[str, Any], schema: dict[str, Any], max_tokens: int, seconds: float) -> dict[str, Any]:
        self.calls += 1
        raw = json.dumps(self.wire)
        return {"raw": raw, "usage": {"input_tokens": packing(self.tokenizer, spec)["input_tokens"],
                                      "output_tokens": len(raw)},
                "diagnostics": {"eos_observed": True, "reached_max_new_tokens": False}}


def check(output: Path) -> dict[str, Any]:
    output.mkdir(mode=0o700)
    base = {"claim": "Synthetic compound improves a synthetic endpoint.",
            "documents": [{"document_id": 9000, "title": "Synthetic study, not real evidence",
                           "sentences": [{"sentence_id": 0, "text": "Synthetic endpoint improved in this fixture."},
                                         {"sentence_id": 1, "text": "Synthetic irrelevant background."}]}]}
    slots: list[tuple[dict[str, Any], dict[str, Any]]] = [(base | {"component": "screening"}, {"decision": "select", "document_ids": [9000]}),
             (base | {"component": "relation"}, {"decision": "classify", "relation": "SUPPORT"}),
             (base | {"component": "rationale", "oracle_relation": "SUPPORT"}, {"decision": "select", "sentence_ids": [0]}),
             (base | {"component": "screening"}, {"decision": "abstain", "document_ids": []})]
    records = [fixture_slot(spec, ScriptedProvider(wire), output / f"slot-{i}") for i, (spec, wire) in enumerate(slots)]
    result = {"fixture_calls": 4, "model_calls": 0, "valid_fixtures": sum(r["status"] == "valid" for r in records),
              "protocol_sha256": sha(encoded(protocol())), "tokenizer": "synthetic_character_counter_not_Qwen",
              "status": "fixture_contract_only_not_model_quality_or_real_preflight"}
    if result["valid_fixtures"] != 4:
        raise ValueError("synthetic_fixture_failure")
    write_once(output / "protocol-template.json", protocol())
    write_once(output / "fixture-summary.json", result)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    print(json.dumps(check(parser.parse_args().output), sort_keys=True))


if __name__ == "__main__":
    main()
