"""Synthetic CPU-only candidate validation. No dataset/model/GPU arguments."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import logging
import subprocess
from pathlib import Path
from typing import Any

from climate_rag.agent_v3 import Source, V3Budget
from climate_rag.evidence_gap_candidate import (
    CANDIDATE_PROTOCOL,
    EvidenceGapCandidate,
    GapProviderAdapter,
    parse_gap_wire,
    render_gap_prompt,
)
from climate_rag.evidence_gap_grammar import OrderedGapPrefix, build_ordered_gap_prefix
from climate_rag.local_scifact_provider import require_clean_grammar_environment
from climate_rag.private_diagnostics_v3 import PrivateDiagnosticStore
from climate_rag.public_v2 import file_sha256
from climate_rag.scifact_agent_v1 import SciFactDocumentAgentV1
from climate_rag.scifact_runtime_smoke import smoke_cases
from climate_rag.scifact_terminal import PROTOCOL, render_scifact_prompt

from smoke_agent_v3_tokenizer import TOKENIZER_HASHES
from smoke_scifact_provider import cpu_tokenizer_data


def wire(decision: dict[str, Any], gap: str = "") -> dict[str, Any]:
    return {
        "evidence_state": {
            "retrieval_need": "uncertain",
            "relevance": "unknown",
            "support": "unknown",
        },
        "gap_claim_span": gap,
        "decision": decision,
    }


class TokenList:
    def __init__(self, ids: list[int]):
        self.ids = ids

    def tolist(self) -> list[int]:
        return list(self.ids)


def callback_accepts(
    callback: Any, tokenizer: Any, prompt: str, value: Any
) -> tuple[bool, int]:
    ids = tokenizer.encode(prompt, add_special_tokens=False)
    output = tokenizer.encode(
        json.dumps(value, separators=(",", ":")), add_special_tokens=False
    )
    for token in output:
        if token not in callback(0, TokenList(ids)):
            return False, len(output)
        ids.append(token)
    return tokenizer.eos_token_id in callback(0, TokenList(ids)), len(output)


class ScriptedBackend:
    name, kind = "synthetic-cached-tokenizer", "fixture"
    wire_protocol, terminal_protocol = CANDIDATE_PROTOCOL, PROTOCOL

    def __init__(
        self, tokenizer: Any, responses: list[Any], root: Path, candidate: bool = True
    ):
        self.tokenizer, self.responses = tokenizer, iter(responses)
        self.store, self.candidate = PrivateDiagnosticStore(root), candidate

    def count_text(self, text: str) -> int:
        return len(self.tokenizer.encode(text, add_special_tokens=False))

    def count_prompt(self, observation: Any, schema: Any) -> int:
        renderer = render_gap_prompt if self.candidate else render_scifact_prompt
        return self.count_text(renderer(self.tokenizer, observation, schema))

    def generate(
        self,
        observation: Any,
        schema: Any,
        max_output_tokens: int,
        remaining_seconds: float,
    ) -> dict[str, Any]:
        raw = json.dumps(next(self.responses), separators=(",", ":"))
        output = self.count_text(raw)
        if output > max_output_tokens:
            raise ValueError("synthetic wire exceeds shared output cap")
        sink = self.store.sink("response", 32768)
        sink.write(raw)
        return {
            "raw": raw,
            "usage": {
                "input_tokens": self.count_prompt(observation, schema),
                "output_tokens": output,
            },
            "diagnostics": {"private_attachment": sink.receipt()},
        }


def prefix_smoke(tokenizer: Any, data: Any, private: Path) -> dict[str, Any]:
    from lmformatenforcer import JsonSchemaParser, TokenEnforcer
    from lmformatenforcer.characterlevelparser import CharacterLevelParserConfig

    records, callbacks = [], []
    for i, case in enumerate(smoke_cases()[:3]):
        backend = ScriptedBackend(tokenizer, [], private / str(i))
        adapter = GapProviderAdapter(backend)
        observation, schema = adapter.prepare(case["observation"], case["schema"])
        prompt = render_gap_prompt(tokenizer, observation, schema)
        encoded = tokenizer.encode(prompt, add_special_tokens=False)
        if encoded != tokenizer(prompt, add_special_tokens=False)["input_ids"]:
            raise ValueError("candidate renderer tokenization mismatch")
        value = wire(
            {
                "action": "answer",
                "documents": [
                    {
                        "source_id": alias,
                        "label": label,
                        "sentence_ids": [f"{alias}:{j}" for j in [3, 1, 2, 0]],
                    }
                    for alias, label in zip(
                        case["aliases"], ["SUPPORTS", "REFUTES"], strict=True
                    )
                ],
            }
        )
        parse_gap_wire(json.dumps(value), observation["immutable_claim"], schema)
        callback = build_ordered_gap_prefix(data, schema)
        callbacks.append(callback)
        valid, output_tokens = callback_accepts(callback, tokenizer, prompt, value)
        decision_first = {
            "decision": value["decision"],
            "evidence_state": value["evidence_state"],
            "gap_claim_span": "",
        }
        no_gap = {
            "evidence_state": value["evidence_state"],
            "decision": value["decision"],
        }
        cross_doc = copy.deepcopy(value)
        cross_doc["decision"]["documents"][0]["sentence_ids"][0] = (
            f"{case['aliases'][1]}:3"
        )
        stale = copy.deepcopy(value)
        other = "c5" if case["aliases"][0] == "c0" else "c0"
        stale["decision"]["documents"][0]["source_id"] = other
        stale["decision"]["documents"][0]["sentence_ids"] = [f"{other}:0"]
        rejected = {
            name: not callback_accepts(
                build_ordered_gap_prefix(data, schema), tokenizer, prompt, bad
            )[0]
            for name, bad in [
                ("decision_first", decision_first),
                ("skipped_gap", no_gap),
                ("cross_document", cross_doc),
                ("stale_alias", stale),
            ]
        }
        if (
            not valid
            or not all(rejected.values())
            or len(encoded) > 8192
            or output_tokens >= 512
        ):
            raise ValueError("candidate ordered callback contract failed")
        # Regression control: parser constructor alone is insufficient in0.11.3.
        constructor_config = CharacterLevelParserConfig(
            alphabet=data.tokenizer_alphabet,
            max_consecutive_whitespaces=12,
            force_json_field_order=True,
            max_json_array_length=20,
        )
        unsafe = OrderedGapPrefix(
            TokenEnforcer(data, JsonSchemaParser(schema, config=constructor_config))
        )
        constructor_only_accepts_wrong_order = callback_accepts(
            unsafe, tokenizer, prompt, decision_first
        )[0]
        if not constructor_only_accepts_wrong_order:
            raise ValueError("pinned LMFE constructor-overwrite control changed")
        records.append(
            {
                "input_tokens": len(encoded),
                "scripted_wire_output_tokens": output_tokens,
                "ordered_mixed_label_wire_accepted": valid,
                "rejected": rejected,
                "constructor_only_config_wrongly_accepts_decision_first": constructor_only_accepts_wrong_order,
                "effective_force_json_field_order": callback.token_enforcer.root_parser.config.force_json_field_order,
                "effective_alphabet_sha256": hashlib.sha256(
                    callback.token_enforcer.root_parser.config.alphabet.encode()
                ).hexdigest(),
            }
        )
    if len({id(c.token_enforcer) for c in callbacks}) != 3:
        raise ValueError("A/B/A did not use fresh prefix state")
    return {
        "fresh_A_B_A": True,
        "records": records,
        "real_model_generations": 0,
        "hf_generate_tested": False,
    }


def packing_smoke(tokenizer: Any, private: Path) -> list[dict[str, Any]]:
    corpus = [
        Source(
            str(i),
            "Synthetic fixture document",
            tuple(
                f"Synthetic valve document {i}, sentence {j}: "
                + "Calibration readings are contextual observations, not real study data. "
                * 10
                for j in range(12)
            ),
        )
        for i in range(6)
    ]
    claim = "Synthetic valve calibration changed in 2020."
    abstain = {"action": "abstain", "reason": "insufficient_evidence"}
    records = []
    for route in ["fixed_retrieval", "fixed_rerank", "deterministic_extra", "adaptive"]:
        row: dict[str, Any] = {
            "route": route,
            "same_input_cap": 2048,
            "same_output_cap": 512,
        }
        for candidate in [False, True]:
            backend = ScriptedBackend(
                tokenizer,
                [wire(abstain) if candidate else abstain],
                private / f"{route}-{candidate}",
                candidate,
            )
            engine = EvidenceGapCandidate if candidate else SciFactDocumentAgentV1
            result = engine(
                backend,
                lambda q, k: corpus,
                rerank=lambda q, rows: rows,
                budget=V3Budget(context_k=2, max_input_tokens=2048),
            ).run(claim, route)
            if result["model_calls"] != 1 or not result["outcome"].startswith(
                "model_abstention:"
            ):
                raise ValueError("synthetic packing did not reach legal abstention")
            attempt = result["generation_attempts"][0]
            row["candidate" if candidate else "frozen_action_template"] = {
                "input_tokens": attempt["input_prompt_tokens"],
                "visible_sentence_ids": list(attempt["visible_sentence_sha256"]),
                "visible_sentence_count": attempt["displayed_sentence_count"],
                "scripted_output_tokens": attempt["usage"]["output_tokens"],
            }
        records.append(row)
    return records


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tokenizer", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    repository = Path(__file__).resolve().parents[1]
    status = subprocess.check_output(
        ["git", "status", "--porcelain", "--untracked-files=all"],
        cwd=repository,
        text=True,
    )
    if status:
        raise ValueError("candidate smoke requires clean tracked/nonignored source")
    source = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=repository, text=True
    ).strip()
    actual = {p.name: file_sha256(p) for p in args.tokenizer.iterdir() if p.is_file()}
    if actual != TOKENIZER_HASHES:
        raise ValueError("frozen tokenizer files/hash mismatch")
    require_clean_grammar_environment()
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(
        str(args.tokenizer), local_files_only=True, trust_remote_code=False
    )
    args.output.mkdir(parents=True, exist_ok=False)
    data = cpu_tokenizer_data(tokenizer)

    class ErrorCounter(logging.Handler):
        count = 0

        def emit(self, record: logging.LogRecord) -> None:
            self.count += 1

    root, counter = logging.getLogger(), ErrorCounter()
    handlers, level = root.handlers[:], root.level
    root.handlers, root.level = [counter], logging.ERROR
    try:
        prefix = prefix_smoke(tokenizer, data, args.output / "private-prefix")
        packing = packing_smoke(tokenizer, args.output / "private-packing")
    finally:
        root.handlers, root.level = handlers, level
        counter.close()
    if counter.count:
        raise ValueError("grammar backend logged errors; no success receipt")
    report = {
        "schema_version": "evidence-gap-candidate-cpu-smoke-v1",
        "source_git": source,
        "candidate_protocol": CANDIDATE_PROTOCOL,
        "status": "passed",
        "tokenizer_revision": "350135a4de9a3407be836fa238cccc1d61503a85",
        "tokenizer_files_sha256": actual,
        "prefix_callback": prefix,
        "packing": packing,
        "real_queries_read": 0,
        "real_model_generations": 0,
        "gpu_jobs_submitted": 0,
        "real_weights_hf_integration_verified": False,
        "classification_or_agent_quality_measured": False,
        "counts_are_scripted_fixture_token_lengths_not_model_performance": True,
    }
    target = args.output / "report.json"
    target.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {
                "status": "passed",
                "report_sha256": file_sha256(target),
                "source_git": source,
            }
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
