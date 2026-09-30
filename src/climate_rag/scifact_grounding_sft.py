"""One bounded TRAIN-only grounding candidate using the production terminal wire.

This is document/rationale task adaptation, not action supervision. Importing
the module never loads weights. Gold is used only to author FIT targets or by
the independent scorer, never to construct evaluation candidate contexts.
"""
from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Mapping, Sequence
from typing import Any

from .scifact_grounding import LABEL_TO_PROJECT, Abstract, GoldClaim
from .scifact_semantic_contract import MODEL_SHA, TOKENIZER_SHA, encoded, sha
from .scifact_terminal import (
    PROTOCOL, action_schema, parse_action, render_answer, render_scifact_prompt,
    source_from_abstract, to_original_prediction,
)

VERSION = "scifact-grounding-sft-cpu-v1"
CONFIG: dict[str, Any] = {
    "version": VERSION, "seed": 20261001, "base_manifest_sha256": MODEL_SHA,
    "tokenizer_sha256": TOKENIZER_SHA, "terminal_protocol": PROTOCOL,
    "lora_r": 8, "lora_alpha": 16, "lora_dropout": 0.0,
    "target_modules": ["q_proj", "v_proj"], "task_type": "CAUSAL_LM",
    "learning_rate": 0.0001, "weight_decay": 0.0,
    "batch_size": 1, "gradient_accumulation": 4,
    "max_epochs": 1, "max_optimizer_steps": 64, "max_fit_records": 192,
    "max_records_per_claim": 4, "max_input_tokens": 8192,
    "max_output_tokens": 512, "max_seconds_per_call": 120,
    "candidate_k": 5, "comparison_calls": 48,
    "decoding": "greedy_nonthinking", "weak_negative_weight": 0.0,
    "annotation_target_weight": 1.0, "checkpoint_policy": "final_only",
    "abstention_threshold": "none_no_posthoc_threshold_selection",
}
QUOTAS = {
    "fit": {"SUPPORTS": 24, "REFUTES": 24},
    "tune": {"SUPPORTS": 4, "REFUTES": 4, "NOT_ENOUGH_INFO": 4},
    "validation": {"SUPPORTS": 4, "REFUTES": 4, "NOT_ENOUGH_INFO": 4},
}


def require(ok: bool, category: str) -> None:
    if not ok:
        raise ValueError(category)  # never include private text/IDs


def config_sha() -> str:
    return sha(encoded(CONFIG))


def select_components(
    assignment: Mapping[str, Any], labels: Mapping[int, str],
    excluded: Sequence[str],
) -> dict[str, list[int]]:
    """Freeze identities before token/target inspection; no replacement on failure.

    One hash-selected representative per frozen component. Fit can include
    historically consumed TRAIN, while tune/validation cannot. Labels only
    establish the preregistered strata; no performance/packing-based sampling.
    """
    ids = assignment["eligible_train_ids"]
    require(len(ids) == len(set(ids)) and set(labels) == set(ids), "eligible_matrix")
    components = assignment["claim_component"]
    members: dict[str, list[int]] = {}
    for i in ids:
        members.setdefault(str(components[str(i)]), []).append(i)

    def key(i: int) -> str:
        return sha(f'{VERSION}:{CONFIG["seed"]}:claim:{i}'.encode())

    representatives = {c: min(v, key=key) for c, v in members.items()}
    ordered = sorted(representatives, key=lambda c: sha(
        f'{VERSION}:{CONFIG["seed"]}:component:{c}'.encode()))
    used: set[str] = set()
    result: dict[str, list[int]] = {}
    # Reserve the two unconsumed partitions before fit; never retry another seed.
    for part in ("tune", "validation", "fit"):
        rows: list[int] = []
        for label, count in QUOTAS[part].items():
            candidates = [c for c in ordered if c not in used
                          and (part == "fit" or c not in excluded)
                          and labels[representatives[c]] == label]
            require(len(candidates) >= count, "insufficient_component_stratum")
            selected = candidates[:count]
            used.update(selected)
            rows.extend(representatives[c] for c in selected)
        result[part] = rows
    require(len(used) == 72, "whole_component_overlap")
    return result


def context(claim: str, docs: Sequence[Abstract]) -> dict[str, Any]:
    """Gold-free complete contexts in the production visible-citation path."""
    require(bool(claim.strip()) and 0 < len(docs) <= 5, "context_bounds")
    require(len({d.doc_id for d in docs}) == len(docs), "duplicate_context_document")
    observation: dict[str, Any] = {
        "immutable_claim": claim, "allowed_actions": ["answer", "abstain"],
        "current_citable": [], "preview_only": [], "feedback": "",
        "remaining_calls": 1, "remaining_tools": 0,
    }
    visible: dict[str, Any] = {}
    aliases: list[str] = []
    for number, abstract in enumerate(docs, 1):
        alias = f"c{number}"
        aliases.append(alias)
        source = source_from_abstract(abstract)
        for index, text in enumerate(source.sentences):
            sid = f"{alias}:{index}"
            observation["current_citable"].append({"sentence_id": sid, "text": text})
            visible[sid] = {
                "source_id": source.source_id, "sentence_index": index,
                "text": text, "text_sha256": hashlib.sha256(text.encode()).hexdigest(),
                "source_text_sha256": source.text_sha256,
            }
    schema = action_schema(["answer", "abstain"], aliases, list(visible), 5)
    return {"observation": observation, "schema": schema, "visible": visible,
            "aliases": aliases, "document_ids": [d.doc_id for d in docs]}


def canonical_context(value: Mapping[str, Any], corpus: Mapping[int, Abstract]) -> dict[str, Any]:
    """Reconstruct field order and reject hidden gold, extra keys or edited text."""
    rebuilt = context(value["observation"]["immutable_claim"],
                      [corpus[i] for i in value["document_ids"]])
    require(dict(value) == rebuilt, "context_identity_or_hidden_fields")
    return rebuilt


def prediction(claim_id: int, raw: Any, ctx: Mapping[str, Any],
               corpus: Mapping[int, Abstract]) -> dict[str, Any]:
    decision = parse_action(raw, ["answer", "abstain"], ctx["visible"], ctx["aliases"], 5)
    result: dict[str, Any] = {"protocol": PROTOCOL, "outcome": "ids_validated_semantics_unmeasured",
              "answer": None}
    if decision["action"] == "answer":
        result["answer"] = render_answer(decision, ctx["visible"])
    else:
        result["outcome"] = "model_abstention:" + decision["reason"]
    return to_original_prediction(claim_id, result, corpus)


def fit_records(claim: GoldClaim, corpus: Mapping[int, Abstract],
                component: str) -> list[dict[str, Any]]:
    """Up to four official document/alternative records, never union targets.

    No sampled unannotated negative, no document-level NEI label, and no label
    or gold markers in the input. Full annotated document is the FIT context;
    evaluation retrieval must not reuse this context selection policy.
    """
    rows: list[dict[str, Any]] = []
    for doc_id, alternatives in sorted(claim.evidence.items()):
        for alternative_index, alternative in enumerate(alternatives):
            if len(rows) == CONFIG["max_records_per_claim"]:
                return rows
            target = {"action": "answer", "documents": [{
                "source_id": "c1", "label": LABEL_TO_PROJECT[alternative.label],
                "sentence_ids": [f"c1:{i}" for i in alternative.sentences],
            }]}
            ctx = context(claim.claim, [corpus[doc_id]])
            # Invalid/too-long targets stop preparation, not truncate or replace.
            prediction(claim.claim_id, target, ctx, corpus)
            rows.append({"claim_id": claim.claim_id, "component": component,
                         "context": ctx, "target": target, "weight": 1.0,
                         "provenance": "official_annotated_document_alternative",
                         "document_id": doc_id, "alternative_index": alternative_index})
    return rows


def token_record(tokenizer: Any, ctx: Mapping[str, Any],
                 target: Mapping[str, Any] | None = None) -> dict[str, Any]:
    prompt = render_scifact_prompt(tokenizer, ctx["observation"], ctx["schema"])
    inputs = list(tokenizer.encode(prompt, add_special_tokens=False))
    require(len(inputs) <= CONFIG["max_input_tokens"], "full_context_over_budget")
    result = {"input_tokens": len(inputs), "prompt_sha256": sha(prompt.encode()),
              "context_sha256": sha(encoded(ctx))}
    if target is None:
        return result
    canonical_target = parse_action(dict(target), ["answer", "abstain"],
                                    ctx["visible"], ctx["aliases"], 5)
    raw = json.dumps(canonical_target, ensure_ascii=False, separators=(",", ":"))
    # Verify the full tokenized sequence shares the actual inference prefix;
    # separate tokenization is not assumed additive at a BPE boundary.
    end = tokenizer.eos_token_id
    require(type(end) is int, "tokenizer_eos_required")
    combined = list(tokenizer.encode(prompt + raw + tokenizer.eos_token,
                                     add_special_tokens=False))
    require(combined[:len(inputs)] == inputs, "assistant_prefix_token_mismatch")
    assistant = combined[len(inputs):]
    require(bool(assistant) and assistant[-1] == end, "assistant_eos_mismatch")
    require(0 < len(assistant) <= CONFIG["max_output_tokens"], "complete_target_over_budget")
    return result | {"input_ids": inputs + assistant,
                     "attention_mask": [1] * (len(inputs) + len(assistant)),
                     "labels": [-100] * len(inputs) + assistant,
                     "target_tokens": len(assistant), "target_json_sha256": sha(raw.encode()),
                     "full_token_ids_sha256": sha(encoded(inputs + assistant)),
                     "loss_mask_sha256": sha(encoded([-100] * len(inputs) + assistant))}


def training_steps(record_count: int) -> int:
    require(type(record_count) is int and 0 < record_count <= 192, "fit_record_ceiling")
    return int(min(64, math.ceil(record_count / CONFIG["gradient_accumulation"])))


def advancement(base: Mapping[str, Any], adapted: Mapping[str, Any]) -> bool:
    """Predeclared count gate; failures never masquerade as abstentions."""
    require(base["input_identity"] == adapted["input_identity"], "paired_input_mismatch")
    require(base["claims"] == adapted["claims"] == 12, "tune_matrix")
    require(base["attempts"] == adapted["attempts"] == 12, "tune_call_matrix")
    return bool(adapted["correctly_rationalized_documents"] > base["correctly_rationalized_documents"]
                and adapted["nei_false_evidence"] <= base["nei_false_evidence"]
                and adapted["planned_unsuccessful"] <= base["planned_unsuccessful"]
                and base["unknown_usage"] == adapted["unknown_usage"] == 0)
