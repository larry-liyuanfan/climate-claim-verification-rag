"""Gold-free Stage A component contracts. Importing never loads a model."""
from __future__ import annotations

import json
from collections.abc import Mapping
from typing import Any

from .scifact_semantic_contract import MODEL_SHA, TOKENIZER_SHA, encoded, sha

VERSION = "scifact-component-oracle-diagnostic-v1"
COMPONENTS = ("screening", "relation", "rationale")
LIMITS = {"input_tokens": 8192, "output_tokens": 512, "seconds": 120,
          "attempts_per_slot": 1, "diagnostic_calls": 33, "preflight_calls": 4}
PROMPTS = {
    "screening": (
        "Select documents that provide scientific evidence supporting or contradicting the claim. "
        "Use only the supplied full abstracts. Select any number of the supplied document IDs, "
        "without duplicates. Select no document by choosing abstain if none is sufficient. "
        "This is document screening, not relation classification. Return only the schema JSON."
    ),
    "relation": (
        "Classify the relation between the claim and the supplied complete scientific abstract: "
        "SUPPORT, CONTRADICT, or NOT_ENOUGH_INFO. NOT_ENOUGH_INFO means this abstract does not "
        "establish either relation. Choose abstain only if you cannot perform the classification. "
        "The document was externally selected; its selection does not reveal the label. "
        "Return only the schema JSON."
    ),
    "rationale": (
        "The supplied oracle_relation is explicitly the correct relation for this document. "
        "This is an oracle-conditioned sentence-selection diagnostic, not classification. "
        "Select one complete minimal alternative rationale from the full abstract. Keep original "
        "sentence indices. Put a complete rationale within your first three selected indices; "
        "up to eight unique indices are permitted, but extra indices are not required. "
        "Choose abstain if you cannot identify a complete rationale. Return only the schema JSON."
    ),
}


class ContractError(ValueError):
    """Category only: never include private text or identifiers in errors."""


def require(ok: bool, category: str) -> None:
    if not ok:
        raise ContractError(category)


def validate_input(spec: Mapping[str, Any]) -> None:
    component = spec.get("component")
    require(component in COMPONENTS, "component")
    keys = {"component", "claim", "documents"}
    if component == "rationale":
        keys.add("oracle_relation")
        require(spec.get("oracle_relation") in {"SUPPORT", "CONTRADICT"}, "oracle_relation")
    require(set(spec) == keys, "input_fields")
    require(isinstance(spec["claim"], str) and bool(spec["claim"].strip()), "claim")
    docs = spec["documents"]
    require(isinstance(docs, list) and bool(docs), "empty_document_context")
    require(component == "screening" or len(docs) == 1, "single_document_required")
    seen: set[int] = set()
    for doc in docs:
        require(isinstance(doc, dict) and set(doc) == {"document_id", "title", "sentences"}, "document_fields")
        i = doc["document_id"]
        require(type(i) is int and i >= 0 and i not in seen, "document_identity")
        seen.add(i)
        require(isinstance(doc["title"], str), "title")
        sentences = doc["sentences"]
        require(isinstance(sentences, list) and bool(sentences), "empty_abstract")
        for index, sentence in enumerate(sentences):
            require(isinstance(sentence, dict) and set(sentence) == {"sentence_id", "text"}, "sentence_fields")
            require(type(sentence["sentence_id"]) is int and sentence["sentence_id"] == index,
                    "original_sentence_order")
            require(isinstance(sentence["text"], str) and bool(sentence["text"].strip()), "sentence_text")


def object_schema(properties: dict[str, Any]) -> dict[str, Any]:
    return {"type": "object", "properties": properties, "required": list(properties), "additionalProperties": False}


def schema_for(spec: Mapping[str, Any]) -> dict[str, Any]:
    validate_input(spec)
    component = spec["component"]
    if component == "relation":
        active = object_schema({"decision": {"const": "classify"},
                                "relation": {"enum": ["SUPPORT", "CONTRADICT", "NOT_ENOUGH_INFO"]}})
        abstain = object_schema({"decision": {"const": "abstain"}, "relation": {"type": "null"}})
    else:
        key = "document_ids" if component == "screening" else "sentence_ids"
        ids = ([d["document_id"] for d in spec["documents"]] if component == "screening"
               else [s["sentence_id"] for s in spec["documents"][0]["sentences"]])
        active = object_schema({"decision": {"const": "select"}, key: {"type": "array",
            "items": {"type": "integer", "enum": ids}, "minItems": 1,
            "maxItems": len(ids) if component == "screening" else min(8, len(ids)), "uniqueItems": True}})
        abstain = object_schema({"decision": {"const": "abstain"}, key: {"type": "array", "maxItems": 0}})
    return {"anyOf": [active, abstain]}


def authored_input(spec: Mapping[str, Any]) -> dict[str, Any]:
    """Restore Stage A field order after canonical JSON persistence.

    Lists and values are never reordered or repaired. These exact object-key
    orders are those used by prepare_claim/document before freezing packing.
    Alphabetically sorting prompt JSON would change the frozen prompt bytes.
    """
    validate_input(spec)
    ordered = {"component": spec["component"], "claim": spec["claim"], "documents": [
        {"document_id": doc["document_id"], "title": doc["title"], "sentences": [
            {"sentence_id": sentence["sentence_id"], "text": sentence["text"]}
            for sentence in doc["sentences"]]} for doc in spec["documents"]]}
    if spec["component"] == "rationale":
        ordered["oracle_relation"] = spec["oracle_relation"]
    return ordered


def messages(spec: Mapping[str, Any]) -> list[dict[str, str]]:
    schema = schema_for(spec)
    return [{"role": "system", "content": PROMPTS[spec["component"]] + "\n" +
             json.dumps(schema, ensure_ascii=False, separators=(",", ":"))},
            {"role": "user", "content": json.dumps(authored_input(spec), ensure_ascii=False, separators=(",", ":"))}]


def render(tokenizer: Any, spec: Mapping[str, Any]) -> str:
    text = tokenizer.apply_chat_template(messages(spec), tokenize=False,
                                        add_generation_prompt=True, enable_thinking=False)
    require(isinstance(text, str), "tokenizer_render")
    return str(text)


def packing(tokenizer: Any, spec: Mapping[str, Any]) -> dict[str, Any]:
    text = render(tokenizer, spec)
    count = len(tokenizer.encode(text, add_special_tokens=False))
    return {"input_tokens": count, "prompt_sha256": sha(text.encode()),
            "schema_sha256": sha(encoded(schema_for(spec))), "input_sha256": sha(encoded(spec)),
            "status": "prepared" if count <= LIMITS["input_tokens"] else "full_abstract_over_budget"}


def unique_pairs(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        require(key not in result, "duplicate_key")
        result[key] = value
    return result


def parse(raw: str, spec: Mapping[str, Any]) -> dict[str, Any]:
    validate_input(spec)
    try:
        obj = json.loads(raw, object_pairs_hook=unique_pairs,
                         parse_constant=lambda _: (_ for _ in ()).throw(ContractError("nonfinite")))
    except json.JSONDecodeError as exc:
        raise ContractError("invalid_json") from exc
    component = spec["component"]
    key = {"screening": "document_ids", "relation": "relation", "rationale": "sentence_ids"}[component]
    require(isinstance(obj, dict) and set(obj) == {"decision", key}, "output_fields")
    decision, value = obj["decision"], obj[key]
    if component == "relation":
        require((decision == "abstain" and value is None) or
                (decision == "classify" and isinstance(value, str) and value in
                 {"SUPPORT", "CONTRADICT", "NOT_ENOUGH_INFO"}), "relation_output")
    else:
        require(isinstance(value, list) and all(type(v) is int for v in value), "index_type")
        require(len(set(value)) == len(value), "duplicate_index")
        allowed = ({d["document_id"] for d in spec["documents"]} if component == "screening" else
                   set(range(len(spec["documents"][0]["sentences"]))))
        require(set(value) <= allowed, "unknown_index")
        require(component != "rationale" or len(value) <= 8, "output_over_budget")
        require((decision == "abstain" and not value) or (decision == "select" and bool(value)), "selection_output")
    return dict(obj)


def protocol() -> dict[str, Any]:
    return {"version": VERSION, "limits": LIMITS, "components": list(COMPONENTS),
        "prompt_sha256": {k: sha(v.encode()) for k, v in PROMPTS.items()},
        "tokenizer_sha256": TOKENIZER_SHA, "model_manifest_sha256": MODEL_SHA,
        "decoding": "greedy_nonthinking", "model_calls": 0, "gpu_authorized": False,
        "scope": "already_consumed_twelve_train_claims_component_diagnostic_not_generalization",
        "candidates": "original_G_fixed_rerank_first_actual_visible_alias_order_no_backfill",
        "input_change": "restore_full_original_abstracts_not_old_prompt_replay",
        "target_document": "minimum_numeric_gold_doc_id_or_minimum_official_cited_doc_id_for_NEI",
        "rationale_condition": "same_target_document_explicit_oracle_relation_no_alternative_leakage",
        "nei_provenance": "official_code_derived_cited_context_not_human_pair_negative",
        "gold_sets": "alternatives_OR_original_SIDs_first_three_in_original_output_order",
        "missing_policy": "retain_gap_no_replacement_no_truncation_no_gold_aware_packing",
        "resource_ceiling_not_authorized": {"gpu_a100": 1, "cpu": 8, "memory_gib": 32, "wall_seconds": 6000},
        "wall_basis": "37_single_attempts_x120_seconds_plus1560_staging_load_margin",
        "stop": "CPU_preparation_only_separate_exact_hash_provider_operator_GPU_release_required"}
