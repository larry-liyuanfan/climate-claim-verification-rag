"""Private, bounded Stage A preparation; scoring targets never enter renderer."""
from __future__ import annotations

import json
from collections.abc import Mapping
from typing import Any

from .scifact_component_contract import packing, require
from .scifact_semantic_contract import sha


def source_hash(doc: Mapping[str, Any]) -> str:
    return sha(json.dumps([doc["title"], doc["abstract"]], ensure_ascii=False, separators=(",", ":")).encode())


def document(doc: Mapping[str, Any]) -> dict[str, Any]:
    return {"document_id": doc["doc_id"], "title": doc["title"],
            "sentences": [{"sentence_id": i, "text": s} for i, s in enumerate(doc["abstract"])]}


def actual_documents(result: Mapping[str, Any], corpus: Mapping[int, Any]) -> dict[str, Any]:
    require(bool(result["visible_attempts"]) and bool(result["generation_attempts"]), "missing_trace")
    first = result["visible_attempts"][0]
    require(type(first.get("attempt_index")) is int and first["attempt_index"] == 0, "first_attempt_identity")
    visible = first["visible"]
    projection: list[dict[str, str]] = []
    aliases: dict[str, int] = {}
    seen_sids: set[str] = set()
    for v in visible:
        alias, doc_id, index = v["alias"], v["doc_id"], v["sentence_index"]
        require(isinstance(alias, str) and type(doc_id) is int and type(index) is int, "trace_types")
        require(doc_id in corpus and 0 <= index < len(corpus[doc_id]["abstract"]), "trace_index")
        require(alias not in aliases or aliases[alias] == doc_id, "alias_mapping_changed")
        require(alias in aliases or doc_id not in aliases.values(), "duplicate_document_alias")
        aliases[alias] = doc_id
        sid = f"{alias}:{index}"
        require(sid not in seen_sids, "duplicate_trace_sentence")
        seen_sids.add(sid)
        require(v["source_text_sha256"] == source_hash(corpus[doc_id]), "source_hash")
        require(v["text_sha256"] == sha(corpus[doc_id]["abstract"][index].encode()), "sentence_hash")
        projection.append({"sentence_id": sid, "sha256": v["text_sha256"]})
    identities = [first["actual_observation"]["visible"], result["initial_context_identity"]["visible"],
                  result["generation_attempts"][0]["diagnostics"]["actual_observation"]["visible"]]
    require(all(identity == projection for identity in identities), "ordered_visible_identity_mismatch")
    requested = result["generation_attempts"][0]["requested_context"]
    require(isinstance(requested, list) and len(requested) == len(set(requested)), "requested_duplicates")
    require(set(aliases) <= set(requested), "visible_outside_requested")
    return {"documents": [document(corpus[i]) for i in aliases.values()],
            "candidate_doc_ids": list(aliases.values()), "original_visible": visible,
            "requested_without_visible_document": len(set(requested) - set(aliases)),
            "requested_documents": len(requested)}


def prepare_claim(gold: Mapping[str, Any], result: Mapping[str, Any], corpus: Mapping[int, Any],
                  tokenizer: Any) -> list[dict[str, Any]]:
    """Gold is used for oracle target selection/scoring only, not screening input."""
    reconstructed = actual_documents(result, corpus)
    evidence = gold["evidence"]
    relevant = [int(k) for k in evidence]
    pool = reconstructed["candidate_doc_ids"]
    screening_target = {"relevant_doc_ids": relevant, "pool_doc_ids": pool,
                        "gold_missing_from_pool": len(set(relevant) - set(pool)),
                        "requested_without_visible_document": reconstructed["requested_without_visible_document"],
                        "requested_documents": reconstructed["requested_documents"],
                        "original_complete_first3_gold_documents": sum(any(
                            0 < len(r["sentences"]) <= 3 and set(r["sentences"]) <=
                            {v["sentence_index"] for v in reconstructed["original_visible"] if v["doc_id"] == d}
                            for r in evidence[str(d)]) for d in relevant)}
    rows: list[dict[str, Any]] = []

    def add(component: str, docs: list[Any], target: dict[str, Any], gap: str | None = None,
            relation: str | None = None) -> None:
        spec = {"component": component, "claim": gold["claim"], "documents": docs}
        if component == "rationale":
            spec["oracle_relation"] = relation
        packed = packing(tokenizer, spec) if not gap else {"status": gap, "input_tokens": None}
        rows.append({"claim_id": gold["id"], "component": component, "input": spec if not gap else None,
                     "packing": packed, "target": target})

    add("screening", reconstructed["documents"], screening_target, "no_actual_visible_context" if not pool else None)
    nei = not relevant
    candidates = gold["cited_doc_ids"] if nei else relevant
    require(isinstance(candidates, list) and all(type(v) is int and v >= 0 for v in candidates), "target_ids")
    provenance = "official_code_derived_cited_context_NEI" if nei else "official_document_relation_oracle_document"
    if not candidates or min(candidates) not in corpus:
        add("relation", [], {"label_source": provenance, "nei_control": nei}, "missing_nei_source" if nei else "missing_gold_document")
        if not nei:
            add("rationale", [], {}, "missing_gold_document")
        return rows
    selected = min(candidates)
    rats = [] if nei else evidence[str(selected)]
    labels = {r["label"] for r in rats}
    require(labels <= {"SUPPORT", "CONTRADICT"}, "unsupported_gold_relation")
    if not nei and len(labels) != 1:
        add("relation", [], {"nei_control": False}, "inconsistent_target_labels")
        add("rationale", [], {}, "inconsistent_target_labels")
        return rows
    relation = "NOT_ENOUGH_INFO" if nei else next(iter(labels))
    docs = [document(corpus[selected])]
    add("relation", docs, {"target_doc_id": selected, "relation": relation, "nei_control": nei,
                           "label_source": provenance})
    if not nei:
        alternatives = [r["sentences"] for r in rats]
        require(all(isinstance(a, list) and bool(a) and all(type(i) is int and 0 <= i < len(docs[0]["sentences"]) for i in a)
                    and len(a) == len(set(a)) for a in alternatives), "gold_sentence_contract")
        reachable = any(len(a) <= 3 for a in alternatives)
        add("rationale", docs, {"target_doc_id": selected, "alternatives": alternatives,
                                "first3_reachable": reachable, "oracle_relation": relation},
            None if reachable else "first3_contract_unreachable", relation)
    return rows


def coverage(rows: list[dict[str, Any]]) -> dict[str, Any]:
    components: dict[str, Any] = {}
    for name in ("screening", "relation", "rationale"):
        subset = [r for r in rows if r["component"] == name]
        reasons = {r["packing"]["status"] for r in subset}
        tokens = [r["packing"]["input_tokens"] for r in subset if r["packing"]["input_tokens"] is not None]
        components[name] = {"planned": len(subset), "prepared": sum(r["packing"]["status"] == "prepared" for r in subset),
                            "statuses": {s: sum(r["packing"]["status"] == s for r in subset) for s in sorted(reasons)},
                            "min_input_tokens": min(tokens) if tokens else None, "max_input_tokens": max(tokens) if tokens else None}
    screen = [r["target"] for r in rows if r["component"] == "screening"]
    gold_docs = sum(len(r["relevant_doc_ids"]) for r in screen)
    missing = sum(r["gold_missing_from_pool"] for r in screen)
    return {"components": components, "gold_documents": gold_docs, "gold_documents_in_actual_pool": gold_docs - missing,
            "gold_document_pool_coverage": (gold_docs - missing) / gold_docs if gold_docs else None,
            "requested_without_visible_documents": sum(r["requested_without_visible_document"] for r in screen),
            "actual_candidate_documents": sum(len(r["pool_doc_ids"]) for r in screen),
            "old_context_first3_complete_gold_documents": sum(r["original_complete_first3_gold_documents"] for r in screen),
            "nei_cited_context_controls": sum(r["component"] == "relation" and r["target"].get("nei_control", False) for r in rows),
            "model_calls": 0, "new_claims_selected": 0}
