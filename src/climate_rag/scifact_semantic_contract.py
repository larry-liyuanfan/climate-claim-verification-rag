"""Stdlib-only frozen handoff contract; no gold/model loading at import."""
from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

RELEASE = "climate-scifact-semantic-pair-20260930-v1"
PAIR = "scifact-semantic-policy-paired-v1"
PREPARATION_GIT = "779e49883570371ce6223cb2b4a8619df5924d3b"
COMPACT_SHA = "4d73e0196fb64167798b89746020b15aca192ebdb94cf34a36d6bd274d7adadd"
INPUT_SHA = "c4907db623b644c9c883a807f07597f87746595da7fc6435714452304e50a44d"
SELECTED_INFERENCE_SHA = "0ac620a9b3e10ed80aac3724689e4d23830450b440d2dc8d35f2ed7f79657e44"
PRIVATE_SHA = {
    "consumption-ledger.json": "65eac7e26e5603ee8672b44eb6b0dce0d6d8544c8759cfa36184daaf0a518a73",
    "current-opportunity.json": "65e91ff669fa1163a479b3594735fc8e8a0899823b5ebeaa90a9c3709edcdc4b",
    "future-pair-protocol-draft.json": "a90c728b900865f84bb872c7d6d4a213357abfbc9c558ea51f619ae292056342",
    "selected-before-probe.json": "118d57233b1205f294d7090b6820d658f447ba09263fe8179bdf8e66497338a8",
    "selected-gold.json": "0725cd9529c5c6c3b25afa25e6c0e6964a68cab9f989dfca0e87ffa0f5698dcb",
    "selected-inference.json": SELECTED_INFERENCE_SHA,
}
TOKENIZER_SHA = {
    "merges.txt": "8831e4f1a044471340f7c0a83d7bd71306a5b867e95fd870f74d0c5308a904d5",
    "tokenizer.json": "aeb13307a71acd8fe81861d94ad54ab689df773318809eed3cbe794b4492dae4",
    "tokenizer_config.json": "5a7303fcb1a27ede63134a2cbd61d5282c247ca6d769ce4746d4ffa124aedd63",
    "vocab.json": "ca10d7e9fb3ed18575dd1e277a2579c16d108e32f27439684afa0e10b1440910",
}
CORPUS_SHA = "5df96a227847260d2877931d62e58288472902db605c5eacb31d9d9703165f76"
CORPUS_MEMBER = "data/scifact-original-20260930/prepared-r1/inference/corpus.jsonl"
POLICIES = ("scifact-gap-original-v1", "scifact-gap-semantic-policy-v1")
PROMPTS = dict(zip(POLICIES, (
    "b4a6b1e5accd5a6b427dfe278245580c9ff16598c03d876c8920324cc2d30dfd",
    "2be482d567d0e57a546eebae440e9b04f86995a0724260b95c7d31278f834fab",
), strict=True))
ROUTES = ("fixed_retrieval", "fixed_rerank", "deterministic_extra", "adaptive")
BUDGET = dict(candidate_k=20, context_k=5, max_calls=5, max_repairs=2,
              max_tools=5, max_input_tokens=8192, max_output_tokens=512, timeout_seconds=120.0)
MODEL_SHA = "d1dd9783afdf4e0fbd21eee824834d71b86982f5a5d5f6f371fe07f2f76f3cf6"
RERANKER_SHA = "de1d4ac39101816774439e68881e2308c5e5f1bd94d0b0dc4c492a56c2681052"
GRAMMAR_SHA = "36ed51e0a9a71ded058cc0e7290a78a7f5f638b91bff517d49ad6ea83c70e410"
SCORING_NAMES = frozenset({"gold.jsonl", "selected-strata.json", "current-opportunity.json",
    "consumption-ledger.json", "selection-freeze.json", "preparation-draft.json", "manifest.json"})
INFERENCE_NAMES = frozenset({"protocol.json", "corpus.jsonl", "claims.jsonl"})
SCOPE = "previously_gold_preparation_seen_eligible_train_not_independent_test"
DECODING = "unchanged greedy nonthinking bounded grammar"
PACKING = "max(oldG empty,newG empty,both versions of active actual history,active actual prompt)"
PROTOCOL_KEYS = frozenset({"release_id", "comparison_protocol", "preparation_source_git",
    "execution_source_git", "preparation_compact_sha256", "scope", "ordered_claim_ids", "policies",
    "routes", "budget", "policy_prompt_sha256", "tokenizer_sha256", "model_manifest_sha256",
    "reranker_manifest_sha256", "slots_per_arm", "paired_slots", "inference_file_sha256",
    "corpus_documents", "model_calls", "official_dev_read", "fresh_baseline_required", "old_G_results_reusable",
    "gpu_authorized", "release_requires_external_coordinator_authorization", "preflight_calls_per_arm", "decoding", "packing"})


def sha(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def checked(path: Path, expected: str) -> bytes:
    if path.is_symlink():
        raise ValueError("symlink_input")
    payload = path.read_bytes()
    if sha(payload) != expected:
        raise ValueError("input_sha_mismatch")
    return payload


def encoded(value: Any) -> bytes:
    return (json.dumps(value, sort_keys=True, ensure_ascii=False, indent=2) + "\n").encode()


def jsonl(rows: Sequence[Mapping[str, Any]]) -> bytes:
    return ("\n".join(json.dumps(r, sort_keys=True, ensure_ascii=False) for r in rows) + "\n").encode()


def write_once(path: Path, value: Any) -> None:
    with path.open("xb") as stream:
        stream.write(encoded(value))


def validate_protocol(p: Mapping[str, Any], claims: Sequence[Mapping[str, Any]], git: str) -> None:
    ids = [r.get("id") for r in claims]
    if (set(p) != PROTOCOL_KEYS or sha(encoded(claims)) != SELECTED_INFERENCE_SHA
            or p.get("scope") != SCOPE or p.get("decoding") != DECODING or p.get("packing") != PACKING
            or p.get("preflight_calls_per_arm") != 4 or p.get("release_requires_external_coordinator_authorization") is not True
            or len(git) != 40 or any(c not in "0123456789abcdef" for c in git)
            or p.get("execution_source_git") != git or p.get("preparation_source_git") != PREPARATION_GIT
            or p.get("preparation_compact_sha256") != COMPACT_SHA
            or p.get("release_id") != RELEASE or p.get("comparison_protocol") != PAIR
            or p.get("policy_prompt_sha256") != PROMPTS or tuple(p.get("policies", ())) != POLICIES
            or tuple(p.get("routes", ())) != ROUTES or p.get("budget") != BUDGET
            or p.get("model_manifest_sha256") != MODEL_SHA or p.get("reranker_manifest_sha256") != RERANKER_SHA
            or p.get("tokenizer_sha256") != TOKENIZER_SHA
            or p.get("ordered_claim_ids") != ids or len(ids) != 12 or len(set(ids)) != 12
            or any(set(r) != {"id", "claim"} or type(r["id"]) is not int
                   or not isinstance(r["claim"], str) or not r["claim"].strip() for r in claims)
            or p.get("corpus_documents") != 5183 or p.get("model_calls") != 0
            or p.get("official_dev_read") is not False or p.get("gpu_authorized") is not False
            or p.get("fresh_baseline_required") is not True or p.get("old_G_results_reusable") is not False
            or p.get("slots_per_arm") != [{"claim_id": i, "route": r} for i in ids for r in ROUTES]
            or p.get("paired_slots") != [{"claim_id": i, "route": r, "policy": a}
                                          for a in POLICIES for i in ids for r in ROUTES]
            or set(p.get("inference_file_sha256", {})) != {"corpus.jsonl", "claims.jsonl"}
            or p["inference_file_sha256"]["corpus.jsonl"] != CORPUS_SHA):
        raise ValueError("semantic_protocol_contract")


def load_inference(directory: Path, protocol_sha: str, git: str) -> tuple[dict[str, Any], list[dict[str, Any]], bytes]:
    if {p.name for p in directory.iterdir()} != INFERENCE_NAMES:
        raise ValueError("inference_exact_allowlist")
    protocol = json.loads(checked(directory / "protocol.json", protocol_sha))
    claims_payload = checked(directory / "claims.jsonl", protocol["inference_file_sha256"]["claims.jsonl"])
    claims = [json.loads(line) for line in claims_payload.splitlines()]
    validate_protocol(protocol, claims, git)
    corpus = checked(directory / "corpus.jsonl", CORPUS_SHA)
    return protocol, claims, corpus
