"""Pinned real tokenizer + actual HF prefix callback, synthetic wires only.

No model weights, train/dev/test, quality metrics, network or GPU calls.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path

from climate_rag.bounded_scifact_grammar import build_bounded_scifact_prefix
from climate_rag.bounded_token_traversal import PlainStringPartition
from climate_rag.evidence_gap_candidate import gap_schema
from climate_rag.scifact_terminal import action_schema
from smoke_agent_v3_tokenizer import TOKENIZER_HASHES


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--tokenizer", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--case", help="Run a single named case for bounded performance diagnosis")
    parser.add_argument("--differential", action="store_true", help="Two bounded real-vocab slow/fast prefix comparisons")
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError("smoke output already exists")
    actual = {p.name: hashlib.sha256(p.read_bytes()).hexdigest()
              for p in args.tokenizer.iterdir() if p.is_file()}
    if actual != TOKENIZER_HASHES:
        raise ValueError("pinned tokenizer mismatch")
    import torch
    from lmformatenforcer.integrations.transformers import build_token_enforcer_tokenizer_data
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer, local_files_only=True, trust_remote_code=False)
    began = time.perf_counter()
    data = build_token_enforcer_tokenizer_data(tokenizer)
    partition = PlainStringPartition(data)
    load_ms = (time.perf_counter() - began) * 1000
    aliases = ["c0", "c1", "c10", "c2", "c3"]
    sids = [f"{a}:{i}" for a in aliases for i in range(10)]
    schema = action_schema(["abstain", "answer", "read", "rewrite", "rerank"], aliases, sids, 5)
    prompt = tokenizer.encode("Synthetic generation-constraint smoke. No scientific claim.", add_special_tokens=False)

    def answer(counts):
        return {"action": "answer", "documents": [
            {"source_id": a, "label": "SUPPORTS" if j % 2 else "REFUTES",
             "sentence_ids": [f"{a}:{i}" for i in reversed(range(n))]}
            for j, (a, n) in enumerate(zip(aliases, counts))]}
    doc = answer([1])["documents"][0]
    cases = [
        ("total20_8_8_4_A1", answer([8, 8, 4]), True),
        ("read_c1_c10_B", {"action": "read", "source_ids": ["c1", "c10"]}, True),
        ("total20_8_8_4_A2", answer([8, 8, 4]), True),
        ("total20_8_7_5", answer([8, 7, 5]), True),
        ("five_documents", answer([4, 4, 4, 4, 4]), True),
        ("total21_rejected", answer([8, 8, 5]), False),
        ("doc9_rejected", answer([9]), False),
        ("duplicate_doc_rejected", {"action": "answer", "documents": [doc, doc]}, False),
        ("duplicate_sid_rejected", {"action": "answer", "documents": [dict(doc, sentence_ids=["c0:0", "c0:0"])]}, False),
        ("duplicate_read_rejected", {"action": "read", "source_ids": ["c1", "c1"]}, False),
        ("rerank", {"action": "rerank"}, True),
        ("abstain", {"action": "abstain", "reason": "insufficient_evidence"}, True),
        ("rewrite", {"action": "rewrite", "query": 'synthetic "quoted" query'}, True),
        ("rewrite_unicode", {"action": "rewrite", "query": 'α 中文 query'}, True),
        ("rewrite_unicode_escaped", {"action": "rewrite", "query": 'α 中文 query'}, True),
        ("gap_answer20", {"evidence_state": {"retrieval_need": "not_needed", "relevance": "relevant",
                           "support": "sufficient"}, "gap_claim_span": "", "decision": answer([8, 8, 4])}, True),
        ("gap_rewrite", {"evidence_state": {"retrieval_need": "needed", "relevance": "mixed",
                          "support": "partial"}, "gap_claim_span": "synthetic claim fragment",
                         "decision": {"action": "rewrite", "query": "synthetic query"}}, True),
    ]
    rows = []
    for name, value, expected in cases:
        if args.case and name != args.case:
            continue
        text = json.dumps(value, ensure_ascii=name.endswith("_escaped"), separators=(",", ":"))
        tokens = tokenizer.encode(text, add_special_tokens=False)
        envelope = gap_schema(schema) if name.startswith("gap_") else None
        prefix = build_bounded_scifact_prefix(data, schema, envelope, plain_partition=partition)
        sent = list(prompt)
        times = []
        accepted = True
        stop = None
        for index, token in enumerate(tokens):
            began = time.perf_counter()
            allowed = prefix(0, torch.tensor(sent))
            times.append((time.perf_counter() - began) * 1000)
            if token not in allowed:
                accepted, stop = False, index
                break
            sent.append(token)
        if accepted:
            began = time.perf_counter()
            allowed = prefix(0, torch.tensor(sent))
            times.append((time.perf_counter() - began) * 1000)
            accepted = tokenizer.eos_token_id in allowed
        rows.append({"name": name, "accepted": accepted, "expected": expected,
                     "wire_tokens": len(tokens), "rejected_token_index": stop,
                     "prefix_calls": len(times),
                     "measured_prefix_ms": sum(times), "max_single_prefix_ms": max(times),
                     "effective_field_order": prefix.token_enforcer.root_parser.config.force_json_field_order})
        print(json.dumps(rows[-1]), flush=True)
        if accepted != expected:
            raise ValueError("HF prefix smoke mismatch: " + name)
    if not rows:
        raise ValueError("no matching smoke case")
    differential = []
    if args.differential:
        for text in ['{"action":"rewrite","query":"ab', '{"action":"rewrite","query":"a\\']:
            slow = build_bounded_scifact_prefix(data, schema)
            fast = build_bounded_scifact_prefix(data, schema, plain_partition=partition)
            sent = list(prompt)
            began = time.perf_counter()
            for token in tokenizer.encode(text, add_special_tokens=False):
                assert set(slow(0, torch.tensor(sent))) == set(fast(0, torch.tensor(sent)))
                sent.append(token)
            a, b = set(slow(0, torch.tensor(sent))), set(fast(0, torch.tensor(sent)))
            if a != b:
                raise ValueError("real-vocab slow/fast allowed-token disagreement")
            if tokenizer.eos_token_id in a:
                raise ValueError("incomplete JSON must not permit EOS")
            differential.append({"prefix_sha256": hashlib.sha256(text.encode()).hexdigest(),
                                 "same_allowed_tokens": len(a), "eos_allowed": False,
                                 "elapsed_ms": (time.perf_counter() - began) * 1000})
    report = {"schema_version": "bounded-scifact-prefix-real-tokenizer-smoke-v1",
              "scope": "synthetic CPU token constraints, no model quality",
              "tokenizer_sha256": actual, "tokenizer_data_build_ms": load_ms,
              "model_calls": 0, "gpu_calls": 0, "rows": rows, "differential": differential}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(report, stream, indent=2, sort_keys=True)
        stream.write("\n")


if __name__ == "__main__":
    main()
