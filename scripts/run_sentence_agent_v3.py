"""Allocated, consumed-authored v3 development runner; never reads benchmark gold."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import os
import sys
from collections import Counter
from pathlib import Path

from climate_rag.agent_v3 import (
    SentenceAgentV3,
    Source,
    V3Budget,
    action_schema,
    parse_action,
)
from climate_rag.bm25 import BM25Index
from climate_rag.io import iter_evidence
from climate_rag.local_agent_model import verify_model_files
from climate_rag.local_agent_v3 import LocalQwenV3Provider, grammar_config_identity
from climate_rag.models import RankedDocument
from climate_rag.rerank import Qwen3CausalLMReranker

from run_budget_agent_full_operator import CORPUS_SHA, MODEL_SHA, RERANKER_SHA, digest

RELEASE = "climate-sentence-v3-20260930-pilot-r1"
ROUTES = ["fixed_retrieval", "fixed_rerank", "deterministic_extra", "adaptive"]


def validate_protocol(protocol):
    if (
        protocol["release_id"] != RELEASE
        or protocol["official_gold_available"]
        or protocol["corpus_sha256"] != CORPUS_SHA
        or protocol["document_count"] != 5240
        or protocol["routes"] != ROUTES
        or len(protocol["tasks"]) != 6
        or len({t["id"] for t in protocol["tasks"]}) != 6
        or any(set(t) != {"id", "claim_text"} for t in protocol["tasks"])
    ):
        raise ValueError("development protocol mismatch")
    budget = V3Budget.model_validate(protocol["budget"])
    if budget != V3Budget():
        raise ValueError("unreleased v3 budget")
    return budget


def write_once(path, value):
    with path.open("x", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2, sort_keys=True)
        stream.write("\n")


def runtime_preflight(provider):
    """Real prefix callback/EOS/fresh-state smoke, not evidence-quality evaluation."""
    records = []
    for source_id, cap in [("c0", 128), ("c1", 128), ("c0", 128), ("c1", 1)]:
        schema = action_schema(["read"], [source_id], [], 1)
        observation = {
            "claim": "Synthetic protocol check",
            "current_citable": [],
            "preview_only": [{"source_id": source_id}],
            "allowed_actions": ["read"],
        }
        record = {"max_new_tokens": cap}
        try:
            response = provider.generate(observation, schema, cap, 45)
            record.update(
                usage=response["usage"], diagnostics=response.get("diagnostics", {})
            )
            diagnostics = response["diagnostics"]
            record.update(
                {
                    "input_tokens": response["usage"]["input_tokens"],
                    "output_tokens": response["usage"]["output_tokens"],
                    "expected_input_tokens": provider.count_prompt(observation, schema),
                    "diagnostics": diagnostics,
                    "max_new_tokens": cap,
                }
            )
            if cap == 1:
                try:
                    json.loads(response["raw"])
                    incomplete = False
                except json.JSONDecodeError:
                    incomplete = True
                passed = (
                    incomplete
                    and record["output_tokens"] == 1
                    and diagnostics["reached_max_new_tokens"]
                    and not diagnostics["eos_observed"]
                )
            else:
                value = parse_action(
                    json.loads(response["raw"]), ["read"], {}, [source_id], 1
                )
                passed = (
                    value == {"action": "read", "source_ids": [source_id]}
                    and diagnostics["eos_observed"]
                )
            record["passed"] = bool(
                passed
                and record["input_tokens"] == record["expected_input_tokens"]
                and 0 < record["output_tokens"] <= cap
                and record["input_tokens"] <= 8192
            )
        except Exception as exc:
            record.update(passed=False, exception_type=type(exc).__name__)
            record.setdefault("usage", getattr(exc, "usage", {}))
            record.setdefault("diagnostics", getattr(exc, "diagnostics", {}))
        records.append(record)
        if not record["passed"]:
            break
    return {
        "passed": len(records) == 4 and all(r["passed"] for r in records),
        "scope": "forced synthetic grammar runtime, not autonomous decisions or quality",
        "records": records,
    }


def compact_run(run, protocol):
    expected = {(t["id"], r) for t in protocol["tasks"] for r in ROUTES}
    actual = [(r["task_id"], r["route"]) for r in run["runs"]]
    if len(actual) != len(expected) or set(actual) != expected:
        raise ValueError("incomplete or duplicate matrix")
    routes = {}
    for route in ROUTES:
        rows = [r for r in run["runs"] if r["route"] == route]
        routes[route] = {
            "slots": len(rows),
            "outcomes": dict(Counter(r["outcome"] for r in rows)),
            "mechanically_accepted_answers": sum(r["answer"] is not None for r in rows),
            "model_selected_completed_tools": dict(
                Counter(
                    e["tool"]
                    for r in rows
                    for e in r["events"]
                    if e.get("model_selected") and e.get("status") == "completed"
                )
            ),
            "generation_attempt_status": dict(
                Counter(a["status"] for r in rows for a in r["generation_attempts"])
            ),
            "known_token_subtotal": {
                k: sum(r["usage"][k] for r in rows)
                for k in ("input_tokens", "output_tokens")
            },
            **{
                key: sum(r[key] for r in rows)
                for key in (
                    "model_calls",
                    "tool_calls",
                    "rerank_pairs",
                    "unknown_usage_attempts",
                    "elapsed_ms",
                    "context_payload_tokens_sum",
                    "preview_payload_tokens_sum",
                )
            },
        }
    return {
        "schema_version": "sentence-id-v3-pilot-compact",
        "source_git": run["source_git"],
        "protocol_sha256": run["protocol_sha256"],
        "corpus_sha256": CORPUS_SHA,
        "model_sha256": run["model_sha256"],
        "reranker_sha256": run["reranker_sha256"],
        "document_count": run["document_count"],
        "complete_slots": len(actual),
        "official_evidence_metrics": None,
        "semantic_quality": None,
        "independent_evaluation": False,
        "dense_enabled": False,
        "boundary": "Consumed authored questions; exact quotes do not prove entailment. Equal caps, not equal actual work.",
        "routes": routes,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for option in (
        "evidence",
        "model-dir",
        "model-manifest",
        "reranker-dir",
        "reranker-manifest",
        "protocol",
        "output-dir",
    ):
        parser.add_argument("--" + option, type=Path, required=True)
    parser.add_argument("--expected-protocol-sha256", required=True)
    args = parser.parse_args()
    if (
        not os.environ.get("SLURM_JOB_ID")
        or not os.environ.get("CUDA_VISIBLE_DEVICES")
        or os.environ.get("CLIMATE_V3_RELEASE") != RELEASE
    ):
        raise ValueError("separately released GPU allocation required")
    if (
        digest(args.protocol) != args.expected_protocol_sha256
        or digest(args.evidence) != CORPUS_SHA
    ):
        raise ValueError("input hash mismatch")
    protocol = json.loads(args.protocol.read_text())
    budget = validate_protocol(protocol)
    source_git = (
        (Path(__file__).resolve().parents[1] / "SOURCE_REVISION").read_text().strip()
    )
    if source_git != os.environ["CLIMATE_SOURCE_GIT"]:
        raise ValueError("source identity mismatch")
    for name, version in {
        "transformers": "4.51.3",
        "lm-format-enforcer": "0.11.3",
        "interegular": "0.3.3",
    }.items():
        if importlib.metadata.version(name) != version:
            raise ValueError("runtime version mismatch")
    documents = list(iter_evidence(args.evidence))
    if len(documents) != 5240 or len({d.evidence_id for d in documents}) != 5240:
        raise ValueError("corpus count/unique ID mismatch")
    model_manifest = json.loads(args.model_manifest.read_text())
    reranker_manifest = json.loads(args.reranker_manifest.read_text())
    if (
        hashlib.sha256(json.dumps(model_manifest, sort_keys=True).encode()).hexdigest()
        != MODEL_SHA
    ):
        raise ValueError("wrong model manifest")
    if verify_model_files(args.reranker_dir, reranker_manifest) != RERANKER_SHA:
        raise ValueError("wrong reranker manifest")
    out = args.output_dir
    write_once(
        out / "runner.consumed.json", {"release_id": RELEASE, "source_git": source_git}
    )
    private = out / "private-responses"
    private.mkdir(mode=0o700)
    provider = LocalQwenV3Provider(args.model_dir, model_manifest, private_dir=private)
    preflight = runtime_preflight(provider)
    preflight["grammar_config"] = grammar_config_identity()
    write_once(out / "runtime-preflight.json", preflight)
    if not preflight["passed"]:
        raise ValueError("real runtime preflight failed; no pilot")
    index = BM25Index().fit(documents)
    sources = {
        d.evidence_id: Source(
            d.evidence_id, str(d.metadata.get("title", "")), (d.text,)
        )
        for d in documents
    }
    reranker = Qwen3CausalLMReranker(
        str(args.reranker_dir),
        device="cuda",
        dtype="bfloat16",
        max_length=2048,
        batch_size=1,
    )

    def retrieve(query, width):
        return [sources[r.evidence_id] for r in index.search(query, width)]

    def rerank(query, candidates):
        ranked = [
            RankedDocument(
                evidence_id=s.source_id,
                text=" ".join(s.sentences),
                score=0.0,
                rank=i + 1,
                source="v3-candidates",
            )
            for i, s in enumerate(candidates)
        ]
        return [
            sources[r.evidence_id] for r in reranker.rerank(query, ranked, len(ranked))
        ]

    agent = SentenceAgentV3(provider, retrieve, rerank=rerank, budget=budget)
    rows = []
    for task in protocol["tasks"]:
        for route in ROUTES:
            row = {"task_id": task["id"], **agent.run(task["claim_text"], route)}
            rows.append(row)
            write_once(out / f"slot-{len(rows):02}.json", row)
    run = {
        "source_git": source_git,
        "protocol_sha256": args.expected_protocol_sha256,
        "document_count": len(documents),
        "model_sha256": provider.base.model_sha256,
        "reranker_sha256": RERANKER_SHA,
        "runs": rows,
    }
    write_once(out / "run.json", run)
    compact = compact_run(run, protocol)
    compact["run_sha256"] = digest(out / "run.json")
    write_once(out / "compact.json", compact)
    print(
        "Completed 24 consumed-authored development slots; semantic quality unmeasured"
    )
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as error:
        print("v3 runner failed: " + type(error).__name__, file=sys.stderr)
        raise SystemExit(1) from None
