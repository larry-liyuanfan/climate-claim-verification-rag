"""CPU pilot or explicitly opted-in local-model run. Never opens benchmark gold."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import subprocess
from pathlib import Path
from typing import Any

from langsmith import tracing_context

from climate_rag.bm25 import BM25Index
from climate_rag.budget_agent import (
    AgentBudget, BudgetedEvidenceAgent, DecisionProvider, HeuristicAbstainingProvider,
)
from climate_rag.dense import DenseRetriever
from climate_rag.io import iter_evidence, write_json
from climate_rag.langchain_evidence import ClimateEvidenceRetriever, create_evidence_tool
from climate_rag.local_agent_model import (
    LocalQwenDecisionProvider, agent_prompt_identity, preflight_local_dependencies, verify_model_files,
)
from climate_rag.pipeline import HybridRetriever
from climate_rag.rerank import DeterministicFeatureReranker, Qwen3CausalLMReranker, Reranker


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def code_identity() -> tuple[str, bool]:
    archive_revision = (Path(__file__).resolve().parents[1] / "SOURCE_REVISION").read_text().strip()
    if re.fullmatch(r"[0-9a-f]{40}", archive_revision):
        return archive_revision, False  # git archive export-subst; archive hash checked by launcher
    commit = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    dirty = bool(subprocess.check_output(
        ["git", "status", "--porcelain", "--untracked-files=normal"], text=True).strip())
    return commit, dirty


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--evidence", type=Path, required=True)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--expected-protocol-sha256", required=True)
    parser.add_argument("--phase", choices=["pilot", "vnext", "validation"], default="pilot")
    parser.add_argument("--provider", choices=["heuristic", "local-qwen"], default="heuristic")
    parser.add_argument("--model-dir", type=Path)
    parser.add_argument("--model-manifest", type=Path)
    parser.add_argument("--dense-index", type=Path)
    parser.add_argument("--reranker-dir", type=Path)
    parser.add_argument("--reranker-manifest", type=Path)
    parser.add_argument("--execution-manifest", type=Path)
    parser.add_argument("--expected-execution-sha256")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--preflight-only", action="store_true")
    args = parser.parse_args()
    if sha(args.protocol) != args.expected_protocol_sha256:
        raise ValueError("protocol changed after freeze")
    protocol = json.loads(args.protocol.read_text(encoding="utf-8"))
    if sha(args.evidence) != protocol["corpus_sha256"]:
        raise ValueError("corpus mismatch")
    if args.phase == "vnext" and args.provider != "local-qwen":
        raise ValueError("vNext is reserved for released real-model execution")
    if args.provider == "heuristic" and (args.dense_index or args.reranker_dir):
        raise ValueError("CPU control may not implicitly load GPU components")
    execution = None
    if args.provider == "local-qwen":
        if not args.model_dir or not args.model_manifest:
            raise ValueError("local generation requires model and complete SHA manifest")
        if not args.execution_manifest or not args.expected_execution_sha256:
            raise ValueError("local-model execution requires a frozen execution manifest")
        if sha(args.execution_manifest) != args.expected_execution_sha256:
            raise ValueError("execution manifest changed after freeze")
        execution = json.loads(args.execution_manifest.read_text())
        expected_inputs = {
            "protocol_sha256": sha(args.protocol),
            "model_manifest_sha256": sha(args.model_manifest) if args.model_manifest else None,
            "reranker_manifest_sha256": sha(args.reranker_manifest) if args.reranker_manifest else None,
            "dense_file_hashes": {p.name: sha(p) for p in args.dense_index.iterdir() if p.is_file()}
            if args.dense_index else None,
        }
        if execution != expected_inputs:
            raise ValueError("model/retrieval inputs differ from frozen execution manifest")
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"
    budget = AgentBudget.model_validate(protocol["budget"])
    if args.preflight_only:
        dependency_report = None
        if args.provider == "local-qwen":
            verify_model_files(args.model_dir, json.loads(args.model_manifest.read_text()))
            model_dirs = [args.model_dir]
            if args.reranker_dir:
                if not args.reranker_manifest:
                    raise ValueError("reranker requires full model SHA manifest")
                verify_model_files(args.reranker_dir, json.loads(args.reranker_manifest.read_text()))
                model_dirs.append(args.reranker_dir)
            dependency_report = preflight_local_dependencies(model_dirs)
        report = {"preflight": "passed", "model_generation": False,
                  "protocol_sha256": sha(args.protocol), "execution_manifest": execution,
                  "dependencies": dependency_report}
        if args.output.exists():
            raise ValueError("preflight output already exists")
        write_json(args.output, report)
        print(json.dumps(report))
        return 0
    # A consumed output path is never silently reused after an interrupted run.
    args.output.parent.mkdir(parents=True, exist_ok=True)
    receipt = args.output.with_suffix(".consumed.json")
    with receipt.open("x", encoding="utf-8") as stream:
        json.dump({"protocol_sha256": sha(args.protocol), "phase": args.phase,
                   "provider": args.provider}, stream)
    if args.output.exists():
        raise ValueError("output already exists")
    provider: DecisionProvider = HeuristicAbstainingProvider()
    model_hash = reranker_hash = None
    if args.provider == "local-qwen":
        if not args.model_dir or not args.model_manifest:
            raise ValueError("local generation requires model and complete SHA manifest")
        manifest = json.loads(args.model_manifest.read_text())
        private_path = os.environ.get("CLIMATE_PRIVATE_RESPONSE_DIR")
        if private_path:
            private_dir = Path(private_path)
            private_root = Path("/data/gpfs/projects/punim2936/portfolio_20260903/climate-public-retrieval-v2/runs")
            if not private_dir.is_absolute() or not private_dir.resolve().is_relative_to(private_root):
                raise ValueError("private model responses must stay in Spartan Climate runs")
        provider = LocalQwenDecisionProvider(
            args.model_dir, manifest,
            private_response_dir=Path(private_path) if private_path else None,
        )
        model_hash = provider.model_sha256
    reranker: Reranker = DeterministicFeatureReranker()
    if args.reranker_dir:
        if not args.reranker_manifest:
            raise ValueError("reranker requires full model SHA manifest")
        reranker_hash = verify_model_files(args.reranker_dir, json.loads(
            args.reranker_manifest.read_text()))
        reranker = Qwen3CausalLMReranker(str(args.reranker_dir), device="cuda",
                                       dtype="bfloat16", max_length=2048, batch_size=1)
    documents = {x.evidence_id: x for x in iter_evidence(args.evidence)}
    dense = DenseRetriever.load(args.dense_index, device="cuda") if args.dense_index else None
    dense_hashes = None
    if dense is not None:
        if dict(zip(dense.doc_ids, dense.texts, strict=True)) != {
            key: row.text for key, row in documents.items()
        } or len(dense.doc_ids) != len(documents):
            raise ValueError("dense corpus identity differs from lexical corpus")
        dense_hashes = {path.name: sha(path) for path in args.dense_index.iterdir()
                        if path.is_file()}
    retriever = ClimateEvidenceRetriever(
        pipeline=HybridRetriever(bm25=BM25Index().fit(documents.values()), dense=dense),
        documents=documents, corpus_sha256=sha(args.evidence),
        corpus_reference="CLIMATE-FEVER evidence-only export",
        evidence_track=protocol["study_kind"],
        recall_k=100, rerank_k=budget.candidate_k, final_k=budget.candidate_k,
    )
    tool = create_evidence_tool(retriever)
    agent = BudgetedEvidenceAgent(lambda query: tool.invoke({"claim_text": query}),
                                 provider, budget=budget, reranker=reranker)
    runs: list[dict[str, Any]] = []
    code_sha, dirty = code_identity()
    with tracing_context(enabled=False):
        for task in protocol[args.phase]:
            for strategy in protocol["strategies"]:
                if (budget.controller_protocol == "feedback-v2"
                        and isinstance(provider, LocalQwenDecisionProvider) and private_path):
                    # One bounded owner-only directory per row preserves <=5
                    # attempts without the historical 32-files-per-study cap.
                    row_key = hashlib.sha256(json.dumps([task["id"], strategy]).encode()).hexdigest()
                    row_private = Path(private_path) / row_key
                    row_private.mkdir(mode=0o700)
                    provider.private_response_dir = row_private
                runs.append({"task_id": task["id"], **agent.run(
                    {"claim_text": task["claim_text"]}, strategy=strategy)})
                if budget.controller_protocol == "feedback-v2":
                    write_json(args.output.with_suffix(".progress.json"), {
                        "status": "partial_not_resumable", "code_sha": code_sha,
                        "protocol_sha256": sha(args.protocol), "completed_slots": len(runs),
                        "expected_slots": len(protocol[args.phase]) * len(protocol["strategies"]), "runs": runs})
    write_json(args.output, {
        "code_sha": code_sha, "working_tree_dirty": dirty,
        "protocol_sha256": sha(args.protocol), "corpus_sha256": sha(args.evidence),
        "document_count": len(documents), "phase": args.phase,
        "study_kind": protocol["study_kind"], "model_sha256": model_hash,
        "reranker_sha256": reranker_hash, "reranker_name": reranker.name,
        "dense_enabled": dense is not None, "budget": budget.model_dump(),
        "dense_file_hashes": dense_hashes,
        "execution_manifest": execution,
        "prompt_identity": agent_prompt_identity(budget.controller_protocol)
        if args.provider == "local-qwen" else None,
        "neural_work_accounting": {
            "dense_query_attempts": sum(row["retrieval_calls"] for row in runs) if dense else 0,
            "reranker_pair_attempts": sum(row["rerank_candidate_pairs"] for row in runs)
            if args.reranker_dir else 0,
            "generation_token_scope": "only autoregressive decision/answer provider; not dense or reranker FLOPs",
        },
        "official_evidence_metrics": None, "human_semantic_evaluation": None,
        "boundary": "CPU heuristic run is not real-model quality. Authored tasks are not independent benchmark gold.",
        "runs": runs,
    })
    print(f"Completed {len(runs)} bounded {args.provider} runs ({args.phase})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
