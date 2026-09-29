"""Read original results only inside the allocated Spartan CPU audit; export no prose."""
from __future__ import annotations

import hashlib
import importlib.util
import json
import os
import subprocess
import sys
import tarfile
from collections import Counter
from pathlib import Path

ROOT = Path("/data/gpfs/projects/punim2936/portfolio_20260903/climate-public-retrieval-v2")
RUN_NAME = "budget-agent-protocol-confirm-72eaa90-20260929"
RESULT_SHA = "4934332558a2b7abddb7467279f819ca93a04e44738202dc51b64947134a842f"
SOURCE_SHA = "72eaa90567e3603a3b940edffbb21b684bf1d1ba"
SOURCE_TAR_SHA = "ad533a3b14a81ab658049d6261cfc22848ebd792f3333e9f635d327d034d4cb1"
SCORER_SHA = "208ff931badff270cbbc9593c8ab54f5c79aec9e"
INPUT_SHA = "563738f0be1f7bf7b99b8de20bcdeab552f7e893166dec260b8f7f1e7951c3c1"
PROTOCOL_SHA = "6ee90b8479192335fc543c3425c3fbbb89fea2d65ba9698fb54d86564bc3c3c2"
PROMPT_SHA = "6b0163a4ef9da624e35dffb51fff34913ccb8a90db395c8dfec68bbe409febaa"


def digest(path: Path) -> str:
    result = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            result.update(block)
    return result.hexdigest()


def member_bytes(archive: tarfile.TarFile, name: str) -> bytes:
    member = archive.getmember(name)
    assert member.isfile() and member.size <= 32 * 1024 * 1024
    stream = archive.extractfile(member)
    assert stream is not None
    return stream.read()


def compact_row(row: dict, outcome: str, reason_code: str) -> dict:
    events = row["events"]
    decisions = [x["decision"] for x in events if x["stage"] == "decision"]
    diagnostics = [x["generation_diagnostics"] for x in events if "generation_diagnostics" in x]
    for decision in decisions:
        assert decision["action"] in {"answer", "abstain", "rewrite", "rerank"}
    result = {key: row[key] for key in (
        "task_id", "strategy", "status", "usage", "usage_known", "tool_calls", "model_calls",
        "generation_calls", "retrieval_calls", "rerank_calls", "rerank_candidate_pairs", "elapsed_ms")}
    result.update(
        outcome=outcome, reason_code=reason_code,
        schema_valid_action_proposals=[x["action"] for x in decisions],
        final_answer_present=row["answer"] is not None,
        final_label=row["answer"]["label"] if row["answer"] else None,
        final_statement_count=len(row["answer"]["statements"]) if row["answer"] else 0,
        candidate_count=len(row["candidate_evidence_ids"]), context_count=len(row["context_evidence_ids"]),
        successful_model_tool_events=[x["stage"] for x in events
            if row["strategy"] == "adaptive" and x["stage"] in {"rerank", "rewrite_retrieve"}],
        stage_elapsed_ms=[{"stage": x["stage"], "elapsed_ms": x["elapsed_ms"]}
                          for x in events if "elapsed_ms" in x],
        diagnostics=[{key: x.get(key) for key in (
            "category", "output_sha256", "output_characters", "output_bytes", "output_tokens",
            "eos_observed", "reached_max_new_tokens", "generation_elapsed_ms", "private_attachment",
            "errors", "line", "column")}
            for x in diagnostics],
    )
    assert result["final_label"] in {None, "SUPPORTS", "REFUTES"}
    return result


def main() -> None:
    assert os.environ.get("SLURM_JOB_ID"), "allocated CPU job required"
    assert not os.environ.get("CUDA_VISIBLE_DEVICES")
    source = Path(os.environ["CLIMATE_SCORE_TREE"])
    assert (source / "SOURCE_REVISION").read_text().strip() == SCORER_SHA
    sys.path.insert(0, str(source / "src"))
    from pydantic import ValidationError

    from climate_rag.budget_agent import AgentDecision, check_answer
    from climate_rag.local_agent_model import agent_prompt_identity
    from climate_rag.model_diagnostics import schema_error_locations

    spec = importlib.util.spec_from_file_location("frozen_score", source / "scripts/score_budget_agent.py")
    assert spec is not None and spec.loader is not None
    scorer = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(scorer)
    assert scorer.scorer_identity()["git_sha"] == SCORER_SHA
    assert digest(ROOT / "envs" / f"budget-agent-protocol-source-{SOURCE_SHA}.tar") == SOURCE_TAR_SHA
    input_path = ROOT / "envs" / f"budget-agent-inputs-20260929-{INPUT_SHA}.tar"
    assert digest(input_path) == INPUT_SHA  # streaming, allocated CPU only; no model extraction
    result_path = ROOT / "runs" / (RUN_NAME + ".tar.gz")
    assert digest(result_path) == RESULT_SHA
    with tarfile.open(result_path) as archive:
        assert set(archive.getnames()) == {"result", "result/run.json", "result/run.consumed.json"}
        raw_run = member_bytes(archive, "result/run.json")
        consumed = json.loads(member_bytes(archive, "result/run.consumed.json"))
    run = json.loads(raw_run)
    assert run["code_sha"] == SOURCE_SHA and run["working_tree_dirty"] is False
    assert run["phase"] == consumed["phase"] == "pilot" and consumed["provider"] == "local-qwen"
    assert run["protocol_sha256"] == consumed["protocol_sha256"] == PROTOCOL_SHA
    assert run["prompt_identity"] == agent_prompt_identity()
    assert run["prompt_identity"]["system_prompt_sha256"] == PROMPT_SHA
    prior = json.loads((source / "docs/verified-runs/budget-agent-diagnostic-canary-31519179.json").read_text())
    assert run["execution_manifest"] == prior["execution_manifest"]
    assert run["model_sha256"] == "d1dd9783afdf4e0fbd21eee824834d71b86982f5a5d5f6f371fe07f2f76f3cf6"
    assert run["reranker_sha256"] == "de1d4ac39101816774439e68881e2308c5e5f1bd94d0b0dc4c492a56c2681052"
    assert not run["dense_enabled"] and run["official_evidence_metrics"] is None
    with tarfile.open(input_path) as archive:
        protocol = member_bytes(archive, "authored-protocol.json")
        evidence_raw = member_bytes(archive, "evidence.jsonl")
    assert hashlib.sha256(evidence_raw).hexdigest() == run["corpus_sha256"]
    assert run["corpus_sha256"] == prior["corpus_sha256"]
    evidence = {str(x.get("evidence_id", x.get("id"))): x for x in
                (json.loads(line) for line in evidence_raw.splitlines() if line.strip())}
    assert len(evidence) == run["document_count"] == 5240
    assert run["budget"] == json.loads(protocol)["budget"]
    score = scorer.score(run, protocol, phase="pilot", expected_protocol_sha256=PROTOCOL_SHA)
    assert score["matrix"]["actual_slots"] == 9
    rows = []
    for row in run["runs"]:
        kind = scorer.outcome(row)
        compact = compact_row(row, kind, scorer.public_reason(row, kind))
        for event in row["events"]:
            if event["stage"] == "decision":
                AgentDecision.model_validate(event["decision"])
        if row["answer"]:
            decision = AgentDecision.model_validate(row["answer"])
            context = [{"evidence_id": key, "text": evidence[key].get("text", evidence[key].get("evidence_text", ""))}
                       for key in row["context_evidence_ids"]]
            compact["independent_mechanical_errors"] = check_answer(decision, context)
            assert not compact["independent_mechanical_errors"]
        rows.append(compact)
    private = ROOT / "runs" / (RUN_NAME + "-private")
    attachments = private / "private-responses"
    assert attachments.stat().st_mode & 0o777 == 0o700
    raw_counts: Counter = Counter()
    raw_hashes: Counter = Counter()
    raw_diagnostics = []
    paths = list(attachments.glob("*.txt"))
    assert len(paths) <= 32
    for path in paths:
        assert path.stat().st_mode & 0o777 == 0o600 and path.stat().st_size <= 32768
        original = path.read_bytes()
        fingerprint = hashlib.sha256(original).hexdigest()
        raw_hashes[fingerprint] += 1
        diagnosis = {"output_sha256": fingerprint, "starts_markdown_fence": original.strip().startswith(b"```")}
        try:
            obj = json.loads(original.decode().strip())
            decision = AgentDecision.model_validate(obj)
        except json.JSONDecodeError as error:
            raw_counts["json_decode_failure"] += 1
            diagnosis.update(category="json_decode", line=error.lineno, column=error.colno)
        except ValidationError as error:
            raw_counts["schema_validation_failure"] += 1
            errors = error.errors(include_url=False, include_context=False, include_input=False)
            known_messages = {
                "Value error, query is only allowed for rewrite": "query_only_allowed_for_rewrite",
                "Value error, rewrite requires query": "rewrite_requires_query",
                "Value error, answer requires label and cited statements": "answer_requires_label_statements_sufficient",
                "Value error, only answer may contain statements/label": "non_answer_contains_statements_or_label",
            }
            diagnosis.update(category="schema_validation", errors=schema_error_locations(errors),
                             cross_field_codes=[known_messages[x["msg"]] for x in errors if x["msg"] in known_messages])
        else:
            raw_counts["schema_valid_" + decision.action] += 1
            diagnosis.update(category="validated", action=decision.action)
        raw_diagnostics.append(diagnosis)
    observed = Counter(x["output_sha256"] for row in rows for x in row["diagnostics"])
    assert raw_hashes == observed
    operator = json.loads((private / "operator-status.json").read_text())
    assert operator["job_id"] == "31520350" and operator["exit_status"] == 0
    slurm = subprocess.check_output([
        "sacct", "-j", "31520350", "--parsable2", "--noheader",
        "--format=JobID,State,ExitCode,ElapsedRaw,TotalCPU,MaxRSS,Start,End,AllocTRES%100"], text=True)
    report = {
        "job_id": "31520350", "audit_cpu_job_id": os.environ["SLURM_JOB_ID"],
        "kind": "protocol confirmation only; no official quality/semantic evaluation",
        "inference_source_git_sha": SOURCE_SHA, "scorer_identity": scorer.scorer_identity(),
        "audit_script_sha256": digest(Path(__file__)),
        "source_archive_sha256": SOURCE_TAR_SHA, "inputs_archive_sha256_verified": INPUT_SHA,
        "result_archive_sha256": RESULT_SHA, "result_archive_bytes": result_path.stat().st_size,
        "run_sha256": hashlib.sha256(raw_run).hexdigest(), "run_bytes": len(raw_run),
        "prompt_identity": run["prompt_identity"], "execution_manifest": run["execution_manifest"],
        "corpus_sha256": run["corpus_sha256"], "documents": len(evidence), "budget": run["budget"],
        "model_sha256": run["model_sha256"], "reranker_sha256": run["reranker_sha256"],
        "compact_score": score, "rows": rows, "private_original_validation": dict(raw_counts),
        "private_original_safe_diagnostics": raw_diagnostics,
        "private_raw_hash_multiset_matches_events": True, "private_originals_exported": False,
        "slurm_rows": slurm.strip().splitlines(), "operator": operator,
        "gpu_memory_peak": None, "semantic_supportability": None,
        "generation_attempts": sum(x["generation_calls"] for x in rows),
        "query_elapsed_ms_sum": sum(x["elapsed_ms"] for x in rows),
        "generation_elapsed_ms_sum": sum(d["generation_elapsed_ms"] for x in rows for d in x["diagnostics"]),
    }
    destination = ROOT / "runs/budget-agent-protocol-confirm-31520350-audit"
    destination.mkdir(mode=0o700)
    with (destination / "compact.json").open("x") as stream:
        json.dump(report, stream, indent=2, sort_keys=True)
        stream.write("\n")
    print(json.dumps({"audit_job": os.environ["SLURM_JOB_ID"], "matrix": score["matrix"],
                      "private_original_validation": dict(raw_counts), "compact_sha256": digest(destination / "compact.json")}))


if __name__ == "__main__":
    try:
        main()
    except Exception as error:
        # Pydantic tracebacks could contain private input; print only exception type.
        print("Private audit failed: " + type(error).__name__, file=sys.stderr)
        raise SystemExit(1) from None
