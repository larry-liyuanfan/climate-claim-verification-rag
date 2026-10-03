"""Local CPU-only draft from an immutable tokenizer receipt; never a job submit.

GenerationConfig expansion belongs to the pinned local validation environment,
not the deliberately Torch-free Spartan tokenizer environment. No weight load.
"""
from __future__ import annotations

import argparse
import ast
import json
from pathlib import Path
from typing import Any

from climate_rag.scifact_evidence_bottleneck import PROTOCOL, ROUTES
from climate_rag.scifact_generation import frozen_contract
from climate_rag.scifact_natural_contract import require
from climate_rag.scifact_read_continuation import ordered_write
from climate_rag.scifact_relation_verifier import RELATION_PROTOCOL
from climate_rag.scifact_semantic_contract import MODEL_SHA, TOKENIZER_SHA, checked
from package_scifact_source import git, validate_archive
from run_scifact_evidence_commit_operator import release_fields
from run_scifact_grounding_train_operator import ROOT
import scifact_evidence_input as adapter


def same_probe_implementation(repo: Path, before: str, after: str) -> None:
    # Reuse one physical tokenizer pass only when model-facing source and the
    # executed probe body are byte/AST-identical. Report the two revisions, not
    # an invented cluster execution of the later draft-only repair.
    for name in ("src/climate_rag/scifact_evidence_bottleneck.py", "src/climate_rag/scifact_relation_verifier.py",
                 "src/climate_rag/scifact_generation.py", "src/climate_rag/scifact_evidence_commit.py",
                 "scripts/scifact_evidence_input.py", "src/climate_rag/scifact_prospective_inputs.py"):
        require(git(repo, "show", before+":"+name) == git(repo, "show", after+":"+name), "preflight_component_changed:"+name)
    name = "scripts/preflight_scifact_evidence_bottleneck.py"
    def probe_body(revision: str) -> str:
        tree = ast.parse(git(repo, "show", revision+":"+name))
        return ast.dump(next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "probe"))
    require(probe_body(before) == probe_body(after), "tokenizer_probe_changed")


def build(receipt: Any, fields: Any, contract: Any, revision: str, archive_sha: str,
          receipt_sha: str) -> dict[str, Any]:
    require(receipt["protocol"] == PROTOCOL and receipt["claims"] == 24 and receipt["overflow_count"] == 0
            and receipt["model_calls"] == 0 and receipt["gold_read"] is False
            and receipt["tokenizer_sha256"] == TOKENIZER_SHA
            and receipt["frames_sha256"] == fields["frames_sha256"]
            and receipt["selection_sha256"] == fields["selection_sha256"], "completed_tokenizer_receipt_required")
    stage = ROOT / "envs" / ("evidence-bottleneck-" + revision[:12])
    return {**{k: fields[k] for k in adapter.HASH_KEYS}, "protocol": PROTOCOL,
        "input_protocol": adapter.PROTOCOL, "scope": adapter.SCOPE, "status": "draft_not_authorized",
        "model_execution_authorized": False, "source_git": revision,
        "source_archive": (stage / "source.tar").as_posix(), "source_archive_sha256": archive_sha,
        "frames_identity": receipt["frames_identity"],
        "prepared": (ROOT / "runs/scifact-evidence-commit-prospective24-v1-20261002-confirmation-v1/prepared").as_posix(),
        "model_sha256": MODEL_SHA, "generation_contract": contract, "tokenizer_sha256": TOKENIZER_SHA,
        "model_directory": (stage / "future-model-input/models/generator/model").as_posix(),
        "model_directory_status": "not_extracted_this_CPU_package", "max_generations": 96,
        "routes": list(ROUTES), "output": (ROOT / "runs" / (PROTOCOL + "-" + revision[:12])).as_posix(),
        "input_token_cap": 8192, "output_token_cap": 512, "per_stage_seconds": 120, "planned_route_results": 72,
        "baseline": "archived v2 fixed_top1 and v3 fixed_top1, never rerun",
        "tokenizer_receipt_sha256": receipt_sha, "tokenizer_executed_source": receipt["source_git"],
        "tokenizer_job": receipt["job_id"], "tokenizer_job_status": "FAILED_tail_draft_export_after_complete_probe",
        "draft_recovery": "local_CPU_only_no_cluster_resubmission",
        "release_requires": "new coordinator exact-source approval plus bounded parent supervision/exit proof"}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--preflight", type=Path, required=True)
    parser.add_argument("--preflight-sha", required=True)
    parser.add_argument("--source-git", required=True)
    parser.add_argument("--archive", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    repo = Path(__file__).resolve().parents[1]
    receipt = json.loads(checked(args.preflight, args.preflight_sha))
    same_probe_implementation(repo, receipt["source_git"], args.source_git)
    archive = validate_archive(repo, args.source_git, args.archive)
    fields = release_fields({"protocol": RELATION_PROTOCOL, "input_protocol": adapter.PROTOCOL})
    result = build(receipt, fields, frozen_contract(), args.source_git, archive["source_archive_sha256"], args.preflight_sha)
    ordered_write(args.output, result)
    print(json.dumps({"status": result["status"], "source_git": result["source_git"],
                      "tokenizer_executed_source": result["tokenizer_executed_source"], "model_calls": 0}))


if __name__ == "__main__":
    main()
