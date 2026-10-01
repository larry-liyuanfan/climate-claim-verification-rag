"""Same official/strict scoring; new physical reference audit, three arms."""
from __future__ import annotations

from functools import partial
from pathlib import Path
from typing import Any

from climate_rag.scifact_evidence_commit import ARMS, PROTOCOL
from climate_rag.scifact_evidence_commit_runtime import audit_episode
from score_scifact_document_verifier import score_after_exit as score_original_entry
from run_scifact_grounding_train_operator import ROOT
import scifact_evidence_input as inputs


def score_after_exit(output: Path, load_tokenizer: Any, release: Any, root: Path = ROOT, *,
                     release_sha: str, report_directory: Path | None = None) -> dict[str, Any]:
    # Scope includes the exact release identity, not a client-supplied saved ref.
    from run_scifact_grounding_train_operator import sha
    import json
    reservation = json.loads((output / "reserved.json").read_bytes())
    proof = json.loads((output / "worker-exit.json").read_bytes())
    from climate_rag.scifact_natural_contract import require
    require(reservation["release_sha256"] == proof["release_sha256"] == release_sha,
            "trusted_release_scope_mismatch")
    protocol = release.get("protocol", PROTOCOL)
    return score_original_entry(output, load_tokenizer, release, root,
        protocol=protocol, arms=ARMS,
        report_directory=report_directory,
        input_adapter=inputs if inputs.prospective(release) else None,
        audit_fn=partial(audit_episode, run_identity=release_sha, protocol=protocol,
                         generation_contract=release.get("generation_contract")), baseline_arm="fixed_all",
        comparison_limits={
            "verify_is_prerequisite_not_spontaneous_demand": True,
            "fixed_top1_rule": "first frozen retrieval document, no gold selection",
            "call_caps": {"fixed_top1": 1, "fixed_all": 4, "adaptive": 5},
            "fixed_empty_positive_is_unresolved_not_NEI": True,
            "no_automatic_agent_gain": True,
            "exact_reservation_sha256": sha(output / "reserved.json"),
        })


def main() -> None:
    import argparse
    import json
    from climate_rag.scifact_semantic_contract import checked
    from run_scifact_evidence_commit_operator import validate_release, output_path
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--release", type=Path, required=True)
    parser.add_argument("--release-sha", required=True)
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--report-directory", type=Path,
                        help="New disjoint directory for CPU replay; original run remains read-only")
    args = parser.parse_args()
    release = json.loads(checked(args.release, args.release_sha))
    validate_release(release)
    def tokenizer() -> Any:
        from transformers import AutoTokenizer
        return AutoTokenizer.from_pretrained(args.model_dir, local_files_only=True)
    result = score_after_exit(output_path(release), tokenizer, release, release_sha=args.release_sha,
                              report_directory=args.report_directory)
    raise SystemExit(0 if result["status"] == "scored" else 2)


if __name__ == "__main__":
    main()
