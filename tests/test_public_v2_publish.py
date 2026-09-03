from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path
from typing import Any


def _repository() -> Path:
    return Path(__file__).parents[1]


def _write(path: Path, value: Any) -> Path:
    path.write_text(json.dumps(value), encoding="utf-8")
    return path


def test_publish_negative_closeout_without_external_transfer(tmp_path: Path) -> None:
    repository = _repository()
    protocol = json.loads(
        (repository / "configs" / "public_retrieval_v2.json").read_text(
            encoding="utf-8"
        )
    )
    prepare = _write(tmp_path / "prepare.json", {"climate_fever": {"claims": 1535}})
    base = _write(
        tmp_path / "base.json",
        {
            "selection_boundary": {"test_claims_loaded": 0},
            "bm25": {},
            "qwen3_embedding_0_6b_base": {},
            "dense_vs_bm25": {},
            "training": {},
        },
    )
    pilot = _write(
        tmp_path / "pilot.json",
        {
            "selected_for_full": [protocol["adapter_matrix"][2]["id"]],
            "all_pilot_results": [
                {
                    "id": row["id"],
                    "pilot": {"advance_eligible": False},
                    "metrics_sha256": "a" * 64,
                }
                for row in protocol["adapter_matrix"]
            ],
        },
    )
    full = _write(
        tmp_path / "full.json",
        {
            "selected_candidate_promoted": False,
            "selected_candidate_id": None,
            "external_transfer_status": "not-authorized",
            "diagnostic_attempt": {"adapter_integrity": "failed"},
        },
    )
    downstream = _write(
        tmp_path / "downstream.json",
        {
            "downstream_mode": "base-only-negative-closeout",
            "systems": {},
            "paired_bootstrap_vs_bm25": {},
            "diagnostic_slices_vs_bm25": {},
            "index": {},
            "timing": {},
            "resource": {"git_commit": "1" * 40},
            "lambdamart": {},
            "reranker": {},
            "selected_adapter": {
                "id": None,
                "promoted": False,
                "used_in_downstream": False,
            },
        },
    )
    sacct = _write(tmp_path / "sacct.json", [{"job_id": "30007124"}])
    output = tmp_path / "publish"
    subprocess.run(
        [
            sys.executable,
            str(repository / "scripts" / "public_v2_publish.py"),
            "--protocol",
            str(repository / "configs" / "public_retrieval_v2.json"),
            "--prepare-metrics",
            str(prepare),
            "--base-metrics",
            str(base),
            "--pilot-selection",
            str(pilot),
            "--full-selection",
            str(full),
            "--downstream-metrics",
            str(downstream),
            "--sacct-json",
            str(sacct),
            "--schema",
            str(repository / "schemas" / "public_v2_compact.schema.json"),
            "--output-dir",
            str(output),
        ],
        check=True,
        cwd=repository,
    )
    artifact = output / "public-v2-compact.json"
    compact = json.loads(artifact.read_text(encoding="utf-8"))
    assert compact["evidence_status"].endswith("negative-adapter-closeout")
    assert compact["external_transfer"]["completed_evaluations"] == 0
    assert compact["external_transfer"]["qrels_opened"] is False
    subprocess.run(
        [
            sys.executable,
            str(repository / "scripts" / "check_public_v2_artifact.py"),
            str(artifact),
            str(repository / "schemas" / "public_v2_compact.schema.json"),
        ],
        check=True,
        cwd=repository,
    )
