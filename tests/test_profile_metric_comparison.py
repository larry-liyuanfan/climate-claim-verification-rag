from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

from climate_rag.io import write_json, write_jsonl


def load_script(monkeypatch):
    scripts = Path(__file__).parents[1] / "scripts"
    monkeypatch.syspath_prepend(str(scripts))
    spec = importlib.util.spec_from_file_location("profile_metric_comparison", scripts / "compare_profile_metrics.py")
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def make_bundle(root, module):
    for name, value in (("rerank20", 0.2), ("rerank100", 0.4)):
        directory = root / name
        directory.mkdir()
        write_jsonl(directory / "query-metrics.jsonl", [
            {"claim_id": qid, **dict.fromkeys(module.METRICS, value)} for qid in ("a", "b")
        ])
        write_json(directory / "summary.json", {
            "query_count": 2, "metrics": dict.fromkeys(module.METRICS, value),
            **dict.fromkeys(("source_hashes", "query_ids_sha256", "git_commit", "model_revisions", "device"), "same"),
        })
    from public_v2_stage_manifest import payload_summary
    write_json(root / "stage-manifest.json", payload_summary(root))


def test_saved_profile_pair_is_direct_and_manifest_checked(tmp_path, monkeypatch):
    module = load_script(monkeypatch)
    make_bundle(tmp_path, module)
    result = module.compare_saved_profiles(tmp_path, "rerank20", "rerank100")
    assert result["bootstrap_samples"] == 5000
    assert result["paired_bootstrap"]["recall@5"]["mean_difference"] == pytest.approx(0.2)
    write_jsonl(tmp_path / "rerank20" / "query-metrics.jsonl", [{"claim_id": "wrong"}])
    with pytest.raises(ValueError, match="archive"):
        module.compare_saved_profiles(tmp_path, "rerank20", "rerank100")


def test_output_cannot_modify_immutable_bundle(tmp_path, monkeypatch):
    module = load_script(monkeypatch)
    monkeypatch.setattr(sys, "argv", ["compare", "--run-dir", str(tmp_path), "--output", str(tmp_path / "new.json")])
    with pytest.raises(ValueError, match="immutable"):
        module.main()
