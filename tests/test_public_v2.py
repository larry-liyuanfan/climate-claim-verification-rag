from __future__ import annotations

import json
import zipfile
from pathlib import Path

import pytest

from climate_rag.io import read_json
from climate_rag.public_v2 import (
    complete_external_transfer,
    export_public_v2_splits,
    file_md5,
    load_public_v2_protocol,
    paired_promotion_decision,
    prepare_scifact_transfer,
    reserve_external_transfer,
    select_full_candidate,
    select_pilot_candidates,
    validate_public_v2_protocol,
)


def _repository() -> Path:
    return Path(__file__).parents[1]


def _comparison(recall_lower: float, secondary_delta: float = 0.01) -> dict:
    return {
        "paired_bootstrap": {
            "recall@5": {
                "lower": recall_lower,
                "mean_difference": recall_lower + 0.01,
            },
            "mrr@10": {"lower": 0.0, "mean_difference": secondary_delta},
            "ndcg@10": {"lower": 0.0, "mean_difference": secondary_delta},
            "evidence_f1": {"lower": 0.0, "mean_difference": secondary_delta},
        },
        "candidate": {
            "recall@5": 0.3 + recall_lower,
            "mrr@10": 0.4,
            "ndcg@10": 0.3,
            "evidence_f1": 0.2,
        },
        "adapter_sha256": "a" * 64,
    }


def test_repository_public_v2_protocol_is_fully_frozen() -> None:
    protocol = load_public_v2_protocol(
        _repository() / "configs" / "public_retrieval_v2.json"
    )
    policy = read_json(_repository() / "configs" / "public_evaluation_policy.json")
    assert isinstance(policy, dict)
    result = validate_public_v2_protocol(protocol, policy)
    assert result["adapter_count"] == 6
    assert result["max_full_candidates"] == 2
    assert result["external_transfer_budget"] == 1


def test_export_public_v2_splits_never_materialises_test_claims(tmp_path: Path) -> None:
    prepared = tmp_path / "prepared"
    prepared.mkdir()
    split = {
        "train": [str(index) for index in range(1_075)],
        "validation": [str(index) for index in range(1_075, 1_305)],
        "test": [str(index) for index in range(1_305, 1_535)],
    }
    claims = {
        str(index): {
            "claim_text": f"claim {index}",
            "claim_label": "SUPPORTS",
            "evidences": [f"e{index}"],
        }
        for index in range(1_535)
    }
    (prepared / "split_manifest.json").write_text(
        json.dumps({"split": split}), encoding="utf-8"
    )
    (prepared / "claims.json").write_text(json.dumps(claims), encoding="utf-8")
    target = tmp_path / "selection"
    exported = export_public_v2_splits(prepared, target)
    assert exported["train"]["claim_count"] == 1_075
    assert exported["validation"]["claim_count"] == 230
    assert not (target / "test-claims.json").exists()
    assert (target / "test-seal.json").exists()


def test_pilot_selects_at_most_two_and_preserves_negative_matrix() -> None:
    matrix = [{"id": f"c{index}"} for index in range(6)]
    metrics = {
        f"c{index}": _comparison(0.01 * index, -0.01 if index == 0 else 0.01)
        for index in range(6)
    }
    result = select_pilot_candidates(matrix, metrics, maximum=2)
    assert result["selected_for_full"] == ["c5", "c4"]
    assert len(result["negative_results_preserved"]) == 4


def test_full_promotion_requires_positive_recall_ci_and_no_secondary_regression() -> (
    None
):
    assert paired_promotion_decision(_comparison(0.001))["promotion_pass"]
    assert not paired_promotion_decision(_comparison(0.0))["promotion_pass"]
    assert not paired_promotion_decision(_comparison(0.001, -0.001))["promotion_pass"]
    selection = select_full_candidate(
        {"pass": _comparison(0.002), "fail": _comparison(-0.01)}
    )
    assert selection["selected_candidate_id"] == "pass"
    assert selection["selected_candidate_promoted"] is True


def test_scifact_adapter_reads_official_beir_layout_once(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    archive = tmp_path / "scifact.zip"
    with zipfile.ZipFile(archive, "w") as output:
        output.writestr(
            "scifact/corpus.jsonl",
            json.dumps({"_id": "d1", "title": "Paper", "text": "Evidence"}) + "\n",
        )
        output.writestr(
            "scifact/queries.jsonl",
            json.dumps({"_id": "q1", "text": "Scientific claim"}) + "\n",
        )
        output.writestr(
            "scifact/qrels/test.tsv", "query-id\tcorpus-id\tscore\nq1\td1\t1\n"
        )
    monkeypatch.setattr("climate_rag.public_v2.BEIR_SCIFACT_MD5", file_md5(archive))
    result = prepare_scifact_transfer(archive, tmp_path / "prepared")
    assert result["corpus_count"] == 1
    assert result["test_query_count"] == 1
    assert result["test_qrel_count"] == 1


def test_external_transfer_ledger_allows_only_same_config_infra_retry(
    tmp_path: Path,
) -> None:
    ledger = tmp_path / "external.json"
    digest = "b" * 64
    reserve_external_transfer(ledger, frozen_config_sha256=digest, attempt_id="job-1")
    reserve_external_transfer(ledger, frozen_config_sha256=digest, attempt_id="job-2")
    with pytest.raises(ValueError, match="another config"):
        reserve_external_transfer(
            ledger, frozen_config_sha256="c" * 64, attempt_id="job-3"
        )
    complete_external_transfer(ledger, attempt_id="job-2", metrics_sha256="d" * 64)
    with pytest.raises(ValueError, match="already completed"):
        reserve_external_transfer(
            ledger, frozen_config_sha256=digest, attempt_id="job-3"
        )
