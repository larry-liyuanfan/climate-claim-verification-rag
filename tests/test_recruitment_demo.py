"""Only recruitment projection/rendering deltas; not new model evaluation."""

import json
import hashlib
from pathlib import Path

import pytest
import demo_recruitment_case as demo

from demo_recruitment_case import (
    ROOT,
    build_demo,
    historical_summary,
    public_packets,
    render_html,
    render_text,
)


def test_replay_keeps_protocols_and_nonpromotion() -> None:
    data = historical_summary()
    assert data["training"]["queries"] == 154
    assert data["training"]["steps"] == 20
    assert data["search"]["selected_offline_candidate"] == "ltr"
    assert data["search"]["live_ltr_assets_available"] is False
    assert data["agent"]["stop"] == 32
    assert data["agent"]["acquire"] == 0


def test_projection_preserves_actual_metrics_and_uncertainty() -> None:
    data = historical_summary()
    profiles = data["search"]["profiles"]
    assert profiles[0]["recall_at_5"] == pytest.approx(0.6054232804232804)
    assert profiles[0]["p95_ms"] == pytest.approx(77.81600579619408)
    assert profiles[2]["p95_ms"] > 9000
    assert data["search"]["rerank100_minus_ltr_recall_ci"][0] < 0
    assert data["search"]["rerank100_minus_ltr_f1_ci"][1] > 0


def test_default_is_saved_public_packets_not_live_search() -> None:
    data = build_demo()
    assert data["public_retrieval"]["mode"].startswith("saved_redacted_public")
    assert len(data["public_retrieval"]["cases"]) == 3
    assert all(c["answer"] is None for c in data["public_retrieval"]["cases"])
    assert data["public_retrieval"]["cases"][-1]["items"] == []
    assert all(
        "text" not in item
        for case in data["public_retrieval"]["cases"]
        for item in case["items"]
    )


def test_wrong_corpus_rejected_before_retrieval(tmp_path: Path) -> None:
    corpus = tmp_path / "unregistered.jsonl"
    corpus.write_text('{"text":"not a public approved corpus"}', encoding="utf8")
    with pytest.raises(ValueError, match="registered public"):
        public_packets(corpus)


def test_render_escapes_public_text_and_does_not_generate_verdict() -> None:
    data = build_demo()
    data["public_retrieval"]["cases"][0]["claim_text"] = "<script>private</script>"
    rendered = render_html(data)
    assert "<script>private</script>" not in rendered
    assert "&lt;script&gt;private&lt;/script&gt;" in rendered
    assert "不是线上 SLA" in rendered
    assert "不是 CPU-only" in rendered
    assert "无 verdict" in rendered


def test_text_contains_scope_and_empty_result() -> None:
    rendered = render_text(build_demo())
    assert "NOT live dense/LTR/LLM" in rendered
    assert "Empty result" in rendered
    assert "No verdict generated" in rendered


def test_semantic_diagnosis_is_complete_but_not_new_model_accuracy() -> None:
    data = json.loads(
        (
            ROOT / "docs/verified-runs/recruitment-semantic-diagnosis-20261003.json"
        ).read_bytes()
    )
    assert data["task_count"] == 12 and data["output_count"] == 36
    assert sum(data["citation_assessment_counts"].values()) == 36
    assert sum(data["initial_packet_counts_unique_tasks"].values()) == 12
    assert sum(data["primary_issue_counts_outputs"].values()) == 36
    assert all(sum(counts.values()) == 12 for counts in data["route_counts"].values())
    assert data["official_scores_unchanged"] is True
    assert data["human_blind_annotation"] is False
    assert data["new_inference_or_official_rescoring"] is False
    assert data["gold_read"] is False
    assert "accuracy" not in data


def test_public_redaction_does_not_attach_hash_to_placeholder_text() -> None:
    for case in public_packets(None)["cases"]:
        for item in case["items"]:
            assert "text" not in item and "text_sha256" not in item
            assert len(item["original_text_sha256"]) == 64


@pytest.fixture
def synthetic_public_corpus(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Test-only corpus/pin override; never a real retrieval quality result."""
    rows = [
        {
            "evidence_id": f"e{index:04}",
            "text": "alpha" if index < 2 else "background",
            "metadata": {"source": "synthetic test-only"},
        }
        for index in range(5240)
    ]
    path = tmp_path / "test-only-corpus.jsonl"
    path.write_text("\n".join(json.dumps(row) for row in rows), encoding="utf8")
    monkeypatch.setattr(
        demo, "PUBLIC_CORPUS_SHA", hashlib.sha256(path.read_bytes()).hexdigest()
    )
    return path


def test_live_claim_returns_candidates_selected_evidence_and_timings(
    synthetic_public_corpus: Path,
) -> None:
    result = public_packets(
        synthetic_public_corpus, claim="  alpha  ", candidate_k=2, top_k=1
    )
    case = result["cases"][0]
    assert result["input_mode"] == "custom_claim"
    assert result["configuration"]["dense"] is False
    assert case["claim_text"] == "alpha" and case["answer"] is None
    assert len(case["candidates"]) == 2 and len(case["items"]) == 1
    assert case["items"][0]["evidence_id"] == "e0000"
    assert case["items"][0]["provenance"]["source"] == "synthetic test-only"
    assert all(value >= 0 for value in case["timings_ms"].values())
    assert case["timings_ms"]["request_total"] == pytest.approx(
        case["timings_ms"]["search_and_rank"] + case["timings_ms"]["evidence_packet"]
    )
    assert all(value >= 0 for value in result["setup_timings_ms"].values())
    assert "not P50/P95" in result["timing_scope"]
    full_demo = historical_summary()
    full_demo["public_retrieval"] = result
    assert "Candidate pool" in render_text(full_demo)
    assert "Current request timings" in render_text(full_demo)


def test_live_empty_claim_result_has_no_fabricated_evidence(
    synthetic_public_corpus: Path,
) -> None:
    case = public_packets(synthetic_public_corpus, claim="zzzznovel")["cases"][0]
    assert case["status"] == "empty_result"
    assert case["candidates"] == case["items"] == []
    assert case["answer"] is None and case["timings_ms"]["request_total"] >= 0


def test_new_claim_cannot_be_answered_by_saved_replay() -> None:
    with pytest.raises(ValueError, match="require --evidence"):
        public_packets(None, claim="new query")
    with pytest.raises(ValueError, match="require --evidence"):
        public_packets(None, top_k=2)


def test_cli_refuses_new_claim_in_replay_mode(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    monkeypatch.setattr(
        "sys.argv", ["demo_recruitment_case.py", "--claim", "new query"]
    )
    with pytest.raises(SystemExit) as exc:
        demo.main()
    assert exc.value.code == 2
    assert "require --evidence" in capsys.readouterr().err


@pytest.mark.parametrize("claim", ["  ", "x" * 2001])
def test_invalid_claim_rejected_before_asset_access(claim: str) -> None:
    with pytest.raises(ValueError, match="1 to 2000"):
        public_packets(Path("must-not-open.jsonl"), claim=claim)


@pytest.mark.parametrize("candidate_k,top_k", [(0, 1), (2, 3), (101, 1)])
def test_invalid_candidate_budget_rejected_before_asset_access(
    candidate_k: int,
    top_k: int,
) -> None:
    with pytest.raises(ValueError, match="top_k <= candidate_k"):
        public_packets(
            Path("must-not-open.jsonl"), candidate_k=candidate_k, top_k=top_k
        )
