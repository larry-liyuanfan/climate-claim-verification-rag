"""Only recruitment projection/rendering deltas; not new model evaluation."""

import json
from pathlib import Path

import pytest

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
