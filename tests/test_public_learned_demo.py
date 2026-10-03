"""Identity refusal fixtures only; no learned model, gold or empirical gain."""
import json

import pytest

from demo_learned_public_search import run
from rerank_public_demo import run as rerank
from build_public_learned_demo import build_index


def test_demo_identity_fails_before_loading_model_or_faiss(tmp_path):
    corpus = tmp_path / "evidence.jsonl"
    corpus.write_text("SYNTHETIC different corpus")
    dense = tmp_path / "dense"
    dense.mkdir()
    (dense / "reconstruction.json").write_text(json.dumps({"corpus_sha256": "changed"}))
    with pytest.raises(ValueError, match="reconstruction_identity"):
        run(corpus, dense, "A hand-authored SYNTHETIC claim.")


def test_claim_refused_before_asset_or_model_access(tmp_path):
    for value in ("", "x" * 2001):
        with pytest.raises(ValueError, match="claim_length"):
            run(tmp_path / "missing", tmp_path / "missing", value)


def test_rerank_requires_bound_live_result_before_model_access(tmp_path):
    source = tmp_path / "input.json"
    source.write_text("SYNTHETIC no model")
    with pytest.raises(ValueError, match="live_search_input_identity"):
        rerank(source, "f" * 64, tmp_path / "absent", tmp_path / "absent")


@pytest.mark.parametrize("name", ["index.faiss", "dense_index.json", "reconstruction.json"])
def test_completed_or_partial_index_never_overwritten(tmp_path, name):
    target = tmp_path / name
    target.write_bytes(b"PRESERVE SYNTHETIC artifact")
    with pytest.raises(ValueError, match="artifact_already_exists"):
        build_index(tmp_path / "absent", tmp_path / "absent", tmp_path)
    assert target.read_bytes() == b"PRESERVE SYNTHETIC artifact"
