import importlib.util
import copy
import io
import json
import tarfile
from pathlib import Path

import pytest

from climate_rag.cross_identity import (
    IdentityClaim,
    audit_identity,
    normalized_identity,
    text_edges,
)


def load_driver():
    script = (
        Path(__file__).resolve().parents[1] / "scripts/audit_cross_dataset_identity.py"
    )
    spec = importlib.util.spec_from_file_location("identity_audit_fixture", script)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_source_identity_ignores_gold_labels_rationales_and_evidence_keys():
    module = load_driver()
    inference = {"id": 3, "claim": "a synthetic claim"}
    source = {
        **inference,
        "cited_doc_ids": [12],
        "evidence": {"4": [{"label": "SUPPORT", "sentences": [0]}]},
    }
    changed = copy.deepcopy(source)
    changed["evidence"] = {"999": [{"label": "CONTRADICT", "sentences": [7, 9]}]}
    first = module.scifact_identity(source, inference)
    second = module.scifact_identity(changed, inference)
    assert first == second
    assert first.sources == ("scifact_document:12",)
    documents = {"scifact_doc:12": "synthetic document"}
    assert audit_identity([first], documents) == audit_identity([second], documents)


def test_edge_order_does_not_depend_on_mapping_order():
    texts = {
        "c": "one two three four",
        "a": "one two three four five",
        "b": "ONE TWO THREE FOUR",
    }
    assert text_edges(texts, 0.8) == text_edges(
        dict(reversed(list(texts.items()))), 0.8
    )


def test_nfkc_casefold_exact_and_fixed_jaccard_keep_distinct_namespaces():
    assert normalized_identity("ＣＯ２   WARMING") == "co2 warming"
    edges = text_edges(
        {
            "a": "one two three four",
            "b": "ONE TWO THREE FOUR",
            "c": "one two three four five",
            "d": "unrelated",
        },
        0.8,
    )
    assert ("a", "b", "normalized_exact") in edges
    assert ("a", "c", "token_jaccard") in edges
    assert all("d" not in edge[:2] for edge in edges)
    with pytest.raises(ValueError):
        text_edges({"empty": "!!!"}, 0.8)


def test_equal_bare_document_ids_do_not_imply_shared_source():
    claims = [
        IdentityClaim(
            "public_train", "1", "ocean temperature", ("climate:1",), ("climate:1",)
        ),
        IdentityClaim(
            "scifact_dev", "1", "blood pressure", ("scifact:1",), ("scifact:1",)
        ),
    ]
    summary, private = audit_identity(
        claims, {"climate:1": "ocean temperature", "scifact:1": "blood pressure"}
    )
    assert summary["target_claims_connected_to_consumed"] == 0
    assert not private["affected_target_ids"]
    assert not summary["independence_certified"]


def test_document_variant_and_claim_bridge_form_cross_track_component():
    claims = [
        IdentityClaim("public_train", "1", "first claim", (), ("climate:1",)),
        IdentityClaim("scifact_dev", "2", "second claim", (), ("scifact:1",)),
        IdentityClaim("authored_consumed", "3", "SECOND CLAIM"),
    ]
    summary, private = audit_identity(
        claims, {"climate:1": "same document", "scifact:1": "same document"}
    )
    assert summary["target_claims_connected_to_consumed"] == 1
    assert summary["target_claims_connected_by_consumed_track"] == {
        "authored_consumed": 1,
        "public_train": 1,
    }
    assert summary["target_unchanged"] and len(claims) == 3
    assert private["affected_target_ids"] == ["scifact_dev:2"]
    assert "first claim" not in json.dumps(summary)


def test_missing_documents_and_duplicate_claim_keys_fail_closed():
    row = IdentityClaim("scifact_dev", "1", "claim", (), ("missing:1",))
    with pytest.raises(ValueError, match="missing"):
        audit_identity([row], {})
    with pytest.raises(ValueError, match="duplicate"):
        audit_identity([row, row], {})


def test_safe_projection_never_extracts_or_decodes_other_archive_members(
    tmp_path, monkeypatch
):
    script = (
        Path(__file__).resolve().parents[1] / "scripts/audit_cross_dataset_identity.py"
    )
    spec = importlib.util.spec_from_file_location("identity_audit_fixture", script)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    archive_path = tmp_path / "safe.tar"
    values = {
        module.MEMBERS["train"]: b'{"a":{"claim_text":"train fixture"}}',
        module.MEMBERS["validation"]: b'{"b":{"claim_text":"val fixture"}}',
        "retired-test.json": b"INVALID JSON MUST NEVER BE DECODED",
    }
    with tarfile.open(archive_path, "w") as archive:
        for name, data in values.items():
            info = tarfile.TarInfo(name)
            info.size = len(data)
            archive.addfile(info, io.BytesIO(data))
    read_names = []
    original = tarfile.TarFile.extractfile

    def spy(archive, member):
        read_names.append(member.name)
        return original(archive, member)

    monkeypatch.setattr(tarfile.TarFile, "extractfile", spy)
    result, _ = module.safe_public_members(
        archive_path, {"train": ["a"], "validation": ["b"], "test": ["c"]}
    )
    assert len(result["train"]) == 1
    assert set(read_names) == set(module.MEMBERS.values())
    with pytest.raises(ValueError):
        module.safe_public_members(
            archive_path, {"train": ["c"], "validation": ["b"], "test": ["a"]}
        )
