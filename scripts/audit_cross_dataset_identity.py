"""CPU identity-only audit of consumed material versus unchanged SciFact dev."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import tarfile
from dataclasses import asdict
from pathlib import Path

from climate_rag.cross_identity import (
    IdentityClaim,
    audit_identity,
    normalized_identity,
)
from climate_rag.io import iter_evidence

ARCHIVE_SHA = "bef810a9a3a4eb2f4a4a2684e1362b0c02b6a231c56202d9d1b1b7d478eab329"
SPLIT_SHA = "66bf9b2c0157505f504459e7b38285a2aeed0c14770f82c74d1f619a03551f16"
CORPUS_SHA = "c14315aee9feecbbbbc3b0c7101d978b52b47cee9a613336304bcaa736460c71"
RESTRICTED_DEV_SHA = "ea9976e861095c5856a13773d15cf149c9b961d4afaf36be2e8b22d2e1ec1fe4"
AUTHORED_SHA = "6ee90b8479192335fc543c3425c3fbbb89fea2d65ba9698fb54d86564bc3c3c2"
SCIFACT_MANIFEST_SHA = "3cd6bc1e1c299ece9195098a3853ebb05bb3401507d8b467c1596ba6afb6e138"
MEMBERS = {
    split: f"./data/climate-fever-v2/selection-only/{split}-claims.json"
    for split in ("train", "validation")
}


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def load_jsonl(path):
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def write_once(path, value):
    with path.open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(value, stream, ensure_ascii=False, sort_keys=True, indent=2)
        stream.write("\n")


def scifact_identity(row, inference_row):
    """Cited-document metadata only; gold evidence keys/labels/rationales ignored."""
    if row["claim"] != inference_row["claim"] or str(row["id"]) != str(
        inference_row["id"]
    ):
        raise ValueError("SciFact source text identity changed")
    ids = sorted({str(i) for i in row["cited_doc_ids"]})
    return IdentityClaim(
        "scifact_dev",
        str(row["id"]),
        inference_row["claim"],
        tuple("scifact_document:" + i for i in ids),
        tuple("scifact_doc:" + i for i in ids),
        "cited_doc_metadata_not_original_claim_paper_origin",
    )


def safe_public_members(archive_path, split):
    """Read only existing safe projections; never the merged claims/test member."""
    payloads, hashes = {}, {}
    with tarfile.open(archive_path) as archive:
        for partition, member in MEMBERS.items():
            info = archive.getmember(member)
            if not info.isfile() or info.size > 5 * 1024 * 1024:
                raise ValueError("unexpected allowlisted member")
            stream = archive.extractfile(info)
            if stream is None:
                raise ValueError("safe projection absent")
            raw = stream.read()
            rows = json.loads(raw)
            expected = set(map(str, split[partition]))
            if set(rows) != expected or expected & set(map(str, split["test"])):
                raise ValueError("safe projection identity mismatch")
            payloads[partition] = rows
            hashes[member] = hashlib.sha256(raw).hexdigest()
    return payloads, hashes


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in (
        "public-archive",
        "split-manifest",
        "public-evidence",
        "restricted-dev",
        "authored-protocol",
        "scifact-prepared",
        "output",
    ):
        parser.add_argument("--" + name, type=Path, required=True)
    args = parser.parse_args()
    inputs = {
        "safe_public_archive": (args.public_archive, ARCHIVE_SHA),
        "public_split_manifest": (args.split_manifest, SPLIT_SHA),
        "public_evidence": (args.public_evidence, CORPUS_SHA),
        "restricted_fixed_dev": (args.restricted_dev, RESTRICTED_DEV_SHA),
        "consumed_authored_protocol": (args.authored_protocol, AUTHORED_SHA),
    }
    for path, expected in inputs.values():
        if sha(path) != expected:
            raise ValueError("frozen audit input hash mismatch")
    split = load_json(args.split_manifest)["split"]
    if (len(split["train"]), len(split["validation"]), len(split["test"])) != (
        1075,
        230,
        230,
    ):
        raise ValueError("split denominator changed")
    public, member_hashes = safe_public_members(args.public_archive, split)
    documents = {}
    public_docs = {d.evidence_id: d for d in iter_evidence(args.public_evidence)}
    if len(public_docs) != 5240:
        raise ValueError("public corpus changed")
    for key, row in public_docs.items():
        documents["climate_public_doc:" + key] = (
            str(row.metadata.get("article", "")) + " " + row.text
        )
    claims = []
    for partition, rows in public.items():
        for identifier, row in rows.items():
            ids = set(map(str, row["annotated_candidate_ids"]))
            if not ids <= set(public_docs):
                raise ValueError("annotated evidence identity missing")
            source_keys = {"climate_public_evidence:" + i for i in ids}
            articles = {
                normalized_identity(str(public_docs[i].metadata.get("article", "")))
                for i in ids
            }
            source_keys.update(
                "climate_public_article:" + article for article in articles if article
            )
            claims.append(
                IdentityClaim(
                    "public_" + partition,
                    str(identifier),
                    row["claim_text"],
                    tuple(sorted(source_keys)),
                    tuple("climate_public_doc:" + i for i in sorted(ids)),
                    "annotation_associated_candidates_and_articles_not_original_source_metadata",
                )
            )
    restricted = load_json(args.restricted_dev)
    if len(restricted) != 154:
        raise ValueError("restricted dev denominator changed")
    for identifier, row in restricted.items():
        claims.append(
            IdentityClaim(
                "restricted_fixed_dev",
                str(identifier),
                row["claim_text"],
                (),
                (),
                "claim_only_source_unknown_gold_evidence_keys_not_used",
            )
        )
    authored = load_json(args.authored_protocol)
    author_rows = authored["pilot"] + authored["vnext"]
    if len({normalized_identity(r["claim_text"]) for r in author_rows}) != 11:
        raise ValueError("authored identities changed")
    for row in author_rows:
        claims.append(IdentityClaim("authored_consumed", row["id"], row["claim_text"]))
    prepared = args.scifact_prepared
    manifest_path = prepared / "preparation-manifest.json"
    if sha(manifest_path) != SCIFACT_MANIFEST_SHA:
        raise ValueError("frozen SciFact preparation manifest changed")
    manifest = load_json(manifest_path)
    safe_paths = (
        "inference/claims_dev.jsonl",
        "inference/corpus.jsonl",
        "gold/claims_dev.jsonl",
    )
    for relative in safe_paths:
        path = prepared / relative
        if sha(path) != manifest["output_file_sha256"][relative]:
            raise ValueError("SciFact preparation changed")
        inputs["scifact_" + relative] = (path, sha(path))
    inputs["scifact_preparation_manifest"] = (manifest_path, sha(manifest_path))
    inference_rows = load_jsonl(prepared / safe_paths[0])
    if len(inference_rows) != 300 or any(
        set(r) != {"id", "claim"} for r in inference_rows
    ):
        raise ValueError("SciFact inference view changed")
    by_id = {str(r["id"]): r for r in inference_rows}
    if len(by_id) != 300:
        raise ValueError("duplicate SciFact dev identity")
    # CPU identity projection uses cited_doc_ids, NOT annotated evidence keys.
    # The gold-containing file is parsed, but those fields never select any edge;
    # this file is NOT an inference input. Its access is disclosed in the manifest.
    source_rows = load_jsonl(prepared / safe_paths[2])
    if {str(r["id"]) for r in source_rows} != set(by_id) or len(source_rows) != 300:
        raise ValueError("SciFact source metadata differs from fixed dev")
    corpus = load_jsonl(prepared / safe_paths[1])
    if len(corpus) != 5183 or len({r["doc_id"] for r in corpus}) != 5183:
        raise ValueError("SciFact document count changed")
    for doc in corpus:
        documents["scifact_doc:" + str(doc["doc_id"])] = (
            doc["title"] + " " + " ".join(doc["abstract"])
        )
    for row in source_rows:
        identifier = str(row["id"])
        claims.append(scifact_identity(row, by_id[identifier]))
    summary, private = audit_identity(claims, documents)
    summary.update(
        input_sha256={name: digest for name, (_, digest) in inputs.items()},
        only_archive_members_read=member_hashes,
        old_test_claim_text_or_labels_read=False,
        merged_public_claims_opened=False,
        scifact_gold_containing_file_parsed=True,
        scifact_identity_fields_used=["id", "claim", "cited_doc_ids"],
        scifact_gold_evidence_keys_labels_rationales_used=False,
        restricted_gold_evidence_keys_used=False,
        public_source_links="annotation-associated candidate IDs and linked articles; not original-source metadata",
        labels_used_for_edges=False,
        labels_passed_to_model=False,
        source_git=subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True
        ).strip(),
        source_worktree_dirty=bool(
            subprocess.check_output(["git", "status", "--porcelain"], text=True).strip()
        ),
    )
    args.output.mkdir(parents=True, mode=0o700)  # unique output, never overwrite
    write_once(args.output / "private-affected-identities.json", private)
    for partition in ("train", "validation"):
        write_once(
            args.output / f"public-{partition}-identity-only.json",
            [
                {"id": key, "claim_text": row["claim_text"]}
                for key, row in sorted(public[partition].items())
            ],
        )
    # Source membership only; no restricted claim text or labels are copied out.
    write_once(
        args.output / "private-source-map.json",
        [{k: v for k, v in asdict(row).items() if k != "text"} for row in claims],
    )
    summary["private_artifact_sha256"] = {
        path.name: sha(path) for path in sorted(args.output.iterdir()) if path.is_file()
    }
    write_once(args.output / "summary.json", summary)
    print(json.dumps(summary, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
