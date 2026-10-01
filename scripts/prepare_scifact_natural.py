"""Build inference inputs from the ONCE-frozen metadata roster, never reselect."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import tarfile
from typing import Any

from climate_rag.scifact_natural_selection import PROTOCOL, digest
from climate_rag.scifact_read_continuation import ordered_write
from climate_rag.scifact_semantic_contract import CORPUS_SHA, checked
from prepare_scifact_state_supervision import member
from prepare_scifact_utility8 import ARCHIVE, ARCHIVE_SHA, CORPUS, MANIFEST_SHA, PREP, select_queries
from run_scifact_grounding_train_operator import ROOT, require, sha

SELECTION_SHA = "107c50a8ecc1fd4d1b352cdc7e2f86b2997d6abc027d5afeb176bab1d09f7acb"
SELECTOR_SHA = "eced2d4226c6b53aa4bbf56336c1255f1c1337c0426a72d57c36f0517930c44f"
QUERY_SHA = "003e0fb7fb6ac8b7f6a6af5949782801cf370d6019d710624b90c3531b8bd914"


def prepare(out: Path, root: Path = ROOT) -> dict[str, Any]:
    selection_bytes = checked(root / "posthoc" / PROTOCOL / "selection.json", SELECTION_SHA)
    selection = json.loads(selection_bytes)
    ids = [r["id"] for r in selection["selected"]]
    require(selection["protocol"] == PROTOCOL and len(ids) == len(set(ids)) == 24
            and len({r["component"] for r in selection["selected"]}) == 24
            and selection["ordered_ids_sha256"] == digest(ids), "frozen_natural_24")
    out.mkdir(mode=0o700)
    with (out / "selection.json").open("xb") as stream:
        stream.write(selection_bytes)  # exact bytes, not a new sample
    bundle = root / "envs" / ARCHIVE
    require(sha(bundle) == ARCHIVE_SHA, "original_archive_identity")
    with tarfile.open(bundle) as archive:
        with member(archive, PREP + "/preparation-manifest.json") as stream:
            raw = stream.read()
        require(hashlib.sha256(raw).hexdigest() == MANIFEST_SHA, "preparation_manifest_identity")
        require(json.loads(raw)["output_file_sha256"]["inference/claims_train_eligible.jsonl"] == QUERY_SHA,
                "query_member_binding")
        with member(archive, PREP + "/inference/claims_train_eligible.jsonl") as stream:
            claims = select_queries(stream, QUERY_SHA, ids)
    inference = out / "inference"
    inference.mkdir(mode=0o700)
    ordered_write(inference / "claims.json", claims)
    with (inference / "corpus.jsonl").open("xb") as stream:
        stream.write(checked(root / CORPUS, CORPUS_SHA))
    receipt = {"protocol": PROTOCOL, "claims_sha256": sha(inference / "claims.json"),
        "corpus_sha256": CORPUS_SHA, "selection_sha256": SELECTION_SHA, "selector_source_sha256": SELECTOR_SHA,
        "query_member_sha256": QUERY_SHA, "ordered_ids_sha256": digest(ids),
        "official_gold_decoded": False, "protected_split_read": False, "reselection": False}
    ordered_write(out / "preparation.json", receipt)
    return receipt
