"""Freeze ONLY original FIT24 actual initial-frame identities, no gold/repacking."""
from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path
from typing import Any

from climate_rag.scifact_natural_contract import valid_physical_prefix
from climate_rag.scifact_semantic_contract import checked
from climate_rag.scifact_terminal import render_scifact_prompt, source_from_abstract
from climate_rag.scifact_utility_contract import identity
from prepare_scifact_natural import SELECTION_SHA
from run_scifact_grounding_train_operator import ROOT, require, sha

ORIGINAL = ROOT / "runs/scifact-natural-fit24-v1-20261001"
ORIGINAL_SOURCE = "723c6a8fc4f0956ba91497b278d5f8fe65dc9fee"
ORIGINAL_RELEASE = "3abdaa9ef67d8a3d026d1d9e6494b99d56670e7abcc6d1e4489ff6171fd24a3a"


def input_inventory(original: Path = ORIGINAL) -> dict[str, Any]:
    selection = json.loads(checked(original / "prepared/selection.json", SELECTION_SHA))
    ids = [r["id"] for r in selection["selected"]]
    require(len(ids) == len(set(ids)) == 24, "all_frozen_24_required")
    reservation = json.loads((original / "reserved.json").read_bytes())
    require(reservation["source_git"] == ORIGINAL_SOURCE and reservation["release_sha256"] == ORIGINAL_RELEASE,
            "original_execution_identity")
    files = {"reserved.json": sha(original / "reserved.json"),
             "prepared/selection.json": SELECTION_SHA}
    for i in ids:
        prefix_path = f"inference/{i}-A/prefix.json"
        prefix = json.loads((original / prefix_path).read_bytes())
        key = prefix["payload"]["attempt"]["diagnostics"]["physical_attempt_id"]
        for relative in (prefix_path, f"inference/{i}-A/initial-frame.json",
                         f"inference/ledger/{key}.reserved.json", f"inference/ledger/{key}.finished.json"):
            files[relative] = sha(original / relative)
    return {"original_source_git": ORIGINAL_SOURCE, "original_release_sha256": ORIGINAL_RELEASE,
            "selection_sha256": SELECTION_SHA, "ordered_claim_ids": ids, "files": files,
            "gold_read": False, "model_calls": 0}


def load_frames(claims: list[Any], corpus: Any, tokenizer: Any, expected: str,
                original: Path = ORIGINAL) -> list[Any]:
    inventory = input_inventory(original)
    require(identity(inventory) == expected and [c["id"] for c in claims] == inventory["ordered_claim_ids"],
            "frozen_actual_input_inventory")
    frames = []
    for claim in claims:
        slot = f"{claim['id']}-A"
        directory = original / "inference" / slot
        prefix = json.loads((directory / "prefix.json").read_bytes())
        outer = json.loads((directory / "initial-frame.json").read_bytes())
        valid_physical_prefix(prefix, original / "inference/ledger", directory / "private-responses", tokenizer, slot)
        p, frame = prefix["payload"], copy.deepcopy(outer["frame"])
        require(frame == p["frame"] and p["claim"] == claim["claim"] == frame["observation"]["immutable_claim"],
                "immutable_initial_frame")
        prompt = render_scifact_prompt(tokenizer, frame["observation"], frame["schema"])
        require(outer["rendered_prompt_sha256"] == identity(prompt)
                and outer["token_ids_sha256"] == identity(tokenizer.encode(prompt, add_special_tokens=False)),
                "actual_initial_prompt")
        visible = frame["visible"]
        require(frame["observation"]["current_citable"] == [
            {"sentence_id": s, "text": v["text"]} for s, v in visible.items()], "actual_visible_order")
        for sid, row in visible.items():
            alias, index = sid.rsplit(":", 1)
            source = source_from_abstract(corpus[int(row["source_id"])])
            require(frame["alias_to_source"][alias] == row["source_id"]
                    and type(row["sentence_index"]) is int and str(row["sentence_index"]) == index
                    and 0 <= row["sentence_index"] < len(source.sentences)
                    and row["text"] == source.sentences[row["sentence_index"]]
                    and row["text_sha256"] == hashlib.sha256(row["text"].encode()).hexdigest()
                    and row["source_text_sha256"] == source.text_sha256, "original_source_sentence_identity")
        aliases = {s.rsplit(":", 1)[0] for s in visible}
        frame["document_order"] = [d for d in p["candidates"] if d in aliases]
        frame["original_initial_frame_sha256"] = sha(directory / "initial-frame.json")
        frames.append(frame)
    return frames  # all24 authenticated before any generation
