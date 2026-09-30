"""Freeze a hash-selected OLD validation replay; read no retired test member."""

import argparse
import hashlib
import json
import tarfile
from pathlib import Path
from typing import Any

from climate_rag.verification import normalise_claim

ARCHIVE_SHA = "bef810a9a3a4eb2f4a4a2684e1362b0c02b6a231c56202d9d1b1b7d478eab329"
MEMBER = "./data/climate-fever-v2/selection-only/validation-claims.json"


def write_json(path: Path, value: Any) -> None:
    # Frozen digests are byte-identical across Windows and Spartan.
    path.write_bytes((json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n").encode())


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--preflight-archive", type=Path, required=True)
    parser.add_argument("--base-protocol", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    if hashlib.sha256(args.preflight_archive.read_bytes()).hexdigest() != ARCHIVE_SHA:
        raise ValueError("preflight archive SHA mismatch")
    # extractfile on this single explicitly named safe member, never extractall.
    with tarfile.open(args.preflight_archive) as archive:
        stream = archive.extractfile(MEMBER)
        if stream is None:
            raise ValueError("validation-only member absent")
        raw = stream.read()
    claims = json.loads(raw)
    decisive = [key for key, row in claims.items() if row["evidences"]]
    nondecisive = [key for key, row in claims.items() if not row["evidences"]]
    if (len(claims), len(decisive), len(nondecisive)) != (230, 126, 104):
        raise ValueError("unexpected validation denominator")

    def order(key: str) -> str:
        return hashlib.sha256(("agent-validation-20260929:" + key).encode()).hexdigest()

    selected = sorted(decisive, key=order)[:24] + sorted(nondecisive, key=order)[:8]
    protocol = json.loads(args.base_protocol.read_text())
    protocol.update({
        "study_kind": "public_v2_repeatedly_used_validation_replay_not_independent_test",
        "source": "Existing CLIMATE-FEVER v2 validation-only export",
        "exposure": "Validation used for previous model selection; no new generalization claim.",
        "official_gold_available": True,
        "validation": [{"id": key, "claim_text": claims[key]["claim_text"]} for key in selected],
    })
    protocol.pop("pilot")
    protocol.pop("vnext")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    protocol_path = args.output_dir / "validation-protocol.json"
    write_json(protocol_path, protocol)
    protocol_sha = hashlib.sha256(protocol_path.read_bytes()).hexdigest()
    gold = {
        "protocol_sha256": protocol_sha,
        "provenance": {"annotation_kind": "official_CLIMATE_FEVER_evidence_and_label",
                       "split": "repeated_validation_replay", "archive_sha256": ARCHIVE_SHA,
                       "member_sha256": hashlib.sha256(raw).hexdigest()},
        "claims": {key: {
            "claim_sha256": hashlib.sha256(normalise_claim(claims[key]["claim_text"]).encode()).hexdigest(),
            "evidence_ids": claims[key]["evidences"], "label": claims[key]["claim_label"],
        } for key in selected},
    }
    write_json(args.output_dir / "validation-gold.json", gold)
    write_json(args.output_dir / "selection-manifest.json", {
        "source_archive_sha256": ARCHIVE_SHA, "only_member_read": MEMBER,
        "member_sha256": hashlib.sha256(raw).hexdigest(),
        "protocol_sha256": protocol_sha,
        "gold_sha256": hashlib.sha256((args.output_dir / "validation-gold.json").read_bytes()).hexdigest(),
        "selection": "first SHA256(agent-validation-20260929:claim_id), without reading model outcomes",
        "selected_ids": selected, "selected_decisive": 24, "selected_nondecisive": 8,
        "available_decisive": 126, "available_nondecisive": 104,
        "retrieval_denominator": 24, "call_latency_denominator_per_strategy": 32,
        "boundary": "No retired test read; no independent test; nondecisive excluded from retrieval averages.",
    })
    print(json.dumps({"selected": 32, "decisive": 24, "protocol_sha256": protocol_sha}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
