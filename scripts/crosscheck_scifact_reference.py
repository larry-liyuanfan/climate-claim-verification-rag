"""Compare synthetic fixtures ONLY with pinned original SciFact scoring code.

Download data.py/metrics.py from the documented official SHA into an ignored
reference directory. Pandas2.2.3 is required only by that original scorer.
This script never reads a real train/dev/test example or model prediction.
"""

from __future__ import annotations

import argparse
import contextlib
import importlib.util
import io
import json
import random
import sys
import types
import warnings
from pathlib import Path
from types import SimpleNamespace

from climate_rag.public_v2 import file_sha256
from climate_rag.scifact_grounding import OFFICIAL_CODE_SHA, GoldClaim, Rationale
from climate_rag.scifact_scoring import (
    ClaimPrediction,
    PredictedAbstract,
    score_original,
)

HASHES = {
    "data.py": "b413aedc068d980520e9a54870591738d9833ca6e24614742ad070b7edcc3ca5",
    "metrics.py": "ffda621f72cf3f1c557cc88be1e02fdee49b3966f5892f9f08db49f1ed29200b",
}


def load_reference(directory):
    # Verify both before executing either. No dynamic download/branch resolution.
    for name, expected in HASHES.items():
        if file_sha256(directory / name) != expected:
            raise ValueError("official scorer file SHA mismatch")
    package = types.ModuleType("_pinned_scifact")
    package.__path__ = [str(directory)]
    sys.modules[package.__name__] = package
    result = {}
    for name in ("data", "metrics"):
        spec = importlib.util.spec_from_file_location(
            f"_pinned_scifact.{name}", directory / f"{name}.py"
        )
        module = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = module
        spec.loader.exec_module(module)
        result[name] = module
    return result


def official_score(reference, gold, predictions):
    data = reference["data"]
    ref_gold = {
        g.claim_id: SimpleNamespace(
            evidence={
                d: data.EvidenceAbstract(
                    d, data.make_label(rats[0].label), [list(r.sentences) for r in rats]
                )
                for d, rats in g.evidence.items()
            }
        )
        for g in gold
    }

    class Predictions(list):
        pass

    ref_predictions = Predictions(
        [
            data.ClaimPredictions(
                p.claim_id,
                {
                    d: data.PredictedAbstract(
                        d, data.make_label(r.label), list(r.sentences)
                    )
                    for d, r in p.evidence.items()
                },
            )
            for p in predictions
        ]
    )
    ref_predictions.gold = SimpleNamespace(get_claim=ref_gold.__getitem__)
    with warnings.catch_warnings(), contextlib.redirect_stdout(io.StringIO()):
        warnings.simplefilter("ignore")
        return reference["metrics"].compute_metrics(ref_predictions).to_dict()


def fixtures():
    yield (
        [
            GoldClaim(
                52,
                "synthetic official-shape fixture",
                {
                    11: (Rationale("SUPPORT", (0, 1)), Rationale("SUPPORT", (11,))),
                    15: (Rationale("SUPPORT", (4,)),),
                },
                (),
            )
        ],
        [
            ClaimPrediction(
                52,
                {
                    11: PredictedAbstract("SUPPORT", (1, 11, 13)),
                    16: PredictedAbstract("CONTRADICT", (18, 20)),
                },
            )
        ],
    )
    rng = random.Random(20260930)
    for _ in range(128):
        gold, predictions = [], []
        for claim_id in range(4):
            gold_evidence, pred_evidence = {}, {}
            for doc in range(3):
                if rng.random() < 0.7:
                    label = rng.choice(("SUPPORT", "CONTRADICT"))
                    indices = rng.sample(range(8), rng.randint(1, 5))
                    gold_evidence[doc] = tuple(
                        Rationale(label, tuple(indices[i : i + 2]))
                        for i in range(0, len(indices), 2)
                    )
                if rng.random() < 0.7:
                    pred_evidence[doc] = PredictedAbstract(
                        rng.choice(("SUPPORT", "CONTRADICT")),
                        tuple(rng.choices(range(8), k=rng.randint(0, 8))),
                    )
            gold.append(GoldClaim(claim_id, "synthetic only", gold_evidence, ()))
            predictions.append(ClaimPrediction(claim_id, pred_evidence))
        yield gold, predictions


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference-dir", type=Path, required=True)
    args = parser.parse_args()
    reference = load_reference(args.reference_dir)
    matrices = 0
    for gold, predictions in fixtures():
        actual = score_original(gold, predictions)["metrics"]
        expected = official_score(reference, gold, predictions)
        for name, metrics in expected.items():
            for key, value in metrics.items():
                if abs(value - actual[name][key]) > 1e-12:
                    raise AssertionError(
                        "fixture metric differs from official reference"
                    )
        matrices += 1
    print(
        json.dumps(
            {
                "status": "passed",
                "fixture_matrices": matrices,
                "official_reference_sha": OFFICIAL_CODE_SHA,
                "reference_file_sha256": HASHES,
                "real_data_or_predictions_read": False,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
