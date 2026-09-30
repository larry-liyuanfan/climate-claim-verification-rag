"""Thin validation-only entrypoint; exact tune receipts do not authorize execution."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from climate_rag.scifact_grounding_sft import advancement
from run_scifact_grounding_train_operator import ROOT, require, sha
from run_scifact_grounding_tune_operator import (
    ADAPTER_SHA, CONFIG_SHA, DATA_SHA, TRAINING_SHA, run_pair,
)

GATE_SHA = '60bcc9ec34a1d5f33f6f665d3cf57d5c81c80ab8cdf2785f8a1b443680ae3d7a'
SCORE_SHA = '232255d657d35e546f3d64480724f02aa5914c528176b998c8f4465377119803'
TUNE = ROOT / 'runs/scifact-grounding-tune-20261001-v1'


def verify_frozen_gate(path: Path, release: dict[str, Any]) -> None:
    require(path.resolve() == TUNE / 'gate.json' and not path.is_symlink()
            and not (TUNE / 'score.json').is_symlink(), 'fixed_tune_gate_path')
    require(release['tune_gate_sha256'] == GATE_SHA == sha(path)
            and release['tune_score_sha256'] == SCORE_SHA == sha(TUNE / 'score.json'),
            'fixed_tune_gate_score_hashes')
    gate = json.loads(path.read_bytes())
    score = json.loads((TUNE / 'score.json').read_bytes())
    require(gate['passed'] is True and gate['score_sha256'] == SCORE_SHA
            and gate['next_validation_calls'] == 24
            and gate['data_manifest_sha256'] == release['data_manifest_sha256'] == DATA_SHA
            and gate['adapter_training_sha256'] == release['adapter_training_sha256'] == TRAINING_SHA
            and advancement(score['base'], score['adapted']), 'closed_validation_gate')


def fixed_validation_release(release: dict[str, Any]) -> None:
    require(release['authorization'] == 'coordinator_exact_hash_release'
            and release['purpose'] == 'evaluate_validation' and release['partition'] == 'validation'
            and release['max_runtime_seconds'] == 1500 and release['max_total_calls'] == 24
            and release['warmup_calls'] == 0 and release['calls_per_arm'] == 12
            and release['training_authorized'] is False and release['validation_authorized'] is True,
            'validation_only_release_required')
    require(release['adapter_training_sha256'] == TRAINING_SHA
            and release['adapter_model_sha256'] == ADAPTER_SHA
            and release['data_manifest_sha256'] == DATA_SHA and release['config_sha256'] == CONFIG_SHA
            and release['output'] == str(ROOT / 'runs/scifact-grounding-validation-20261001-v1'),
            'frozen_validation_identity')
    verify_frozen_gate(TUNE / 'gate.json', release)


def main() -> None:
    run_pair('validation', fixed_validation_release)


if __name__ == '__main__':
    main()
