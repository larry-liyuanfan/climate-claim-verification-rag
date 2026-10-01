"""Thin prospective input adapter; old policy, model and scoring stay frozen."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from climate_rag.scifact_natural_contract import require
from climate_rag.scifact_prospective_inputs import PROTOCOL as PROTOCOL, SCOPE as SCOPE, reserved_selection, validate_frames
from climate_rag.scifact_semantic_contract import CORPUS_SHA, checked, sha
from climate_rag.scifact_utility_contract import identity
from run_scifact_grounding_train_operator import ROOT

PREPARED = ROOT / 'posthoc' / PROTOCOL
OUTPUT = ROOT / 'runs' / (PROTOCOL + '-confirmation-v1')
HASH_KEYS = ('selection_sha256', 'component_reservations_sha256', 'preparation_sha256',
             'claims_sha256', 'frames_sha256', 'ordered_ids_sha256')


def prospective(release: Any) -> bool:
    if 'input_protocol' not in release:
        return False
    require(release['input_protocol'] == PROTOCOL, 'unknown_input_protocol')
    return True


def frozen_fields(original: Any) -> dict[str, Any]:
    fields = dict(original)
    for key in ('selection_sha256', 'initial_inventory_sha256'):
        del fields[key]
    fields.update(input_protocol=PROTOCOL, input_scope=SCOPE, prepared=PREPARED.as_posix(),
                  output=OUTPUT.as_posix(), attempt_id='prospective24-confirmation-v1',
                  comparison_job_id='31944832',
                  comparison_quality_sha256=None,
                  decision_policy_git='2ab219bb13564790b004797354dcc89554da2bf6')
    return fields


def validate_hashes(release: Any) -> None:
    require('initial_inventory_sha256' not in release, 'prospective_has_no_old_frame_inventory')
    for key in HASH_KEYS:
        value = release.get(key)
        require(isinstance(value, str) and len(value) == 64 and set(value) <= set('0123456789abcdef'),
                'prospective_input_hash:' + key)


def check_prepared(prepared: Path, release: Any) -> tuple[Any, Any]:
    validate_hashes(release)
    selection = reserved_selection(prepared, release['selection_sha256'], release['component_reservations_sha256'])
    receipt = json.loads(checked(prepared/'preparation.json', release['preparation_sha256']))
    require(receipt['protocol'] == PROTOCOL and receipt['scope'] == SCOPE
            and receipt['model_calls'] == 0 and receipt['official_gold_decoded'] is False
            and receipt['protected_split_read'] is False and receipt['reselection'] is False,
            'prospective_preparation_contract')
    for key in HASH_KEYS:
        if key != 'preparation_sha256':
            require(receipt[key] == release[key], 'preparation_release_binding:' + key)
    claims = json.loads(checked(prepared/'inference/claims.json', release['claims_sha256']))
    require([c['id'] for c in claims] == [r['id'] for r in selection['selected']]
            and identity([c['id'] for c in claims]) == release['ordered_ids_sha256']
            and all(set(c) == {'id', 'claim'} for c in claims), 'selected_ordered_claims_only')
    checked(prepared/'inference/initial-frames.json', release['frames_sha256'])
    checked(prepared/'inference/corpus.jsonl', CORPUS_SHA)
    return receipt, claims


def copy_prepared(out: Path, release: Any) -> Any:
    receipt, _ = check_prepared(PREPARED, release)
    files = {'selection.json': release['selection_sha256'],
        'component-reservations.json': release['component_reservations_sha256'],
        'preparation.json': release['preparation_sha256'],
        'inference/claims.json': release['claims_sha256'],
        'inference/initial-frames.json': release['frames_sha256'],
        'inference/corpus.jsonl': CORPUS_SHA}
    out.mkdir(mode=0o700)
    (out/'inference').mkdir(mode=0o700)
    for name, digest in files.items():
        with (out/name).open('xb') as stream:
            stream.write(checked(PREPARED/name, digest))
    check_prepared(out, release)
    return receipt


def load_frames(prepared: Path, claims: Any, corpus: Any, tokenizer: Any, release: Any) -> list[Any]:
    _, bound_claims = check_prepared(prepared, release)
    require(claims == bound_claims, 'worker_claims_prepared_mismatch')
    frames = json.loads(checked(prepared/'inference/initial-frames.json', release['frames_sha256']))
    return list(validate_frames(claims, frames, corpus, tokenizer))


def release_inputs(prepared: Path) -> dict[str, Any]:
    receipt = json.loads((prepared/'preparation.json').read_bytes())
    fields = {k: receipt[k] for k in HASH_KEYS if k != 'preparation_sha256'}
    fields['preparation_sha256'] = sha((prepared/'preparation.json').read_bytes())
    check_prepared(prepared, fields)
    return fields
