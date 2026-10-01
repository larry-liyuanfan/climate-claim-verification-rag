"""Identity adapter for accepted mixed training, not another evaluator or trainer."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping, Sequence

from .scifact_mixed_inputs import unseal
from .scifact_mixed_launch import ROOT, preparation_fields
from .scifact_mixed_training import CONFIG, VERSION, config_sha
from .scifact_semantic_contract import checked, encoded, sha
from .scifact_state_supervision import require

KIND = 'mixed_claim_mean_144_223_36_v1'


def binding_identity(binding: Mapping[str, Any]) -> dict[str, Any]:
    require(binding['kind'] == KIND and len(binding['training_source_git']) == 40
            and set(binding['training_source_git']) <= set('0123456789abcdef'),
            'explicit_mixed_checkpoint_kind')
    require(set(binding['adapter_files']) == {'adapter_config.json', 'adapter_model.safetensors', 'README.md'},
            'exact_final_adapter_files')
    digest_fields = ('training_complete_sha256', 'training_release_sha256', 'training_source_archive_sha256',
        'worker_exit_sha256', 'training_runtime_sha256', 'started_sha256', 'prepared_envelope_sha256',
        'plan_envelope_sha256', 'prepared_file_sha256', 'plan_file_sha256', 'roster_file_sha256')
    digests = [binding[n] for n in digest_fields] + list(binding['adapter_files'].values())
    require(all(isinstance(d, str) and len(d) == 64
                and set(d) <= set('0123456789abcdef') for d in digests), 'checkpoint_hash_fields')
    return {'checkpoint_kind': KIND, 'adapter_training_sha256': binding['training_complete_sha256'],
        'adapter_model_sha256': binding['adapter_files']['adapter_model.safetensors'],
        'training_source_git': binding['training_source_git'], 'binding_sha256': sha(encoded(dict(binding)))}


def checkpoint_metadata(adapter: Path, binding: Mapping[str, Any]) -> dict[str, Any]:
    identity = binding_identity(binding)
    require(adapter.as_posix() == binding['directory']
            and adapter == ROOT/'runs'/('scifact-mixed-training-'+binding['training_source_git'][:12]),
            'fixed_mixed_checkpoint_directory')
    release_path = Path(binding['training_release_file'])
    require(release_path.resolve().is_relative_to((ROOT/'envs').resolve()), 'training_release_project_storage')
    release = json.loads(checked(release_path, binding['training_release_sha256']))
    require(release['authorization'] == 'coordinator_exact_hash_release' and release['purpose'] == VERSION
            and release['config_sha256'] == config_sha() and release['output'] == adapter.as_posix()
            and release['source_git'] == binding['training_source_git']
            and release['source_archive_sha256'] == binding['training_source_archive_sha256'],
            'training_release_source_config_binding')
    training = json.loads(checked(adapter/'complete.json', binding['training_complete_sha256']))
    require(training['version'] == VERSION and training['status'] == 'complete'
            and training['actual_claims'] == 144 and training['decision_records'] == 223
            and training['optimizer_steps'] == 36 and training['automatic_retry'] is False
            and training['artifacts'] == binding['adapter_files'], 'mixed_complete_shape')
    require({p.name for p in (adapter/'final').iterdir()} == set(binding['adapter_files'])
            and not any(p.is_symlink() for p in (adapter/'final').iterdir()), 'checkpoint_regular_exact_files')
    for name, digest in binding['adapter_files'].items():
        checked(adapter/'final'/name, digest)
    execution = adapter.with_name(adapter.name+'-execution')
    exit_proof = json.loads(checked(execution/'worker-exit.json', binding['worker_exit_sha256']))
    reservation = json.loads((execution/'reserved.json').read_bytes())
    require(exit_proof['returncode'] == 0 and exit_proof['child_started'] is True
            and exit_proof['child_reaped'] is True and exit_proof['interrupted'] is None
            and reservation['release_sha256'] == binding['training_release_sha256'], 'training_child_exited_successfully')
    started = json.loads(checked(adapter/'started.json', binding['started_sha256']))
    runtime = json.loads(checked(adapter/'training-runtime.json', binding['training_runtime_sha256']))
    require(started['config_sha256'] == config_sha() and runtime['config'] == CONFIG
            and runtime['fresh_base_and_adapter'] is True, 'actual_training_config_and_freshness')
    bundle = Path(release['prepared_directory'])
    prepared_fields = preparation_fields(checked(bundle/'complete.json', release['preparation_receipt_sha256']))
    require(all(release[k] == v for k, v in prepared_fields.items()), 'accepted_CPU_preparation_binding')
    # The accepted CPU receipt attests these physical files. Inference must NOT
    # open prepared.json/plan.json: they contain training targets, and exposed
    # regression queries can overlap FIT. No target is needed to load an adapter.
    for name in ('prepared', 'plan', 'roster'):
        field = name+'_file_sha256'
        require(release[field] == binding[field], 'physical_CPU_file_identity')
    # These are envelope identities, NOT SHA-256 of serialized files above.
    require(training['prepared_sha256'] == binding['prepared_envelope_sha256']
            == started['prepared_sha256'] and training['plan_sha256'] == binding['plan_envelope_sha256']
            == started['plan']['sha256'], 'logical_envelope_vs_physical_file_binding')
    return {**identity, 'training': training, 'prepared_directory': bundle.as_posix(),
            'roster_file_sha256': binding['roster_file_sha256'], 'training_exit_verified': True,
            'training_payload_opened': False}


def fit_overlap(binding: Mapping[str, Any], selected: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """IDs/components only; no training records or labels, no assumption of equality."""
    release = json.loads(checked(Path(binding['training_release_file']), binding['training_release_sha256']))
    path = Path(release['prepared_directory'])/'roster.json'
    roster = unseal(json.loads(checked(path, binding['roster_file_sha256'])))
    ids = set(roster['claim_ids'])
    require(len(ids) == len(roster['claim_ids']) == 144
            and [len(roster['cohorts'][c]) for c in ('old48', 'supp49', 'NEI47')] == [48, 49, 47],
            'all_144_fit_roster_metadata')
    components = set(roster['components'].values())
    direct = sorted(s['id'] for s in selected if s['id'] in ids)
    shared = sorted(s['id'] for s in selected if s['component'] in components)
    return {'direct_claim_overlap': direct, 'component_overlap': shared, 'fit_claims': 144,
            'roster_sha256': binding['roster_file_sha256'], 'no_overlap_is_not_independent_holdout': True}
