"""Fixed-source CPU producer; no annotation reads, retrieval, teacher or model.

Only the allocated CLI can supply production files. Dependency-injected fixture
mode exists for CPU tests and cannot be selected by that CLI.
"""
from __future__ import annotations

from collections import Counter
from collections.abc import Callable, Mapping
import json
import os
from pathlib import Path
import time
from typing import Any

from .scifact_mixed_inputs import (
    LEGACY_SHA, NEI_COMPACT_SHA, NEI_SELECTION_SHA, ROSTER_SHA, SCOPE,
    build_inputs, input_config_sha, legacy_envelopes, make_plan, nei_envelope,
    nei_inventory, seal, validate_inputs,
)
from .scifact_nei_preparation import fixed_selection
from .scifact_read_continuation import ordered_write
from .scifact_semantic_contract import encoded, sha
from .scifact_state_supervision import require
from .scifact_utility_contract import identity

VERSION = 'scifact-mixed-program-preparation-v1-20261001'
OLD = 'posthoc/scifact-grounding-candidate-40d84a377bd1'
SHARED = 'posthoc/scifact-shared-corpus-fit-6fd92eb0ac91'
SUPPLEMENT = 'posthoc/scifact-supplemental-fit-09f3d9e72216'
NEI = 'posthoc/scifact-nei47-program-782892d57365'
PINS = {
    OLD+'/private/selection-before-packing.json': '578f57ec2955e9d51c1da7d4d219c15c339be7aa0790f283e3872221f3fb432a',
    OLD+'/private/families.json': '8b151a3a031b3cf7937c6394632a26dc64d9e7284bf976fffcdffc50d74e70cc',
    SHARED+'/compact.json': '27ad7cec3bb9aa1d9c12cd8618d9f9546956ed8b171e0483950333a3959c686d',
    SHARED+'/claim-reports.json': '4120b3f65793f5807d26ce256e9b154f11cf133655b46a25a7c5bcf06da0faea',
    SHARED+'/fit-records.json': LEGACY_SHA['old48'],
    SUPPLEMENT+'/compact.json': '4ef3d782ecde3efd36ca9dfef1482b9117f3f0ed695cca020bacdec8c047284e',
    SUPPLEMENT+'/selection-before-content.json': '72d0012fd12bf76396fb88c5f0d9581ef8d154996608e5fd167ea8f23e0a1355',
    SUPPLEMENT+'/component-reservations.json': '5afb9f5c5843ac083ea046b2276a0bdf117a8046beaa43190b469ae3409c56d2',
    SUPPLEMENT+'/claim-reports.json': '9595ac7a58807459a3e8d55e441acb76cd072dbb16d668e1daf341aea1c5b82e',
    SUPPLEMENT+'/fit-records.json': LEGACY_SHA['supp49'],
    NEI+'/compact.json': NEI_COMPACT_SHA,
    NEI+'/NEI-selection.json': NEI_SELECTION_SHA,
    NEI+'/claim-status.json': 'daae7ff23d2fc83bca31f221406687b5ac0b92232713d93b5e2d2af163aeb42b',
}
COUNTS = (48, 49, 47)


def preparation_config_sha() -> str:
    return sha(encoded({'version': VERSION, 'pins': PINS, 'input_config_sha256': input_config_sha(),
        'counts': COUNTS, 'raw_order_preserved': True, 'model_calls': 0, 'optimizer_steps': 0}))


def roster_metadata(objects: Mapping[str, Any], *, scope: str, counts: tuple[int, int, int]) -> dict[str, Any]:
    split = objects[OLD+'/private/selection-before-packing.json']
    families = objects[OLD+'/private/families.json']
    old_reports = objects[SHARED+'/claim-reports.json']
    shared = objects[SHARED+'/compact.json']
    compact = objects[SUPPLEMENT+'/compact.json']
    selection = objects[SUPPLEMENT+'/selection-before-content.json']
    reservation = objects[SUPPLEMENT+'/component-reservations.json']
    reports = objects[SUPPLEMENT+'/claim-reports.json']
    selected_nei = fixed_selection(compact, selection, reservation, reports, total=counts[1]+counts[2], nei=counts[2])
    require(selected_nei == objects[NEI+'/NEI-selection.json'], 'existing_NEI_selection_unchanged')
    old = split['fit']
    require(len(old) == counts[0] and [r['claim_id'] for r in old_reports] == old
            and all(r['training_ready'] for r in old_reports), 'original_roster_from_metadata_not_rows')
    ready = [r['claim_id'] for r in reports if r['status'] == 'prepared' and r['training_ready']]
    nei_ids = selected_nei['selected_ordered_ids']
    require(len(ready) == counts[1] and set(ready).isdisjoint(nei_ids)
            and set(ready) | set(nei_ids) == set(selection['selected_ordered_ids']), 'fixed_49_47_partition')
    cohorts = {'old48': old, 'supp49': ready, 'NEI47': nei_ids}
    ids = old+ready+nei_ids
    components = {str(r['claim_id']): r['component'] for r in old_reports}
    components.update({str(i): c for i, c in zip(selection['selected_ordered_ids'],
                        selection['selected_ordered_components'], strict=True)})
    require(len(ids) == len(set(ids)) == sum(counts), 'explicit_complete_roster')
    require(all(families['component_partition'][components[str(i)]] == 'fit' for i in ids), 'fit_components_only')
    for directory, record_sha in ((SHARED, LEGACY_SHA['old48']), (SUPPLEMENT, LEGACY_SHA['supp49'])):
        require(objects[directory+'/compact.json']['private_file_sha256']['fit-records.json'] == record_sha
                == PINS[directory+'/fit-records.json'], 'compact_to_legacy_source_binding')
    if scope == SCOPE:
        require(all(sha(encoded(cohorts[c])) == digest for c, digest in ROSTER_SHA.items())
                and shared['source_git'] == '6fd92eb0ac91fab206222d321c86448046dd2879'
                and compact['source_git'] == '09f3d9e72216dd6d61adff27c10179176a41656c', 'fixed_rosters_and_source_commits')
    return {'claim_ids': ids, 'cohorts': cohorts, 'components': components,
        'original96_ids': selection['selected_ordered_ids'],
        'excluded_claim_ids': sorted(set(split['tune']) | set(split['validation'])),
        'excluded_components': sorted(c for c, part in families['component_partition'].items() if part != 'fit'),
        'exclusion_scope': 'frozen_split_metadata_only_no_protected_content_or_regrouping',
        'source_files_sha256': dict(PINS)}


def prepare_bundle(output: Path, release: Mapping[str, Any], read_raw: Callable[[str], bytes],
                   assets: Callable[[], tuple[Any, Any]], *, scope: str = SCOPE,
                   counts: tuple[int, int, int] = COUNTS, final_check: Callable[[], None] | None = None) -> dict[str, Any]:
    require(scope in {SCOPE, 'synthetic_fixture'} and (scope != SCOPE or counts == COUNTS), 'preparation_scope')
    output.mkdir(mode=0o700, exist_ok=False)
    start = time.perf_counter()
    denominator = sum(counts)
    states: list[dict[str, Any]] = [{'index': i, 'status': 'unknown'} for i in range(denominator)]
    ordered_write(output/'started.json', {'version': VERSION, 'release': dict(release),
        'denominator': denominator, 'automatic_retry': False, 'training_authorized': False})
    actual_files: dict[str, str] = {}
    def read(name: str, expected: str) -> bytes:
        require(name in PINS or (name.startswith(NEI+'/claim-') and Path(name).name == name.removeprefix(NEI+'/')),
                'source_read_allowlist')
        raw = read_raw(name)
        require(sha(raw) == expected, 'physical_source_hash:'+name)
        actual_files[name] = expected
        return raw
    stage, fatal = 'metadata', None
    prepared: dict[str, Any] | None = None
    try:
        objects = {name: json.loads(read(name, digest)) for name, digest in PINS.items() if not name.endswith('/fit-records.json')}
        roster = roster_metadata(objects, scope=scope, counts=counts)
        for state, claim in zip(states, roster['claim_ids'], strict=True):
            state['claim_id'] = claim
        ordered_write(output/'input-roster-confirmed.json', {'roster': roster, 'physical_files': dict(actual_files)})
        compact = objects[NEI+'/compact.json']
        statuses = objects[NEI+'/claim-status.json']
        require(compact['whole_cohort_data_ready'] is True and compact['status_counts'] ==
                {'ready': counts[2], 'gap': 0, 'failed': 0, 'unknown': 0}
                and [r['claim_id'] for r in statuses] == roster['cohorts']['NEI47']
                and all(r['status'] == 'ready' for r in statuses), 'accepted_NEI47_complete')
        require(all(compact['private_file_sha256'][name] == PINS[NEI+'/'+name]
                    for name in ('NEI-selection.json', 'claim-status.json')), 'NEI_inventory_metadata_binding')
        inventory = nei_inventory(read(NEI+'/compact.json', PINS[NEI+'/compact.json']), scope=scope)
        stage = 'assets'
        corpus, tokenizer = assets()
        stage = 'legacy_factories'
        envelopes = []
        for cohort, directory in (('old48', SHARED), ('supp49', SUPPLEMENT)):
            name = directory+'/fit-records.json'
            raw = read(name, PINS[name])
            envelopes.extend(legacy_envelopes(raw, cohort, roster['cohorts'][cohort], corpus, tokenizer, scope=scope))
            for state in states:
                if state['claim_id'] in roster['cohorts'][cohort]:
                    state['status'] = 'ready'
            ordered_write(output/f'{cohort}-checkpoint.json', {'states': states, 'physical_files': dict(actual_files)})
        stage = 'NEI_factory'
        for status in statuses:
            name = status['artifact']
            require(Path(name).name == name and name.startswith('claim-') and name.endswith('.json')
                    and compact['private_file_sha256'][name] == status['artifact_sha256'], 'NEI_artifact_inventory_binding')
            claim = status['claim_id']
            state = next(s for s in states if s['claim_id'] == claim)
            try:
                raw = read(NEI+'/'+name, status['artifact_sha256'])
                envelope = nei_envelope(raw, expected_sha=status['artifact_sha256'], component=roster['components'][str(claim)],
                    inventory=inventory, artifact_name=name, corpus=corpus, tokenizer=tokenizer, scope=scope)
                require(envelope['artifact']['claim_id'] == claim, 'NEI_status_to_artifact_claim')
                envelopes.append(envelope)
                state['status'] = 'ready'
            except BaseException:
                state['status'] = 'failed'
                raise
        stage = 'derive_and_write'
        roster['source_envelope_sha256'] = [identity(e) for e in envelopes]
        frozen = seal(roster)
        prepared = build_inputs(envelopes, frozen, scope=scope, inventory=inventory)
        plan = make_plan(prepared)
        ordered_write(output/'roster.json', frozen)
        ordered_write(output/'prepared.json', prepared)
        ordered_write(output/'plan.json', plan)
        stage = 'physical_readback'
        restored = json.loads((output/'prepared.json').read_bytes())
        require(identity(restored) == identity(prepared), 'readback_order_and_full_identity')
        validate_inputs(restored)
        require(restored['payload']['tokenized'] == [e['tokenized'] for e in envelopes]
                and [r['source_envelope'] for r in restored['payload']['records']] == envelopes,
                'readback_original_factory_tokens_and_packing')
        require(json.loads((output/'plan.json').read_bytes()) == make_plan(restored)
                and identity(json.loads((output/'roster.json').read_bytes())) == identity(frozen), 'readback_plan_roster')
        if final_check is not None:
            final_check()
        require(all(s['status'] == 'ready' for s in states), 'all_denominator_claims_ready')
    except BaseException as exc:
        fatal = type(exc).__name__
        # Keep source-contract diagnostics private; never add messages or samples
        # to the exportable compact receipt.
        private = output/'private'
        private.mkdir(mode=0o700, exist_ok=False)
        message = str(exc)
        diagnostic = private/'failure-diagnostic.json'
        ordered_write(diagnostic, {'stage': stage, 'exception_type': fatal,
            'message': message[:2048], 'message_truncated': len(message) > 2048,
            'private_only': True})
        diagnostic.chmod(0o600)
    resource_fields: dict[str, Any] = {'available': False, 'cpu_seconds': None, 'maxrss_kib_linux': None}
    if os.name == 'posix':
        import resource
        posix_resource: Any = resource
        usage = posix_resource.getrusage(posix_resource.RUSAGE_SELF)
        resource_fields = {'available': True, 'cpu_seconds': usage.ru_utime+usage.ru_stime, 'maxrss_kib_linux': usage.ru_maxrss}
    success = fatal is None and prepared is not None
    status_counts = Counter(s['status'] for s in states)
    files = {name: sha((output/name).read_bytes()) for name in ('prepared.json', 'plan.json', 'roster.json')
             if (output/name).is_file()}
    compact = {'version': VERSION, 'status': 'complete' if success else 'failed', 'scope': scope,
        'denominator': denominator, 'status_counts': {k: status_counts[k] for k in ('ready', 'failed', 'unknown')},
        'rows': len(prepared['payload']['records']) if prepared is not None else None,
        'planned_updates': make_plan(prepared)['payload']['optimizer_steps'] if prepared is not None else None,
        'physical_source_files_sha256': actual_files, 'files': files, 'source_git': release['source_git'],
        'release_sha256': release.get('release_sha256'), 'input_config_sha256': input_config_sha(),
        'preparation_config_sha256': preparation_config_sha(), 'source_bridge_validated': success,
        'last_stage': stage, 'fatal_type': fatal, 'resources': resource_fields,
        'elapsed_seconds': time.perf_counter()-start, 'model_calls': 0, 'optimizer_steps': 0,
        'training_authorized': False, 'automatic_retry': False}
    ordered_write(output/'claim-status.json', states)
    ordered_write(output/'compact.json', compact)
    if success:
        require(set(files) == {'prepared.json', 'plan.json', 'roster.json'}, 'all_three_outputs_required')
        ordered_write(output/'complete.pending.json', compact)
        os.link(output/'complete.pending.json', output/'complete.json')
    else:
        ordered_write(output/'failed.json', compact)
    return compact
