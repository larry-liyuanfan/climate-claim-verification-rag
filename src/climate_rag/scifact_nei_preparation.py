"""Versioned reprocessing of already exposed NEI inside the frozen supplemental96."""
from __future__ import annotations

import json
import os
import time
from collections import Counter
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any

from .scifact_grounding import Abstract, parse_gold
from .scifact_program_capture import VERSION as PROGRAM_VERSION, capture_initial, program_candidates
from .scifact_read_continuation import ordered_write
from .scifact_retrieval import SciFactBM25
from .scifact_semantic_contract import checked, encoded, sha
from .scifact_state_supervision import require
from .scifact_state_training_driver import corpus_identity
from .scifact_terminal_supervision import _unique_object
from .scifact_utility_contract import identity

VERSION = 'scifact-fixed-NEI47-preparation-v1-20261001'
PARENT = 'posthoc/scifact-supplemental-fit-09f3d9e72216'
INPUT_SHA = {
    'compact.json': '4ef3d782ecde3efd36ca9dfef1482b9117f3f0ed695cca020bacdec8c047284e',
    'selection-before-content.json': '72d0012fd12bf76396fb88c5f0d9581ef8d154996608e5fd167ea8f23e0a1355',
    'component-reservations.json': '5afb9f5c5843ac083ea046b2276a0bdf117a8046beaa43190b469ae3409c56d2',
    'claim-reports.json': '9595ac7a58807459a3e8d55e441acb76cd072dbb16d668e1daf341aea1c5b82e',
    'complete-selected-fit-gold.json': 'cb948810eb48fc455c1ef26582cc6e4a09ad66e34c3c61475a7e32ff26d31ba1',
}
CONFIG = {'version': VERSION, 'original_claims': 96, 'official_NEI_claims': 47,
          'selection': 'already_exposed_official_NEI_in_original96_order_no_reselection',
          'candidate_k': 20, 'candidate_pool_documents': 5183, 'real_model_calls': 0,
          'optimizer_steps': 0, 'training_authorized': False, 'automatic_retry': False}


def fixed_selection(compact: Mapping[str, Any], selection: Mapping[str, Any],
                    reservation: Mapping[str, Any], reports: list[dict[str, Any]],
                    *, total: int, nei: int) -> dict[str, Any]:
    ids, components = selection['selected_ordered_ids'], selection['selected_ordered_components']
    require(len(ids) == len(set(ids)) == len(components) == len(set(components)) == total
            and all(type(i) is int for i in ids), 'original_selection_identity')
    require(ids == reservation['ordered_claim_ids'] == [r['claim_id'] for r in reports]
            and components == reservation['ordered_components'] == [r['component'] for r in reports]
            and compact['actual_selected_claims'] == total
            and compact['selected_ids_sha256'] == sha(encoded(ids))
            and compact['selected_components_sha256'] == sha(encoded(components)), 'original96_order_binding')
    require(reservation['selection_sha256'] == compact['private_file_sha256']['selection-before-content.json']
            and reservation['config_sha256'] == selection['config_sha256'] == compact['config_sha256'],
            'original_reservation_binding')
    chosen = []
    for row in reports:
        if row['status'] == 'not_applicable' or row['official_label_scope'] == 'NOT_ENOUGH_INFO':
            require(row['status'] == 'not_applicable' and row['official_label_scope'] == 'NOT_ENOUGH_INFO'
                    and row['reason'] == 'official_nei_not_supported_by_frozen_teacher'
                    and row['retained_records'] == 0, 'official_NEI_report_contract')
            chosen.append(row['claim_id'])
    require(len(chosen) == nei, 'exact_NEI_count_no_replacement')
    return {'version': VERSION, 'original_ordered_ids': ids, 'original_ordered_components': components,
        'selected_ordered_ids': chosen, 'selected_components': [components[ids.index(i)] for i in chosen],
        'source_label_exposure': 'already_exposed_by_supplemental96_preparation',
        'selection_before_first_label_exposure': False, 'reselection_allowed': False,
        'original_ids_sha256': sha(encoded(ids)), 'NEI_ids_sha256': sha(encoded(chosen))}


def freeze(out: Path, selection: Mapping[str, Any]) -> dict[str, str]:
    ordered_write(out/'NEI-selection.json', dict(selection))
    digest = sha((out/'NEI-selection.json').read_bytes())
    ordered_write(out/'NEI-reservation.json', {'version': VERSION, 'selection_sha256': digest,
        'ordered_claim_ids': selection['selected_ordered_ids'], 'automatic_retry': False,
        'state': 'reserved_before_complete96_gold_reopen_no_replacement'})
    return {name: sha((out/name).read_bytes()) for name in ('NEI-selection.json', 'NEI-reservation.json')}


def reserved_gold(open_gold: Callable[[], bytes], gold_sha: str, out: Path, frozen: Mapping[str, str],
                  corpus: Mapping[int, Abstract]) -> dict[int, dict[str, Any]]:
    # The callback cannot open gold before both durable files have been checked.
    selection, reservation = [json.loads(checked(out/name, frozen[name]))
                             for name in ('NEI-selection.json', 'NEI-reservation.json')]
    require(reservation['selection_sha256'] == frozen['NEI-selection.json']
            and reservation['ordered_claim_ids'] == selection['selected_ordered_ids'], 'NEI_reservation_binding')
    raw = open_gold()
    require(sha(raw) == gold_sha, 'complete96_gold_hash')
    rows = json.loads(raw, object_pairs_hook=_unique_object)
    require(isinstance(rows, list) and [r['id'] for r in rows] == selection['original_ordered_ids']
            and all(set(r) == {'id', 'claim', 'evidence', 'cited_doc_ids'} and type(r['id']) is int for r in rows),
            'complete96_fullrow_order_identity')
    claims = [parse_gold(r, corpus) for r in rows]
    require([c.claim_id for c in claims if not c.evidence] == selection['selected_ordered_ids'],
            'frozen47_complete_gold_NEI_identity')
    return {r['id']: r for r in rows if r['id'] in selection['selected_ordered_ids']}


def prepare_reserved(out: Path, release: Mapping[str, Any],
                     metadata: Callable[[], dict[str, Any]], open_gold: Callable[[], bytes], gold_sha: str,
                     assets: Callable[[], tuple[Mapping[int, Abstract], Any]], *,
                     total: int = 96, nei: int = 47, seconds: float = 875,
                     final_check: Callable[[], None] | None = None) -> dict[str, Any]:
    """Exclusive bounded preparation. Small synthetic sizes are for tests, not CLI options."""
    start = time.perf_counter()
    out.mkdir(mode=0o700)  # A failed/partial run is not reusable.
    ordered_write(out/'run-reservation.json', {'version': VERSION, 'release': dict(release),
        'planned_claims': nei, 'automatic_retry': False, 'training_authorized': False})
    ordered_write(out/'started.json', {'version': VERSION, 'state': 'reserved', 'planned_claims': nei})
    states: list[dict[str, Any]] = [{'index': i, 'status': 'unknown', 'scripted_responses': None} for i in range(nei)]
    frozen: dict[str, str] = {}
    fatal: str | None = None
    stage = 'metadata'
    try:
        selection = fixed_selection(**metadata(), total=total, nei=nei)
        frozen = freeze(out, selection)
        for row, claim_id in zip(states, selection['selected_ordered_ids'], strict=True):
            row['claim_id'] = claim_id
        stage = 'assets'
        corpus, tokenizer = assets()
        stage = 'complete96_gold_after_freeze'
        claims = reserved_gold(open_gold, gold_sha, out, frozen, corpus)
        ordered_write(out/'gold-read-check.json', {'read_after_frozen_files_verified': True,
            'complete96_order_verified': True, 'actual_NEI_matches_frozen': True,
            'parent_gold_sha256': gold_sha, 'frozen_file_sha256': frozen})
        retrieve = SciFactBM25(corpus)
        corpus_sha = corpus_identity(corpus)
        stage = 'program_capture'
        for row in states:
            require(time.perf_counter()-start < seconds, 'CPU_preparation_deadline')
            try:
                original = claims[row['claim_id']]
                receipt = capture_initial({k: original[k] for k in ('id', 'claim')}, retrieve, corpus, tokenizer)
                row['scripted_responses'] = receipt['payload']['counts']['scripted_responses']
                annotation = encoded(original)
                p = {'version': PROGRAM_VERSION, 'scope': release['scope'], 'annotation_sha256': sha(annotation),
                    'capture_sha256': receipt['sha256'], 'corpus_sha256': corpus_sha,
                    'complete_original_row_declared': True, 'selection_sha256': frozen['NEI-selection.json'],
                    'parent_gold_sha256': gold_sha}
                result = program_candidates(receipt, annotation, {'payload': p, 'sha256': identity(p)}, corpus, tokenizer)
                name = f"claim-{row['index']:02d}.json"
                ordered_write(out/name, result)
                written = (out/name).read_bytes()
                restored = json.loads(written)
                require(identity(restored) == identity(result), 'persisted_program_result_identity')
                rebuilt = program_candidates(restored['program_capture'], annotation, restored['provenance'], corpus, tokenizer)
                require(identity(rebuilt) == identity(result), 'persisted_prompt_target_token_identity')
                row.update(status='ready' if result['status'] == 'candidates_enumerated' else 'gap',
                           gaps=result['gaps'], artifact=name, artifact_sha256=sha(written))
            except TimeoutError:
                raise
            except Exception as exc:
                row.update(status='failed', error_type=type(exc).__name__, reason=str(exc)[:180])
            ordered_write(out/f"checkpoint-{row['index']:02d}.json", row)
        stage = 'verify_artifacts'
        for name, digest in frozen.items():
            checked(out/name, digest)
        for row in states:
            if 'artifact' in row:
                checked(out/row['artifact'], row['artifact_sha256'])
        if final_check is not None:
            final_check()
    except (Exception, KeyboardInterrupt) as exc:
        fatal = type(exc).__name__ + ':' + str(exc)[:180]
    try:
        return _finalize(out, states, release, frozen, stage, fatal, start, nei, gold_sha)
    except (Exception, KeyboardInterrupt) as exc:
        # No complete means incomplete. Disk/SIGKILL may also prevent a failure marker.
        if not (out/'complete.json').exists() and not (out/'failed.json').exists():
            try:
                ordered_write(out/'failed.json', {'version': VERSION, 'stage': 'finalization',
                    'error_type': type(exc).__name__, 'denominator': nei,
                    'whole_cohort_data_ready': False, 'automatic_retry': False})
            except (Exception, KeyboardInterrupt):
                pass
        raise


def _finalize(out: Path, states: list[dict[str, Any]], release: Mapping[str, Any], frozen: Mapping[str, str],
              stage: str, fatal: str | None, start: float, nei: int, gold_sha: str) -> dict[str, Any]:
    counts = Counter(r['status'] for r in states)
    success = counts['ready'] == nei and fatal is None
    ordered_write(out/'claim-status.json', states)
    resources: dict[str, Any] = {'cpu_seconds': None, 'maxrss_kib': None, 'available': False}
    if os.name == 'posix':
        import resource
        usage = resource.getrusage(resource.RUSAGE_SELF)
        resources = {'cpu_seconds': usage.ru_utime+usage.ru_stime, 'maxrss_kib': usage.ru_maxrss,
                     'available': True, 'scope': 'whole_preparation_process'}
    compact = {'version': VERSION, 'config': CONFIG, 'config_sha256': sha(encoded(CONFIG)),
        'release': dict(release), 'input_parent_gold_sha256': gold_sha, 'selection_file_sha256': frozen,
        'planned_claims': nei, 'denominator': nei, 'actual_records': counts['ready']+counts['gap'],
        'status_counts': {s: counts[s] for s in ('ready', 'gap', 'failed', 'unknown')},
        'whole_cohort_data_ready': success, 'last_stage': stage, 'fatal': fatal,
        'scripted_responses_known': sum(r['scripted_responses'] or 0 for r in states),
        'scripted_cost_unknown_claims': sum(r['scripted_responses'] is None for r in states),
        'model_calls': 0, 'real_model_calls': 0, 'optimizer_steps': 0, 'training_authorized': False,
        'original48_reexecuted': False, 'supplemental49_reexecuted': False,
        'roster_created': False, 'weights_created': False, 'automatic_retry': False,
        'elapsed_seconds': time.perf_counter()-start, 'resources': resources,
        'private_file_sha256': {p.name: sha(p.read_bytes()) for p in sorted(out.iterdir()) if p.is_file()}}
    ordered_write(out/'compact.json', compact)
    marker = {'version': VERSION, 'compact_sha256': sha((out/'compact.json').read_bytes()),
              'whole_cohort_data_ready': success, 'automatic_retry': False}
    if success:
        ordered_write(out/'complete.pending.json', marker)
        os.link(out/'complete.pending.json', out/'complete.json')  # Atomic no-clobber publication.
    else:
        ordered_write(out/'failed.json', marker)
    return compact
