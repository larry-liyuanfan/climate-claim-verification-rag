"""Hash-bound metadata audit; no retrieval, model, new labels or cohort selection.

Only previously exposed FIT frames/targets and saved old witness metadata are
compared. The source archive is opened for two metadata members only. All other
inputs are manifests, family/group mappings and physical reservation receipts.
The sole output contains counts, hashes and one anonymised case, never text/IDs.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import re
import tarfile
from typing import Any

ROOT = Path('/data/gpfs/projects/punim2936/portfolio_20260903/climate-public-retrieval-v2')
OLD = 'posthoc/scifact-grounding-candidate-40d84a377bd1'
READ = 'posthoc/scifact-selected-read-opportunity-dcbe368c1c02'
NEW = 'posthoc/scifact-state-supervision-fit-v2-9bf5fdad2ec9'
PREP = 'data/scifact-original-20260930/prepared-r1'
ARCHIVE = 'envs/scifact-semantic-inputs-c4907db6.tar'
ARCHIVE_SHA = 'c4907db623b644c9c883a807f07597f87746595da7fc6435714452304e50a44d'
MEMBERS = {
    'preparation-manifest.json': '3cd6bc1e1c299ece9195098a3853ebb05bb3401507d8b467c1596ba6afb6e138',
    'private-group-assignment.json': '6d79861daef5d385878913fe10681dff95df444e6f7b1b76ab7c07f4d4d70a12',
}
FILES = {
    OLD+'/manifest.json': '28de2c5d1d531aabdb757ccb45db0b91232a54ac7089ac7dc5ebf42e65ba3b2f',
    OLD+'/private/selection-before-packing.json': '578f57ec2955e9d51c1da7d4d219c15c339be7aa0790f283e3872221f3fb432a',
    OLD+'/private/families.json': '8b151a3a031b3cf7937c6394632a26dc64d9e7284bf976fffcdffc50d74e70cc',
    OLD+'/private/ledger.json': 'df55427ad67f1f45f3c9d37929b756d948f2dbb94f02d2f9a3f3b7f0584923c7',
    READ+'/private/states-before-gold.json': '5a49b53aa6ab7790231b224b5966f3e8e6f2fff1856219c1c177af16ded714b8',
    READ+'/private/diagnosis.json': 'a0e95cd626430a365599579948d6f5ea22630f457c6fe3c4a968fc52cc6cebd0',
    NEW+'/compact.json': '3cfc8acffcb001488ce841310349fa05a806a2216d66336afe331d9cd04e1bde',
    NEW+'/captured-frames.json': '1a2dcc53c55c2013ed11b8b3818ca364d0b4ecc223f14ba1a1128713ecd6c7fa',
    NEW+'/fit-records.json': 'd33769569ff4d2460575d5f35e3e04abd0876335582ada77b26e64b147438df9',
    'runs/scifact-grounding-training-20261001-v1/complete.json': 'f5e6a865ec09cb67a520e36ba4646fb297a381383d4d0357208488ef6e9168a0',
    'runs/scifact-grounding-tune-20261001-v1/adapted/complete.json': 'ad4dc447ba4d50dfdf5fa44a67d8c3365fd164a86ef92cb952f81a5189b91799',
    'runs/scifact-grounding-tune-20261001-v1/base/complete.json': 'a3a5ca96365b1eef12bb5102a15aab422912070a0cef29266877999c9be2c652',
    'runs/scifact-adapter-bare-regression-20261001-v1/cost-before-quality.json': '87910b370e6265170957c991c6c6750e3f8c43d46983e2ee12722e13db551c9e',
    'runs/scifact-read-conditional-20261001-v1/cost-before-quality.json': '4f8170e588939bad502b1a9cb19726b4fd10768aa3526449cb562835377f3373',
}


def require(ok: bool, message: str) -> None:
    if not ok:
        raise ValueError(message)


def digest(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def encoded(value: Any) -> bytes:
    # Match semantic_contract.encoded used for the frozen FIT-pool digest.
    return (json.dumps(value, sort_keys=True, ensure_ascii=False, indent=2)+'\n').encode()


def describe(values: set[Any]) -> dict[str, Any]:
    return {'count': len(values), 'sha256': digest(encoded(sorted(values)))}


class Inputs:
    def __init__(self, root: Path) -> None:
        self.root = root.resolve()
        self.reads: dict[str, str] = {}

    def read(self, name: str, expected: str | None = None) -> Any:
        expected = FILES[name] if expected is None else expected
        path = self.root/name
        require(not path.is_symlink() and path.resolve().is_relative_to(self.root), 'input_path')
        raw = path.read_bytes()
        require(digest(raw) == expected, 'input_hash_mismatch')
        self.reads[name] = expected
        return json.loads(raw)

    def metadata_members(self) -> tuple[dict[str, Any], dict[str, Any]]:
        path = self.root/ARCHIVE
        require(not path.is_symlink() and path.resolve().is_relative_to(self.root), 'archive_path')
        checksum = hashlib.sha256()
        with path.open('rb') as stream:
            for chunk in iter(lambda: stream.read(1024*1024), b''):
                checksum.update(chunk)
        require(checksum.hexdigest() == ARCHIVE_SHA, 'archive_hash_mismatch')
        self.reads[ARCHIVE] = ARCHIVE_SHA
        result = []
        with tarfile.open(path) as bundle:
            for name, expected in MEMBERS.items():
                matches = [m for m in bundle.getmembers() if m.name == PREP+'/'+name]
                require(len(matches) == 1 and matches[0].isfile(), 'unique_metadata_member')
                member_stream = bundle.extractfile(matches[0])
                require(member_stream is not None, 'missing_metadata_member')
                assert member_stream is not None
                raw = member_stream.read()
                require(digest(raw) == expected, 'metadata_member_hash')
                self.reads[ARCHIVE+'::'+PREP+'/'+name] = expected
                result.append(json.loads(raw))
        return result[0], result[1]

    def reservations(self, receipt: str, count: int, tune: bool = False) -> set[int]:
        index = self.read(receipt)
        directory = str(Path(receipt).parent).replace('\\', '/')
        pattern = r'slot-\d{2}/started\.json' if tune else r'slot-\d{2}-reserved\.json'
        hashes = index['physical_files_sha256' if tune else 'physical_sha256']
        names = [n for n in hashes if re.fullmatch(pattern, n)]
        require(len(names) == count, 'reservation_count_or_filename')
        ids = set()
        for name in names:
            row = self.read(directory+('' if tune else '/inference')+'/'+name, hashes[name])
            require(type(row['claim_id']) is int, 'reservation_claim_type')
            ids.add(row['claim_id'])
        return ids


def remaining_components(all_fit: set[str], exclusions: list[tuple[str, set[str]]]) -> dict[str, Any]:
    remaining = set(all_fit)
    steps = []
    for name, excluded in exclusions:
        removed = remaining & excluded
        steps.append({'reason': name, 'overlap_with_original_fit_partition': len(all_fit & excluded),
                      'newly_removed': describe(removed)})
        remaining -= excluded
    return {'initial': describe(all_fit), 'sequential_exclusions': steps, 'remaining': describe(remaining)}


def case_comparison(states: list[dict[str, Any]], diagnosis: list[dict[str, Any]],
                    records: list[dict[str, Any]], frames: list[dict[str, Any]],
                    split: dict[str, Any], doc_partition: dict[int, str]) -> dict[str, Any]:
    read_cases = [r for r in states if r['frozen_read_ids']]
    overlap = [r for r in read_cases if r['claim_id'] in split['fit']]
    require(len(states) == 12 and len(read_cases) == 3 and len(overlap) == 1, 'exact_overlap_case')
    old = overlap[0]
    rows = [r for r in records if r['claim_id'] == old['claim_id']]
    captured = [r for r in frames if r['claim_id'] == old['claim_id']]
    require(bool(rows) and len(captured) == 1 and len(captured[0]['frames']) == 1, 'frozen_initial_frame')
    new = rows[0]
    require(all(r['candidates'] == new['candidates'] and r['frame'] == new['frame'] for r in rows)
            and new['frame'] == captured[0]['frames'][0], 'new_record_frame_identity')
    old_ids = old['candidate_doc_ids']
    new_ids = [r['doc_id'] for r in new['candidates']]
    require(len(old['frozen_read_ids']) == 1 and len(old['states']) == 2, 'one_saved_read')
    alias = old['frozen_read_ids'][0]
    require(re.fullmatch(r'c\d+', alias) is not None, 'source_alias')
    doc = old_ids[int(alias[1:])]
    require(doc in new_ids, 'witness_in_new_candidates')
    new_alias = 'c'+str(new_ids.index(doc))
    sources = [{r['doc_id']: r['source_sha256'] for r in items}
               for items in (old['ordered_candidates'], new['candidates'])]
    common = set(old_ids) & set(new_ids)
    require(all(sources[0][i] == sources[1][i] for i in common), 'common_source_text_hash_changed')

    def old_visible(state: dict[str, Any]) -> dict[int, str]:
        return {int(s['sentence_id'].split(':')[1]): s['sha256'] for s in state['identity']['visible']
                if s['sentence_id'].split(':')[0] == alias}

    first, after = [old_visible(s) for s in old['states']]
    initial = {int(s['sentence_id'].split(':')[1]): digest(s['text'].encode())
               for s in new['frame']['observation']['current_citable']
               if s['sentence_id'].split(':')[0] == new_alias}
    require(all(after[i] == initial[i] for i in set(after) & set(initial)), 'sentence_hash_changed')
    witness = next(r for r in diagnosis if r['claim_id'] == old['claim_id'])
    target_sets = {frozenset(int(s.split(':')[1]) for s in d['sentence_ids']) for r in rows
                   for d in r['target']['documents'] if d['source_id'] == new_alias}
    require(bool(target_sets), 'no_existing_target_for_saved_document')
    filtered = [d for d in old_ids if doc_partition[d] == 'fit']
    before = old_ids[:old_ids.index(doc)]
    return {
        'case': 'only_original_fit_overlap_of_three_saved_reads',
        'old_candidate_count': len(old_ids), 'new_candidate_count': len(new_ids),
        'old_rank': old_ids.index(doc)+1, 'new_rank': new_ids.index(doc)+1,
        'old_top20_fit_filter_rank_without_rescoring': filtered.index(doc)+1,
        'old_preceding_candidates_by_partition': {p: sum(doc_partition[d] == p for d in before)
                                                 for p in ('fit', 'tune', 'validation', 'unowned')},
        'common_candidate_count': len(common),
        'common_relative_order_unchanged': [i for i in old_ids if i in common] == [i for i in new_ids if i in common],
        'common_source_hashes_unchanged': True,
        'witness_source_sha256': sources[0][doc],
        'old_initial_citable_positions': sorted(first), 'old_after_read_citable_positions': sorted(after),
        'new_initial_citable_positions': sorted(initial),
        'old_initial_sentence_map_sha256': digest(encoded(first)),
        'old_after_read_sentence_map_sha256': digest(encoded(after)),
        'new_initial_sentence_map_sha256': digest(encoded(initial)),
        'old_initial_preview_present': any(p['source_id'] == alias and not p['citable']
                                           for p in old['states'][0]['identity']['previews']),
        'saved_old_diagnosis_initial_eligible': doc in witness['initial_eligible_gold_doc_ids'],
        'saved_old_diagnosis_after_read_eligible': doc in witness['final_eligible_gold_doc_ids'],
        'existing_new_target_rationale_sets': len(target_sets),
        'existing_target_fits_old_after_read': sum(set(t) <= set(after) for t in target_sets),
        'existing_target_fits_new_initial': sum(set(t) <= set(initial) for t in target_sets),
        'newly_scored_witnesses': 0, 'causal_effect_identified': False,
    }


def audit(inputs: Inputs) -> dict[str, Any]:
    original, assignment = inputs.metadata_members()
    manifest = inputs.read(OLD+'/manifest.json')
    require(original['output_file_sha256']['private-group-assignment.json'] == MEMBERS['private-group-assignment.json']
            == manifest['assignment_sha256'], 'assignment_manifest_binding')
    split, families, ledger = [inputs.read(OLD+'/private/'+n) for n in
                               ('selection-before-packing.json', 'families.json', 'ledger.json')]
    require(all(manifest['files']['private/'+n] == FILES[OLD+'/private/'+n] for n in
                ('selection-before-packing.json', 'families.json', 'ledger.json')), 'old_manifest_bindings')
    eligible = set(assignment['eligible_train_ids'])
    components = {int(i): str(c) for i, c in assignment['claim_component'].items()}
    def groups(ids: set[int]) -> set[str]:
        require(ids <= set(components), 'unmapped_claim')
        return {components[i] for i in ids}
    partitions = families['component_partition']
    owners = families['family_component']
    eligible_groups, dev_groups = groups(eligible), groups(set(assignment['dev_ids']))
    require(not eligible_groups & dev_groups and set(partitions) == eligible_groups
            and set(owners.values()) <= eligible_groups, 'eligible_dev_group_isolation')
    selected = {p: groups(set(split[p])) for p in ('fit', 'tune', 'validation')}
    require([len(split[p]) for p in selected] == [48, 12, 12]
            and all(len(selected[p]) == len(split[p]) for p in selected)
            and all(partitions[c] == p for p in selected for c in selected[p])
            and sum(map(len, selected.values())) == len(set().union(*selected.values())), 'selected_partition_isolation')
    doc_partition = {int(d): partitions[owners[str(f)]] if str(f) in owners else 'unowned'
                     for d, f in families['document_family'].items()}
    pools = {p: {d for d, part in doc_partition.items() if part == p}
             for p in ('fit', 'tune', 'validation', 'unowned')}
    new_compact = inputs.read(NEW+'/compact.json')
    require(len(doc_partition) == 5183 and digest(encoded(sorted(pools['fit'])))
            == new_compact['fit_pool_doc_ids_sha256'], 'original_fit_pool_binding')
    states = inputs.read(READ+'/private/states-before-gold.json')
    old_ids = {r['claim_id'] for r in states}
    read_ids = {r['claim_id'] for r in states if r['frozen_read_ids']}
    train = inputs.read('runs/scifact-grounding-training-20261001-v1/complete.json')
    require(train['data_manifest_sha256'] == FILES[OLD+'/manifest.json'] and train['records_seen_once'] == 96,
            'training_manifest_or_record_binding')
    later_exposures = set(split['fit'])
    for arm in ('base', 'adapted'):
        tune_ids = inputs.reservations('runs/scifact-grounding-tune-20261001-v1/'+arm+'/complete.json', 12, True)
        require(tune_ids == set(split['tune']), 'tune_reservation_identity')
        later_exposures.update(tune_ids)
    regression_ids = inputs.reservations('runs/scifact-adapter-bare-regression-20261001-v1/cost-before-quality.json', 48)
    require(regression_ids == old_ids, 'regression_reservation_identity')
    conditional_ids = inputs.reservations('runs/scifact-read-conditional-20261001-v1/cost-before-quality.json', 3)
    require(conditional_ids == read_ids, 'conditional_reservation_identity')
    later_exposures.update(regression_ids | conditional_ids)
    consumed, unknown = groups(set(ledger['model_consumed_ids'])), groups(set(ledger['uncertain_ids']))
    require(consumed | unknown == set(ledger['component_excluded']), 'ledger_component_exclusion')
    all_fit = {c for c, part in partitions.items() if part == 'fit'}
    failed_attempt_ids = {e['claim_id'] for r in ledger['runs'] for e in r['attempts']
                          if e['claim_id'] is not None and e['status'] in ('failed', 'unknown')}
    failed_groups = groups(failed_attempt_ids)
    exclusions = [('original_fit48', selected['fit']), ('reserved_tune', selected['tune']),
                  ('reserved_validation', selected['validation']), ('prior_model_consumed', consumed),
                  ('prior_uncertain', unknown), ('later_training', selected['fit']),
                  ('later_tune', selected['tune']), ('later_regression', groups(old_ids)),
                  ('later_conditional', groups(read_ids)), ('failed_or_unknown_attempt', failed_groups)]
    counts = remaining_components(all_fit, exclusions)
    remaining = all_fit - set().union(*(c for _, c in exclusions))
    counts['eligible_claim_ids_in_remaining_components'] = describe({i for i in eligible if components[i] in remaining})
    counts['gold_preparation_seen_eligible_claims'] = len(eligible)
    counts['original_ledger_gold_preparation_count_matches'] = ledger['gold_preparation_seen_count'] == len(eligible)
    counts['checked_receipt_exposures_outside_original_fit_tune_and_old_selected'] = len(
        later_exposures - set(split['fit']) - set(split['tune']) - old_ids)
    counts['scope'] = 'conditional_on_listed_hash_bound_ledger_and_receipts_not_global_exposure_attestation'
    counts['label_or_read_opportunity_eligibility_checked'] = False
    comparison = case_comparison(states, inputs.read(READ+'/private/diagnosis.json'),
        inputs.read(NEW+'/fit-records.json'), inputs.read(NEW+'/captured-frames.json'), split, doc_partition)
    return {
        'version': 'scifact-pool-metadata-audit-20261001-v1',
        'script_sha256': digest(Path(__file__).read_bytes()), 'input_sha256': inputs.reads,
        'overlap_case': comparison,
        'family_inventory': {
            'corpus_documents': len(doc_partition), 'pools': {p: describe(d) for p, d in pools.items()},
            'partition_components': {p: describe({c for c, part in partitions.items() if part == p})
                                     for p in ('fit', 'tune', 'validation')},
            'eligible_train_dev_components_disjoint': True,
            'eligible_train_only_family_owner_mapping': True,
            'complete_dev_quarantined_test_family_inventory_persisted': False,
            'provable_additional_safe_documents_outside_current_fit_pool': describe(set()),
            'unowned_does_not_mean_safe': True,
            'protection_scope': 'saved lexical claim/family grouping, not semantic or pretraining decontamination',
            'missing_proof': 'complete family-to-claim edges for dev/quarantined/other heldout; official test not inventoried',
        },
        'train_metadata_eligibility': counts,
        'decision': 'metadata_only_protocol_change_pending_no_pool_expansion_or_cohort_selection_authorized',
        'boundary': {'retrieval_executions': 0, 'model_calls': 0, 'training_steps': 0,
                     'new_claim_text_or_gold_read': False, 'corpus_text_deserialized': False,
                     'new_cohort_selected': False, 'protected_metadata_only': True,
                     'no_safe_background_proof_is_not_proof_that_none_exists': True},
    }


def main() -> None:
    result = audit(Inputs(ROOT))
    out = ROOT/'posthoc'/('scifact-pool-metadata-'+result['script_sha256'][:12])
    out.mkdir(mode=0o700)
    with (out/'compact.json').open('xb') as stream:
        stream.write(json.dumps(result, ensure_ascii=False, sort_keys=True, indent=2).encode()+b'\n')
    print(json.dumps({'output': str(out/'compact.json'), 'sha256': digest((out/'compact.json').read_bytes()),
                      'overlap_case': result['overlap_case'],
                      'family_inventory': result['family_inventory'],
                      'train_metadata_eligibility': result['train_metadata_eligibility']}, sort_keys=True))


if __name__ == '__main__':
    main()
