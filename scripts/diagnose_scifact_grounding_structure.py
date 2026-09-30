"""Existing receipts/fit metadata only: no gold loader, rescoring or model calls."""
from __future__ import annotations

from collections import Counter
import json
from pathlib import Path
from typing import Any

from climate_rag.scifact_semantic_contract import PRIVATE_SHA, checked, sha, write_once

ROOT = Path('/data/gpfs/projects/punim2936/portfolio_20260903/climate-public-retrieval-v2')
FIT_SHA = '70e40db85b0e51292ee1235be6bb708272846a0fcd4be2d123cfb4f2238d15a5'
MANIFEST_SHA = '28de2c5d1d531aabdb757ccb45db0b91232a54ac7089ac7dc5ebf42e65ba3b2f'
RESPONSE_SHA = {
    'base': '44a111271b144d5a4112dd67bd1d5c0d9c279acd836ac0cd9472a2095294163f',
    'adapted': 'fab6431b50941b71b94059a80b12cddcc5a0aa86641a9b800d378a9c94d4d287',
}


def summarise_fit(rows: list[dict[str, Any]], selection: list[dict[str, Any]]) -> dict[str, Any]:
    ids, components = {r['claim_id'] for r in rows}, {r['component'] for r in rows}
    return {
        'fit_records': len(rows), 'fit_claims': len(ids), 'fit_components': len(components),
        'context_document_counts': dict(Counter(len(r['context']['document_ids']) for r in rows)),
        'target_actions': dict(Counter(r['target']['action'] for r in rows)),
        'target_source_aliases': dict(Counter(d['source_id'] for r in rows for d in r['target']['documents'])),
        'target_relation_counts': dict(Counter(d['label'] for r in rows for d in r['target']['documents'])),
        'old_selected_claims': len(selection),
        'exact_claim_overlap': sum(s['id'] in ids for s in selection),
        'component_overlap': sum(s['component'] in components for s in selection),
        'overlap_by_legacy_stratum': {
            group: {'selected': sum(s['legacy_stratum'] == group for s in selection),
                    'fit_exact': sum(s['legacy_stratum'] == group and s['id'] in ids for s in selection),
                    'fit_component': sum(s['legacy_stratum'] == group and s['component'] in components for s in selection)}
            for group in sorted({s['legacy_stratum'] for s in selection})},
    }


def main() -> None:
    prepared = ROOT / 'posthoc/scifact-grounding-candidate-40d84a377bd1'
    manifest = json.loads(checked(prepared / 'manifest.json', MANIFEST_SHA))
    rows = json.loads(checked(prepared / 'fit/records.json', FIT_SHA))
    assert manifest['files']['fit/records.json'] == FIT_SHA
    selection = json.loads(checked(ROOT / 'posthoc/scifact-semantic-preparation-779e49883570/private/selected-before-probe.json',
                                   PRIVATE_SHA['selected-before-probe.json']))['selection']
    run = ROOT / 'runs/scifact-grounding-tune-20261001-v1'
    arms: dict[str, Any] = {}
    for arm in ('base', 'adapted'):
        paths = sorted((run / arm).glob('slot-*/response.json'))
        assert len(paths) == 12
        hashes = {p.relative_to(run / arm).as_posix(): sha(p.read_bytes()) for p in paths}
        index_sha = sha(json.dumps(hashes, sort_keys=True, separators=(',', ':')).encode())
        assert index_sha == RESPONSE_SHA[arm]
        raw = [json.loads(json.loads(p.read_bytes())['raw']) for p in paths]
        docs = [d for r in raw for d in r.get('documents', [])]
        arms[arm] = {'queries': len(raw), 'response_manifest_sha256': index_sha,
            'actions': dict(Counter(r['action'] for r in raw)),
            'documents_per_response': dict(Counter(len(r.get('documents', [])) for r in raw)),
            'source_aliases': dict(Counter(d['source_id'] for d in docs)),
            'relations': dict(Counter(d['label'] for d in docs)),
            'sentence_index_counts': dict(Counter(int(s.split(':')[1]) for d in docs for s in d['sentence_ids']))}
    compact = {'scope': 'posthoc_existing_train_internal_artifacts_not_new_experiment',
        'fit_sha256': FIT_SHA, 'data_manifest_sha256': MANIFEST_SHA,
        'selection_sha256': PRIVATE_SHA['selected-before-probe.json'],
        'fit_and_overlap': summarise_fit(rows, selection), 'tune_responses': arms,
        'script_sha256': sha(Path(__file__).read_bytes()), 'new_model_calls': 0,
        'gold_loaded': False, 'scorer_rerun': False, 'original_gate_modified': False,
        'interpretation': ['Single-document c1 supervision and c1-only tune predictions suggest alias/top1 convergence, not demonstrated multi-document selection.',
            'Sentence indices vary; this is not simply copying the first sentence.',
            'The three historic read witnesses split into one FIT-overlap and two non-direct-FIT-overlap queries; all remain exposed regression inputs.',
            'Retain all twelve queries and all four routes; tune/validation grouping is unchanged.']}
    output = ROOT / 'posthoc/scifact-grounding-structure-20261001-v1'
    output.mkdir(mode=0o700)
    write_once(output / 'compact.json', compact)
    print(json.dumps({'output': str(output), 'sha256': sha((output / 'compact.json').read_bytes()),
                      'fit': compact['fit_and_overlap'], 'tune': arms}, sort_keys=True))


if __name__ == '__main__':
    main()
