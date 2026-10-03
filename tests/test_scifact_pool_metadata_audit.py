"""Small synthetic metadata checks; never load real claims or scoring data."""
from __future__ import annotations

import importlib.util
from pathlib import Path
import tempfile
import unittest
from typing import Any

spec = importlib.util.spec_from_file_location('pool_metadata', Path(__file__).parents[1]/
                                             'scripts/audit_scifact_candidate_pool_metadata.py')
assert spec is not None and spec.loader is not None
audit = importlib.util.module_from_spec(spec)
spec.loader.exec_module(audit)


class MetadataAuditTests(unittest.TestCase):
    def test_overlapping_exclusions_not_double_counted(self) -> None:
        result = audit.remaining_components({'a', 'b', 'c', 'd'}, [
            ('fit', {'a'}), ('consumed', {'a', 'b'}), ('unknown', {'b', 'c'}), ('failed', {'b'})])
        self.assertEqual([r['newly_removed']['count'] for r in result['sequential_exclusions']], [1, 1, 1, 0])
        self.assertEqual(result['remaining'], audit.describe({'d'}))

    def test_hash_mismatch_fails_before_deserialization(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root/'metadata.json').write_bytes(b'not json')
            with self.assertRaisesRegex(ValueError, 'input_hash_mismatch'):
                audit.Inputs(root).read('metadata.json', '0'*64)

    def test_path_escape_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaisesRegex(ValueError, 'input_path'):
                audit.Inputs(Path(directory)).read('../outside.json', '0'*64)

    def example(self) -> tuple[Any, ...]:
        text_hash = audit.digest(b'synthetic sentence')
        first = {'identity': {'visible': [], 'previews': [{'source_id': 'c2', 'citable': False}]}}
        after = {'identity': {'visible': [{'sentence_id': 'c2:2', 'sha256': text_hash}], 'previews': []}}
        old = {'claim_id': 1, 'candidate_doc_ids': [10, 20, 30], 'frozen_read_ids': ['c2'],
               'ordered_candidates': [{'doc_id': i, 'source_sha256': str(i)} for i in [10, 20, 30]],
               'states': [first, after]}
        states = [old, {'claim_id': 2, 'frozen_read_ids': ['c0']}, {'claim_id': 3, 'frozen_read_ids': ['c0']}]
        states += [{'claim_id': i, 'frozen_read_ids': []} for i in range(4, 13)]
        frame = {'observation': {'current_citable': [{'sentence_id': 'c0:2', 'text': 'synthetic sentence'}]}}
        rows = [{'claim_id': 1, 'frame': frame,
                 'candidates': [{'doc_id': i, 'source_sha256': str(i)} for i in [30, 20]],
                 'target': {'documents': [{'source_id': 'c0', 'sentence_ids': ['c0:2']}]}}]
        return (states, [{'claim_id': 1, 'initial_eligible_gold_doc_ids': [], 'final_eligible_gold_doc_ids': [30]}],
                rows, [{'claim_id': 1, 'frames': [frame]}], {'fit': [1]}, {10: 'unowned', 20: 'fit', 30: 'fit'})

    def test_rank_and_sentence_identity_not_alias_identity(self) -> None:
        result = audit.case_comparison(*self.example())
        self.assertEqual((result['old_rank'], result['new_rank']), (3, 1))
        self.assertEqual(result['old_top20_fit_filter_rank_without_rescoring'], 2)
        self.assertFalse(result['common_relative_order_unchanged'])
        self.assertTrue(result['old_initial_preview_present'])
        self.assertEqual(result['old_after_read_sentence_map_sha256'], result['new_initial_sentence_map_sha256'])
        self.assertEqual(result['existing_target_fits_new_initial'], 1)

    def test_sentence_content_change_rejected(self) -> None:
        args = self.example()
        args[2][0]['frame']['observation']['current_citable'][0]['text'] = 'different'
        with self.assertRaisesRegex(ValueError, 'sentence_hash_changed'):
            audit.case_comparison(*args)


if __name__ == '__main__':
    unittest.main()
