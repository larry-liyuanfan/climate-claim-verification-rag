"""Regression for CI run 36841857351: collect CLI tests without shell path setup."""
from __future__ import annotations

import os
from pathlib import Path
import subprocess
import sys


def test_script_entry_tests_collect_without_external_pythonpath() -> None:
    names = (
        'test_scifact_adapter_regression.py', 'test_scifact_evidence_note.py',
        'test_scifact_grounding_train.py', 'test_scifact_grounding_tune.py',
        'test_scifact_grounding_validation.py', 'test_scifact_mixed_checkpoint.py',
        'test_scifact_mixed_launch.py', 'test_scifact_read_continuation.py',
        'test_scifact_selected_read.py',
    )
    env = dict(os.environ)
    env.pop('PYTHONPATH', None)
    env.pop('PYTEST_ADDOPTS', None)
    env['PYTHONDONTWRITEBYTECODE'] = '1'
    # Collect only the formerly failing modules, not this test (no recursion).
    result = subprocess.run([sys.executable, '-m', 'pytest', '--collect-only', '-q',
        '-p', 'no:cacheprovider', *('tests/' + name for name in names)],
        cwd=Path(__file__).resolve().parents[1], env=env, capture_output=True,
        text=True, encoding='utf-8', errors='replace', timeout=120, check=False)
    assert result.returncode == 0, result.stdout + result.stderr
    assert all(name in result.stdout for name in names)
    assert 'ModuleNotFoundError' not in result.stdout + result.stderr
