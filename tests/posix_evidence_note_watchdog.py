"""Stdlib-only POSIX seam tests of the EXACT operator function, no model imports.

Run in WSL/Linux: python3 tests/posix_evidence_note_watchdog.py
AST selection avoids requiring the GPU/runtime dependency tree on the local host.
"""
from __future__ import annotations

import ast
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import tempfile
import time
from typing import Any
import unittest
from unittest.mock import patch


def require(ok: bool, reason: str) -> None:
    if not ok:
        raise ValueError(reason)


source = Path(__file__).resolve().parents[1] / 'scripts/run_scifact_evidence_note_operator.py'
tree = ast.parse(source.read_text())
node = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == 'bounded_worker')
namespace: dict[str, Any] = {'os': os, 'Path': Path, 'signal': signal, 'subprocess': subprocess,
    'time': time, 'Any': Any, 'json': json, 'require': require}
exec(compile(ast.Module(body=[node], type_ignores=[]), str(source), 'exec'), namespace)
bounded_worker = namespace['bounded_worker']


@unittest.skipUnless(os.name == 'posix', 'real POSIX process-group test')
class WatchdogTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp = tempfile.TemporaryDirectory(prefix='climate-note-watchdog-')
        self.root = Path(self.temp.name)
        self.allocation = self.root / 'allocation'
        self.allocation.mkdir()
        self.inference = self.root / 'inference'
        self.inference.mkdir()

    def tearDown(self) -> None:
        self.temp.cleanup()

    def run_child(self, code: str, timeout: float = 3) -> dict[str, Any]:
        result: dict[str, Any] = bounded_worker([sys.executable, '-c', code], self.allocation, self.inference, timeout)
        return result

    def test_success_reaped(self) -> None:
        proof = self.run_child('pass')
        self.assertTrue(proof['child_reaped'])
        self.assertEqual(proof['returncode'], 0)

    def test_total_timeout_kills_only_own_group(self) -> None:
        proof = self.run_child('import time; time.sleep(30)', 0.2)
        self.assertTrue(proof['child_reaped'])
        self.assertEqual(proof['returncode'], -signal.SIGKILL)
        self.assertEqual(proof['interrupted'], 'TimeoutError')

    def test_stage_deadline_and_partial_reservation(self) -> None:
        directory = self.inference / 'case-01/note'
        directory.mkdir(parents=True)
        path = directory / 'reserved.json'
        path.write_text('{')
        os.utime(path, (time.time() - 121, time.time() - 121))
        proof = self.run_child('import time; time.sleep(30)')
        self.assertTrue(proof['child_reaped'])
        self.assertEqual(proof['returncode'], -signal.SIGKILL)

    def test_signal_in_popen_return_window_reaped(self) -> None:
        actual = subprocess.Popen
        pids = []
        def interrupted_launch(*args: Any, **kwargs: Any) -> Any:
            process = actual(*args, **kwargs)
            pids.append(process.pid)
            os.kill(os.getpid(), signal.SIGTERM)
            return process
        with patch.object(subprocess, 'Popen', interrupted_launch):
            proof = self.run_child('import time; time.sleep(30)')
        self.assertTrue(proof['child_reaped'])
        self.assertEqual(proof['interrupted'], 'InterruptedError')
        with self.assertRaises(ProcessLookupError):
            os.kill(pids[0], 0)


if __name__ == '__main__':
    unittest.main()
