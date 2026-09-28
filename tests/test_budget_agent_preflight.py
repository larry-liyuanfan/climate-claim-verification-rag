import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path


def test_cpu_preflight_persists_receipt_without_generation(tmp_path):
    root = Path(__file__).resolve().parents[1]
    evidence = tmp_path / "evidence.jsonl"
    evidence.write_bytes(b"{}\n")
    protocol = tmp_path / "protocol.json"
    protocol.write_text(json.dumps({"budget": {}, "corpus_sha256": hashlib.sha256(evidence.read_bytes()).hexdigest()}))
    output = tmp_path / "result.json"
    command = [sys.executable, "scripts/run_budget_agent.py", "--evidence", str(evidence),
               "--protocol", str(protocol), "--expected-protocol-sha256",
               hashlib.sha256(protocol.read_bytes()).hexdigest(), "--preflight-only", "--output", str(output)]
    env = {**os.environ, "PYTHONPATH": str(root / "src")}
    subprocess.run(command, check=True, cwd=root, env=env, capture_output=True, timeout=30)
    report = json.loads(output.read_text())
    assert report["preflight"] == "passed" and report["model_generation"] is False
    assert report["dependencies"] is None  # CPU metadata check is not Linux model readiness
    assert not output.with_suffix(".consumed.json").exists()
    repeated = subprocess.run(command, check=False, cwd=root, env=env, capture_output=True, text=True, timeout=30)
    assert repeated.returncode != 0 and "preflight output already exists" in repeated.stderr
    assert json.loads(output.read_text()) == report
