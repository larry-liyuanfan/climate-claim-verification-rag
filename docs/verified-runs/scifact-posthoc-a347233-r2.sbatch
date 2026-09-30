#!/bin/bash
#SBATCH --job-name=climate-pair-closeout-r2
#SBATCH --account=punim2936
#SBATCH --partition=sapphire,cascade
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --time=00:05:00
set -euo pipefail
umask 077
CLIMATE_AUDIT_MODE=${1:---audit}
case "$CLIMATE_AUDIT_MODE" in
  --imports-only) ;;
  --audit) : "${SLURM_JOB_ID:?CPU allocation required}" ;;
  *) exit 2 ;;
esac
CLIMATE_ROOT=/data/gpfs/projects/punim2936/portfolio_20260903/climate-public-retrieval-v2
CLIMATE_AUDIT_GIT=a347233b88c103cb23c76845527278f960a48406
CLIMATE_AUDIT_TAR="$CLIMATE_ROOT/envs/scifact-posthoc-a347233.tar"
CLIMATE_RUNTIME="$CLIMATE_ROOT/envs/runtime-py310-2423a755979c426ecae5d17deead154393de712efd10e84884d8c87ff00d29f8.tar.gz"
CLIMATE_OVERLAY="$CLIMATE_ROOT/envs/budget-agent-linux-overlay-20260929-e7223ac2f4c08a736e763d485a47dcb088ce62c28fb9f1da556661a3f71da0ce.tar.gz"
test "$(sha256sum "$CLIMATE_AUDIT_TAR" | cut -d' ' -f1)" = bee2d9a9af76bdbbe9625f08cdc5f13f1d3ea0816ad4973825ef1a371fc30619
test "$(sha256sum "$CLIMATE_RUNTIME" | cut -d' ' -f1)" = 2423a755979c426ecae5d17deead154393de712efd10e84884d8c87ff00d29f8
test "$(sha256sum "$CLIMATE_OVERLAY" | cut -d' ' -f1)" = e7223ac2f4c08a736e763d485a47dcb088ce62c28fb9f1da556661a3f71da0ce
CLIMATE_AUDIT_PARENT="${SLURM_TMPDIR:-${TMPDIR:-/tmp}}"
CLIMATE_AUDIT_WORK=$(mktemp -d "$CLIMATE_AUDIT_PARENT/climate-pair-audit-r2-${SLURM_JOB_ID:-preflight}-XXXXXX")
mkdir "$CLIMATE_AUDIT_WORK/source"
tar --no-same-owner -xf "$CLIMATE_AUDIT_TAR" -C "$CLIMATE_AUDIT_WORK/source"
test "$(cat "$CLIMATE_AUDIT_WORK/source/SOURCE_REVISION")" = "$CLIMATE_AUDIT_GIT"
test "$(sha256sum "$CLIMATE_AUDIT_WORK/source/scripts/audit_scifact_bounded_closeout.py" | cut -d' ' -f1)" = 1d18a54e883fc2c5e7a70af25fc6c469b38abcd11144b451c01f86451981fde9
module purge
module load GCC/11.3.0 OpenMPI/4.1.4 PyTorch/2.1.2-CUDA-12.2.0
export PYTHONDONTWRITEBYTECODE=1 PYTHONNOUSERSITE=1
export CUDA_VISIBLE_DEVICES=""
python - "$CLIMATE_AUDIT_WORK" "$CLIMATE_RUNTIME" "$CLIMATE_OVERLAY" "$CLIMATE_AUDIT_MODE" "$CLIMATE_ROOT" "$CLIMATE_AUDIT_GIT" <<'PY'
import importlib.util
import os
import subprocess
import sys
from pathlib import Path

work, runtime, overlay, mode, root, audit_git = sys.argv[1:]
work, root = Path(work), Path(root)
source = work / "source"
sys.path.insert(0, str(source / "scripts"))
from run_budget_agent_full_operator import safe_extract
from run_sentence_agent_v3_operator import v3_runtime_environment

print("runtime_extracted_bytes=" + str(safe_extract(Path(runtime), work / "runtime", runtime=True)), flush=True)
print("overlay_extracted_bytes=" + str(safe_extract(Path(overlay), work / "overlay")), flush=True)
subprocess.run([
    sys.executable, "-m", "pip", "install", "--no-index", "--no-deps", "--require-hashes",
    "--no-compile", "--no-cache-dir", "--disable-pip-version-check", "--target", str(work / "overlay-site"),
    "--find-links", str(work / "overlay/wheels"), "-r", str(work / "overlay/requirements.lock"),
], check=True)
# Exact frozen inference helper: overlay, runtime, source, then module PYTHONPATH.
env = v3_runtime_environment(work, source)
check = r'''
import importlib.metadata
import json
import sys
from pathlib import Path
scripts = Path(sys.argv[1])
sys.path.insert(0, str(scripts))
import numpy
import pydantic
import jsonschema
import audit_scifact_bounded_closeout as audit
print(json.dumps({
    "import_only": True,
    "python": sys.version.split()[0],
    "numpy": {"version": numpy.__version__, "path": numpy.__file__},
    "pydantic": {"version": pydantic.__version__, "path": pydantic.__file__},
    "jsonschema": {"version": importlib.metadata.version("jsonschema"), "path": jsonschema.__file__},
    "audit_entry": audit.__file__,
    "audit_callable": callable(audit.main),
}, sort_keys=True), flush=True)
'''
subprocess.run([sys.executable, "-c", check, str(source / "scripts")], env=env, check=True)
if mode == "--imports-only":
    print("import_only_passed_no_audit_or_model_calls", flush=True)
else:
    if not os.environ.get("SLURM_JOB_ID"):
        raise RuntimeError("CPU allocation required")
    output = root / "posthoc/scifact-pair-audit-a347233-r2"
    output.mkdir(mode=0o700)  # New output only; preserve first attempt and refuse overwrite.
    subprocess.run([
        sys.executable, str(source / "scripts/audit_scifact_bounded_closeout.py"),
        "--root", str(root),
        "--submission", str(source / "docs/verified-runs/scifact-bounded-pair-submission-31686182-31686183.json"),
        "--protocol", str(source / "docs/protocols/scifact-bounded-gap-pair-20260930.json"),
        "--audit-git", audit_git, "--output", str(output / "compact.json"),
    ], env=env, check=True)
    print("cpu_audit_job_id=" + os.environ["SLURM_JOB_ID"], flush=True)
PY
