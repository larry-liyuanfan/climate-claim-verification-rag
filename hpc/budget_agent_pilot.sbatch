#!/bin/bash
# Preparation only: explicit resource release required before submission.
#SBATCH --job-name=climate-agent-pilot
#SBATCH --account=punim2936
#SBATCH --partition=gpu-a100
#SBATCH --gres=gpu:A100:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=00:15:00
#SBATCH --tmp=30G
set -euo pipefail
: "${CLIMATE_SOURCE_TAR:?exact committed source archive}"
: "${CLIMATE_SOURCE_SHA256:?archive SHA256}"
: "${CLIMATE_RUN_BUNDLE:?preverified input bundle tar: evidence/protocol/model manifests/args.json}"
: "${CLIMATE_BUNDLE_SHA256:?bundle SHA256}"
: "${CLIMATE_PYTHON:?existing locked runtime python executable}"
: "${CLIMATE_RESULT_TAR:?new single output tar under the Climate root}"
case "${CLIMATE_RESULT_TAR}" in
  /data/gpfs/projects/punim2936/portfolio_20260903/climate-public-retrieval-v2/runs/*) ;;
  *) exit 91 ;;
esac
test ! -e "${CLIMATE_RESULT_TAR}"
test -x "${CLIMATE_PYTHON}"
test "$(sha256sum "${CLIMATE_SOURCE_TAR}" | cut -d' ' -f1)" = "${CLIMATE_SOURCE_SHA256}"
test "$(sha256sum "${CLIMATE_RUN_BUNDLE}" | cut -d' ' -f1)" = "${CLIMATE_BUNDLE_SHA256}"
TASK_TMP="${SLURM_TMPDIR:-${TMPDIR:-}}"
test -n "${TASK_TMP}"
case "$(readlink -f "${TASK_TMP}")" in /tmp/*|/var/tmp/*|/jobfs/*) ;; *) exit 92 ;; esac
TASK_ROOT="$(mktemp -d "${TASK_TMP}/climate-agent-${SLURM_JOB_ID}-XXXXXX")"
export TASK_ROOT CLIMATE_SOURCE_TAR CLIMATE_RUN_BUNDLE
"${CLIMATE_PYTHON}" - <<'PY'
import os, pathlib, tarfile
root = pathlib.Path(os.environ['TASK_ROOT'])
for env, name in [('CLIMATE_SOURCE_TAR', 'source'), ('CLIMATE_RUN_BUNDLE', 'input')]:
    target = root / name
    target.mkdir()
    with tarfile.open(os.environ[env]) as archive:
        for member in archive.getmembers():
            path = (target / member.name).resolve()
            if not path.is_relative_to(target) or not (member.isfile() or member.isdir()):
                raise ValueError('unsafe archive member')
        archive.extractall(target)
PY
export PYTHONPATH="${TASK_ROOT}/source/src${PYTHONPATH:+:${PYTHONPATH}}"
export PYTHONDONTWRITEBYTECODE=1 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export HF_HUB_DISABLE_TELEMETRY=1
cd "${TASK_ROOT}/source"
"${CLIMATE_PYTHON}" - <<'PY'
import importlib.metadata, json, os, pathlib, subprocess, sys
assert importlib.metadata.version('langchain-core') == '1.6.5'
root = pathlib.Path(os.environ['TASK_ROOT'])
args = json.loads((root / 'input/args.json').read_text())
assert isinstance(args, list) and all(isinstance(x, str) for x in args)
assert '--provider' in args and args[args.index('--provider') + 1] == 'local-qwen'
args = [x.replace('{INPUT}', str(root / 'input')) for x in args]
if os.environ.get('CLIMATE_PREFLIGHT_ONLY') == '1':
    args += ['--preflight-only']
# Trusted operator arguments only, never model-generated commands.
subprocess.run([sys.executable, 'scripts/run_budget_agent.py', *args,
                '--output', str(root / 'result/run.json')], check=True)
PY
if test "${CLIMATE_PREFLIGHT_ONLY:-0}" != 1; then
  tar -czf "${CLIMATE_RESULT_TAR}" -C "${TASK_ROOT}" result
  sha256sum "${CLIMATE_RESULT_TAR}"
fi
