#!/bin/bash
# Shared body read from a SHA-verified immutable source archive inside Slurm.
set -euo pipefail
: "${CLIMATE_SOURCE_TAR:?exact committed source archive}"
: "${CLIMATE_SOURCE_SHA256:?archive SHA256}"
: "${CLIMATE_RUN_BUNDLE:?preverified input bundle tar: evidence/protocol/model manifests/args.json}"
: "${CLIMATE_BUNDLE_SHA256:?bundle SHA256}"
module purge
module load GCC/11.3.0 OpenMPI/4.1.4 PyTorch/2.1.2-CUDA-12.2.0
CLIMATE_PYTHON="$(command -v python)"
for archive in "${CLIMATE_SOURCE_TAR}" "${CLIMATE_RUN_BUNDLE}" "${CLIMATE_RUNTIME_TAR:?}" "${CLIMATE_OVERLAY_TAR:?}"; do
  case "$(readlink -f "${archive}")" in
    /data/gpfs/projects/punim2936/portfolio_20260903/climate-public-retrieval-v2/*) ;;
    *) exit 90 ;;
  esac
done
: "${CLIMATE_RESULT_TAR:?new single output tar under the Climate root}"
case "$(readlink -f "$(dirname "${CLIMATE_RESULT_TAR}")")" in
  /data/gpfs/projects/punim2936/portfolio_20260903/climate-public-retrieval-v2/runs|/data/gpfs/projects/punim2936/portfolio_20260903/climate-public-retrieval-v2/runs/*) ;;
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
source "${TASK_ROOT}/source/hpc/budget_agent_runtime.sh"
budget_agent_activate_runtime
cd "${TASK_ROOT}/source"
"${CLIMATE_PYTHON}" - <<'PY'
import importlib.metadata, json, os, pathlib, subprocess, sys
from climate_rag.runtime_bundle import resolve_bundle_args
assert importlib.metadata.version('langchain-core') == '1.6.5'
root = pathlib.Path(os.environ['TASK_ROOT'])
phase = os.environ.get('CLIMATE_AGENT_PHASE', 'pilot')
assert phase in ('pilot', 'validation', 'vnext')
args = resolve_bundle_args(json.loads((root / f'input/args-{phase}.json').read_text()), root / 'input')
assert args[args.index('--phase') + 1] == phase
if os.environ.get('CLIMATE_PREFLIGHT_ONLY') == '1':
    args += ['--preflight-only']
# Trusted operator arguments only, never model-generated commands.
subprocess.run([sys.executable, 'scripts/run_budget_agent.py', *args,
                '--output', str(root / 'result/run.json')], check=True)
PY
tar -czf "${CLIMATE_RESULT_TAR}" -C "${TASK_ROOT}" result
sha256sum "${CLIMATE_RESULT_TAR}"
