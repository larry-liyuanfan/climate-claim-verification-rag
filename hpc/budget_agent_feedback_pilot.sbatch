#!/bin/bash
# Released development pilot only. One node-local preflight followed by one pilot.
#SBATCH --job-name=climate-agent-feedback-v2
#SBATCH --account=punim2936
#SBATCH --partition=gpu-a100,gpu-a100-short
#SBATCH --gres=gpu:A100:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --tmp=30G
#SBATCH --time=00:45:00
#SBATCH --no-requeue
#SBATCH --signal=B:USR1@90
set -euo pipefail
umask 077
: "${SLURM_JOB_ID:?allocated job required}"
: "${CLIMATE_SOURCE_TAR:?}"
: "${CLIMATE_SOURCE_SHA256:?}"
: "${CLIMATE_SOURCE_GIT:?}"
test "${CLIMATE_FEEDBACK_RELEASE:-}" = climate-feedback-v2-20260930-pilot-r1
case "$(readlink -f "${CLIMATE_SOURCE_TAR}")" in
  /data/gpfs/projects/punim2936/portfolio_20260903/climate-public-retrieval-v2/envs/*) ;; *) exit 90 ;;
esac
test "$(sha256sum "${CLIMATE_SOURCE_TAR}" | cut -d' ' -f1)" = "${CLIMATE_SOURCE_SHA256}"
test "$(tar -xOf "${CLIMATE_SOURCE_TAR}" SOURCE_REVISION | tr -d '\r\n')" = "${CLIMATE_SOURCE_GIT}"
test "$(sha256sum "$0" | cut -d' ' -f1)" = "$(tar -xOf "${CLIMATE_SOURCE_TAR}" hpc/budget_agent_feedback_pilot.sbatch | sha256sum | cut -d' ' -f1)"
module purge
module load GCC/11.3.0 OpenMPI/4.1.4 PyTorch/2.1.2-CUDA-12.2.0
CLIMATE_SCRATCH_PARENT="${SLURM_TMPDIR:-${TMPDIR:-}}"
case "$(readlink -f "${CLIMATE_SCRATCH_PARENT}")" in
  /tmp|/tmp/*|/var/tmp|/var/tmp/*|/jobfs/*) ;; *) exit 91 ;;
esac
CLIMATE_FEEDBACK_WORK=$(mktemp -d "${CLIMATE_SCRATCH_PARENT}/climate-feedback-${SLURM_JOB_ID}-XXXXXX")
export CLIMATE_FEEDBACK_WORK CLIMATE_SOURCE_TAR CLIMATE_SOURCE_GIT CLIMATE_SOURCE_SHA256 CLIMATE_FEEDBACK_RELEASE
export PYTHONDONTWRITEBYTECODE=1 PYTHONNOUSERSITE=1
python - <<'PY'
import os, pathlib, tarfile
target = pathlib.Path(os.environ['CLIMATE_FEEDBACK_WORK']) / 'source'
target.mkdir()
with tarfile.open(os.environ['CLIMATE_SOURCE_TAR']) as archive:
    names = set()
    for member in archive.getmembers():
        path = (target / member.name).resolve()
        if not path.is_relative_to(target) or not (member.isfile() or member.isdir()) or path in names:
            raise ValueError('unsafe source archive')
        names.add(path)
    archive.extractall(target)
PY
exec python "${CLIMATE_FEEDBACK_WORK}/source/scripts/run_budget_agent_feedback_operator.py"
