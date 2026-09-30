#!/bin/bash
# PREPARATION ONLY: queue availability is NOT permission to submit this script.
#SBATCH --job-name=climate-agent-full-frozen
#SBATCH --account=punim2936
#SBATCH --partition=gpu-a100
#SBATCH --gres=gpu:A100:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --tmp=30G
#SBATCH --time=02:00:00
#SBATCH --no-requeue
#SBATCH --signal=B:USR1@90
set -euo pipefail
umask 077
: "${SLURM_JOB_ID:?separate released allocation required}"
: "${CLIMATE_FULL_RELEASE_ID:?explicit coordinator full release required}"
: "${CLIMATE_OPERATOR_TAR:?immutable operator archive}"
: "${CLIMATE_OPERATOR_SHA256:?operator archive hash}"
: "${CLIMATE_OPERATOR_GIT:?operator source identity, distinct from inference/scorer}"
case "$(readlink -f "${CLIMATE_OPERATOR_TAR}")" in
  /data/gpfs/projects/punim2936/portfolio_20260903/climate-public-retrieval-v2/envs/*) ;; *) exit 90 ;;
esac
test "$(sha256sum "${CLIMATE_OPERATOR_TAR}" | cut -d' ' -f1)" = "${CLIMATE_OPERATOR_SHA256}"
test "$(tar -xOf "${CLIMATE_OPERATOR_TAR}" SOURCE_REVISION | tr -d '\r\n')" = "${CLIMATE_OPERATOR_GIT}"
module purge
module load GCC/11.3.0 OpenMPI/4.1.4 PyTorch/2.1.2-CUDA-12.2.0
CLIMATE_FULL_SCRATCH_PARENT="${SLURM_TMPDIR:-${TMPDIR:-}}"
case "$(readlink -f "${CLIMATE_FULL_SCRATCH_PARENT}")" in
  /tmp|/tmp/*|/var/tmp|/var/tmp/*|/jobfs/*) ;; *) exit 91 ;;
esac
CLIMATE_FULL_TASK_ROOT=$(mktemp -d "${CLIMATE_FULL_SCRATCH_PARENT}/climate-agent-full-${SLURM_JOB_ID}-XXXXXX")
CLIMATE_FULL_WRAPPER_SHA256=$(sha256sum "$0" | cut -d' ' -f1)
test "${CLIMATE_FULL_WRAPPER_SHA256}" = "$(tar -xOf "${CLIMATE_OPERATOR_TAR}" hpc/budget_agent_full.sbatch | sha256sum | cut -d' ' -f1)"
export CLIMATE_FULL_TASK_ROOT CLIMATE_FULL_SCRATCH_PARENT CLIMATE_FULL_WRAPPER_SHA256
export CLIMATE_OPERATOR_GIT CLIMATE_OPERATOR_SHA256 CLIMATE_FULL_RELEASE_ID
export PYTHONDONTWRITEBYTECODE=1 PYTHONNOUSERSITE=1
python - <<'PY'
import os, pathlib, tarfile
target = pathlib.Path(os.environ['CLIMATE_FULL_TASK_ROOT']) / 'operator.py'
with tarfile.open(os.environ['CLIMATE_OPERATOR_TAR']) as archive:
    member = archive.getmember('scripts/run_budget_agent_full_operator.py')
    assert member.isfile() and member.size < 100000
    with target.open('xb') as output:
        output.write(archive.extractfile(member).read())
PY
# Replacing the batch shell lets Slurm B:USR1 reach the operator signal handler.
exec python "${CLIMATE_FULL_TASK_ROOT}/operator.py"
