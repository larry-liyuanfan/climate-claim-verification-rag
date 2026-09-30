#!/bin/bash
# CPU-prepared only; requires a separate exact-source coordinator release.
#SBATCH --job-name=climate-component-diag
#SBATCH --account=punim2936
#SBATCH --partition=gpu-a100,gpu-a100-short
#SBATCH --gres=gpu:A100:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --tmp=30G
#SBATCH --time=01:40:00
#SBATCH --no-requeue
#SBATCH --signal=B:USR1@90
set -euo pipefail
umask 077
: "${SLURM_JOB_ID:?allocated job required}"
: "${CLIMATE_SOURCE_TAR:?}"
: "${CLIMATE_SOURCE_SHA256:?}"
: "${CLIMATE_SOURCE_GIT:?}"
test "${CLIMATE_COMPONENT_RELEASE:-}" = scifact-component-single-attempt-20261001-v1
ROOT=/data/gpfs/projects/punim2936/portfolio_20260903/climate-public-retrieval-v2
case "$(readlink -f "$CLIMATE_SOURCE_TAR")" in "${ROOT}/envs/"*) ;; *) exit 90 ;; esac
test "$(sha256sum "$CLIMATE_SOURCE_TAR" | cut -d' ' -f1)" = "$CLIMATE_SOURCE_SHA256"
# BEGIN SOURCE_ARCHIVE_GUARD
test "$(tar -tf "$CLIMATE_SOURCE_TAR" | grep -cx 'SOURCE_REVISION')" -eq 1
test "$(tar -xOf "$CLIMATE_SOURCE_TAR" SOURCE_REVISION | wc -c)" -eq 41
test "$(tar -xOf "$CLIMATE_SOURCE_TAR" SOURCE_REVISION)" = "$CLIMATE_SOURCE_GIT"
test "$(sha256sum "$0" | cut -d' ' -f1)" = "$(tar -xOf "$CLIMATE_SOURCE_TAR" hpc/scifact_components.sbatch | sha256sum | cut -d' ' -f1)"
# END SOURCE_ARCHIVE_GUARD
module purge
module load GCC/11.3.0 OpenMPI/4.1.4 PyTorch/2.1.2-CUDA-12.2.0
CLIMATE_SCRATCH_PARENT="${SLURM_TMPDIR:-${TMPDIR:-}}"
case "$(readlink -f "$CLIMATE_SCRATCH_PARENT")" in /tmp|/tmp/*|/var/tmp|/var/tmp/*|/jobfs/*) ;; *) exit 91 ;; esac
CLIMATE_COMPONENT_WORK=$(mktemp -d "${CLIMATE_SCRATCH_PARENT}/climate-component-${SLURM_JOB_ID}-XXXXXX")
export CLIMATE_COMPONENT_WORK PYTHONDONTWRITEBYTECODE=1 PYTHONNOUSERSITE=1 PYTHONHASHSEED=0
python - <<'PY'
import os, pathlib, tarfile
target = pathlib.Path(os.environ['CLIMATE_COMPONENT_WORK']) / 'source'
target.mkdir(mode=0o700)
with tarfile.open(os.environ['CLIMATE_SOURCE_TAR']) as archive:
    paths = set()
    for member in archive.getmembers():
        path = (target / member.name).resolve()
        if not path.is_relative_to(target) or not (member.isfile() or member.isdir()) or path in paths:
            raise ValueError('unsafe source archive')
        paths.add(path)
    archive.extractall(target)
PY
export PYTHONPATH="$CLIMATE_COMPONENT_WORK/source/src${PYTHONPATH:+:$PYTHONPATH}"
exec python "$CLIMATE_COMPONENT_WORK/source/scripts/run_scifact_component_operator.py"
