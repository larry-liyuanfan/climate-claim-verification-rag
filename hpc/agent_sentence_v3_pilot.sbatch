#!/bin/bash
# Preparation only; separately released source/protocol/grammar hashes required.
#SBATCH --job-name=climate-agent-sentence-v3
#SBATCH --account=punim2936
#SBATCH --partition=gpu-a100,gpu-a100-short
#SBATCH --gres=gpu:A100:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --tmp=30G
#SBATCH --time=01:00:00
#SBATCH --no-requeue
#SBATCH --signal=B:USR1@90
set -euo pipefail
umask 077
: "${SLURM_JOB_ID:?allocated job required}"
: "${CLIMATE_SOURCE_TAR:?}"
: "${CLIMATE_SOURCE_SHA256:?}"
: "${CLIMATE_SOURCE_GIT:?}"
: "${CLIMATE_V3_GRAMMAR_TAR:?}"
: "${CLIMATE_V3_GRAMMAR_SHA256:?}"
: "${CLIMATE_V3_PROTOCOL_SHA256:?}"
test "${CLIMATE_V3_RELEASE:-}" = climate-sentence-v3-20260930-pilot-r1
ROOT=/data/gpfs/projects/punim2936/portfolio_20260903/climate-public-retrieval-v2
case "$(readlink -f "${CLIMATE_SOURCE_TAR}")" in "${ROOT}/envs/"*) ;; *) exit 90 ;; esac
case "$(readlink -f "${CLIMATE_V3_GRAMMAR_TAR}")" in "${ROOT}/envs/"*) ;; *) exit 90 ;; esac
test "$(sha256sum "${CLIMATE_SOURCE_TAR}" | cut -d' ' -f1)" = "${CLIMATE_SOURCE_SHA256}"
test "$(sha256sum "${CLIMATE_V3_GRAMMAR_TAR}" | cut -d' ' -f1)" = "${CLIMATE_V3_GRAMMAR_SHA256}"
test "$(tar -xOf "${CLIMATE_SOURCE_TAR}" SOURCE_REVISION | tr -d '\r\n')" = "${CLIMATE_SOURCE_GIT}"
test "$(tar -xOf "${CLIMATE_SOURCE_TAR}" configs/agent_sentence_v3_pilot_20260930.json | sha256sum | cut -d' ' -f1)" = "${CLIMATE_V3_PROTOCOL_SHA256}"
test "$(sha256sum "$0" | cut -d' ' -f1)" = "$(tar -xOf "${CLIMATE_SOURCE_TAR}" hpc/agent_sentence_v3_pilot.sbatch | sha256sum | cut -d' ' -f1)"
module purge
module load GCC/11.3.0 OpenMPI/4.1.4 PyTorch/2.1.2-CUDA-12.2.0
CLIMATE_SCRATCH_PARENT="${SLURM_TMPDIR:-${TMPDIR:-}}"
case "$(readlink -f "${CLIMATE_SCRATCH_PARENT}")" in
  /tmp|/tmp/*|/var/tmp|/var/tmp/*|/jobfs/*) ;; *) exit 91 ;;
esac
CLIMATE_V3_WORK=$(mktemp -d "${CLIMATE_SCRATCH_PARENT}/climate-sentence-v3-${SLURM_JOB_ID}-XXXXXX")
export CLIMATE_V3_WORK CLIMATE_SOURCE_TAR CLIMATE_SOURCE_GIT CLIMATE_SOURCE_SHA256
export CLIMATE_V3_RELEASE CLIMATE_V3_GRAMMAR_TAR CLIMATE_V3_GRAMMAR_SHA256 CLIMATE_V3_PROTOCOL_SHA256
export PYTHONDONTWRITEBYTECODE=1 PYTHONNOUSERSITE=1
python - <<'PY'
import os, pathlib, tarfile
target = pathlib.Path(os.environ['CLIMATE_V3_WORK']) / 'source'
target.mkdir(mode=0o700)
with tarfile.open(os.environ['CLIMATE_SOURCE_TAR']) as archive:
    names = set()
    for member in archive.getmembers():
        path = (target / member.name).resolve()
        if not path.is_relative_to(target) or not (member.isfile() or member.isdir()) or path in names:
            raise ValueError('unsafe source archive')
        names.add(path)
    archive.extractall(target)
PY
exec python "${CLIMATE_V3_WORK}/source/scripts/run_sentence_agent_v3_operator.py"
