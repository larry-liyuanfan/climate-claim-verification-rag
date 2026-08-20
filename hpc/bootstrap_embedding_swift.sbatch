#!/bin/bash
#SBATCH --job-name=climate-swift-env
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=12G
#SBATCH --time=00:20:00
#SBATCH --output=slurm-%x-%j.out
#SBATCH --error=slurm-%x-%j.err

set -euo pipefail
: "${SWIFT_ENV:?set project-storage SWIFT_ENV}"
: "${PIP_CACHE_DIR:?set project-storage PIP_CACHE_DIR}"

module purge
module load GCC/11.3.0 OpenMPI/4.1.4 PyTorch/2.1.2-CUDA-12.2.0
export PYTHONNOUSERSITE=1
export PIP_DISABLE_PIP_VERSION_CHECK=1
mkdir -p "${PIP_CACHE_DIR}"

if [[ ! -x "${SWIFT_ENV}/bin/python" ]]; then
  python -m venv --system-site-packages "${SWIFT_ENV}"
fi
"${SWIFT_ENV}/bin/python" -m pip install --upgrade pip
"${SWIFT_ENV}/bin/python" -m pip install "ms-swift==3.9.3"
"${SWIFT_ENV}/bin/python" - <<'PY'
import importlib.metadata

print("ms-swift", importlib.metadata.version("ms-swift"))
PY
"${SWIFT_ENV}/bin/swift" sft --help >/dev/null
