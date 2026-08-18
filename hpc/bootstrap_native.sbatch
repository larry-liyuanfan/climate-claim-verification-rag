#!/bin/bash
#SBATCH --job-name=climate-native-env
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=01:00:00
#SBATCH --output=slurm-%x-%j.out

set -euo pipefail
: "${REPO_DIR:?set REPO_DIR}"
: "${NATIVE_ENV:?set NATIVE_ENV}"
: "${PIP_CACHE_DIR:?set PIP_CACHE_DIR}"

module purge
module load GCC/11.3.0 OpenMPI/4.1.4 PyTorch/2.1.2-CUDA-12.2.0
python -m venv --system-site-packages "${NATIVE_ENV}"
source "${NATIVE_ENV}/bin/activate"
python -m pip install --upgrade pip
python -m pip install --no-cache-dir \
  "transformers>=4.51,<5" \
  "sentence-transformers>=4,<6" \
  "faiss-cpu>=1.8" \
  "lightgbm>=4.3" \
  "pydantic>=2.7" \
  "PyYAML>=6.0" \
  "fastapi>=0.111" \
  "uvicorn>=0.30"
python -m pip install --no-deps -e "${REPO_DIR}"
python -c "import torch, faiss, lightgbm, sentence_transformers; print(torch.__version__, torch.version.cuda)"
