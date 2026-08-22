#!/bin/bash
#SBATCH --job-name=climate-eval-native
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=128G
#SBATCH --time=24:00:00
#SBATCH --output=slurm-%x-%j.out

set -euo pipefail
: "${DATA_DIR:?set DATA_DIR}"
: "${ARTIFACT_DIR:?set ARTIFACT_DIR}"
: "${NATIVE_ENV:?set NATIVE_ENV}"
: "${REPO_DIR:?set REPO_DIR}"

module purge
module load GCC/11.3.0 OpenMPI/4.1.4 PyTorch/2.1.2-CUDA-12.2.0
source "${NATIVE_ENV}/bin/activate"
export PYTHONNOUSERSITE=1
export PYTHONPATH="${NATIVE_ENV}/lib/python3.10/site-packages:${PYTHONPATH:-}"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
export HF_HOME="${ARTIFACT_DIR}/.cache/huggingface"
export HF_HUB_CACHE="${HF_HOME}/hub"
export TRANSFORMERS_CACHE="${HF_HOME}/transformers"

climate-rag evaluate \
  --claims "${DATA_DIR}/dev-claims.json" \
  --experiment-config "${REPO_DIR}/configs/five_stage.native.yaml" \
  --bootstrap-samples 5000 \
  --output-dir "${ARTIFACT_DIR}/five-stage"
