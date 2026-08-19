#!/bin/bash
#SBATCH --job-name=climate-ltr-data-native
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

module purge
module load GCC/11.3.0 OpenMPI/4.1.4 PyTorch/2.1.2-CUDA-12.2.0
source "${NATIVE_ENV}/bin/activate"
export PYTHONNOUSERSITE=1
export PYTHONPATH="${NATIVE_ENV}/lib/python3.10/site-packages:${PYTHONPATH:-}"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
export HF_HOME="${ARTIFACT_DIR}/.cache/huggingface"
export HF_HUB_CACHE="${HF_HOME}/hub"
export TRANSFORMERS_CACHE="${HF_HOME}/transformers"

climate-rag mine-negatives \
  --claims "${DATA_DIR}/train-claims.json" \
  --bm25-index "${ARTIFACT_DIR}/bm25/bm25.pkl.gz" \
  --dense-index "${ARTIFACT_DIR}/dense-hnsw/dense" \
  --device cpu \
  --recall-k 1000 \
  --limit 20 \
  --output-dir "${ARTIFACT_DIR}/ltr-data"
