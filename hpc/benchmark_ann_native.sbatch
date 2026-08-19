#!/bin/bash
#SBATCH --job-name=climate-ann-bench
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --gres=gpu:1
#SBATCH --mem=128G
#SBATCH --time=06:00:00
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

climate-rag benchmark-ann \
  --claims "${DATA_DIR}/dev-claims.json" \
  --index "flat=${ARTIFACT_DIR}/dense-flat/dense/index.faiss" \
  --index "hnsw=${ARTIFACT_DIR}/dense-hnsw/dense/index.faiss" \
  --index "ivfpq=${ARTIFACT_DIR}/dense-ivfpq/dense/index.faiss" \
  --index-manifest "flat=${ARTIFACT_DIR}/dense-flat/run_manifest.json" \
  --index-manifest "hnsw=${ARTIFACT_DIR}/dense-hnsw/run_manifest.json" \
  --index-manifest "ivfpq=${ARTIFACT_DIR}/dense-ivfpq/run_manifest.json" \
  --model Qwen/Qwen3-Embedding-0.6B \
  --device cuda \
  --batch-size 64 \
  --faiss-threads "${SLURM_CPUS_PER_TASK}" \
  --hnsw-ef-search 64 \
  --ivf-nprobe 32 \
  --output-dir "${ARTIFACT_DIR}/ann-benchmark"
