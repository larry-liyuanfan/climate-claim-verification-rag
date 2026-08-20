#!/bin/bash
#SBATCH --job-name=climate-qwen-rerank-compare
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --time=00:05:00
#SBATCH --output=slurm-%x-%j.out
#SBATCH --error=slurm-%x-%j.err

set -euo pipefail
: "${DATA_DIR:?set DATA_DIR}"
: "${NATIVE_ENV:?set NATIVE_ENV}"
: "${REPO_DIR:?set REPO_DIR}"
: "${RERANK_RUN_DIR:?set completed Qwen reranker run directory}"
: "${OUTPUT_DIR:?set a new comparison output directory}"

cd "${REPO_DIR}"
module purge
module load GCC/11.3.0 OpenMPI/4.1.4 PyTorch/2.1.2-CUDA-12.2.0
source "${NATIVE_ENV}/bin/activate"
export PYTHONNOUSERSITE=1
export PYTHONPATH="${NATIVE_ENV}/lib/python3.10/site-packages:${PYTHONPATH:-}"
export PYTHONUNBUFFERED=1

climate-rag evaluate \
  --claims "${DATA_DIR}/dev-claims.json" \
  --predictions "${RERANK_RUN_DIR}/predictions_rrf_reranker.json" \
  --baseline-predictions "${RERANK_RUN_DIR}/predictions_rrf.json" \
  --ks 5 \
  --bootstrap-samples 5000 \
  --output-dir "${OUTPUT_DIR}"
sync
