#!/bin/bash
#SBATCH --job-name=climate-ann-native
#SBATCH --array=0-1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=128G
#SBATCH --time=24:00:00
#SBATCH --output=slurm-%x-%A_%a.out

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

KINDS=(hnsw ivfpq)
KIND="${KINDS[$SLURM_ARRAY_TASK_ID]}"
EXTRA=()
if [[ "${KIND}" == "hnsw" ]]; then
  EXTRA=(--hnsw-m 32 --hnsw-ef-construction 200)
else
  EXTRA=(--nlist 4096 --pq-m 32 --nbits 8 --nprobe 32 --ivf-train-size 200000)
fi

climate-rag index \
  --evidence "${DATA_DIR}/evidence.json" \
  --backend dense \
  --encoder sentence-transformer \
  --model Qwen/Qwen3-Embedding-0.6B \
  --device cpu \
  --ann "${KIND}" \
  --embeddings "${ARTIFACT_DIR}/qwen3-0.6b.npy" \
  --output-dir "${ARTIFACT_DIR}/dense-${KIND}" \
  "${EXTRA[@]}"
