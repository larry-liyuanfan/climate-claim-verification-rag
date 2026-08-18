#!/bin/bash
#SBATCH --job-name=climate-dense-native
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=96G
#SBATCH --time=12:00:00
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
climate-rag index \
  --evidence "${DATA_DIR}/evidence.json" \
  --backend dense \
  --encoder sentence-transformer \
  --model Qwen/Qwen3-Embedding-0.6B \
  --device cuda \
  --ann flat \
  --batch-size 64 \
  --embeddings "${ARTIFACT_DIR}/qwen3-0.6b.npy" \
  --output-dir "${ARTIFACT_DIR}/dense-flat"
