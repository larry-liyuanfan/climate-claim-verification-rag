#!/bin/bash
#SBATCH --job-name=climate-bm25-native
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --output=slurm-%x-%j.out

set -euo pipefail
: "${DATA_DIR:?set DATA_DIR}"
: "${ARTIFACT_DIR:?set ARTIFACT_DIR}"
: "${NATIVE_ENV:?set NATIVE_ENV}"

module purge
module load GCC/11.3.0 OpenMPI/4.1.4 PyTorch/2.1.2-CUDA-12.2.0
source "${NATIVE_ENV}/bin/activate"
unset PYTHONPATH
export PYTHONNOUSERSITE=1
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
climate-rag index \
  --evidence "${DATA_DIR}/evidence.json" \
  --backend bm25 \
  --output-dir "${ARTIFACT_DIR}/bm25"
