#!/bin/bash
#SBATCH --job-name=climate-ltr-native
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --output=slurm-%x-%j.out

set -euo pipefail
: "${ARTIFACT_DIR:?set ARTIFACT_DIR}"
: "${NATIVE_ENV:?set NATIVE_ENV}"

module purge
module load GCC/11.3.0 OpenMPI/4.1.4 PyTorch/2.1.2-CUDA-12.2.0
source "${NATIVE_ENV}/bin/activate"
export PYTHONNOUSERSITE=1
export PYTHONPATH="${NATIVE_ENV}/lib/python3.10/site-packages:${PYTHONPATH:-}"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK}"

climate-rag train-fusion \
  --features "${ARTIFACT_DIR}/ltr-data/ltr_features.jsonl" \
  --algorithm lambdamart \
  --output-dir "${ARTIFACT_DIR}/ltr"
