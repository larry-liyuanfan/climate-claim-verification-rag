#!/bin/bash
#SBATCH --job-name=climate-ltr
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --output=slurm-%x-%j.out

set -euo pipefail
: "${REPO_DIR:?set REPO_DIR}"
ARTIFACT_DIR="${ARTIFACT_DIR:-/data/gpfs/projects/punim2936/climate-rag-artifacts}"
: "${CONTAINER_IMAGE:?set CONTAINER_IMAGE}"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
module load apptainer

apptainer exec \
  --bind "${REPO_DIR}:/workspace:ro,${ARTIFACT_DIR}:/artifacts" \
  "${CONTAINER_IMAGE}" \
  climate-rag train-fusion \
    --features /artifacts/ltr-data/ltr_features.jsonl \
    --algorithm lambdamart \
    --output-dir /artifacts/ltr
