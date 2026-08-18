#!/bin/bash
#SBATCH --job-name=climate-bm25
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --output=slurm-%x-%j.out

set -euo pipefail
: "${REPO_DIR:?set REPO_DIR}"
DATA_DIR="${DATA_DIR:-/data/gpfs/projects/punim2936/nlp/COMP90042_2026-main/data}"
ARTIFACT_DIR="${ARTIFACT_DIR:-/data/gpfs/projects/punim2936/climate-rag-artifacts}"
: "${CONTAINER_IMAGE:?set CONTAINER_IMAGE}"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
module load GCCcore/13.3.0 Apptainer/1.4.5

apptainer exec \
  --bind "${REPO_DIR}:/workspace:ro,${DATA_DIR}:/data:ro,${ARTIFACT_DIR}:/artifacts" \
  "${CONTAINER_IMAGE}" \
  climate-rag index \
    --evidence /data/evidence.json \
    --backend bm25 \
    --output-dir /artifacts/bm25
