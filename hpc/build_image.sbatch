#!/bin/bash
#SBATCH --job-name=climate-image
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=01:00:00
#SBATCH --output=slurm-%x-%j.out

set -euo pipefail
: "${REPO_DIR:?set REPO_DIR}"
: "${ARTIFACT_DIR:?set ARTIFACT_DIR}"

mkdir -p "${ARTIFACT_DIR}/apptainer-cache" "${ARTIFACT_DIR}/apptainer-tmp"
export APPTAINER_CACHEDIR="${ARTIFACT_DIR}/apptainer-cache"
export APPTAINER_TMPDIR="${ARTIFACT_DIR}/apptainer-tmp"
CONTAINER_IMAGE="${CONTAINER_IMAGE:-${ARTIFACT_DIR}/climate-rag.sif}"

module load GCCcore/13.3.0 Apptainer/1.4.5
cd "${REPO_DIR}"
apptainer build "${CONTAINER_IMAGE}" hpc/Apptainer.def
