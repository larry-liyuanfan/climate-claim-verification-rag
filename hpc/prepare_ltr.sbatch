#!/bin/bash
#SBATCH --job-name=climate-ltr-data
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --time=06:00:00
#SBATCH --output=slurm-%x-%j.out

set -euo pipefail
: "${REPO_DIR:?set REPO_DIR}"
DATA_DIR="${DATA_DIR:-/data/gpfs/projects/punim2936/nlp/COMP90042_2026-main/data}"
ARTIFACT_DIR="${ARTIFACT_DIR:-/data/gpfs/projects/punim2936/climate-rag-artifacts}"
: "${CONTAINER_IMAGE:?set CONTAINER_IMAGE}"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
module load Apptainer/1.4.5

apptainer exec --nv \
  --bind "${REPO_DIR}:/workspace:ro,${DATA_DIR}:/data:ro,${ARTIFACT_DIR}:/artifacts" \
  "${CONTAINER_IMAGE}" \
  climate-rag mine-negatives \
    --claims /data/train-claims.json \
    --bm25-index /artifacts/bm25/bm25.pkl.gz \
    --dense-index /artifacts/dense-hnsw/dense \
    --device cuda \
    --recall-k 1000 \
    --limit 20 \
    --output-dir /artifacts/ltr-data
