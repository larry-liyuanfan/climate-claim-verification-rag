#!/bin/bash
#SBATCH --job-name=climate-embed-data
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=00:10:00
#SBATCH --output=slurm-%x-%j.out
#SBATCH --error=slurm-%x-%j.err

set -euo pipefail
: "${DATA_DIR:?set DATA_DIR}"
: "${NATIVE_ENV:?set NATIVE_ENV}"
: "${REPO_DIR:?set detached, exact-SHA REPO_DIR}"
: "${HARD_NEGATIVES:?set HARD_NEGATIVES}"
: "${OUTPUT_DIR:?set a new OUTPUT_DIR}"

cd "${REPO_DIR}"
source "${NATIVE_ENV}/bin/activate"
export PYTHONNOUSERSITE=1
export PYTHONUNBUFFERED=1

python scripts/prepare_embedding_training.py \
  --claims "${DATA_DIR}/train-claims.json" \
  --evidence "${DATA_DIR}/evidence.json" \
  --hard-negatives "${HARD_NEGATIVES}" \
  --output-dir "${OUTPUT_DIR}" \
  --negatives-per-positive "${NEGATIVES_PER_POSITIVE:-4}" \
  --eval-ratio "${EVAL_RATIO:-0.1}" \
  --split-seed "${SPLIT_SEED:-45}"
