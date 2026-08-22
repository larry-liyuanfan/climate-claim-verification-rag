#!/bin/bash
#SBATCH --job-name=climate-ltr-supported
#SBATCH --account=punim2936
#SBATCH --partition=sapphire
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=24G
#SBATCH --time=00:30:00
#SBATCH --output=/data/gpfs/projects/punim2936/portfolio_20260818/climate-artifacts/logs/%x-%j.out

set -euo pipefail
: "${REPO_DIR:?Set the Spartan Climate checkout}"
: "${CLIMATE_GIT_COMMIT:?Set an exact pushed Git commit}"
: "${DATA_DIR:?Set the restricted course data directory}"
: "${ARTIFACT_DIR:?Set the existing verified Climate artifact root}"
: "${OUTPUT_ROOT:?Set a new candidate-supported LTR artifact root}"
: "${NATIVE_ENV:?Set the verified native Python environment}"

git -C "${REPO_DIR}" cat-file -e "${CLIMATE_GIT_COMMIT}^{commit}"
CODE_ROOT="${SLURM_TMPDIR:-${TMPDIR:-/tmp}}/climate-ltr-supported-${SLURM_JOB_ID:?}"
trap 'rm -rf -- "${CODE_ROOT}"' EXIT
# Copy the object database into node-local storage. A shared clone leaves an
# alternates pointer back to GPFS and previously failed during checkout with an
# intermittently unreadable tree even though the source repository passed fsck.
git clone --no-local --no-checkout "${REPO_DIR}" "${CODE_ROOT}"
git -C "${CODE_ROOT}" checkout --detach "${CLIMATE_GIT_COMMIT}"
cd "${CODE_ROOT}"

module purge
module load GCC/11.3.0 OpenMPI/4.1.4 PyTorch/2.1.2-CUDA-12.2.0
source "${NATIVE_ENV}/bin/activate"
export PYTHONNOUSERSITE=1
export PYTHONPATH="${CODE_ROOT}/src:${NATIVE_ENV}/lib/python3.10/site-packages:${PYTHONPATH:-}"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
export HF_HOME="${ARTIFACT_DIR}/.cache/huggingface"
export HF_HUB_CACHE="${HF_HOME}/hub"
export TRANSFORMERS_CACHE="${HF_HOME}/transformers"

test ! -e "${OUTPUT_ROOT}"
mkdir -p "${OUTPUT_ROOT}"

climate-rag mine-negatives \
  --claims "${DATA_DIR}/train-claims.json" \
  --bm25-index "${ARTIFACT_DIR}/bm25/bm25.pkl.gz" \
  --dense-index "${ARTIFACT_DIR}/dense-hnsw/dense" \
  --device cpu \
  --recall-k 1000 \
  --rrf-k 60 \
  --limit 100 \
  --output-dir "${OUTPUT_ROOT}/ltr-data"

climate-rag train-fusion \
  --features "${OUTPUT_ROOT}/ltr-data/ltr_features.jsonl" \
  --algorithm lambdamart \
  --output-dir "${OUTPUT_ROOT}/ltr"

export LTR_MODEL="${OUTPUT_ROOT}/ltr/ltr_model.txt"
climate-rag evaluate \
  --claims "${DATA_DIR}/dev-claims.json" \
  --experiment-config "${CODE_ROOT}/configs/five_stage.candidate_supported_ltr.yaml" \
  --bootstrap-samples 5000 \
  --output-dir "${OUTPUT_ROOT}/five-stage"
