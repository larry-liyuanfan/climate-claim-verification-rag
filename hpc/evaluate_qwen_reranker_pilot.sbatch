#!/bin/bash
#SBATCH --job-name=climate-qwen-rerank-pilot
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=00:20:00
#SBATCH --output=slurm-%x-%j.out
#SBATCH --error=slurm-%x-%j.err

set -euo pipefail
: "${DATA_DIR:?set DATA_DIR}"
: "${ARTIFACT_DIR:?set ARTIFACT_DIR}"
: "${NATIVE_ENV:?set NATIVE_ENV}"
: "${REPO_DIR:?set REPO_DIR}"
: "${OUTPUT_DIR:?set a new pilot output directory}"
EXPERIMENT_CONFIG="${EXPERIMENT_CONFIG:-${REPO_DIR}/configs/five_stage.qwen_reranker.pilot.yaml}"
BOOTSTRAP_SAMPLES="${BOOTSTRAP_SAMPLES:-500}"

cd "${REPO_DIR}"
module purge
module load GCC/11.3.0 OpenMPI/4.1.4 PyTorch/2.1.2-CUDA-12.2.0
source "${NATIVE_ENV}/bin/activate"
export PYTHONNOUSERSITE=1
export PYTHONPATH="${NATIVE_ENV}/lib/python3.10/site-packages:${PYTHONPATH:-}"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
if [[ "${MODEL_CACHE_MODE:-project}" == "node-local" ]]; then
  NODE_LOCAL_ROOT="${SLURM_TMPDIR:-${TMPDIR:-/tmp}}"
  export HF_HOME="${NODE_LOCAL_ROOT}/climate-hf-${SLURM_JOB_ID}"
else
  export HF_HOME="${MODEL_CACHE_ROOT:-${ARTIFACT_DIR}/.cache/huggingface}"
fi
mkdir -p "${HF_HOME}"
export HF_HUB_CACHE="${HF_HOME}/hub"
export TRANSFORMERS_CACHE="${HF_HOME}/transformers"
export HF_HUB_DISABLE_TELEMETRY=1
export PYTHONUNBUFFERED=1
export CLIMATE_GIT_COMMIT="$(git rev-parse HEAD)"

climate-rag evaluate \
  --claims "${DATA_DIR}/dev-claims.json" \
  --experiment-config "${EXPERIMENT_CONFIG}" \
  --bootstrap-samples "${BOOTSTRAP_SAMPLES}" \
  --output-dir "${OUTPUT_DIR}"
sync
