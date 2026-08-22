#!/bin/bash
#SBATCH --job-name=climate-dense-model-gate
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
: "${NATIVE_ENV:?set NATIVE_ENV}"
: "${REPO_DIR:?set REPO_DIR}"
: "${OUTPUT_DIR:?set a new output directory}"

cd "${REPO_DIR}"
module purge
module load GCC/11.3.0 OpenMPI/4.1.4 PyTorch/2.1.2-CUDA-12.2.0
source "${NATIVE_ENV}/bin/activate"
export PYTHONNOUSERSITE=1
export PYTHONPATH="${NATIVE_ENV}/lib/python3.10/site-packages:${PYTHONPATH:-}"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
NODE_LOCAL_ROOT="${SLURM_TMPDIR:-${TMPDIR:-/tmp}}"
export HF_HOME="${NODE_LOCAL_ROOT}/climate-dense-gate-${SLURM_JOB_ID}"
export HF_HUB_CACHE="${HF_HOME}/hub"
export TRANSFORMERS_CACHE="${HF_HOME}/transformers"
export HF_HUB_DISABLE_TELEMETRY=1
export PYTHONUNBUFFERED=1
mkdir -p "${HF_HOME}" "${OUTPUT_DIR}"

/usr/bin/time -v python scripts/evaluate_dense_model_gate.py \
  --claims "${DATA_DIR}/dev-claims.json" \
  --evidence "${DATA_DIR}/evidence.json" \
  --output-dir "${OUTPUT_DIR}" \
  --sample-size "${SAMPLE_SIZE:-10000}" \
  --claim-limit "${CLAIM_LIMIT:-8}" \
  --truncate-dim "${TRUNCATE_DIM:-1024}" \
  --batch-size "${BATCH_SIZE:-16}" \
  --bootstrap-samples "${BOOTSTRAP_SAMPLES:-500}"
sync
