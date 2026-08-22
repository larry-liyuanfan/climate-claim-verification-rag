#!/bin/bash
#SBATCH --job-name=climate-embed-adapter-full
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --partition=gpu-h100
#SBATCH --gpus=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=01:15:00
#SBATCH --output=slurm-%x-%j.out
#SBATCH --error=slurm-%x-%j.err

set -euo pipefail
: "${REPO_DIR:?set detached, exact-SHA REPO_DIR}"
: "${DATA_DIR:?set DATA_DIR}"
: "${BASE_INDEX:?set verified full-corpus base FlatIP index}"
: "${BASE_INDEX_METADATA:?set base index metadata with doc_ids}"
: "${BASE_EMBEDDING_METADATA:?set base embedding count/hash metadata}"
: "${ADAPTER_DIR:?set verified LoRA checkpoint}"
: "${OUTPUT_DIR:?set a new OUTPUT_DIR}"
: "${NATIVE_ENV:?set existing native climate virtual environment}"
: "${ARTIFACT_ROOT:?set project artifact/cache root}"

cd "${REPO_DIR}"
module purge
module load GCC/11.3.0 OpenMPI/4.1.4 PyTorch/2.1.2-CUDA-12.2.0
source "${NATIVE_ENV}/bin/activate"
export PYTHONNOUSERSITE=1
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
export HF_HOME="${ARTIFACT_ROOT}/.cache/huggingface"
export HF_HUB_CACHE="${HF_HOME}/hub"
export TRANSFORMERS_CACHE="${HF_HOME}/transformers"
export HF_HUB_DISABLE_TELEMETRY=1
mkdir -p "${HF_HOME}" "${OUTPUT_DIR}"
source hpc/adapter_runtime.sh
setup_climate_adapter_runtime

/usr/bin/time -v python scripts/evaluate_embedding_adapter_full_gate.py \
  --claims "${DATA_DIR}/dev-claims.json" \
  --evidence "${DATA_DIR}/evidence.json" \
  --base-index "${BASE_INDEX}" \
  --base-index-metadata "${BASE_INDEX_METADATA}" \
  --base-embedding-metadata "${BASE_EMBEDDING_METADATA}" \
  --adapter "${ADAPTER_DIR}" \
  --output-dir "${OUTPUT_DIR}" \
  --batch-size "${BATCH_SIZE:-64}" \
  --bootstrap-samples "${BOOTSTRAP_SAMPLES:-5000}"
sync
