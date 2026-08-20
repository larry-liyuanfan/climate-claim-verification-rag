#!/bin/bash
#SBATCH --job-name=climate-embed-lora
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=00:30:00
#SBATCH --output=slurm-%x-%j.out
#SBATCH --error=slurm-%x-%j.err

set -euo pipefail
: "${REPO_DIR:?set detached, exact-SHA REPO_DIR}"
: "${DATASET_DIR:?set DATASET_DIR from prepare_embedding_training.sbatch}"
: "${OUTPUT_DIR:?set a new OUTPUT_DIR}"
: "${ARTIFACT_ROOT:?set project ARTIFACT_ROOT for model caches}"

cd "${REPO_DIR}"
module purge
module load GCC/11.3.0 OpenMPI/4.1.4 PyTorch/2.1.2-CUDA-12.2.0
NODE_LOCAL_ROOT="${SLURM_TMPDIR:-${TMPDIR:-/tmp}}"
RUNTIME_ENV="${SWIFT_ENV:-${NODE_LOCAL_ROOT}/climate-swift-venv-3.9.3}"
if [[ ! -x "${RUNTIME_ENV}/bin/swift" ]]; then
  python -m venv --system-site-packages "${RUNTIME_ENV}"
  export PIP_CACHE_DIR="${NODE_LOCAL_ROOT}/climate-swift-pip-cache"
  "${RUNTIME_ENV}/bin/python" -m pip install --no-input "ms-swift==3.9.3"
fi
source "${RUNTIME_ENV}/bin/activate"
export PYTHONNOUSERSITE=1
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
export HF_HOME="${ARTIFACT_ROOT}/.cache/huggingface"
export HF_HUB_CACHE="${HF_HOME}/hub"
export TRANSFORMERS_CACHE="${HF_HOME}/transformers"
export HF_HUB_DISABLE_TELEMETRY=1
export INFONCE_TEMPERATURE="${INFONCE_TEMPERATURE:-0.05}"
export INFONCE_USE_BATCH=true
export INFONCE_HARD_NEGATIVES="${INFONCE_HARD_NEGATIVES:-4}"
export INFONCE_MASK_FAKE_NEGATIVE=true
mkdir -p "${HF_HOME}" "${OUTPUT_DIR}"

# This is a bounded resource/contract pilot, not a resume-quality effectiveness run.
# The official Qwen3-Embedding/ms-swift recipe uses task_type=embedding,
# InfoNCE and LoRA. Promotion requires a separate held-out retrieval gate.
/usr/bin/time -v swift sft \
  --model Qwen/Qwen3-Embedding-0.6B \
  --model_type qwen3_emb \
  --task_type embedding \
  --tuner_type lora \
  --lora_rank 8 \
  --lora_alpha 32 \
  --target_modules all-linear \
  --dataset "${DATASET_DIR}/train.jsonl" \
  --val_dataset "${DATASET_DIR}/eval.jsonl" \
  --split_dataset_ratio 0 \
  --torch_dtype bfloat16 \
  --max_length 512 \
  --max_steps "${MAX_STEPS:-20}" \
  --per_device_train_batch_size 8 \
  --per_device_eval_batch_size 8 \
  --gradient_accumulation_steps 4 \
  --learning_rate 5e-5 \
  --warmup_ratio 0.05 \
  --loss_type infonce \
  --eval_strategy steps \
  --eval_steps 10 \
  --save_steps 10 \
  --save_total_limit 2 \
  --logging_steps 1 \
  --dataloader_num_workers 4 \
  --dataset_num_proc 4 \
  --dataloader_drop_last true \
  --output_dir "${OUTPUT_DIR}"
