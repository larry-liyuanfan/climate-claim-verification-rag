#!/bin/bash
#SBATCH --job-name=climate-embed-lora
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --partition=gpu-a100-mig
#SBATCH --gres=gpu:1g.20gb:1
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
# EasyBuild exports an older typing_extensions through PYTHONPATH.  Keep the
# module-provided CUDA/PyTorch path, but put the node-local venv first so its
# current typing_extensions wins.  Dropping PYTHONPATH entirely makes pip
# install a CUDA 13 torch wheel that the Spartan driver cannot load.
MODULE_PYTHONPATH="${PYTHONPATH:-}"
unset PYTHONPATH
NODE_LOCAL_ROOT="${SLURM_TMPDIR:-${TMPDIR:-/tmp}}"
RUNTIME_ENV="${SWIFT_ENV:-${NODE_LOCAL_ROOT}/climate-swift-venv-3.9.3}"
if [[ ! -x "${RUNTIME_ENV}/bin/swift" ]]; then
  python -m venv --system-site-packages "${RUNTIME_ENV}"
  export PIP_CACHE_DIR="${NODE_LOCAL_ROOT}/climate-swift-pip-cache"
  export PIP_DEFAULT_TIMEOUT="${PIP_DEFAULT_TIMEOUT:-120}"
  VENV_SITE=$("${RUNTIME_ENV}/bin/python" -c 'import site; print(site.getsitepackages()[0])')
  export PYTHONPATH="${VENV_SITE}${MODULE_PYTHONPATH:+:${MODULE_PYTHONPATH}}"
  "${RUNTIME_ENV}/bin/python" -m pip install --no-input --retries 5 "typing_extensions>=4.12"
  "${RUNTIME_ENV}/bin/python" -m pip install \
    --no-input \
    --retries 5 \
    "transformers==4.51.3" \
    "ms-swift==3.9.3"
fi
source "${RUNTIME_ENV}/bin/activate"
VENV_SITE=$(python -c 'import site; print(site.getsitepackages()[0])')
export PYTHONPATH="${VENV_SITE}${MODULE_PYTHONPATH:+:${MODULE_PYTHONPATH}}"
export PYTHONNOUSERSITE=1
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
export HF_HOME="${ARTIFACT_ROOT}/.cache/huggingface"
export HF_HUB_CACHE="${HF_HOME}/hub"
export TRANSFORMERS_CACHE="${HF_HOME}/transformers"
export HF_HUB_DISABLE_TELEMETRY=1
export MODELSCOPE_CACHE="${ARTIFACT_ROOT}/.cache/modelscope"
export MODELSCOPE_HOME="${ARTIFACT_ROOT}/.cache/modelscope"
export INFONCE_TEMPERATURE="${INFONCE_TEMPERATURE:-0.05}"
export INFONCE_USE_BATCH=true
export INFONCE_HARD_NEGATIVES="${INFONCE_HARD_NEGATIVES:-4}"
export INFONCE_MASK_FAKE_NEGATIVE=true
mkdir -p "${HF_HOME}" "${MODELSCOPE_CACHE}" "${OUTPUT_DIR}"

python - <<'PY'
from typing_extensions import TypeVar
import torch
import transformers
import swift

print(
    "runtime_preflight",
    {"torch": torch.__version__, "transformers": transformers.__version__, "swift": swift.__version__},
    flush=True,
)
assert torch.__version__.startswith("2.1.2"), torch.__version__
assert torch.version.cuda == "12.2", torch.version.cuda
assert torch.cuda.is_available(), "Spartan GPU is not visible to the pinned CUDA runtime"
PY

# This is a bounded resource/contract pilot, not a resume-quality effectiveness run.
# The official Qwen3-Embedding/ms-swift recipe uses task_type=embedding,
# InfoNCE and LoRA. Promotion requires a separate held-out retrieval gate.
/usr/bin/time -v swift sft \
  --model Qwen/Qwen3-Embedding-0.6B \
  --model_type qwen3_emb \
  --task_type embedding \
  --train_type lora \
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
