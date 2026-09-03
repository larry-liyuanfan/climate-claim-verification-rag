#!/bin/bash
#SBATCH --job-name=climate-v2-pilots
#SBATCH --account=punim2936
#SBATCH --partition=gpu-a100-mig
#SBATCH --gres=gpu:1g.20gb:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=01:30:00
#SBATCH --tmp=40G

set -euo pipefail
: "${CLIMATE_V2_ROOT:?Set the isolated public-v2 root}"
: "${REPO_DIR:?Set the exact detached repository}"
: "${CLIMATE_GIT_COMMIT:?Set the exact pushed Git commit}"
: "${PERSIST_DIR:?Set the pre-existing archive destination directory}"

test "$(readlink -f "${CLIMATE_V2_ROOT}")" = \
  "/data/gpfs/projects/punim2936/portfolio_20260903/climate-public-retrieval-v2"
test "$(git -C "${REPO_DIR}" rev-parse HEAD)" = "${CLIMATE_GIT_COMMIT}"
test "${SLURM_ARRAY_TASK_COUNT}" = "6"
test -d "${PERSIST_DIR}"

module purge
module load GCC/11.3.0 OpenMPI/4.1.4 PyTorch/2.1.2-CUDA-12.2.0
source "${REPO_DIR}/hpc/public_v2_runtime.sh"
public_v2_activate_runtime
NODE_TMP="$(public_v2_node_tmp)"
WORK_ROOT="${NODE_TMP}/climate-v2-pilot-${SLURM_ARRAY_JOB_ID}-${SLURM_ARRAY_TASK_ID}"
test ! -e "${WORK_ROOT}"
mkdir -p "${WORK_ROOT}"
public_v2_unpack_stage "${PERSIST_DIR}/preflight" "${WORK_ROOT}/preflight"
public_v2_unpack_stage "${PERSIST_DIR}/base" "${WORK_ROOT}/base"
BASE_DIR="${WORK_ROOT}/base"
TASK_ROOT="${WORK_ROOT}/pilot"
export BASE_DIR
export PYTHONPATH="${REPO_DIR}/src"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
export TOKENIZERS_PARALLELISM=false
cd "${REPO_DIR}"

mapfile -t CONFIG_FIELDS < <(python - "${SLURM_ARRAY_TASK_ID}" <<'PY'
import json
import sys
from pathlib import Path

config = json.loads(Path("configs/public_retrieval_v2.json").read_text())
row = config["adapter_matrix"][int(sys.argv[1])]
print(row["id"])
print(row["max_steps"])
print(row["lora_rank"])
print(row["hard_negatives"])
print(row["temperature"])
print(json.dumps(row, sort_keys=True, separators=(",", ":")))
PY
)
CONFIG_ID="${CONFIG_FIELDS[0]}"
MAX_STEPS="${CONFIG_FIELDS[1]}"
LORA_RANK="${CONFIG_FIELDS[2]}"
HARD_NEGATIVES="${CONFIG_FIELDS[3]}"
INFONCE_TEMPERATURE="${CONFIG_FIELDS[4]}"
CONFIG_JSON="${CONFIG_FIELDS[5]}"
LORA_ALPHA="$((LORA_RANK * 4))"
mkdir -p "${TASK_ROOT}/training"

python - "${TASK_ROOT}/adapter_config.json" "${CONFIG_JSON}" <<'PY'
import hashlib
import json
import os
import sys
from pathlib import Path

target = Path(sys.argv[1])
row = json.loads(sys.argv[2])
dataset = Path(os.environ["BASE_DIR"]) / f"training-n{row['hard_negatives']}" / "train.jsonl"
record = {
    "schema_version": 1,
    "adapter_config": row,
    "lora_alpha": row["lora_rank"] * 4,
    "model": "Qwen/Qwen3-Embedding-0.6B",
    "model_revision": "97b0c614be4d77ee51c0cef4e5f07c00f9eb65b3",
    "dataset_sha256": hashlib.sha256(dataset.read_bytes()).hexdigest(),
    "git_commit": os.environ["CLIMATE_GIT_COMMIT"],
    "slurm_job_id": os.environ["SLURM_ARRAY_JOB_ID"],
    "slurm_array_task_id": os.environ["SLURM_ARRAY_TASK_ID"],
    "resources": "gpu-a100-mig:1g.20gb:1,8CPU,32G,01:30:00",
}
target.write_text(json.dumps(record, indent=2, sort_keys=True) + "\n")
PY

export INFONCE_TEMPERATURE
export INFONCE_USE_BATCH=true
export INFONCE_HARD_NEGATIVES="${HARD_NEGATIVES}"
export INFONCE_MASK_FAKE_NEGATIVE=true
DATASET_DIR="${BASE_DIR}/training-n${HARD_NEGATIVES}"

/usr/bin/time -v swift sft \
  --model Qwen/Qwen3-Embedding-0.6B \
  --model_revision 97b0c614be4d77ee51c0cef4e5f07c00f9eb65b3 \
  --model_type qwen3_emb \
  --task_type embedding \
  --train_type lora \
  --lora_rank "${LORA_RANK}" \
  --lora_alpha "${LORA_ALPHA}" \
  --target_modules all-linear \
  --dataset "${DATASET_DIR}/train.jsonl" \
  --val_dataset "${DATASET_DIR}/contract-eval.jsonl" \
  --split_dataset_ratio 0 \
  --torch_dtype bfloat16 \
  --max_length 512 \
  --max_steps "${MAX_STEPS}" \
  --per_device_train_batch_size 8 \
  --per_device_eval_batch_size 8 \
  --gradient_accumulation_steps 4 \
  --learning_rate 5e-5 \
  --warmup_ratio 0.05 \
  --loss_type infonce \
  --eval_strategy steps \
  --eval_steps "${MAX_STEPS}" \
  --save_steps "${MAX_STEPS}" \
  --save_total_limit 1 \
  --logging_steps 10 \
  --dataloader_num_workers 4 \
  --dataset_num_proc 4 \
  --dataloader_drop_last true \
  --output_dir "${TASK_ROOT}/training"

mapfile -t CHECKPOINTS < <(
  find "${TASK_ROOT}/training" -mindepth 1 -maxdepth 1 -type d \
    -name 'checkpoint-*' -print | sort -V
)
test "${#CHECKPOINTS[@]}" -eq 1
ADAPTER_DIR="${CHECKPOINTS[0]}"

/usr/bin/time -v python scripts/public_v2_evaluate_adapter.py \
  --protocol configs/public_retrieval_v2.json \
  --policy configs/public_evaluation_policy.json \
  --prepared-dir "${WORK_ROOT}/preflight/data/climate-fever-v2/prepared" \
  --selection-dir "${WORK_ROOT}/preflight/data/climate-fever-v2/selection-only" \
  --base-dir "${BASE_DIR}" \
  --adapter-dir "${ADAPTER_DIR}" \
  --training-record "${TASK_ROOT}/adapter_config.json" \
  --adapter-config-id "${CONFIG_ID}" \
  --mode pilot \
  --output-dir "${TASK_ROOT}/evaluation" \
  --device cuda \
  --batch-size 32
public_v2_pack_stage \
  "${TASK_ROOT}" "${PERSIST_DIR}/pilot-${CONFIG_ID}" "pilot-${CONFIG_ID}"
sync
