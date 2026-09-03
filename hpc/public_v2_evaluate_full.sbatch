#!/bin/bash
#SBATCH --job-name=climate-v2-full
#SBATCH --account=punim2936
#SBATCH --partition=gpu-a100-mig
#SBATCH --gres=gpu:1g.20gb:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=00:30:00
#SBATCH --tmp=30G

set -euo pipefail
: "${CLIMATE_V2_ROOT:?Set the isolated public-v2 root}"
: "${REPO_DIR:?Set the exact detached repository}"
: "${CLIMATE_GIT_COMMIT:?Set the exact pushed Git commit}"
: "${PERSIST_DIR:?Set the pre-existing archive destination directory}"

test "$(readlink -f "${CLIMATE_V2_ROOT}")" = \
  "/data/gpfs/projects/punim2936/portfolio_20260903/climate-public-retrieval-v2"
test "$(git -C "${REPO_DIR}" rev-parse HEAD)" = "${CLIMATE_GIT_COMMIT}"
test -d "${PERSIST_DIR}"

module purge
module load GCC/11.3.0 OpenMPI/4.1.4 PyTorch/2.1.2-CUDA-12.2.0
source "${REPO_DIR}/hpc/public_v2_runtime.sh"
public_v2_activate_runtime
NODE_TMP="$(public_v2_node_tmp)"
WORK_ROOT="${NODE_TMP}/climate-v2-full-${SLURM_ARRAY_JOB_ID}-${SLURM_ARRAY_TASK_ID}"
test ! -e "${WORK_ROOT}"
mkdir -p "${WORK_ROOT}"
public_v2_unpack_stage "${PERSIST_DIR}/preflight" "${WORK_ROOT}/preflight"
public_v2_unpack_stage "${PERSIST_DIR}/base" "${WORK_ROOT}/base"
public_v2_unpack_stage \
  "${PERSIST_DIR}/pilot-selection" "${WORK_ROOT}/pilot-selection"
PILOT_SELECTION="${WORK_ROOT}/pilot-selection/pilot_selection.json"
export PYTHONPATH="${REPO_DIR}/src"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
cd "${REPO_DIR}"

CONFIG_ID=$(python - "${PILOT_SELECTION}" "${SLURM_ARRAY_TASK_ID}" <<'PY'
import json
import sys
from pathlib import Path

selection = json.loads(Path(sys.argv[1]).read_text())
print(selection["selected_for_full"][int(sys.argv[2])])
PY
)
public_v2_unpack_stage \
  "${PERSIST_DIR}/pilot-${CONFIG_ID}" "${WORK_ROOT}/pilot"
mapfile -t CHECKPOINTS < <(
  find "${WORK_ROOT}/pilot/training" -mindepth 1 -maxdepth 1 -type d \
    -name 'checkpoint-*' -print | sort -V
)
test "${#CHECKPOINTS[@]}" -eq 1
OUTPUT_DIR="${WORK_ROOT}/full"

/usr/bin/time -v python scripts/public_v2_evaluate_adapter.py \
  --protocol configs/public_retrieval_v2.json \
  --policy configs/public_evaluation_policy.json \
  --prepared-dir "${WORK_ROOT}/preflight/data/climate-fever-v2/prepared" \
  --selection-dir "${WORK_ROOT}/preflight/data/climate-fever-v2/selection-only" \
  --base-dir "${WORK_ROOT}/base" \
  --adapter-dir "${CHECKPOINTS[0]}" \
  --training-record "${WORK_ROOT}/pilot/adapter_config.json" \
  --adapter-config-id "${CONFIG_ID}" \
  --mode full \
  --output-dir "${OUTPUT_DIR}/evaluation" \
  --device cuda \
  --batch-size 32
public_v2_pack_stage \
  "${OUTPUT_DIR}" "${PERSIST_DIR}/full-${CONFIG_ID}" "full-${CONFIG_ID}"
sync
