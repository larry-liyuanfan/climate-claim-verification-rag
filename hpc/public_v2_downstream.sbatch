#!/bin/bash
#SBATCH --job-name=climate-v2-downstream
#SBATCH --account=punim2936
#SBATCH --partition=gpu-a100-mig
#SBATCH --gres=gpu:1g.20gb:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=01:15:00
#SBATCH --tmp=40G

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
WORK_ROOT="${NODE_TMP}/climate-v2-downstream-${SLURM_JOB_ID}"
test ! -e "${WORK_ROOT}"
mkdir -p "${WORK_ROOT}"
public_v2_unpack_stage "${PERSIST_DIR}/preflight" "${WORK_ROOT}/preflight"
public_v2_unpack_stage "${PERSIST_DIR}/base" "${WORK_ROOT}/base"
public_v2_unpack_stage "${PERSIST_DIR}/full-selection" "${WORK_ROOT}/full-selection"
FROZEN_CONFIG="${WORK_ROOT}/full-selection/frozen_config.json"
export PYTHONPATH="${REPO_DIR}/src"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
cd "${REPO_DIR}"

CONFIG_ID=$(python - "${FROZEN_CONFIG}" <<'PY'
import json
import sys
from pathlib import Path

print(json.loads(Path(sys.argv[1]).read_text())["selected_candidate_id"])
PY
)
public_v2_unpack_stage \
  "${PERSIST_DIR}/pilot-${CONFIG_ID}" "${WORK_ROOT}/pilot"
public_v2_unpack_stage \
  "${PERSIST_DIR}/full-${CONFIG_ID}" "${WORK_ROOT}/full"
mapfile -t CHECKPOINTS < <(
  find "${WORK_ROOT}/pilot/training" -mindepth 1 -maxdepth 1 -type d \
    -name 'checkpoint-*' -print | sort -V
)
test "${#CHECKPOINTS[@]}" -eq 1
EMBEDDINGS="${WORK_ROOT}/full/evaluation/candidate_embeddings.npy"
test -f "${EMBEDDINGS}"
OUTPUT_DIR="${WORK_ROOT}/downstream"

/usr/bin/time -v python scripts/public_v2_downstream.py \
  --protocol configs/public_retrieval_v2.json \
  --prepared-dir "${WORK_ROOT}/preflight/data/climate-fever-v2/prepared" \
  --selection-dir "${WORK_ROOT}/preflight/data/climate-fever-v2/selection-only" \
  --base-dir "${WORK_ROOT}/base" \
  --frozen-config "${FROZEN_CONFIG}" \
  --adapter-dir "${CHECKPOINTS[0]}" \
  --candidate-embeddings "${EMBEDDINGS}" \
  --output-dir "${OUTPUT_DIR}" \
  --device cuda \
  --batch-size 32
public_v2_pack_stage \
  "${OUTPUT_DIR}" "${PERSIST_DIR}/downstream" "downstream"
sync
