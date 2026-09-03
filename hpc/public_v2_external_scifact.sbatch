#!/bin/bash
#SBATCH --job-name=climate-v2-scifact
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
: "${LEDGER:?Set the one-shot external-transfer ledger}"

test "$(readlink -f "${CLIMATE_V2_ROOT}")" = \
  "/data/gpfs/projects/punim2936/portfolio_20260903/climate-public-retrieval-v2"
test "$(git -C "${REPO_DIR}" rev-parse HEAD)" = "${CLIMATE_GIT_COMMIT}"
test -d "${PERSIST_DIR}"

module purge
module load GCC/11.3.0 OpenMPI/4.1.4 PyTorch/2.1.2-CUDA-12.2.0
source "${REPO_DIR}/hpc/public_v2_runtime.sh"
public_v2_assert_persist_prefix "${LEDGER%.json}"
public_v2_activate_runtime
NODE_TMP="$(public_v2_node_tmp)"
WORK_ROOT="${NODE_TMP}/climate-v2-scifact-${SLURM_JOB_ID}"
test ! -e "${WORK_ROOT}"
mkdir -p "${WORK_ROOT}"
public_v2_unpack_stage "${PERSIST_DIR}/preflight" "${WORK_ROOT}/preflight"
public_v2_unpack_stage "${PERSIST_DIR}/full-selection" "${WORK_ROOT}/full-selection"
FROZEN_CONFIG="${WORK_ROOT}/full-selection/frozen_config.json"
export PYTHONPATH="${REPO_DIR}/src${PYTHONPATH:+:${PYTHONPATH}}"
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
mapfile -t CHECKPOINTS < <(
  find "${WORK_ROOT}/pilot/training" -mindepth 1 -maxdepth 1 -type d \
    -name 'checkpoint-*' -print | sort -V
)
test "${#CHECKPOINTS[@]}" -eq 1
OUTPUT_DIR="${WORK_ROOT}/external"

/usr/bin/time -v python scripts/public_v2_external_scifact.py \
  --protocol configs/public_retrieval_v2.json \
  --frozen-config "${FROZEN_CONFIG}" \
  --adapter-dir "${CHECKPOINTS[0]}" \
  --archive "${WORK_ROOT}/preflight/data/scifact/source/scifact.zip" \
  --ledger "${LEDGER}" \
  --output-dir "${OUTPUT_DIR}" \
  --device cuda \
  --batch-size 32
public_v2_pack_stage \
  "${OUTPUT_DIR}" "${PERSIST_DIR}/external" "external-scifact"
sync
