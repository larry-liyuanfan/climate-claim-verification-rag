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
: "${BASE_DIR:?Set the verified public-v2 base directory}"
: "${PILOT_ROOT:?Set the completed six-pilot root}"
: "${PILOT_SELECTION:?Set pilot_selection.json}"
: "${FULL_ROOT:?Set a new full-evaluation root}"

test "$(readlink -f "${CLIMATE_V2_ROOT}")" = \
  "/data/gpfs/projects/punim2936/portfolio_20260903/climate-public-retrieval-v2"
test "$(git -C "${REPO_DIR}" rev-parse HEAD)" = "${CLIMATE_GIT_COMMIT}"

module purge
module load GCC/11.3.0 OpenMPI/4.1.4 PyTorch/2.1.2-CUDA-12.2.0
source "${REPO_DIR}/hpc/public_v2_runtime.sh"
public_v2_activate_runtime
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
mapfile -t PILOT_MATCHES < <(
  find "${PILOT_ROOT}" -mindepth 1 -maxdepth 1 -type d \
    -name "${CONFIG_ID}-*" -print
)
test "${#PILOT_MATCHES[@]}" -eq 1
PILOT_DIR="${PILOT_MATCHES[0]}"
mapfile -t CHECKPOINTS < <(
  find "${PILOT_DIR}/training" -mindepth 1 -maxdepth 1 -type d \
    -name 'checkpoint-*' -print | sort -V
)
test "${#CHECKPOINTS[@]}" -eq 1
TASK_ROOT="${FULL_ROOT}/${CONFIG_ID}-${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}"
test ! -e "${TASK_ROOT}"
mkdir -p "${TASK_ROOT}"

/usr/bin/time -v python scripts/public_v2_evaluate_adapter.py \
  --protocol configs/public_retrieval_v2.json \
  --policy configs/public_evaluation_policy.json \
  --prepared-dir "${CLIMATE_V2_ROOT}/data/climate-fever-v2/prepared" \
  --selection-dir "${CLIMATE_V2_ROOT}/data/climate-fever-v2/selection-only" \
  --base-dir "${BASE_DIR}" \
  --adapter-dir "${CHECKPOINTS[0]}" \
  --training-record "${PILOT_DIR}/adapter_config.json" \
  --adapter-config-id "${CONFIG_ID}" \
  --mode full \
  --output-dir "${TASK_ROOT}/evaluation" \
  --device cuda \
  --batch-size 32
sync
