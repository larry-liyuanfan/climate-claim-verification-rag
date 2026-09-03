#!/bin/bash
#SBATCH --job-name=climate-v2-select
#SBATCH --account=punim2936
#SBATCH --partition=sapphire
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --time=00:10:00
#SBATCH --tmp=10G

set -euo pipefail
: "${CLIMATE_V2_ROOT:?Set the isolated public-v2 root}"
: "${REPO_DIR:?Set the exact detached repository}"
: "${CLIMATE_GIT_COMMIT:?Set the exact pushed Git commit}"
: "${PHASE:?Set pilot or full}"
: "${METRICS_ROOT:?Set the pilot/full metrics root}"
: "${OUTPUT_DIR:?Set a unique selection output directory}"

test "$(readlink -f "${CLIMATE_V2_ROOT}")" = \
  "/data/gpfs/projects/punim2936/portfolio_20260903/climate-public-retrieval-v2"
test "$(git -C "${REPO_DIR}" rev-parse HEAD)" = "${CLIMATE_GIT_COMMIT}"
test ! -e "${OUTPUT_DIR}"

module purge
module load GCC/11.3.0 OpenMPI/4.1.4 Python/3.10.4
source "${REPO_DIR}/hpc/public_v2_runtime.sh"
public_v2_activate_runtime
export PYTHONPATH="${REPO_DIR}/src"
cd "${REPO_DIR}"

mapfile -t METRICS < <(
  find "${METRICS_ROOT}" -mindepth 2 -maxdepth 2 -type f \
    -path '*/evaluation/metrics.json' -print | sort
)
if [[ "${PHASE}" = "pilot" ]]; then
  test "${#METRICS[@]}" -eq 6
else
  test "${#METRICS[@]}" -ge 1
  test "${#METRICS[@]}" -le 2
fi
ARGS=()
for path in "${METRICS[@]}"; do
  ARGS+=(--metrics "${path}")
done
python scripts/public_v2_select.py \
  --protocol configs/public_retrieval_v2.json \
  --phase "${PHASE}" \
  "${ARGS[@]}" \
  --output-dir "${OUTPUT_DIR}"
