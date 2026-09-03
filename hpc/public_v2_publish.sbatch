#!/bin/bash
#SBATCH --job-name=climate-v2-publish
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
: "${PREP_METRICS:?Set preparation metrics}"
: "${BASE_METRICS:?Set base metrics}"
: "${PILOT_SELECTION:?Set pilot selection}"
: "${FULL_SELECTION:?Set full selection}"
: "${FULL_ROOT:?Set full evaluation root}"
: "${DOWNSTREAM_METRICS:?Set downstream metrics}"
: "${EXTERNAL_METRICS:?Set external metrics}"
: "${EXTERNAL_LEDGER:?Set external ledger}"
: "${SACCT_JSON:?Set compact Slurm accounting JSON}"
: "${OUTPUT_DIR:?Set a unique compact publish directory}"

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

mapfile -t FULL_METRICS < <(
  find "${FULL_ROOT}" -mindepth 2 -maxdepth 2 -type f \
    -path '*/evaluation/metrics.json' -print | sort
)
test "${#FULL_METRICS[@]}" -ge 1
test "${#FULL_METRICS[@]}" -le 2
FULL_ARGS=()
for path in "${FULL_METRICS[@]}"; do
  FULL_ARGS+=(--full-metrics "${path}")
done
python scripts/public_v2_publish.py \
  --protocol configs/public_retrieval_v2.json \
  --prepare-metrics "${PREP_METRICS}" \
  --base-metrics "${BASE_METRICS}" \
  --pilot-selection "${PILOT_SELECTION}" \
  --full-selection "${FULL_SELECTION}" \
  "${FULL_ARGS[@]}" \
  --downstream-metrics "${DOWNSTREAM_METRICS}" \
  --external-metrics "${EXTERNAL_METRICS}" \
  --external-ledger "${EXTERNAL_LEDGER}" \
  --sacct-json "${SACCT_JSON}" \
  --schema schemas/public_v2_compact.schema.json \
  --output-dir "${OUTPUT_DIR}"
python scripts/check_public_v2_artifact.py \
  "${OUTPUT_DIR}/public-v2-compact.json" \
  schemas/public_v2_compact.schema.json
