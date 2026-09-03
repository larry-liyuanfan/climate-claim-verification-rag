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
: "${PERSIST_DIR:?Set the pre-existing archive destination directory}"
: "${LEDGER:?Set the completed external-transfer ledger}"
: "${JOB_IDS:?Set the comma-separated Slurm job IDs for accounting}"

test "$(readlink -f "${CLIMATE_V2_ROOT}")" = \
  "/data/gpfs/projects/punim2936/portfolio_20260903/climate-public-retrieval-v2"
test "$(git -C "${REPO_DIR}" rev-parse HEAD)" = "${CLIMATE_GIT_COMMIT}"
test -d "${PERSIST_DIR}"

module purge
module load GCC/11.3.0 OpenMPI/4.1.4 Python/3.10.4
source "${REPO_DIR}/hpc/public_v2_runtime.sh"
public_v2_activate_runtime
NODE_TMP="$(public_v2_node_tmp)"
WORK_ROOT="${NODE_TMP}/climate-v2-publish-${SLURM_JOB_ID}"
test ! -e "${WORK_ROOT}"
mkdir -p "${WORK_ROOT}/full"
public_v2_unpack_stage "${PERSIST_DIR}/preflight" "${WORK_ROOT}/preflight"
public_v2_unpack_stage "${PERSIST_DIR}/base" "${WORK_ROOT}/base"
public_v2_unpack_stage \
  "${PERSIST_DIR}/pilot-selection" "${WORK_ROOT}/pilot-selection"
public_v2_unpack_stage \
  "${PERSIST_DIR}/full-selection" "${WORK_ROOT}/full-selection"
public_v2_unpack_stage "${PERSIST_DIR}/downstream" "${WORK_ROOT}/downstream"
public_v2_unpack_stage "${PERSIST_DIR}/external" "${WORK_ROOT}/external"
export PYTHONPATH="${REPO_DIR}/src"
cd "${REPO_DIR}"

shopt -s nullglob
FULL_ARCHIVES=("${PERSIST_DIR}"/full-s*.tar.gz)
shopt -u nullglob
test "${#FULL_ARCHIVES[@]}" -ge 1
test "${#FULL_ARCHIVES[@]}" -le 2
for index in "${!FULL_ARCHIVES[@]}"; do
  archive="${FULL_ARCHIVES[${index}]}"
  stem="${archive%.tar.gz}"
  digest="${stem##*-}"
  prefix="${stem%-${digest}}"
  public_v2_unpack_stage "${prefix}" "${WORK_ROOT}/full/${index}"
done
mapfile -t FULL_METRICS < <(
  find "${WORK_ROOT}/full" -mindepth 3 -maxdepth 3 -type f \
    -path '*/evaluation/metrics.json' -print | sort
)
test "${#FULL_METRICS[@]}" -eq "${#FULL_ARCHIVES[@]}"
FULL_ARGS=()
for path in "${FULL_METRICS[@]}"; do
  FULL_ARGS+=(--full-metrics "${path}")
done

SACCT_TSV="${WORK_ROOT}/sacct.tsv"
SACCT_JSON="${WORK_ROOT}/sacct.json"
sacct -j "${JOB_IDS}" --starttime 2026-09-03 -n -P \
  -o JobID,JobName,Partition,AllocTRES,ReqMem,Elapsed,MaxRSS,State,ExitCode \
  > "${SACCT_TSV}"
python scripts/collect_slurm_accounting.py \
  --input "${SACCT_TSV}" --output "${SACCT_JSON}"

OUTPUT_DIR="${WORK_ROOT}/publish"
python scripts/public_v2_publish.py \
  --protocol configs/public_retrieval_v2.json \
  --prepare-metrics "${WORK_ROOT}/preflight/prepare/metrics.json" \
  --base-metrics "${WORK_ROOT}/base/metrics.json" \
  --pilot-selection "${WORK_ROOT}/pilot-selection/pilot_selection.json" \
  --full-selection "${WORK_ROOT}/full-selection/full_selection.json" \
  "${FULL_ARGS[@]}" \
  --downstream-metrics "${WORK_ROOT}/downstream/metrics.json" \
  --external-metrics "${WORK_ROOT}/external/metrics.json" \
  --external-ledger "${LEDGER}" \
  --sacct-json "${SACCT_JSON}" \
  --schema schemas/public_v2_compact.schema.json \
  --output-dir "${OUTPUT_DIR}"
python scripts/check_public_v2_artifact.py \
  "${OUTPUT_DIR}/public-v2-compact.json" \
  schemas/public_v2_compact.schema.json
public_v2_pack_stage "${OUTPUT_DIR}" "${PERSIST_DIR}/publish" "publish"
sync
