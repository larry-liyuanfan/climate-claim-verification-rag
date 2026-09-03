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
: "${PHASE:?Set pilot, full, or negative-closeout}"
: "${PERSIST_DIR:?Set the pre-existing archive destination directory}"
test "${PHASE}" = "pilot" || test "${PHASE}" = "full" || \
  test "${PHASE}" = "negative-closeout"

test "$(readlink -f "${CLIMATE_V2_ROOT}")" = \
  "/data/gpfs/projects/punim2936/portfolio_20260903/climate-public-retrieval-v2"
test "$(git -C "${REPO_DIR}" rev-parse HEAD)" = "${CLIMATE_GIT_COMMIT}"
test -d "${PERSIST_DIR}"

module purge
module load GCC/11.3.0 OpenMPI/4.1.4 Python/3.10.4
source "${REPO_DIR}/hpc/public_v2_runtime.sh"
public_v2_activate_runtime
NODE_TMP="$(public_v2_node_tmp)"
WORK_ROOT="${NODE_TMP}/climate-v2-${PHASE}-selection-${SLURM_JOB_ID}"
METRICS_ROOT="${WORK_ROOT}/metrics"
OUTPUT_DIR="${WORK_ROOT}/selection"
test ! -e "${WORK_ROOT}"
mkdir -p "${METRICS_ROOT}"
export PYTHONPATH="${REPO_DIR}/src${PYTHONPATH:+:${PYTHONPATH}}"
cd "${REPO_DIR}"

ARGS=()
if [[ "${PHASE}" = "negative-closeout" ]]; then
  : "${DIAGNOSTIC_JOB_ID:?Set the failed diagnostic Slurm job ID}"
  : "${DIAGNOSTIC_EXIT_CODE:?Set the failed diagnostic exit code}"
  : "${DIAGNOSTIC_LOG:?Set the failed diagnostic merged log}"
  case "$(readlink -f "${DIAGNOSTIC_LOG}")" in
    "$(readlink -f "${CLIMATE_V2_ROOT}")"/logs/*) ;;
    *) echo "diagnostic log is outside the isolated root" >&2; exit 1 ;;
  esac
  test -f "${DIAGNOSTIC_LOG}"
  public_v2_unpack_stage \
    "${PERSIST_DIR}/pilot-selection" "${WORK_ROOT}/pilot-selection"
  public_v2_unpack_stage "${PERSIST_DIR}/base" "${WORK_ROOT}/base"
  ARGS+=(
    --pilot-selection "${WORK_ROOT}/pilot-selection/pilot_selection.json"
    --base-embeddings "${WORK_ROOT}/base/base_embeddings.npy"
    --diagnostic-log "${DIAGNOSTIC_LOG}"
    --diagnostic-job-id "${DIAGNOSTIC_JOB_ID}"
    --diagnostic-exit-code "${DIAGNOSTIC_EXIT_CODE}"
  )
else
  shopt -s nullglob
  ARCHIVES=("${PERSIST_DIR}/${PHASE}"-*.tar.gz)
  shopt -u nullglob
  if [[ "${PHASE}" = "pilot" ]]; then
    test "${#ARCHIVES[@]}" -eq 6
  else
    test "${#ARCHIVES[@]}" -ge 1
    test "${#ARCHIVES[@]}" -le 2
  fi
  for index in "${!ARCHIVES[@]}"; do
    archive="${ARCHIVES[${index}]}"
    stem="${archive%.tar.gz}"
    digest="${stem##*-}"
    prefix="${stem%-${digest}}"
    public_v2_unpack_stage "${prefix}" "${METRICS_ROOT}/${index}"
  done
  mapfile -t METRICS < <(
    find "${METRICS_ROOT}" -mindepth 3 -maxdepth 3 -type f \
      -path '*/evaluation/metrics.json' -print | sort
  )
  test "${#METRICS[@]}" -eq "${#ARCHIVES[@]}"
  for path in "${METRICS[@]}"; do
    ARGS+=(--metrics "${path}")
  done
fi
python scripts/public_v2_select.py \
  --protocol configs/public_retrieval_v2.json \
  --phase "${PHASE}" \
  "${ARGS[@]}" \
  --output-dir "${OUTPUT_DIR}"
if [[ "${PHASE}" = "negative-closeout" ]]; then
  public_v2_pack_stage \
    "${OUTPUT_DIR}" "${PERSIST_DIR}/full-selection" "full-selection"
else
  public_v2_pack_stage \
    "${OUTPUT_DIR}" "${PERSIST_DIR}/${PHASE}-selection" "${PHASE}-selection"
fi
sync
