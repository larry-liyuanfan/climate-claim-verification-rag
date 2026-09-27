#!/bin/bash
#SBATCH --job-name=climate-search-tradeoffs
#SBATCH --account=punim2936
#SBATCH --partition=gpu-a100-mig
#SBATCH --gres=gpu:1g.20gb:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=00:45:00
#SBATCH --tmp=40G

set -euo pipefail
: "${CLIMATE_V2_ROOT:?Existing public-v2 archive root}"
: "${REPO_DIR:?Isolated exact-commit checkout}"
: "${CLIMATE_GIT_COMMIT:?Exact pushed commit}"
: "${SOURCE_RUN:?Read-only input run}"
: "${PERSIST_DIR:?New run destination}"
test "$(id -un)" = yzhang3504
test "$(git -C "${REPO_DIR}" rev-parse HEAD)" = "${CLIMATE_GIT_COMMIT}"
test -z "$(git -C "${REPO_DIR}" status --porcelain)"
test -d "${PERSIST_DIR}"
module purge
module load GCC/11.3.0 OpenMPI/4.1.4 PyTorch/2.1.2-CUDA-12.2.0
source "${REPO_DIR}/hpc/public_v2_runtime.sh"
public_v2_activate_runtime
NODE_TMP="$(public_v2_node_tmp)"
WORK_ROOT="${NODE_TMP}/climate-search-tradeoffs-${SLURM_JOB_ID}"
test ! -e "${WORK_ROOT}"
mkdir -p "${WORK_ROOT}/results"
public_v2_unpack_stage "${SOURCE_RUN}/preflight" "${WORK_ROOT}/preflight"
public_v2_unpack_stage "${SOURCE_RUN}/base" "${WORK_ROOT}/base"
public_v2_unpack_stage "${SOURCE_RUN}/downstream" "${WORK_ROOT}/downstream"
export PYTHONPATH="${REPO_DIR}/src${PYTHONPATH:+:${PYTHONPATH}}"
export PYTHONHASHSEED=0
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK}" MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
cd "${REPO_DIR}"
python -m pytest -q tests/test_adapter_integrity.py
PREPARED="${WORK_ROOT}/preflight/data/climate-fever-v2/prepared"
SELECTION="${WORK_ROOT}/preflight/data/climate-fever-v2/selection-only"
# Fresh interpreter for each profile isolates resident models and peak memory.
# Preserve a completed profile before beginning the next (no blind full reruns).
for PROFILE in ltr rerank20 rerank100; do
  /usr/bin/time -v python scripts/profile_public_search.py \
    --protocol configs/public_retrieval_v2.json --policy configs/public_evaluation_policy.json \
    --prepared-dir "${PREPARED}" --selection-dir "${SELECTION}" \
    --base-dir "${WORK_ROOT}/base" --downstream-dir "${WORK_ROOT}/downstream" \
    --output-dir "${WORK_ROOT}/results/${PROFILE}" --profile "${PROFILE}" --device cuda
  public_v2_pack_stage "${WORK_ROOT}/results/${PROFILE}" \
    "${PERSIST_DIR}/profile-${PROFILE}" "profile-${PROFILE}"
done
python scripts/summarize_search_profiles.py --run-dir "${WORK_ROOT}/results" \
  --prepared-dir "${PREPARED}" --selection-dir "${SELECTION}"
public_v2_pack_stage "${WORK_ROOT}/results" "${PERSIST_DIR}/comparison" "search-tradeoffs"
