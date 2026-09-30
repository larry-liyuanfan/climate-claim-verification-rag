#!/bin/bash
#SBATCH --job-name=climate-adapter-integrity
#SBATCH --account=punim2936
#SBATCH --partition=sapphire
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:15:00
#SBATCH --tmp=20G

set -euo pipefail
: "${CLIMATE_V2_ROOT:?Set existing public-v2 archive root}"
: "${REPO_DIR:?Set isolated exact-commit checkout}"
: "${CLIMATE_GIT_COMMIT:?Set exact pushed commit}"
: "${PERSIST_DIR:?Set new, unique pre-existing run directory}"
: "${SOURCE_RUN:?Set existing run archive directory}"
: "${CLIMATE_EXPECTED_USER:?Set the authorised SSH account outside the repository}"
test "$(id -un)" = "${CLIMATE_EXPECTED_USER}"
test "$(git -C "${REPO_DIR}" rev-parse HEAD)" = "${CLIMATE_GIT_COMMIT}"
test -z "$(git -C "${REPO_DIR}" status --porcelain)"
test -d "${PERSIST_DIR}"

module purge
module load GCC/11.3.0 OpenMPI/4.1.4 PyTorch/2.1.2-CUDA-12.2.0
source "${REPO_DIR}/hpc/public_v2_runtime.sh"
public_v2_activate_runtime
NODE_TMP="$(public_v2_node_tmp)"
WORK_ROOT="${NODE_TMP}/climate-adapter-integrity-${SLURM_JOB_ID}"
test ! -e "${WORK_ROOT}"
mkdir -p "${WORK_ROOT}/result"
export PYTHONPATH="${REPO_DIR}/src${PYTHONPATH:+:${PYTHONPATH}}"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
cd "${REPO_DIR}"

# Require the real integration dependencies; importorskip cannot mask a missing runtime.
python -c 'from climate_rag.torch_compat import ensure_torch_pytree_compat; ensure_torch_pytree_compat(); import torch, peft, safetensors; assert peft.__version__ == "0.15.2"'
python -m pytest -q -p no:cacheprovider tests/test_adapter_integrity.py \
  | tee "${WORK_ROOT}/result/integration-tests.txt"
public_v2_unpack_stage \
  "${SOURCE_RUN}/pilot-s100-r16-n4-t005" "${WORK_ROOT}/pilot"
mapfile -t CHECKPOINTS < <(
  find "${WORK_ROOT}/pilot/training" -mindepth 2 -maxdepth 2 -type d \
    -name 'checkpoint-*' -print | sort -V
)
test "${#CHECKPOINTS[@]}" -eq 1
set +e
/usr/bin/time -v python scripts/probe_embedding_adapter.py \
  --adapter-dir "${CHECKPOINTS[0]}" --output "${WORK_ROOT}/result/probe.json" \
  --device cpu
PROBE_EXIT=$?
set -e
public_v2_pack_stage \
  "${WORK_ROOT}/result" "${PERSIST_DIR}/adapter-integrity" "adapter-integrity"
exit "${PROBE_EXIT}"
