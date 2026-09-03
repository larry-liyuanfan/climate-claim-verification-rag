#!/bin/bash
#SBATCH --job-name=climate-v2-base
#SBATCH --account=punim2936
#SBATCH --partition=gpu-a100-mig
#SBATCH --gres=gpu:1g.20gb:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=00:45:00
#SBATCH --tmp=30G

set -euo pipefail
: "${CLIMATE_V2_ROOT:?Set the isolated public-v2 root}"
: "${REPO_DIR:?Set the exact detached repository}"
: "${CLIMATE_GIT_COMMIT:?Set the exact pushed Git commit}"
: "${OUTPUT_DIR:?Set a unique base output directory}"

test "$(readlink -f "${CLIMATE_V2_ROOT}")" = \
  "/data/gpfs/projects/punim2936/portfolio_20260903/climate-public-retrieval-v2"
test "$(git -C "${REPO_DIR}" rev-parse HEAD)" = "${CLIMATE_GIT_COMMIT}"
test ! -e "${OUTPUT_DIR}"

module purge
module load GCC/11.3.0 OpenMPI/4.1.4 PyTorch/2.1.2-CUDA-12.2.0
source "${REPO_DIR}/hpc/public_v2_runtime.sh"
public_v2_activate_runtime
export PYTHONPATH="${REPO_DIR}/src"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
cd "${REPO_DIR}"

/usr/bin/time -v python scripts/public_v2_build_base.py \
  --protocol configs/public_retrieval_v2.json \
  --prepared-dir "${CLIMATE_V2_ROOT}/data/climate-fever-v2/prepared" \
  --selection-dir "${CLIMATE_V2_ROOT}/data/climate-fever-v2/selection-only" \
  --output-dir "${OUTPUT_DIR}" \
  --device cuda \
  --batch-size 32
sync
