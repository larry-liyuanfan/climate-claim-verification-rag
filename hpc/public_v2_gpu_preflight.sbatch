#!/bin/bash
#SBATCH --job-name=climate-v2-gpu-preflight
#SBATCH --account=punim2936
#SBATCH --partition=gpu-a100-mig
#SBATCH --gres=gpu:1g.20gb:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:15:00
#SBATCH --tmp=20G

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
WORK_ROOT="${NODE_TMP}/climate-v2-gpu-preflight-${SLURM_JOB_ID}"
OUTPUT_DIR="${WORK_ROOT}/gpu-preflight"
test ! -e "${WORK_ROOT}"
mkdir -p "${OUTPUT_DIR}"
export PYTHONPATH="${REPO_DIR}/src${PYTHONPATH:+:${PYTHONPATH}}"

cd "${REPO_DIR}"
python - <<'PY' > "${OUTPUT_DIR}/runtime.json"
import json
import sys
import torch
from climate_rag.dense import SentenceTransformerEncoder

revision = "97b0c614be4d77ee51c0cef4e5f07c00f9eb65b3"
encoder = SentenceTransformerEncoder(
    "Qwen/Qwen3-Embedding-0.6B",
    device="cuda",
    truncate_dim=1024,
    revision=revision,
)
vector = encoder.encode_queries(["climate evidence retrieval preflight"], batch_size=1)
assert vector.shape == (1, 1024), vector.shape
assert torch.__version__.startswith("2.1.2"), torch.__version__
assert torch.version.cuda == "12.2", torch.version.cuda
assert torch.cuda.is_available()
print(json.dumps({
    "status": "passed",
    "model_revision": revision,
    "shape": list(vector.shape),
    "torch": torch.__version__,
    "torch_module": torch.__file__,
    "cuda": torch.version.cuda,
    "python_executable": sys.executable,
    "python_prefix": sys.prefix,
    "python_path": sys.path,
    "peak_torch_gpu_bytes": int(torch.cuda.max_memory_allocated()),
}, sort_keys=True))
PY
sha256sum "${OUTPUT_DIR}/runtime.json" > "${OUTPUT_DIR}/runtime.sha256"
public_v2_pack_stage \
  "${OUTPUT_DIR}" "${PERSIST_DIR}/gpu-preflight" "gpu-preflight"
sync
