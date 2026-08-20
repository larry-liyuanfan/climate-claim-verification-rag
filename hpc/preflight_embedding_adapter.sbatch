#!/bin/bash
#SBATCH --job-name=climate-embed-adapter-preflight
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --partition=cascade
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=00:10:00
#SBATCH --output=slurm-%x-%j.out
#SBATCH --error=slurm-%x-%j.err

set -euo pipefail
: "${REPO_DIR:?set detached, exact-SHA REPO_DIR}"
: "${ADAPTER_DIR:?set the verified LoRA checkpoint directory}"
: "${NATIVE_ENV:?set the existing native climate virtual environment}"
: "${ARTIFACT_ROOT:?set project ARTIFACT_ROOT for pip/model caches}"

cd "${REPO_DIR}"
module purge
module load GCC/11.3.0 OpenMPI/4.1.4 PyTorch/2.1.2-CUDA-12.2.0
source "${NATIVE_ENV}/bin/activate"
export PYTHONNOUSERSITE=1
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
export HF_HOME="${ARTIFACT_ROOT}/.cache/huggingface"
export HF_HUB_CACHE="${HF_HOME}/hub"
export TRANSFORMERS_CACHE="${HF_HOME}/transformers"
export HF_HUB_DISABLE_TELEMETRY=1
source hpc/adapter_runtime.sh
setup_climate_adapter_runtime

python - <<'PY'
import json
import os
from pathlib import Path

from climate_rag.dense import SentenceTransformerEncoder

adapter = Path(os.environ["ADAPTER_DIR"])
encoder = SentenceTransformerEncoder(
    "Qwen/Qwen3-Embedding-0.6B",
    device="cpu",
    truncate_dim=1024,
    adapter_path=str(adapter),
)
vector = encoder.encode_queries(["held-out climate retrieval adapter preflight"], batch_size=1)
assert vector.shape == (1, 1024), vector.shape
print(
    json.dumps(
        {
            "event": "adapter_cpu_preflight_completed",
            "adapter": str(adapter),
            "dimension": encoder.dimension,
            "vector_shape": list(vector.shape),
        },
        sort_keys=True,
    ),
    flush=True,
)
PY
