#!/bin/bash
#SBATCH --job-name=climate-embed-adapter-gate
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --partition=gpu-a100-mig
#SBATCH --gres=gpu:1g.20gb:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=00:20:00
#SBATCH --output=slurm-%x-%j.out
#SBATCH --error=slurm-%x-%j.err

set -euo pipefail
: "${REPO_DIR:?set detached, exact-SHA REPO_DIR}"
: "${DATA_DIR:?set DATA_DIR with dev-claims.json and evidence.json}"
: "${EVAL_DATASET:?set the claim-grouped held-out eval JSONL}"
: "${ADAPTER_DIR:?set the verified LoRA checkpoint directory}"
: "${OUTPUT_DIR:?set a new OUTPUT_DIR}"
: "${NATIVE_ENV:?set the existing native climate virtual environment}"
: "${ARTIFACT_ROOT:?set project ARTIFACT_ROOT for model caches}"

cd "${REPO_DIR}"
module purge
module load GCC/11.3.0 OpenMPI/4.1.4 PyTorch/2.1.2-CUDA-12.2.0
source "${NATIVE_ENV}/bin/activate"
export PYTHONNOUSERSITE=1
export PYTHONPATH="${NATIVE_ENV}/lib/python3.10/site-packages:${PYTHONPATH:-}"
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
export HF_HOME="${ARTIFACT_ROOT}/.cache/huggingface"
export HF_HUB_CACHE="${HF_HOME}/hub"
export TRANSFORMERS_CACHE="${HF_HOME}/transformers"
export HF_HUB_DISABLE_TELEMETRY=1
mkdir -p "${HF_HOME}" "${OUTPUT_DIR}"

python - <<'PY'
import torch

print(
    "runtime_preflight",
    {"torch": torch.__version__, "cuda": torch.version.cuda},
    flush=True,
)
assert torch.__version__.startswith("2.1.2"), torch.__version__
assert torch.version.cuda == "12.2", torch.version.cuda
assert torch.cuda.is_available(), "Spartan GPU is not visible to the pinned CUDA runtime"
PY

# This sampled held-out comparison is a promotion gate, not final full-corpus
# evidence. A pass only authorises a fixed-dev full-corpus evaluation.
/usr/bin/time -v python scripts/evaluate_embedding_adapter_gate.py \
  --claims "${DATA_DIR}/${CLAIMS_FILE:-train-claims.json}" \
  --evidence "${DATA_DIR}/evidence.json" \
  --eval-dataset "${EVAL_DATASET}" \
  --adapter "${ADAPTER_DIR}" \
  --output-dir "${OUTPUT_DIR}" \
  --sample-size "${SAMPLE_SIZE:-5000}" \
  --top-k "${TOP_K:-50}" \
  --truncate-dim "${TRUNCATE_DIM:-1024}" \
  --batch-size "${BATCH_SIZE:-16}" \
  --bootstrap-samples "${BOOTSTRAP_SAMPLES:-2000}"
sync
