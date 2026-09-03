#!/bin/bash
#SBATCH --job-name=climate-v2-preflight
#SBATCH --account=punim2936
#SBATCH --partition=sapphire
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:45:00
#SBATCH --tmp=20G

set -euo pipefail
: "${CLIMATE_V2_ROOT:?Set the isolated public-v2 root}"
: "${REPO_DIR:?Set the exact detached repository}"
: "${CLIMATE_GIT_COMMIT:?Set the exact pushed Git commit}"
: "${PREP_OUTPUT:?Set a unique preparation output directory}"

test "$(readlink -f "${CLIMATE_V2_ROOT}")" = \
  "/data/gpfs/projects/punim2936/portfolio_20260903/climate-public-retrieval-v2"
test "$(git -C "${REPO_DIR}" rev-parse HEAD)" = "${CLIMATE_GIT_COMMIT}"
test -z "$(git -C "${REPO_DIR}" status --porcelain)"
test ! -e "${PREP_OUTPUT}"

module purge
module load GCC/11.3.0 OpenMPI/4.1.4 PyTorch/2.1.2-CUDA-12.2.0

NODE_TMP="${SLURM_TMPDIR:-${TMPDIR:-}}"
: "${NODE_TMP:?Spartan did not expose node-local temporary storage}"
case "$(readlink -f "${NODE_TMP}")" in
  /tmp|/tmp/*|/var/tmp|/var/tmp/*|/jobfs/*) ;;
  *) echo "Refusing non-local temporary directory: ${NODE_TMP}" >&2; exit 92 ;;
esac
RUNTIME_ENV="${NODE_TMP}/climate-v2-runtime-${SLURM_JOB_ID}"
RUNTIME_ARCHIVE="${CLIMATE_V2_ROOT}/envs/runtime-py310.tar.gz"
PIP_CACHE_DIR="${NODE_TMP}/climate-v2-pip-${SLURM_JOB_ID}"
test ! -e "${RUNTIME_ARCHIVE}"
python -m venv --system-site-packages "${RUNTIME_ENV}"
source "${RUNTIME_ENV}/bin/activate"
export PIP_CACHE_DIR
export PIP_DEFAULT_TIMEOUT="${PIP_DEFAULT_TIMEOUT:-120}"
python -m pip install --no-input --retries 5 --upgrade pip
python -m pip install --no-input --retries 5 \
  "numpy==1.26.4" \
  "pytest==8.3.5" \
  "httpx==0.28.1" \
  "fastapi==0.115.12" \
  "PyYAML==6.0.2" \
  "faiss-cpu==1.7.4" \
  "sentence-transformers==4.1.0" \
  "transformers==4.51.3" \
  "peft==0.15.2" \
  "lightgbm==4.6.0" \
  "scikit-learn==1.6.1" \
  "ms-swift==3.9.3" \
  "ruff==0.11.8" \
  "mypy==1.18.2" \
  "jsonschema==4.23.0"

export PYTHONNOUSERSITE=1
export PYTHONDONTWRITEBYTECODE=1
export PYTHONUNBUFFERED=1
export PYTHONPATH="${REPO_DIR}/src"
export XDG_CACHE_HOME="${NODE_TMP}/climate-v2-cache-${SLURM_JOB_ID}/xdg"
export HF_HOME="${NODE_TMP}/climate-v2-cache-${SLURM_JOB_ID}/huggingface"
export HF_HUB_CACHE="${HF_HOME}/hub"
export TRANSFORMERS_CACHE="${HF_HOME}/transformers"
export SENTENCE_TRANSFORMERS_HOME="${HF_HOME}/sentence-transformers"
export MODELSCOPE_CACHE="${NODE_TMP}/climate-v2-cache-${SLURM_JOB_ID}/modelscope"
export MODELSCOPE_HOME="${MODELSCOPE_CACHE}"
export HF_HUB_DISABLE_TELEMETRY=1
mkdir -p "${XDG_CACHE_HOME}" "${HF_HOME}" "${MODELSCOPE_CACHE}"

cd "${REPO_DIR}"
python -m pytest -q -p no:cacheprovider
python -m climate_rag.cli build-pareto \
  --profiles configs/search_profiles.verified.json \
  --output-dir "${PREP_OUTPUT}-pareto-smoke"
python scripts/public_v2_prepare.py \
  --protocol configs/public_retrieval_v2.json \
  --policy configs/public_evaluation_policy.json \
  --data-root "${CLIMATE_V2_ROOT}/data" \
  --output-dir "${PREP_OUTPUT}"
python -m pip freeze > "${PREP_OUTPUT}/environment-freeze.txt"
sha256sum "${PREP_OUTPUT}/environment-freeze.txt" \
  > "${PREP_OUTPUT}/environment-freeze.sha256"
mkdir -p "${CLIMATE_V2_ROOT}/envs"
TEMP_ARCHIVE="${CLIMATE_V2_ROOT}/envs/.runtime-py310-${SLURM_JOB_ID}.tar.gz"
tar -czf "${TEMP_ARCHIVE}" -C "${RUNTIME_ENV}" .
mv "${TEMP_ARCHIVE}" "${RUNTIME_ARCHIVE}"
sha256sum "${RUNTIME_ARCHIVE}" > "${CLIMATE_V2_ROOT}/envs/runtime-py310.tar.gz.sha256"
