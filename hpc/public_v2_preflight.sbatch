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
: "${PERSIST_DIR:?Set the pre-existing archive destination directory}"

test "$(readlink -f "${CLIMATE_V2_ROOT}")" = \
  "/data/gpfs/projects/punim2936/portfolio_20260903/climate-public-retrieval-v2"
test "$(git -C "${REPO_DIR}" rev-parse HEAD)" = "${CLIMATE_GIT_COMMIT}"
test -z "$(git -C "${REPO_DIR}" status --porcelain)"
test -d "${PERSIST_DIR}"
AVAILABLE_INODES="$(df --output=iavail "${PERSIST_DIR}" | tail -n 1 | tr -d ' ')"
MINIMUM_INODES=68
printf 'persistent_inode_available=%s\npersistent_inode_minimum=%s\n' \
  "${AVAILABLE_INODES}" "${MINIMUM_INODES}"
test "${AVAILABLE_INODES}" -ge "${MINIMUM_INODES}"

module purge
module load GCC/11.3.0 OpenMPI/4.1.4 PyTorch/2.1.2-CUDA-12.2.0
source "${REPO_DIR}/hpc/public_v2_runtime.sh"
NODE_TMP="$(public_v2_node_tmp)"
WORK_ROOT="${NODE_TMP}/climate-v2-preflight-${SLURM_JOB_ID}"
RUNTIME_ENV="${NODE_TMP}/climate-v2-runtime-${SLURM_JOB_ID}"
PIP_CACHE_DIR="${NODE_TMP}/climate-v2-pip-${SLURM_JOB_ID}"
test ! -e "${WORK_ROOT}"
mkdir -p "${WORK_ROOT}/preflight"
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
export PYTHONPATH="${REPO_DIR}/src${PYTHONPATH:+:${PYTHONPATH}}"
export XDG_CACHE_HOME="${NODE_TMP}/climate-v2-cache-${SLURM_JOB_ID}/xdg"
export HF_HOME="${NODE_TMP}/climate-v2-cache-${SLURM_JOB_ID}/huggingface"
export HF_HUB_CACHE="${HF_HOME}/hub"
export TRANSFORMERS_CACHE="${HF_HOME}/transformers"
export SENTENCE_TRANSFORMERS_HOME="${HF_HOME}/sentence-transformers"
export MODELSCOPE_CACHE="${NODE_TMP}/climate-v2-cache-${SLURM_JOB_ID}/modelscope"
export MODELSCOPE_HOME="${MODELSCOPE_CACHE}"
export HF_HUB_DISABLE_TELEMETRY=1
mkdir -p "${XDG_CACHE_HOME}" "${HF_HOME}" "${MODELSCOPE_CACHE}"

PREP_OUTPUT="${WORK_ROOT}/preflight/prepare"
DATA_ROOT="${WORK_ROOT}/preflight/data"
cd "${REPO_DIR}"
python -m pytest -q -p no:cacheprovider
python -m climate_rag.cli build-pareto \
  --profiles configs/search_profiles.verified.json \
  --output-dir "${WORK_ROOT}/preflight/pareto-smoke"
python scripts/public_v2_prepare.py \
  --protocol configs/public_retrieval_v2.json \
  --policy configs/public_evaluation_policy.json \
  --data-root "${DATA_ROOT}" \
  --output-dir "${PREP_OUTPUT}"
python -m pip freeze > "${PREP_OUTPUT}/environment-freeze.txt"
sha256sum "${PREP_OUTPUT}/environment-freeze.txt" \
  > "${PREP_OUTPUT}/environment-freeze.sha256"

public_v2_pack_stage \
  "${RUNTIME_ENV}" "${CLIMATE_V2_ROOT}/envs/runtime-py310" "runtime-py310"
public_v2_pack_stage \
  "${WORK_ROOT}/preflight" "${PERSIST_DIR}/preflight" "preflight"
sync
