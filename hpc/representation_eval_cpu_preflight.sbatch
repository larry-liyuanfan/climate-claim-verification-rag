#!/bin/bash
#SBATCH --job-name=climate-rep-preflight
#SBATCH --account=punim2936
#SBATCH --partition=sapphire
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=2G
#SBATCH --time=00:05:00

set -euo pipefail
: "${CLIMATE_PREFLIGHT_ROOT:?Set the isolated project-storage root}"
: "${CLIMATE_GIT_COMMIT:?Set the exact pushed commit}"

REPO_DIR="${CLIMATE_PREFLIGHT_ROOT}/repo"
PYTHON_BIN="${CLIMATE_PREFLIGHT_ROOT}/.venv/bin/python"
test "$(readlink -f "${CLIMATE_PREFLIGHT_ROOT}")" = \
  "/data/gpfs/projects/punim2936/portfolio_20260903/climate-representation-eval"
test "$(git -C "${REPO_DIR}" rev-parse HEAD)" = "${CLIMATE_GIT_COMMIT}"
test -x "${PYTHON_BIN}"

export PYTHONNOUSERSITE=1
export PYTHONDONTWRITEBYTECODE=1
export PYTHONPATH="${REPO_DIR}/src"
export XDG_CACHE_HOME="${CLIMATE_PREFLIGHT_ROOT}/cache/xdg"
export HF_HOME="${CLIMATE_PREFLIGHT_ROOT}/cache/huggingface"
export TRANSFORMERS_CACHE="${HF_HOME}/transformers"
export SENTENCE_TRANSFORMERS_HOME="${HF_HOME}/sentence-transformers"
export TOKENIZERS_PARALLELISM=false

cd "${REPO_DIR}"
"${PYTHON_BIN}" -m pytest -q -p no:cacheprovider \
  tests/test_evaluation_protocol.py \
  tests/test_representation_eval.py \
  tests/test_public_grounded_rag.py
"${PYTHON_BIN}" -m climate_rag.cli build-pareto \
  --profiles configs/search_profiles.verified.json \
  --output-dir "${CLIMATE_PREFLIGHT_ROOT}/artifacts/public/pareto-preflight-${SLURM_JOB_ID}"
