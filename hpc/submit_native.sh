#!/bin/bash
set -euo pipefail

REPO_DIR="$(git rev-parse --show-toplevel)"
PORTFOLIO_ROOT="$(dirname "${REPO_DIR}")"
DATA_DIR="${DATA_DIR:-/data/gpfs/projects/punim2936/nlp/COMP90042_2026-main/data}"
ARTIFACT_DIR="${ARTIFACT_DIR:-${PORTFOLIO_ROOT}/climate-artifacts}"
NATIVE_ENV="${NATIVE_ENV:-${ARTIFACT_DIR}/native-venv-standalone}"
PIP_CACHE_DIR="${ARTIFACT_DIR}/pip-cache"
mkdir -p "${ARTIFACT_DIR}" "${PIP_CACHE_DIR}"

BOOTSTRAP_ID=$(sbatch --parsable --account=punim2936 --partition=sapphire \
  --export=ALL,REPO_DIR="${REPO_DIR}",NATIVE_ENV="${NATIVE_ENV}",PIP_CACHE_DIR="${PIP_CACHE_DIR}" \
  hpc/bootstrap_native.sbatch)
BM25_ID=$(sbatch --parsable --dependency="afterok:${BOOTSTRAP_ID}" --account=punim2936 --partition=sapphire \
  --export=ALL,DATA_DIR="${DATA_DIR}",ARTIFACT_DIR="${ARTIFACT_DIR}",NATIVE_ENV="${NATIVE_ENV}" \
  hpc/build_bm25_native.sbatch)
DENSE_ID=$(sbatch --parsable --dependency="afterok:${BOOTSTRAP_ID}" --account=punim2936 --partition=gpu-h100 \
  --export=ALL,DATA_DIR="${DATA_DIR}",ARTIFACT_DIR="${ARTIFACT_DIR}",NATIVE_ENV="${NATIVE_ENV}" \
  hpc/build_dense_native.sbatch)
printf 'bootstrap=%s bm25=%s dense=%s\n' "${BOOTSTRAP_ID}" "${BM25_ID}" "${DENSE_ID}"
