#!/bin/bash
# Not submitted by packager: coordinator exact-hash CPU release required.
#SBATCH --job-name=climate-nei47-cpu
#SBATCH --account=punim2936
#SBATCH --partition=sapphire
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --time=00:15:00
#SBATCH --no-requeue
set -euo pipefail
umask 077
: "${SLURM_JOB_ID:?CPU allocation required}"
: "${CLIMATE_SOURCE_STAGE:?}"
: "${CLIMATE_SOURCE_TAR:?}"
: "${CLIMATE_SOURCE_SHA:?}"
: "${CLIMATE_CPU_WRAPPER_SHA:?}"
: "${CLIMATE_NEI_RELEASE:?}"
: "${CLIMATE_NEI_RELEASE_SHA:?}"
ROOT=/data/gpfs/projects/punim2936/portfolio_20260903/climate-public-retrieval-v2
case "$(readlink -f "$CLIMATE_SOURCE_STAGE")" in "$ROOT/envs/nei47-source-"*) ;; *) exit 90 ;; esac
test "$(readlink -f "$CLIMATE_SOURCE_TAR")" = "$(readlink -f "$CLIMATE_SOURCE_STAGE/source.tar")"
test "$(readlink -f "$CLIMATE_NEI_RELEASE")" = "$(readlink -f "$CLIMATE_SOURCE_STAGE/nei47-release.json")"
test "$(sha256sum "$CLIMATE_SOURCE_TAR" | cut -d' ' -f1)" = "$CLIMATE_SOURCE_SHA"
test "$(sha256sum "$CLIMATE_NEI_RELEASE" | cut -d' ' -f1)" = "$CLIMATE_NEI_RELEASE_SHA"
test "$(sha256sum "$0" | cut -d' ' -f1)" = "$CLIMATE_CPU_WRAPPER_SHA"
# BEGIN SOURCE_ARCHIVE_GUARD
test "$(tar -tf "$CLIMATE_SOURCE_TAR" | grep -cx 'SOURCE_REVISION')" -eq 1
test "$(tar -xOf "$CLIMATE_SOURCE_TAR" SOURCE_REVISION | wc -c)" -eq 41
test "$(tar -xOf "$CLIMATE_SOURCE_TAR" SOURCE_REVISION)" = "$CLIMATE_SOURCE_GIT"
test "$(sha256sum "$0" | cut -d' ' -f1)" = "$(tar -xOf "$CLIMATE_SOURCE_TAR" hpc/scifact_nei47_cpu.sbatch | sha256sum | cut -d' ' -f1)"
# END SOURCE_ARCHIVE_GUARD
test "$(cat "$CLIMATE_SOURCE_STAGE/source/SOURCE_REVISION")" = "$CLIMATE_SOURCE_GIT"
module purge
module load GCC/11.3.0 OpenMPI/4.1.4 PyTorch/2.1.2-CUDA-12.2.0
export PYTHONHASHSEED=0 USE_TORCH=0 USE_TF=0 PYTHONOPTIMIZE=0 PYTHONDONTWRITEBYTECODE=1 PYTHONNOUSERSITE=1
export TOKENIZERS_PARALLELISM=false OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 CUDA_VISIBLE_DEVICES=""
DEPS="$ROOT/envs/grounding-training-deps-20261001"
export PYTHONPATH="$CLIMATE_SOURCE_STAGE/source/src:$CLIMATE_SOURCE_STAGE/source/scripts:$DEPS/numeric-site:$DEPS/site:$ROOT/envs/grounding-cpu-40d84a3/site:$ROOT/envs/scifact-component-cpu-426ff7343fb3/tokenizer-site${PYTHONPATH:+:$PYTHONPATH}"
exec python "$CLIMATE_SOURCE_STAGE/source/scripts/prepare_scifact_nei47.py" \
  --source-git "$CLIMATE_SOURCE_GIT" --source-sha "$CLIMATE_SOURCE_SHA" \
  --source-archive "$CLIMATE_SOURCE_TAR" --wrapper-sha "$CLIMATE_CPU_WRAPPER_SHA" \
  --release-file "$CLIMATE_NEI_RELEASE" --release-sha "$CLIMATE_NEI_RELEASE_SHA"
