#!/bin/bash
# One approved shared-corpus FIT preparation; no GPU/model/training/evaluation.
#SBATCH --job-name=climate-shared-fit-cpu
#SBATCH --account=punim2936
#SBATCH --partition=sapphire
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --time=00:10:00
#SBATCH --no-requeue
set -euo pipefail
umask 077
: "${SLURM_JOB_ID:?CPU allocation required}"
ROOT=/data/gpfs/projects/punim2936/portfolio_20260903/climate-public-retrieval-v2
case "$(readlink -f "$CLIMATE_SOURCE_STAGE")" in "$ROOT/envs/shared-supervision-source-"*) ;; *) exit 90 ;; esac
# BEGIN SHARED_CPU_SOURCE_GUARD
: "${CLIMATE_SOURCE_GIT:?}"
: "${CLIMATE_SOURCE_SHA:?}"
: "${CLIMATE_SOURCE_STAGE:?}"
: "${CLIMATE_CPU_WRAPPER_SHA:?}"
test "$(sha256sum "$CLIMATE_SOURCE_STAGE/source.tar" | cut -d' ' -f1)" = "$CLIMATE_SOURCE_SHA"
test "$(sha256sum "$0" | cut -d' ' -f1)" = "$CLIMATE_CPU_WRAPPER_SHA"
test "$(tar -xOf "$CLIMATE_SOURCE_STAGE/source.tar" hpc/scifact_shared_supervision_cpu.sbatch | sha256sum | cut -d' ' -f1)" = "$CLIMATE_CPU_WRAPPER_SHA"
test "$(cat "$CLIMATE_SOURCE_STAGE/source/SOURCE_REVISION")" = "$CLIMATE_SOURCE_GIT"
# END SHARED_CPU_SOURCE_GUARD
module purge
module load GCC/11.3.0 OpenMPI/4.1.4 PyTorch/2.1.2-CUDA-12.2.0
export PYTHONHASHSEED=0 USE_TORCH=0 USE_TF=0 PYTHONOPTIMIZE=0 PYTHONDONTWRITEBYTECODE=1 PYTHONNOUSERSITE=1
export TOKENIZERS_PARALLELISM=false OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
DEPS="$ROOT/envs/grounding-training-deps-20261001"
export PYTHONPATH="$CLIMATE_SOURCE_STAGE/source/src:$CLIMATE_SOURCE_STAGE/source/scripts:$DEPS/numeric-site:$DEPS/site:$ROOT/envs/grounding-cpu-40d84a3/site:$ROOT/envs/scifact-component-cpu-426ff7343fb3/tokenizer-site${PYTHONPATH:+:$PYTHONPATH}"
exec python "$CLIMATE_SOURCE_STAGE/source/scripts/prepare_scifact_shared_supervision.py" \
  --source-git "$CLIMATE_SOURCE_GIT" --source-sha "$CLIMATE_SOURCE_SHA" \
  --source-archive "$CLIMATE_SOURCE_STAGE/source.tar" --wrapper-sha "$CLIMATE_CPU_WRAPPER_SHA"
