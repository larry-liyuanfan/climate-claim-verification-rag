#!/bin/bash
# Tokenizer only: never load model weights, gold or a protected split.
#SBATCH --job-name=climate-bottleneck-cpu
#SBATCH --account=punim2936
#SBATCH --partition=sapphire,cascade
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=00:10:00
#SBATCH --no-requeue
set -euo pipefail
umask 077
: "${SLURM_JOB_ID:?CPU allocation required}"
: "${CLIMATE_SOURCE_STAGE:?}"
: "${CLIMATE_SOURCE_GIT:?}"
: "${CLIMATE_SOURCE_SHA:?}"
: "${CLIMATE_CPU_WRAPPER_SHA:?}"
ROOT=/data/gpfs/projects/punim2936/portfolio_20260903/climate-public-retrieval-v2
case "$(readlink -f "$CLIMATE_SOURCE_STAGE")" in "$ROOT/envs/evidence-bottleneck-"*) ;; *) exit 90 ;; esac
test "$(sha256sum "$CLIMATE_SOURCE_STAGE/source.tar" | cut -d' ' -f1)" = "$CLIMATE_SOURCE_SHA"
test "$(sha256sum "$0" | cut -d' ' -f1)" = "$CLIMATE_CPU_WRAPPER_SHA"
test "$(tar -xOf "$CLIMATE_SOURCE_STAGE/source.tar" hpc/scifact_evidence_bottleneck_cpu.sbatch | sha256sum | cut -d' ' -f1)" = "$CLIMATE_CPU_WRAPPER_SHA"
module purge
module load GCC/11.3.0 OpenMPI/4.1.4 PyTorch/2.1.2-CUDA-12.2.0
export PYTHONHASHSEED=0 USE_TORCH=0 USE_TF=0 PYTHONDONTWRITEBYTECODE=1 PYTHONNOUSERSITE=1
export TOKENIZERS_PARALLELISM=false OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 CUDA_VISIBLE_DEVICES=""
DEPS="$ROOT/envs/grounding-training-deps-20261001"
export PYTHONPATH="$CLIMATE_SOURCE_STAGE/source/src:$CLIMATE_SOURCE_STAGE/source/scripts:$DEPS/numeric-site:$DEPS/site:$ROOT/envs/grounding-cpu-40d84a3/site:$ROOT/envs/scifact-component-cpu-426ff7343fb3/tokenizer-site"
python - <<'PY'
import os
from pathlib import Path
from prepare_scifact_semantic_pair import verify_source_tree
stage = Path(os.environ['CLIMATE_SOURCE_STAGE'])
verify_source_tree(stage/'source', stage/'source.tar', os.environ['CLIMATE_SOURCE_SHA'], os.environ['CLIMATE_SOURCE_GIT'])
PY
exec python "$CLIMATE_SOURCE_STAGE/source/scripts/preflight_scifact_evidence_bottleneck.py" \
  --output "$CLIMATE_SOURCE_STAGE/bottleneck-tokenizer-preflight.json"
