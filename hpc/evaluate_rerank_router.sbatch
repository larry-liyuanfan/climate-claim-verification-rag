#!/bin/bash
#SBATCH --job-name=climate-router
#SBATCH --account=punim2936
#SBATCH --partition=sapphire
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=4G
#SBATCH --time=00:10:00
#SBATCH --output=slurm-climate-router-%j.out

set -euo pipefail

: "${CLIMATE_REPO:?Set CLIMATE_REPO to the Spartan checkout}"
: "${SOURCE_RUN:?Set SOURCE_RUN to the verified 4B reranker run directory}"
: "${OUTPUT_DIR:?Set OUTPUT_DIR outside the Git checkout}"
: "${NATIVE_ENV:?Set NATIVE_ENV to the verified Climate virtual environment}"

cd "$CLIMATE_REPO"
module purge
module load GCC/11.3.0 OpenMPI/4.1.4 PyTorch/2.1.2-CUDA-12.2.0
source "$NATIVE_ENV/bin/activate"
export PYTHONNOUSERSITE=1
export PYTHONPATH="$CLIMATE_REPO/src:$NATIVE_ENV/lib/python3.10/site-packages${PYTHONPATH:+:$PYTHONPATH}"
python scripts/evaluate_rerank_router.py \
  --predictions "$SOURCE_RUN/predictions.jsonl" \
  --source-metrics "$SOURCE_RUN/metrics.json" \
  --output-dir "$OUTPUT_DIR"
