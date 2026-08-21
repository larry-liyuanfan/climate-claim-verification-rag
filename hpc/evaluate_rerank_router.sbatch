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
module load GCCcore/11.3.0 Python/3.10.4
source "$NATIVE_ENV/bin/activate"
export PYTHONPATH="$CLIMATE_REPO/src${PYTHONPATH:+:$PYTHONPATH}"
python scripts/evaluate_rerank_router.py \
  --predictions "$SOURCE_RUN/predictions.jsonl" \
  --source-metrics "$SOURCE_RUN/metrics.json" \
  --output-dir "$OUTPUT_DIR"
