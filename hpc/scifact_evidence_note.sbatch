#!/bin/bash
# Old12 note+terminal draft: exact coordinator release required; no automatic retry.
#SBATCH --job-name=climate-evidence-note
#SBATCH --account=punim2936
#SBATCH --partition=gpu-a100,gpu-a100-short
#SBATCH --gres=gpu:A100:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --tmp=30G
#SBATCH --time=01:10:00
#SBATCH --no-requeue
set -euo pipefail
umask 077
# Cover source checks/extraction, runtime imports, inference and scoring, not only wait().
# Slurm's 4200s allocation is the final cgroup-wide hard bound for every descendant.
if [ "${1:-}" != "--bounded-body" ]; then
  exec timeout --signal=TERM --kill-after=30s 3900s bash "$0" --bounded-body
fi
: "${SLURM_JOB_ID:?allocated job required}"
: "${CLIMATE_SOURCE_TAR:?}"
: "${CLIMATE_SOURCE_SHA256:?}"
: "${CLIMATE_SOURCE_GIT:?}"
: "${CLIMATE_GROUNDING_RELEASE_FILE:?}"
: "${CLIMATE_GROUNDING_RELEASE_SHA:?}"
ROOT=/data/gpfs/projects/punim2936/portfolio_20260903/climate-public-retrieval-v2
for CLIMATE_ASSET in "$CLIMATE_SOURCE_TAR" "$CLIMATE_GROUNDING_RELEASE_FILE"; do
  case "$(readlink -f "$CLIMATE_ASSET")" in "${ROOT}/envs/"*) ;; *) exit 90 ;; esac
done
test "$(sha256sum "$CLIMATE_SOURCE_TAR" | cut -d' ' -f1)" = "$CLIMATE_SOURCE_SHA256"
test "$(sha256sum "$CLIMATE_GROUNDING_RELEASE_FILE" | cut -d' ' -f1)" = "$CLIMATE_GROUNDING_RELEASE_SHA"
# BEGIN SOURCE_ARCHIVE_GUARD
test "$(tar -tf "$CLIMATE_SOURCE_TAR" | grep -cx 'SOURCE_REVISION')" -eq 1
test "$(tar -xOf "$CLIMATE_SOURCE_TAR" SOURCE_REVISION | wc -c)" -eq 41
test "$(tar -xOf "$CLIMATE_SOURCE_TAR" SOURCE_REVISION)" = "$CLIMATE_SOURCE_GIT"
test "$(sha256sum "$0" | cut -d' ' -f1)" = "$(tar -xOf "$CLIMATE_SOURCE_TAR" hpc/scifact_evidence_note.sbatch | sha256sum | cut -d' ' -f1)"
# END SOURCE_ARCHIVE_GUARD
module purge
module load GCC/11.3.0 OpenMPI/4.1.4 PyTorch/2.1.2-CUDA-12.2.0
CLIMATE_SCRATCH_PARENT="${SLURM_TMPDIR:-${TMPDIR:-}}"
case "$(readlink -f "$CLIMATE_SCRATCH_PARENT")" in /tmp|/tmp/*|/var/tmp|/var/tmp/*|/jobfs/*) ;; *) exit 91 ;; esac
CLIMATE_GROUNDING_WORK=$(mktemp -d "${CLIMATE_SCRATCH_PARENT}/climate-grounding-${SLURM_JOB_ID}-XXXXXX")
export CLIMATE_GROUNDING_WORK PYTHONDONTWRITEBYTECODE=1 PYTHONNOUSERSITE=1 PYTHONHASHSEED=0
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 USE_TORCH=1 USE_TF=0 TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=8 MKL_NUM_THREADS=8
export HF_HOME="$CLIMATE_GROUNDING_WORK/cache" TMPDIR="$CLIMATE_GROUNDING_WORK/tmp"
mkdir -m 700 "$HF_HOME" "$TMPDIR"
export XDG_CACHE_HOME="$HF_HOME" TORCH_HOME="$HF_HOME/torch" CUDA_CACHE_PATH="$HF_HOME/cuda"
export TORCH_EXTENSIONS_DIR="$HF_HOME/extensions" TRITON_CACHE_DIR="$HF_HOME/triton"
mkdir -m 700 "$TORCH_HOME" "$CUDA_CACHE_PATH" "$TORCH_EXTENSIONS_DIR" "$TRITON_CACHE_DIR"
python - <<'PY'
import os, pathlib, tarfile
target = pathlib.Path(os.environ['CLIMATE_GROUNDING_WORK']) / 'source'
target.mkdir(mode=0o700)
with tarfile.open(os.environ['CLIMATE_SOURCE_TAR']) as archive:
    paths = set()
    for member in archive.getmembers():
        path = (target / member.name).resolve()
        if not path.is_relative_to(target) or not (member.isfile() or member.isdir()) or path in paths:
            raise ValueError('unsafe source archive')
        paths.add(path)
    archive.extractall(target)
PY
DEPS="$ROOT/envs/grounding-training-deps-20261001"
export PYTHONPATH="$CLIMATE_GROUNDING_WORK/source/src:$CLIMATE_GROUNDING_WORK/source/scripts:$DEPS/numeric-site:$DEPS/site:$ROOT/envs/grounding-cpu-40d84a3/site:$ROOT/envs/scifact-component-cpu-426ff7343fb3/tokenizer-site${PYTHONPATH:+:$PYTHONPATH}"
exec python "$CLIMATE_GROUNDING_WORK/source/scripts/run_scifact_evidence_note_operator.py"
