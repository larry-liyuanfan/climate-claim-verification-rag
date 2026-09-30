#!/bin/bash
#SBATCH --job-name=climate-semantic-prep
#SBATCH --account=punim2936
#SBATCH --partition=sapphire
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --time=00:10:00
#SBATCH --no-requeue
set -euo pipefail
umask 077
: "${SLURM_JOB_ID:?CPU allocation required}"
: "${CLIMATE_PREP_GIT:?frozen source required}"
: "${CLIMATE_PREP_SOURCE_TAR:?source archive required}"
: "${CLIMATE_PREP_SOURCE_SHA:?source hash required}"
: "${CLIMATE_PREP_INPUT_TAR:?allowlisted private input required}"
: "${CLIMATE_PREP_INPUT_SHA:?input hash required}"
CLIMATE_ROOT=/data/gpfs/projects/punim2936/portfolio_20260903/climate-public-retrieval-v2
case "$CLIMATE_PREP_SOURCE_TAR" in "$CLIMATE_ROOT"/envs/scifact-semantic-source-*.tar) ;; *) exit 2 ;; esac
case "$CLIMATE_PREP_INPUT_TAR" in "$CLIMATE_ROOT"/envs/scifact-semantic-inputs-*.tar) ;; *) exit 2 ;; esac
test "$(sha256sum "$CLIMATE_PREP_SOURCE_TAR" | cut -d' ' -f1)" = "$CLIMATE_PREP_SOURCE_SHA"
test "$(sha256sum "$CLIMATE_PREP_INPUT_TAR" | cut -d' ' -f1)" = "$CLIMATE_PREP_INPUT_SHA"
test "$(sha256sum "$0" | cut -d' ' -f1)" = "$(tar -xOf "$CLIMATE_PREP_SOURCE_TAR" hpc/scifact_semantic_cpu_prepare.sbatch | sha256sum | cut -d' ' -f1)"
CLIMATE_WORK=$(mktemp -d "${SLURM_TMPDIR:-${TMPDIR:-/tmp}}/climate-semantic-prep-$SLURM_JOB_ID-XXXXXX")
mkdir "$CLIMATE_WORK/source"
tar --no-same-owner -xf "$CLIMATE_PREP_SOURCE_TAR" -C "$CLIMATE_WORK/source"
test "$(cat "$CLIMATE_WORK/source/SOURCE_REVISION")" = "$CLIMATE_PREP_GIT"
module purge
module load GCC/11.3.0 OpenMPI/4.1.4 PyTorch/2.1.2-CUDA-12.2.0
export PYTHONDONTWRITEBYTECODE=1 PYTHONNOUSERSITE=1 PYTHONHASHSEED=0 CUDA_VISIBLE_DEVICES=""
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 TOKENIZERS_PARALLELISM=false
python - "$CLIMATE_WORK" "$CLIMATE_ROOT" "$CLIMATE_PREP_SOURCE_TAR" "$CLIMATE_PREP_SOURCE_SHA" "$CLIMATE_PREP_GIT" "$CLIMATE_PREP_INPUT_TAR" <<'PY'
import subprocess
import sys
from pathlib import Path

work, root, source_tar, source_sha, git, input_tar = sys.argv[1:]
work, root = Path(work), Path(root)
source = work / "source"
sys.path.insert(0, str(source / "scripts"))
from run_budget_agent_full_operator import ARCHIVES, digest, safe_extract, read_only_tree
from run_sentence_agent_v3_operator import v3_runtime_environment

for name in ("runtime", "overlay"):
    filename, sha = ARCHIVES[name]
    path = root / "envs" / filename
    if digest(path) != sha:
        raise ValueError("frozen_runtime_archive_mismatch")
    safe_extract(path, work / name, runtime=name == "runtime")
safe_extract(Path(input_tar), work / "inputs")
read_only_tree(work / "inputs")
read_only_tree(source)
subprocess.run([sys.executable, "-m", "pip", "install", "--no-index", "--no-deps", "--require-hashes",
    "--no-compile", "--no-cache-dir", "--disable-pip-version-check", "--target", str(work / "overlay-site"),
    "--find-links", str(work / "overlay/wheels"), "-r", str(work / "overlay/requirements.lock")], check=True)
env = v3_runtime_environment(work, source)
subprocess.run([sys.executable, str(source / "scripts/prepare_scifact_semantic_pair.py"),
    "--root", str(root), "--inputs", str(work / "inputs"),
    "--output", str(root / "posthoc" / ("scifact-semantic-preparation-" + git[:12])),
    "--source-git", git, "--source-archive", source_tar, "--source-sha", source_sha], env=env, check=True)
PY
