#!/bin/bash
# CPU-only receipt audit; never calls a model or extracts model weights.
#SBATCH --job-name=clim-confirm-audit
#SBATCH --account=punim2936
#SBATCH --partition=sapphire
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=2G
#SBATCH --tmp=1G
#SBATCH --time=00:05:00
#SBATCH --no-requeue
set -euo pipefail
: "${SLURM_JOB_ID:?CPU allocation required}"
export CUDA_VISIBLE_DEVICES="" PYTHONDONTWRITEBYTECODE=1 PYTHONNOUSERSITE=1
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
module purge
module load GCC/11.3.0 OpenMPI/4.1.4 PyTorch/2.1.2-CUDA-12.2.0
CLIMATE_AUDIT_TMP=$(mktemp -d "${SLURM_TMPDIR:-${TMPDIR:?}}/climate-confirm-audit-XXXXXX")
export CLIMATE_AUDIT_TMP
python - <<'PY'
import hashlib, io, os, pathlib, tarfile, zipfile
root = pathlib.Path('/data/gpfs/projects/punim2936/portfolio_20260903/climate-public-retrieval-v2/envs')
target = pathlib.Path(os.environ['CLIMATE_AUDIT_TMP'])
archive = root / 'budget-agent-scorer-source-208ff931badff270cbbc9593c8ab54f5c79aec9e.tar'
assert hashlib.sha256(archive.read_bytes()).hexdigest() == '7498d367565eeb59bb7cf724c33f835608a0c9f01da3768d7308700a380c00a1'
source = target / 'source'
source.mkdir()
with tarfile.open(archive) as bundle:
    for member in bundle.getmembers():
        assert (source / member.name).resolve().is_relative_to(source)
        assert member.isfile() or member.isdir()
    bundle.extractall(source)
overlay = root / 'budget-agent-linux-overlay-20260929-e7223ac2f4c08a736e763d485a47dcb088ce62c28fb9f1da556661a3f71da0ce.tar.gz'
assert hashlib.sha256(overlay.read_bytes()).hexdigest() == 'e7223ac2f4c08a736e763d485a47dcb088ce62c28fb9f1da556661a3f71da0ce'
site = target / 'site'
site.mkdir()
count = 0
with tarfile.open(overlay) as bundle:
    for member in bundle.getmembers():
        if pathlib.PurePosixPath(member.name).name.startswith(('pydantic-', 'pydantic_core-', 'typing_extensions-', 'typing_inspection-', 'annotated_types-')) and member.name.endswith('.whl'):
            with zipfile.ZipFile(io.BytesIO(bundle.extractfile(member).read())) as wheel:
                assert all((site / p).resolve().is_relative_to(site) for p in wheel.namelist())
                wheel.extractall(site)
            count += 1
assert count == 5
PY
export CLIMATE_SCORE_TREE="${CLIMATE_AUDIT_TMP}/source"
export PYTHONPATH="${CLIMATE_AUDIT_TMP}/site:${CLIMATE_SCORE_TREE}/src${PYTHONPATH:+:${PYTHONPATH}}"
umask 077
python /data/gpfs/projects/punim2936/portfolio_20260903/climate-public-retrieval-v2/envs/audit_budget_confirmation_31520350.py
