#!/bin/bash
# CPU-only audit/scoring replay of immutable, already-reaped model output.
#SBATCH --job-name=climate-pair-score
#SBATCH --account=punim2936
#SBATCH --partition=sapphire,cascade
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=00:10:00
#SBATCH --no-requeue
set -euo pipefail
umask 077
: "${SLURM_JOB_ID:?CPU allocation required}"
: "${CLIMATE_REPLAY_STAGE:?}"
: "${CLIMATE_REPLAY_GIT:?}"
: "${CLIMATE_REPLAY_ARCHIVE_SHA:?}"
test -z "${CUDA_VISIBLE_DEVICES:-}"
ROOT=/data/gpfs/projects/punim2936/portfolio_20260903/climate-public-retrieval-v2
case "$(readlink -f "$CLIMATE_REPLAY_STAGE")" in "$ROOT/envs/paired-score-"*) ;; *) exit 90 ;; esac
test "$(sha256sum "$CLIMATE_REPLAY_STAGE/source.tar" | cut -d' ' -f1)" = "$CLIMATE_REPLAY_ARCHIVE_SHA"
test "$(cat "$CLIMATE_REPLAY_STAGE/source/SOURCE_REVISION")" = "$CLIMATE_REPLAY_GIT"
test "$(sha256sum "$0" | cut -d' ' -f1)" = "$(tar -xOf "$CLIMATE_REPLAY_STAGE/source.tar" hpc/scifact_paired_score_cpu.sbatch | sha256sum | cut -d' ' -f1)"
module purge
module load GCC/11.3.0 OpenMPI/4.1.4 PyTorch/2.1.2-CUDA-12.2.0
export PYTHONDONTWRITEBYTECODE=1 PYTHONNOUSERSITE=1 PYTHONHASHSEED=0
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 USE_TORCH=1 USE_TF=0 TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2
DEPS="$ROOT/envs/grounding-training-deps-20261001"
export PYTHONPATH="$CLIMATE_REPLAY_STAGE/source/src:$CLIMATE_REPLAY_STAGE/source/scripts:$DEPS/numeric-site:$DEPS/site:$ROOT/envs/grounding-cpu-40d84a3/site:$ROOT/envs/scifact-component-cpu-426ff7343fb3/tokenizer-site${PYTHONPATH:+:$PYTHONPATH}"
timeout --signal=TERM --kill-after=10s 570s python - <<'PY'
import json, os, pathlib, subprocess, sys, time
from climate_rag.scifact_read_continuation import ordered_write
from climate_rag.scifact_semantic_contract import TOKENIZER_SHA, checked
from climate_rag.scifact_utility_contract import identity
from run_scifact_grounding_train_operator import ROOT, require, sha
from scifact_paired_comparison import validate_pair, compare_quality
from prepare_scifact_semantic_pair import verify_source_tree

stage = pathlib.Path(os.environ['CLIMATE_REPLAY_STAGE'])
source = stage/'source'
verify_source_tree(source, stage/'source.tar', os.environ['CLIMATE_REPLAY_ARCHIVE_SHA'],
                   os.environ['CLIMATE_REPLAY_GIT'])
original = ROOT/'runs/scifact-evidence-commit-paired-v2-v3-20261002'
release_path = ROOT/'envs/paired-verifier-5a12fd9f398a/paired-release.authorized.json'
release_sha = 'a09e0488d2b2bcc1fee05ea242f3820b91eadc44c126121f48f4007c1b80d96e'
release = json.loads(checked(release_path, release_sha))
validate_pair(release)
reservation = json.loads((original/'reserved.json').read_bytes())
require(reservation['job_id'] == '31980221' and reservation['release_sha256'] == release_sha
        and reservation['source_git'] == release['source_git'], 'original_execution_identity')
for index, v in enumerate(['v2', 'v3']):
    child = json.loads((original/('release-'+v+'.json')).read_bytes())
    require(identity(child) == identity(release['children'][index])
            and pathlib.Path(child['output']) == original/v, 'child_release_bound_to_parent')
    proof = json.loads((original/v/'worker-exit.json').read_bytes())
    require(proof['child_reaped'] and proof['returncode'] == 0 and proof['interrupted'] is None,
            'both_original_workers_reaped')
tokenizer = ROOT/'posthoc/scifact-read-continuation-fc4ffd61a761/tokenizer'
for name, digest in TOKENIZER_SHA.items():
    checked(tokenizer/name, digest)
out = ROOT/'posthoc'/('paired-score-'+os.environ['CLIMATE_REPLAY_GIT'][:12])
out.mkdir(mode=0o700)
def snapshot():
    paths = sorted(original.rglob('*'))
    require(not any(p.is_symlink() for p in paths), 'original_symlink')
    return {p.relative_to(original).as_posix(): sha(p) for p in paths if p.is_file()}
before = snapshot()
ordered_write(out/'original-manifest-before.json', before)
ordered_write(out/'replay-reserved.json', {'original_job': '31980221',
    'executed_source_git': release['source_git'], 'replay_source_git': os.environ['CLIMATE_REPLAY_GIT'],
    'replay_archive_sha256': os.environ['CLIMATE_REPLAY_ARCHIVE_SHA'],
    'release_sha256': release_sha, 'cpu_job': os.environ['SLURM_JOB_ID'],
    'model_calls': 0, 'training': False, 'original_manifest_sha256': identity(before)})
reports, statuses = [], []
began = time.monotonic()
try:
    for v in ['v2', 'v3']:
        child_release = original/('release-'+v+'.json')
        command = [sys.executable, str(source/'scripts/score_scifact_evidence_commit.py'),
            '--release', str(child_release), '--release-sha', sha(child_release),
            '--model-dir', str(tokenizer), '--report-directory', str(out/v)]
        with (out/(v+'-score.log')).open('xb') as stream:
            result = subprocess.run(command, stdout=stream, stderr=stream, timeout=240, check=False)
        statuses.append({'version': v, 'returncode': result.returncode})
        name = 'quality.json' if result.returncode == 0 else 'no-quality.json'
        reports.append(json.loads((out/v/name).read_bytes()))
    ordered_write(out/'comparison.json', {**compare_quality(reports), 'protocol_results': reports,
        'original_cost_sha256': sha(original/'cost-before-gold.json'),
        'original_failure_sha256': sha(original/'comparison.json'),
        'replay_elapsed_seconds': time.monotonic()-began, 'model_calls': 0})
finally:
    after = snapshot()
    ordered_write(out/'original-manifest-after.json', after)
    ordered_write(out/'replay-exit.json', {'original_unchanged': before == after,
        'original_manifest_sha256': identity(after), 'child_statuses': statuses,
        'elapsed_seconds': time.monotonic()-began, 'model_calls': 0})
    require(before == after, 'original_artifacts_changed')
require(all(r['status'] == 'scored' for r in reports) and len(reports) == 2, 'replay_no_quality')
print(json.dumps({'status':'paired_scored_cpu_replay','output':str(out),'model_calls':0}))
PY
