"""One allocated training-only launch; no inference, retry, or data selection."""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from typing import Any

ROOT = Path('/data/gpfs/projects/punim2936/portfolio_20260903/climate-public-retrieval-v2')
DEPS = ROOT / 'envs/grounding-training-deps-20261001'
SITES = [DEPS / 'numeric-site', DEPS / 'site', ROOT / 'envs/grounding-cpu-40d84a3/site',
         ROOT / 'envs/scifact-component-cpu-426ff7343fb3/tokenizer-site']


def sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def require(value: bool, message: str) -> None:
    if not value:
        raise ValueError(message)


def import_observation(names: list[str]) -> dict[str, Any]:
    import importlib.metadata as md
    import platform
    from climate_rag.torch_compat import ensure_torch_pytree_compat
    ensure_torch_pytree_compat()
    import torch
    import peft
    import accelerate
    import transformers
    import safetensors
    return {'python': platform.python_version(), 'os_name': os.name,
        'torch': torch.__version__, 'python_executable': sys.executable,
        'versions': {n: md.version(n) for n in names},
        'module_files': {m.__name__: m.__file__ for m in
                         (torch, peft, accelerate, transformers, safetensors)},
        'model_loaded': False, 'generation_calls': 0}


def runtime_check(release: dict[str, Any]) -> dict[str, Any]:
    receipt = DEPS / 'final-receipt.json'
    manifest = DEPS / 'runtime-files.json'
    require(sha(receipt) == release['runtime_receipt_sha256'], 'runtime_receipt_hash')
    info = json.loads(receipt.read_bytes())
    require(info['status'] == 'imports_verified' and info['stderr_empty'] is True
            and not info['dependency_errors'], 'runtime_probe_failed')
    require(sha(manifest) == release['runtime_files_sha256'] == info['runtime_files_sha256'],
            'runtime_manifest_hash')
    expected = json.loads(manifest.read_bytes())
    actual = {p.relative_to(ROOT).as_posix(): sha(p) for site in SITES
              for p in site.rglob('*') if p.is_file()}
    require(not any(p.is_symlink() for site in SITES for p in site.rglob('*'))
            and actual == expected, 'runtime_files_changed')
    observed = import_observation(list(info['versions']))
    require(observed['python_executable'] == release['python_executable'], 'runtime_interpreter_changed')
    require(all(observed[k] == info[k] for k in ('python', 'os_name', 'torch', 'versions', 'module_files')),
            'runtime_actual_import_changed')
    return observed


def main() -> None:
    require(os.name == 'posix' and bool(os.environ.get('SLURM_JOB_ID'))
            and bool(os.environ.get('CUDA_VISIBLE_DEVICES')), 'allocated_gpu_required')
    source = Path(__file__).resolve().parents[1]
    work = Path(os.environ['CLIMATE_GROUNDING_WORK']).resolve()
    require(source.parent == work and work.name.startswith('climate-grounding-'), 'private_scratch')
    release_path = Path(os.environ['CLIMATE_GROUNDING_RELEASE_FILE'])
    release_sha = os.environ['CLIMATE_GROUNDING_RELEASE_SHA']
    require(sha(release_path) == release_sha, 'release_hash')
    release = json.loads(release_path.read_bytes())
    require(release['authorization'] == 'coordinator_exact_hash_release'
            and release['purpose'] == 'train' and release['max_runtime_seconds'] == 1500,
            'training_only_release_required')
    require((source / 'SOURCE_REVISION').read_text().strip() == release['source_git']
            == os.environ['CLIMATE_SOURCE_GIT'], 'source_identity')
    require(release['source_archive_sha256'] == os.environ['CLIMATE_SOURCE_SHA256']
            and sha(source / 'hpc/scifact_grounding_train.sbatch') == release['wrapper_sha256'],
            'source_wrapper_identity')
    output = ROOT / 'runs/scifact-grounding-training-20261001-v1'
    require(release['output'] == str(output) and not output.exists(), 'fixed_unused_output')
    allocation = output.with_name(output.name + '-allocation')
    allocation.mkdir(mode=0o700)
    observed = runtime_check(release)
    # Heavy extraction/model work happens only inside the GPU allocation.
    from climate_rag.component_execution import durable
    from run_budget_agent_full_operator import ARCHIVES
    from run_scifact_component_operator import generator_only
    durable(allocation / 'runtime-observed.json', observed)
    durable(allocation / 'started.json', {'source_git': release['source_git'],
        'release_sha256': release_sha, 'job_id': os.environ['SLURM_JOB_ID'],
        'started_unix': time.time(), 'training_only': True})
    filename, digest = ARCHIVES['input']
    require(release['model_archive_sha256'] == digest, 'frozen_model_archive')
    generator_only(ROOT / 'envs' / filename, work / 'input', digest)
    bundle = ROOT / 'posthoc/scifact-grounding-candidate-40d84a377bd1'
    command = [sys.executable, str(source / 'scripts/run_scifact_grounding_candidate.py'),
        'train', '--bundle', str(bundle), '--data-sha', release['data_manifest_sha256'],
        '--output', str(output), '--model', str(work / 'input/models/generator/model'),
        '--model-manifest', str(work / 'input/models/generator/model_manifest.json'),
        '--release', str(release_path), '--release-sha', release_sha]
    completed = subprocess.run(command, check=False)
    durable(allocation / 'ended.json', {'returncode': completed.returncode,
        'ended_unix': time.time(), 'no_automatic_retry': True})
    raise SystemExit(completed.returncode)


if __name__ == '__main__':
    main()
