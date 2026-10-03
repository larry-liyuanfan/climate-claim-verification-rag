"""Frozen launch identity and scratch-relative model paths; no model imports."""
from __future__ import annotations

from collections.abc import Iterable, Mapping
import json
import os
from pathlib import Path
from typing import Any

from .scifact_mixed_inputs import SCOPE, input_config_sha
from .scifact_semantic_contract import MODEL_SHA, sha
from .scifact_state_supervision import require

ROOT = Path('/data/gpfs/projects/punim2936/portfolio_20260903/climate-public-retrieval-v2')
WRAPPER = 'hpc/scifact_mixed_train.sbatch'
ARCHIVE_SHA = '563738f0be1f7bf7b99b8de20bcdeab552f7e893166dec260b8f7f1e7951c3c1'
ARCHIVE_NAME = 'budget-agent-inputs-20260929-'+ARCHIVE_SHA+'.tar'
MODEL_RELATIVE = 'models/generator/model'
MANIFEST_RELATIVE = 'models/generator/model_manifest.json'
PREPARATION_GIT = 'cb1e76462537ea597d0d0c163daee2275f8487aa'
PREPARATION_SHA = '131cc07468b1fedd04dc270f9b2170250ec7d35cdb8254c1d941f7c6a0a6f039'
RUNTIME_RECEIPT_SHA = '9878d3e4d45735828b4260d87656fd2827481663266a875725bfe7abab7d390a'
RUNTIME_FILES_SHA = '8d231f4980e0bf94fe26273074588a0f6c0ea67646117871fd97443ad1142697'
PYTHON_EXECUTABLE = '/apps/easybuild-2022/easybuild/software/Compiler/GCCcore/11.3.0/Python/3.10.4/bin/python'
SCRATCH_PARENTS = (Path('/tmp'), Path('/var/tmp'), Path('/jobfs'))
RESOURCE_CAP = {'gpu': 'A100:1', 'gpu_memory_not_presumed': True, 'cpus': 4,
                'host_ram_gib': 32, 'scratch_gib': 30, 'slurm_seconds': 1800}


def executable_names(names: Iterable[str]) -> set[str]:
    # Cover the operator and all transitive script imports, not a partial list.
    return {n for n in names if n.endswith('.py') and n.startswith(('src/', 'scripts/'))}


def preparation_fields(raw: bytes) -> dict[str, Any]:
    require(sha(raw) == PREPARATION_SHA, 'accepted_CPU_complete_hash')
    value = json.loads(raw)
    require(value['status'] == 'complete' and value['scope'] == SCOPE
            and value['source_git'] == PREPARATION_GIT and value['source_bridge_validated'] is True
            and value['input_config_sha256'] == input_config_sha()
            and value['denominator'] == 144 and value['rows'] == 223 and value['planned_updates'] == 36
            and value['status_counts'] == {'ready': 144, 'failed': 0, 'unknown': 0}
            and value['model_calls'] == value['optimizer_steps'] == 0
            and set(value['files']) == {'prepared.json', 'plan.json', 'roster.json'},
            'accepted_complete_144_223_36_only')
    return {'prepared_directory': (ROOT/'posthoc'/('scifact-mixed-program-'+PREPARATION_GIT[:12])).as_posix(),
        'preparation_receipt_sha256': PREPARATION_SHA, 'input_config_sha256': input_config_sha(),
        'prepared_file_sha256': value['files']['prepared.json'],
        'plan_file_sha256': value['files']['plan.json'], 'roster_file_sha256': value['files']['roster.json']}


def model_contract(release: Mapping[str, Any]) -> None:
    require(release['model_archive_name'] == ARCHIVE_NAME and release['model_archive_sha256'] == ARCHIVE_SHA
            and release['model_relative_directory'] == MODEL_RELATIVE
            and release['model_relative_manifest'] == MANIFEST_RELATIVE
            and release['model_manifest_sha256'] == MODEL_SHA
            and 'model_directory' not in release and 'model_manifest' not in release,
            'frozen_generator_relative_identity')


def scratch_model_paths(release: Mapping[str, Any], model_root: Path, *, source: Path,
                        work: Path, job_id: str) -> tuple[Path, Path]:
    model_contract(release)
    require(job_id.isdigit() and work.is_absolute() and work == work.resolve()
            and any(work.is_relative_to(p) for p in SCRATCH_PARENTS)
            and work.name.startswith('climate-mixed-'+job_id+'-')
            and source.resolve() == work/'source', 'current_allocation_private_scratch')
    require(model_root.is_absolute() and model_root == work/'input' and model_root.resolve() == model_root
            and model_root.is_dir() and not any(p.is_symlink() for p in model_root.rglob('*')),
            'fixed_scratch_model_root_only')
    model, manifest = model_root/MODEL_RELATIVE, model_root/MANIFEST_RELATIVE
    require(model.is_dir() and manifest.is_file(), 'extracted_generator_paths_required')
    return model, manifest


def launch_identity(release: Mapping[str, Any], source: Path) -> None:
    model_contract(release)
    require(release['max_runtime_seconds'] == 1500 and release['resource_cap'] == RESOURCE_CAP
            and os.environ.get('SLURM_CPUS_PER_TASK') == '4', 'frozen_mixed_resource_request')
    require(release['source_git'] == os.environ.get('CLIMATE_SOURCE_GIT')
            and release['source_archive_sha256'] == os.environ.get('CLIMATE_SOURCE_SHA256')
            and release['wrapper_path'] == WRAPPER
            and sha((source/WRAPPER).read_bytes()) == release['wrapper_sha256'], 'mixed_launch_source_identity')
    paths = {p.relative_to(source).as_posix(): p for parent in ('src', 'scripts')
             for p in (source/parent).rglob('*.py')}
    require(set(release['source_files_sha256']) == executable_names(paths)
            and all(not p.is_symlink() and sha(p.read_bytes()) == release['source_files_sha256'][n]
                    for n, p in paths.items()), 'complete_executable_source_manifest')
    require(release['runtime_receipt_sha256'] == RUNTIME_RECEIPT_SHA
            and release['runtime_files_sha256'] == RUNTIME_FILES_SHA
            and release['python_executable'] == PYTHON_EXECUTABLE, 'frozen_existing_runtime')


def parent_command(python: str, source: Path, release: Path, digest: str, model_root: Path) -> list[str]:
    return [python, str(source/'scripts/run_scifact_mixed_training.py'), '--release', str(release),
            '--release-sha', digest, '--model-root', str(model_root)]
