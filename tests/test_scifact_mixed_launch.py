"""Launch-only synthetic checks: no real data, model, scheduler or math rerun."""
import copy
import json
import sys

import pytest

import climate_rag.scifact_mixed_launch as launch
from climate_rag.scifact_mixed_inputs import SCOPE, input_config_sha
from climate_rag.scifact_semantic_contract import MODEL_SHA, sha
import run_scifact_mixed_training as trainer


def model_release():
    return {'model_archive_name': launch.ARCHIVE_NAME, 'model_archive_sha256': launch.ARCHIVE_SHA,
            'model_relative_directory': launch.MODEL_RELATIVE, 'model_relative_manifest': launch.MANIFEST_RELATIVE,
            'model_manifest_sha256': MODEL_SHA}


def test_model_root_is_only_fixed_input_in_this_allocation(tmp_path, monkeypatch):
    work = (tmp_path/'climate-mixed-123-unique').resolve()
    source = work/'source'
    source.mkdir(parents=True)
    model_root = work/'input'
    (model_root/launch.MODEL_RELATIVE).mkdir(parents=True)
    (model_root/launch.MANIFEST_RELATIVE).write_text('{}')
    monkeypatch.setattr(launch, 'SCRATCH_PARENTS', (tmp_path.resolve(),))
    kwargs = {'source': source, 'work': work, 'job_id': '123'}
    expected = (model_root/launch.MODEL_RELATIVE, model_root/launch.MANIFEST_RELATIVE)
    assert launch.scratch_model_paths(model_release(), model_root, **kwargs) == expected
    for root in (work, tmp_path, model_root/'models'):
        with pytest.raises(ValueError, match='fixed_scratch_model_root_only'):
            launch.scratch_model_paths(model_release(), root, **kwargs)
    with pytest.raises(ValueError, match='current_allocation_private_scratch'):
        launch.scratch_model_paths(model_release(), model_root, **{**kwargs, 'job_id': '124'})
    for field, bad in (('model_relative_directory', '../other'), ('model_archive_sha256', '0'*64),
                       ('model_manifest_sha256', '0'*64), ('model_directory', '/tmp/anything')):
        with pytest.raises(ValueError, match='frozen_generator_relative_identity'):
            launch.scratch_model_paths({**model_release(), field: bad}, model_root, **kwargs)


def test_CPU_complete_is_required_before_release_construction(monkeypatch):
    receipt = {'status': 'complete', 'scope': SCOPE, 'source_git': launch.PREPARATION_GIT,
        'source_bridge_validated': True, 'input_config_sha256': input_config_sha(),
        'denominator': 144, 'rows': 223, 'planned_updates': 36,
        'status_counts': {'ready': 144, 'failed': 0, 'unknown': 0}, 'model_calls': 0, 'optimizer_steps': 0,
        'files': {'prepared.json': 'a'*64, 'plan.json': 'b'*64, 'roster.json': 'c'*64}}
    raw = json.dumps(receipt).encode()
    monkeypatch.setattr(launch, 'PREPARATION_SHA', sha(raw))
    fields = launch.preparation_fields(raw)
    assert fields['prepared_file_sha256'] == 'a'*64 and fields['preparation_receipt_sha256'] == sha(raw)
    assert fields['prepared_directory'].startswith('/data/gpfs/') and '\\' not in fields['prepared_directory']
    with pytest.raises(ValueError, match='accepted_CPU_complete_hash'):
        launch.preparation_fields(raw+b' ')
    for key, value in (('status', 'failed'), ('rows', 222), ('source_bridge_validated', False),
                       ('status_counts', {'ready': 143, 'failed': 1, 'unknown': 0})):
        bad = json.dumps({**receipt, key: value}).encode()
        monkeypatch.setattr(launch, 'PREPARATION_SHA', sha(bad))
        with pytest.raises(ValueError, match='accepted_complete_144_223_36_only'):
            launch.preparation_fields(bad)


def test_full_transitive_source_manifest_and_runtime_caps(tmp_path, monkeypatch):
    source = tmp_path/'source'
    files = {'src/climate_rag/a.py': b'# fixture\n', 'scripts/run_scifact_mixed_operator.py': b'# operator\n',
             'scripts/transitive.py': b'# dependency\n'}
    for name, value in {**files, launch.WRAPPER: b'# wrapper\n'}.items():
        path = source/name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(value)
    release = {**model_release(), 'max_runtime_seconds': 1500, 'resource_cap': launch.RESOURCE_CAP,
        'source_git': 'a'*40, 'source_archive_sha256': 'b'*64, 'wrapper_path': launch.WRAPPER,
        'wrapper_sha256': sha(b'# wrapper\n'), 'source_files_sha256': {n: sha(v) for n, v in files.items()},
        'runtime_receipt_sha256': launch.RUNTIME_RECEIPT_SHA, 'runtime_files_sha256': launch.RUNTIME_FILES_SHA,
        'python_executable': launch.PYTHON_EXECUTABLE}
    for name, value in {'SLURM_CPUS_PER_TASK': '4', 'CLIMATE_SOURCE_GIT': 'a'*40,
                       'CLIMATE_SOURCE_SHA256': 'b'*64}.items():
        monkeypatch.setenv(name, value)
    launch.launch_identity(release, source)
    missing = copy.deepcopy(release)
    del missing['source_files_sha256']['scripts/transitive.py']
    with pytest.raises(ValueError, match='complete_executable_source_manifest'):
        launch.launch_identity(missing, source)
    with pytest.raises(ValueError, match='frozen_mixed_resource_request'):
        launch.launch_identity({**release, 'max_runtime_seconds': 3600}, source)


def test_operator_calls_parent_and_parent_owns_single_worker(tmp_path, monkeypatch):
    model_root = tmp_path/'private-input'
    release_path = tmp_path/'release.json'
    command = launch.parent_command('python', tmp_path/'source', release_path, 'f'*64, model_root)
    assert '--worker' not in command and command[-2:] == ['--model-root', str(model_root)]
    release = {'output': str(tmp_path/'persistent-output'), 'max_runtime_seconds': 1500}
    monkeypatch.setattr(trainer, 'release_inputs', lambda *args: (release, {}, {}))
    monkeypatch.setattr(trainer, 'scratch_model_paths', lambda *args, **kwargs: None)
    monkeypatch.setenv('CLIMATE_MIXED_WORK', str(tmp_path))
    monkeypatch.setenv('SLURM_JOB_ID', '123')
    monkeypatch.setattr(sys, 'argv', ['run', '--release', str(release_path), '--release-sha', 'f'*64,
                                     '--model-root', str(model_root)])
    seen = []
    def child(cmd, execution, seconds):
        seen.append(cmd)
        assert seconds == 1500 and execution.is_dir() and (execution/'reserved.json').is_file()
        assert cmd.count('--worker') == 1 and cmd[-3:] == ['--model-root', str(model_root), '--worker']
        assert not (tmp_path/'persistent-output').exists()
        return 0
    monkeypatch.setattr(trainer, 'bounded_worker', child)
    trainer.main()
    with pytest.raises(FileExistsError):
        trainer.main()
    assert len(seen) == 1
