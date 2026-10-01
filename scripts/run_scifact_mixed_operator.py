"""Allocated mixed launch: frozen generator extraction then exec the parent."""
from __future__ import annotations

import os
from pathlib import Path
import sys
import time

from climate_rag.scifact_mixed_launch import ARCHIVE_NAME, ARCHIVE_SHA, ROOT, parent_command, scratch_model_paths
from climate_rag.scifact_read_continuation import ordered_write
from climate_rag.scifact_state_supervision import require
from run_budget_agent_full_operator import ARCHIVES
from run_scifact_component_operator import generator_only
from run_scifact_mixed_training import release_inputs
from prepare_scifact_semantic_pair import verify_source_tree


def main() -> None:
    source = Path(__file__).resolve().parents[1]
    work = Path(os.environ['CLIMATE_MIXED_WORK'])
    require(work == work.resolve() and source == work/'source', 'actual_mixed_scratch_source')
    verify_source_tree(source, Path(os.environ['CLIMATE_SOURCE_TAR']),
                       os.environ['CLIMATE_SOURCE_SHA256'], os.environ['CLIMATE_SOURCE_GIT'])
    path = Path(os.environ['CLIMATE_MIXED_RELEASE_FILE'])
    digest = os.environ['CLIMATE_MIXED_RELEASE_SHA']
    release, _, _ = release_inputs(path, digest)
    output = Path(release['output'])
    require(not Path(str(output)+'-execution').exists(), 'unused_parent_execution_required')
    allocation = Path(str(output)+'-allocation')
    allocation.mkdir(mode=0o700, exist_ok=False)
    ordered_write(allocation/'started.json', {'source_git': release['source_git'], 'release_sha256': digest,
        'job_id': os.environ['SLURM_JOB_ID'], 'started_unix': time.time(), 'automatic_retry': False})
    try:
        require(ARCHIVES['input'] == (ARCHIVE_NAME, ARCHIVE_SHA), 'existing_generator_archive_binding')
        size = generator_only(ROOT/'envs'/ARCHIVE_NAME, work/'input', ARCHIVE_SHA)
        scratch_model_paths(release, work/'input', source=source, work=work, job_id=os.environ['SLURM_JOB_ID'])
        ordered_write(allocation/'generator-extracted.json', {'bytes': size, 'archive_sha256': ARCHIVE_SHA,
            'release_sha256': digest, 'model_loaded': False, 'release_rewritten': False})
        command = parent_command(sys.executable, source, path, digest, work/'input')
        os.execv(sys.executable, command)  # Parent owns execution reservation and sole bounded child.
    except BaseException as exc:
        ordered_write(allocation/'launch-failed.json', {'exception_type': type(exc).__name__, 'automatic_retry': False})
        raise


if __name__ == '__main__':
    main()
