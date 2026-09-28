#!/bin/bash
# Source inside an allocated node only. No project environment/cache installation.
budget_agent_activate_runtime() {
  : "${CLIMATE_RUNTIME_TAR:?verified old Climate Py3.10 overlay archive}"
  : "${CLIMATE_RUNTIME_SHA256:?runtime archive digest}"
  : "${CLIMATE_OVERLAY_TAR:?CPython 3.10 Linux LangChain wheels archive}"
  : "${CLIMATE_OVERLAY_SHA256:?wheel archive digest}"
  test "$(sha256sum "${CLIMATE_RUNTIME_TAR}" | cut -d' ' -f1)" = "${CLIMATE_RUNTIME_SHA256}"
  test "$(sha256sum "${CLIMATE_OVERLAY_TAR}" | cut -d' ' -f1)" = "${CLIMATE_OVERLAY_SHA256}"
  export CLIMATE_RUNTIME_TAR CLIMATE_OVERLAY_TAR
  "${CLIMATE_PYTHON}" - <<'PY'
import os, pathlib, tarfile
root = pathlib.Path(os.environ['TASK_ROOT'])
for env, name in [('CLIMATE_RUNTIME_TAR', 'runtime'), ('CLIMATE_OVERLAY_TAR', 'overlay')]:
    destination = root / name
    destination.mkdir()
    with tarfile.open(os.environ[env]) as archive:
        members = archive.getmembers()
        if name == 'runtime':
            # Do not restore nonportable venv Python symlinks or console scripts.
            members = [m for m in members if pathlib.PurePosixPath(m.name).as_posix().startswith('lib/python3.10/site-packages/')]
            if not members:
                raise ValueError('runtime site-packages absent')
        for member in members:
            path = (destination / member.name).resolve()
            if not path.is_relative_to(destination) or not (member.isfile() or member.isdir()):
                raise ValueError('unsafe runtime archive member')
        archive.extractall(destination, members=members)
PY
  "${CLIMATE_PYTHON}" -m pip install --no-index --no-deps --require-hashes \
    --no-compile --no-cache-dir --disable-pip-version-check \
    --target "${TASK_ROOT}/overlay-site" \
    --find-links "${TASK_ROOT}/overlay/wheels" -r "${TASK_ROOT}/overlay/requirements.lock"
  export PYTHONPATH="${TASK_ROOT}/overlay-site:${TASK_ROOT}/runtime/lib/python3.10/site-packages:${PYTHONPATH:-}"
  export PYTHONNOUSERSITE=1
}
