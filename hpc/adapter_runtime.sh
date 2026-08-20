#!/bin/bash

setup_climate_adapter_runtime() {
  : "${NATIVE_ENV:?set the existing native climate virtual environment}"
  : "${ARTIFACT_ROOT:?set project ARTIFACT_ROOT for pip/model caches}"

  local node_local_root dependency_dir
  node_local_root="${SLURM_TMPDIR:-${TMPDIR:-/tmp}}"
  dependency_dir="${node_local_root}/climate-adapter-deps"
  mkdir -p "${dependency_dir}" "${ARTIFACT_ROOT}/.cache/pip"
  export PIP_CACHE_DIR="${ARTIFACT_ROOT}/.cache/pip"
  python -m pip install \
    --no-input \
    --retries 5 \
    --target "${dependency_dir}" \
    --no-deps \
    "peft==0.17.1"
  export PYTHONPATH="${dependency_dir}:${NATIVE_ENV}/lib/python3.10/site-packages:${PYTHONPATH:-}"

  python - <<'PY'
import accelerate
import peft
import torch
import transformers

print(
    "adapter_runtime",
    {
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "transformers": transformers.__version__,
        "peft": peft.__version__,
        "accelerate": accelerate.__version__,
    },
    flush=True,
)
assert peft.__version__ == "0.17.1", peft.__version__
PY
}
