#!/bin/bash

public_v2_activate_runtime() {
  : "${CLIMATE_V2_ROOT:?Set the isolated public-v2 root}"
  local node_tmp="${SLURM_TMPDIR:-${TMPDIR:-}}"
  : "${node_tmp:?Spartan did not expose node-local temporary storage}"
  case "$(readlink -f "${node_tmp}")" in
    /tmp|/tmp/*|/var/tmp|/var/tmp/*|/jobfs/*) ;;
    *) echo "Refusing non-local temporary directory: ${node_tmp}" >&2; return 92 ;;
  esac

  local archive="${CLIMATE_V2_ROOT}/envs/runtime-py310.tar.gz"
  local runtime_env="${node_tmp}/climate-v2-runtime-${SLURM_JOB_ID}"
  test -f "${archive}"
  test ! -e "${runtime_env}"
  mkdir -p "${runtime_env}"
  tar -xzf "${archive}" -C "${runtime_env}"

  # Console entry points contain the preflight JOBFS prefix. Python itself is
  # relocatable, so make only those text shebangs resolve through this job's PATH.
  local entrypoint first_line
  for entrypoint in "${runtime_env}"/bin/*; do
    test -f "${entrypoint}" || continue
    IFS= read -r first_line < "${entrypoint}" || true
    if [[ "${first_line}" == '#!'*python* ]]; then
      sed -i '1c #!/usr/bin/env python' "${entrypoint}"
    fi
  done

  export VIRTUAL_ENV="${runtime_env}"
  export PATH="${runtime_env}/bin:${PATH}"
  export PYTHONNOUSERSITE=1
  export PYTHONDONTWRITEBYTECODE=1
  export PYTHONUNBUFFERED=1
  export XDG_CACHE_HOME="${node_tmp}/climate-v2-cache-${SLURM_JOB_ID}/xdg"
  export HF_HOME="${node_tmp}/climate-v2-cache-${SLURM_JOB_ID}/huggingface"
  export HF_HUB_CACHE="${HF_HOME}/hub"
  export TRANSFORMERS_CACHE="${HF_HOME}/transformers"
  export SENTENCE_TRANSFORMERS_HOME="${HF_HOME}/sentence-transformers"
  export MODELSCOPE_CACHE="${node_tmp}/climate-v2-cache-${SLURM_JOB_ID}/modelscope"
  export MODELSCOPE_HOME="${MODELSCOPE_CACHE}"
  export HF_HUB_DISABLE_TELEMETRY=1
  mkdir -p "${XDG_CACHE_HOME}" "${HF_HOME}" "${MODELSCOPE_CACHE}"
}
