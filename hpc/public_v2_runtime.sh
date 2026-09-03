#!/bin/bash

public_v2_node_tmp() {
  local node_tmp="${SLURM_TMPDIR:-${TMPDIR:-}}"
  : "${node_tmp:?Spartan did not expose node-local temporary storage}"
  node_tmp="$(readlink -f "${node_tmp}")"
  case "${node_tmp}" in
    /tmp|/tmp/*|/var/tmp|/var/tmp/*|/jobfs/*) ;;
    *) echo "Refusing non-local temporary directory: ${node_tmp}" >&2; return 92 ;;
  esac
  printf '%s\n' "${node_tmp}"
}

public_v2_assert_persist_prefix() {
  : "${CLIMATE_V2_ROOT:?Set the isolated public-v2 root}"
  local prefix="$1"
  local root parent
  root="$(readlink -f "${CLIMATE_V2_ROOT}")"
  test "${root}" = \
    "/data/gpfs/projects/punim2936/portfolio_20260903/climate-public-retrieval-v2"
  parent="$(readlink -f "$(dirname "${prefix}")")"
  case "${parent}" in
    "${root}"|"${root}"/*) ;;
    *) echo "Persistent archive prefix is outside the isolated root: ${prefix}" >&2; return 93 ;;
  esac
}

public_v2_write_manifest() {
  local source="$1"
  local stage_id="$2"
  python "${REPO_DIR}/scripts/public_v2_stage_manifest.py" \
    write "${source}" --stage-id "${stage_id}"
}

public_v2_verify_manifest() {
  local source="$1"
  python "${REPO_DIR}/scripts/public_v2_stage_manifest.py" verify "${source}"
}

public_v2_find_archive() {
  local prefix="$1"
  public_v2_assert_persist_prefix "${prefix}"
  local -a matches
  shopt -s nullglob
  matches=("${prefix}"-*.tar.gz)
  shopt -u nullglob
  test "${#matches[@]}" -eq 1
  local archive stem expected actual
  archive="${matches[0]}"
  stem="${archive%.tar.gz}"
  expected="${stem##*-}"
  test "${#expected}" -eq 64
  actual="$(sha256sum "${archive}" | cut -d' ' -f1)"
  test "${actual}" = "${expected}"
  printf '%s\n' "${archive}"
}

public_v2_pack_stage() {
  local source="$1"
  local prefix="$2"
  local stage_id="$3"
  public_v2_assert_persist_prefix "${prefix}"
  test -d "${source}"
  local -a existing
  shopt -s nullglob
  existing=("${prefix}"-*.tar.gz)
  shopt -u nullglob
  test "${#existing[@]}" -eq 0
  public_v2_write_manifest "${source}" "${stage_id}"
  public_v2_verify_manifest "${source}"
  local temporary digest final
  temporary="${prefix}.tmp-${SLURM_JOB_ID}.tar.gz"
  test ! -e "${temporary}"
  tar -czf "${temporary}" -C "${source}" .
  digest="$(sha256sum "${temporary}" | cut -d' ' -f1)"
  final="${prefix}-${digest}.tar.gz"
  test ! -e "${final}"
  mv "${temporary}" "${final}"
  test "$(sha256sum "${final}" | cut -d' ' -f1)" = "${digest}"
  printf 'archive=%s\nsha256=%s\n' "${final}" "${digest}"
}

public_v2_unpack_stage() {
  local prefix="$1"
  local destination="$2"
  local archive
  archive="$(public_v2_find_archive "${prefix}")"
  test ! -e "${destination}"
  mkdir -p "${destination}"
  tar -xzf "${archive}" -C "${destination}"
  public_v2_verify_manifest "${destination}"
}

public_v2_activate_runtime() {
  : "${CLIMATE_V2_ROOT:?Set the isolated public-v2 root}"
  local node_tmp runtime_env
  node_tmp="$(public_v2_node_tmp)"
  runtime_env="${node_tmp}/climate-v2-runtime-${SLURM_JOB_ID}"
  public_v2_unpack_stage \
    "${CLIMATE_V2_ROOT}/envs/runtime-py310" "${runtime_env}"

  # Console entry points contain the preflight temporary prefix. Python itself
  # is relocatable, so make only those text shebangs resolve through this job's PATH.
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
