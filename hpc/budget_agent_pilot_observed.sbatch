#!/bin/bash
# Operator observation only; executes the frozen ca9fa53 GPU wrapper unchanged.
#SBATCH --job-name=climate-agent-pilot-ca9fa53
#SBATCH --account=punim2936
#SBATCH --partition=gpu-a100
#SBATCH --gres=gpu:A100:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=00:15:00
#SBATCH --tmp=30G
#SBATCH --signal=B:USR1@60
set -euo pipefail
: "${SLURM_JOB_ID:?allocated job required}"
: "${CLIMATE_SOURCE_TAR:?}"
: "${CLIMATE_SOURCE_SHA256:?}"
: "${CLIMATE_RESULT_TAR:?}"
: "${CLIMATE_DIAGNOSTICS:?}"
test "${CLIMATE_RELEASE_ID:-}" = climate-authored-pilot-ca9fa53-20260929-user-resume
test "${CLIMATE_AGENT_PHASE:-}" = pilot
test "${CLIMATE_PREFLIGHT_ONLY:-}" != 1
test -n "${CUDA_VISIBLE_DEVICES:-}"
case "$(readlink -f "${CLIMATE_SOURCE_TAR}")" in
  /data/gpfs/projects/punim2936/portfolio_20260903/climate-public-retrieval-v2/envs/*) ;;
  *) exit 90 ;;
esac
case "$(readlink -f "$(dirname "${CLIMATE_DIAGNOSTICS}")")" in
  /data/gpfs/projects/punim2936/portfolio_20260903/climate-public-retrieval-v2/runs) ;;
  *) exit 91 ;;
esac
test ! -e "${CLIMATE_DIAGNOSTICS}"
mkdir "${CLIMATE_DIAGNOSTICS}"
OBS_STARTED=$(date -u +%Y-%m-%dT%H:%M:%SZ)
OBS_START_SECONDS=$(date +%s)
OBS_SAMPLER_PID=
snapshot() {
  local phase="$1" scratch="${SLURM_TMPDIR:-${TMPDIR:-}}" file directory
  printf '%s %s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "${phase}" >> "${CLIMATE_DIAGNOSTICS}/events.log"
  case "${scratch}" in /tmp|/tmp/*|/var/tmp|/var/tmp/*|/jobfs/*) ;; *) return 0 ;; esac
  shopt -s nullglob
  for directory in "${scratch}"/climate-agent-"${SLURM_JOB_ID}"-*/result; do
    for file in "${directory}/run.json" "${directory}/run.consumed.json"; do
      if test -f "${file}" && test "$(stat -c %s "${file}")" -le 2097152; then
        cp -n "${file}" "${CLIMATE_DIAGNOSTICS}/${phase}-$(basename "${file}")" || true
      fi
    done
  done
}
finish() {
  local status="$?" ended elapsed output=false
  trap - EXIT USR1 TERM INT
  set +e
  if test -n "${OBS_SAMPLER_PID}"; then
    kill "${OBS_SAMPLER_PID}" 2>/dev/null
    wait "${OBS_SAMPLER_PID}" 2>/dev/null
  fi
  snapshot final
  ended=$(date -u +%Y-%m-%dT%H:%M:%SZ)
  elapsed=$(( $(date +%s) - OBS_START_SECONDS ))
  test -s "${CLIMATE_RESULT_TAR}" && output=true
  printf '{"job_id":"%s","started_utc":"%s","ended_utc":"%s","elapsed_seconds":%s,"exit_status":%s,"result_archive_exists":%s,"gpu_sample_interval_seconds":1,"precise_cuda_peak_measured":false,"separate_model_load_time_measured":false}\n' \
    "${SLURM_JOB_ID}" "${OBS_STARTED}" "${ended}" "${elapsed}" "${status}" "${output}" \
    > "${CLIMATE_DIAGNOSTICS}/operator-status.json"
  printf 'CLIMATE_OPERATOR_END status=%s elapsed_seconds=%s result=%s\n' "${status}" "${elapsed}" "${output}"
  exit "${status}"
}
trap finish EXIT
trap 'snapshot pre_timeout' USR1
trap 'snapshot terminated; exit 143' TERM
trap 'snapshot interrupted; exit 130' INT
printf 'CLIMATE_OPERATOR_START job=%s utc=%s release=%s\n' "${SLURM_JOB_ID}" "${OBS_STARTED}" "${CLIMATE_RELEASE_ID}"
test "$(sha256sum "${CLIMATE_SOURCE_TAR}" | cut -d' ' -f1)" = "${CLIMATE_SOURCE_SHA256}"
test "$(tar -xOf "${CLIMATE_SOURCE_TAR}" SOURCE_REVISION | tr -d '\r\n')" = ca9fa53f2d4b70091fd08a8a1512e263f329fc61
(
  while true; do
    nvidia-smi --id="${SLURM_JOB_GPUS:-${CUDA_VISIBLE_DEVICES}}" \
      --query-gpu=timestamp,index,name,memory.used,memory.total,utilization.gpu \
      --format=csv,noheader,nounits || true
    sleep 1
  done
) >> "${CLIMATE_DIAGNOSTICS}/gpu-samples.csv" 2>> "${CLIMATE_DIAGNOSTICS}/gpu-sampler-errors.log" &
OBS_SAMPLER_PID=$!
# Do not rewrite Python source, model inputs, arguments, RNG or CUDA visibility.
# A Slurm pre-timeout signal saves available receipts, never fabricates run results.
set +e
bash <(tar -xOf "${CLIMATE_SOURCE_TAR}" hpc/budget_agent_pilot.sbatch) &
OBS_CHILD_PID=$!
while true; do
  wait "${OBS_CHILD_PID}"
  OBS_STATUS=$?
  kill -0 "${OBS_CHILD_PID}" 2>/dev/null || break
done
exit "${OBS_STATUS}"
