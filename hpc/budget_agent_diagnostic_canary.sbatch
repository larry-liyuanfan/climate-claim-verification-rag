#!/bin/bash
# Prepared only. A NEW explicit canary release and immutable identities are required.
#SBATCH --job-name=climate-agent-diagnostic-canary
#SBATCH --account=punim2936
#SBATCH --partition=gpu-a100
#SBATCH --gres=gpu:A100:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --tmp=30G
#SBATCH --time=00:15:00
#SBATCH --no-requeue
#SBATCH --signal=B:USR1@60
set -euo pipefail
: "${SLURM_JOB_ID:?allocated job required}"
: "${CLIMATE_CANARY_RELEASE_ID:?new explicit coordinator release required}"
: "${CLIMATE_EXPECTED_SOURCE_GIT:?new diagnostic source commit}"
: "${CLIMATE_SOURCE_TAR:?}"
: "${CLIMATE_SOURCE_SHA256:?}"
: "${CLIMATE_DIAGNOSTICS:?fresh owner-only Climate directory}"
test "${CLIMATE_CANARY_RELEASE_ID}" != climate-authored-pilot-ca9fa53-20260929-user-resume
test "${CLIMATE_AGENT_PHASE:-}" = pilot
test "${CLIMATE_PREFLIGHT_ONLY:-}" != 1
test -n "${CUDA_VISIBLE_DEVICES:-}"
case "$(readlink -f "${CLIMATE_SOURCE_TAR}")" in
  /data/gpfs/projects/punim2936/portfolio_20260903/climate-public-retrieval-v2/envs/*) ;; *) exit 90 ;;
esac
test "$(sha256sum "${CLIMATE_SOURCE_TAR}" | cut -d' ' -f1)" = "${CLIMATE_SOURCE_SHA256}"
test "$(tar -xOf "${CLIMATE_SOURCE_TAR}" SOURCE_REVISION | tr -d '\r\n')" = "${CLIMATE_EXPECTED_SOURCE_GIT}"
case "$(readlink -f "$(dirname "${CLIMATE_DIAGNOSTICS}")")" in
  /data/gpfs/projects/punim2936/portfolio_20260903/climate-public-retrieval-v2/runs) ;; *) exit 91 ;;
esac
test ! -e "${CLIMATE_DIAGNOSTICS}"
umask 077
mkdir -m 700 "${CLIMATE_DIAGNOSTICS}"
export CLIMATE_PRIVATE_RESPONSE_DIR="${CLIMATE_DIAGNOSTICS}/private-responses"
mkdir -m 700 "${CLIMATE_PRIVATE_RESPONSE_DIR}"
CANARY_START=$(date +%s)
snapshot() {
  local phase="$1" scratch="${SLURM_TMPDIR:-${TMPDIR:-}}" directory file
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
  local status="$?"
  trap - EXIT USR1 TERM INT
  set +e
  snapshot final
  printf '{"job_id":"%s","exit_status":%s,"elapsed_seconds":%s,"gpu_memory_measured":false}\n' \
    "${SLURM_JOB_ID}" "${status}" "$(( $(date +%s) - CANARY_START ))" \
    > "${CLIMATE_DIAGNOSTICS}/operator-status.json"
  exit "${status}"
}
trap finish EXIT
trap 'snapshot pre_timeout' USR1
trap 'snapshot terminated; exit 143' TERM
trap 'snapshot interrupted; exit 130' INT
# The prior NVML sampler returned no valid records. Do not manufacture a memory peak.
# Private responses are never placed inside result/ or the public result tar.
set +e
bash <(tar -xOf "${CLIMATE_SOURCE_TAR}" hpc/budget_agent_pilot.sbatch) &
CANARY_CHILD=$!
while true; do
  wait "${CANARY_CHILD}"
  CANARY_STATUS=$?
  kill -0 "${CANARY_CHILD}" 2>/dev/null || break
done
exit "${CANARY_STATUS}"
