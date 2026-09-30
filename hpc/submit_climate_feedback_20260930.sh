#!/bin/bash
# One authorized development allocation; no retry, cancellation or priority edits.
set -euo pipefail
umask 077
ROOT=/data/gpfs/projects/punim2936/portfolio_20260903/climate-public-retrieval-v2
RELEASE=climate-feedback-v2-20260930-pilot-r1
: "${CLIMATE_SOURCE_GIT:?}"
: "${CLIMATE_SOURCE_SHA256:?}"
: "${CLIMATE_SOURCE_TAR:?}"
: "${CLIMATE_WRAPPER:?}"
test "${CLIMATE_FEEDBACK_RELEASE:-}" = "${RELEASE}"
[[ "${CLIMATE_SOURCE_GIT}" =~ ^[0-9a-f]{40}$ ]]
[[ "${CLIMATE_SOURCE_SHA256}" =~ ^[0-9a-f]{64}$ ]]
test "$(readlink -f "${CLIMATE_SOURCE_TAR}")" = "${ROOT}/envs/feedback-source-${CLIMATE_SOURCE_GIT}.tar"
test "$(readlink -f "${CLIMATE_WRAPPER}")" = "${ROOT}/envs/feedback-wrapper-${CLIMATE_SOURCE_GIT}.sbatch"
test "$(sha256sum "${CLIMATE_SOURCE_TAR}" | cut -d' ' -f1)" = "${CLIMATE_SOURCE_SHA256}"
test "$(tar -xOf "${CLIMATE_SOURCE_TAR}" SOURCE_REVISION | tr -d '\r\n')" = "${CLIMATE_SOURCE_GIT}"
test "$(sha256sum "${CLIMATE_WRAPPER}" | cut -d' ' -f1)" = "$(tar -xOf "${CLIMATE_SOURCE_TAR}" hpc/budget_agent_feedback_pilot.sbatch | sha256sum | cut -d' ' -f1)"
test ! -e "${ROOT}/runs/${RELEASE}"
queued=$(squeue -u "$(id -un)" -h -o '%j')
if printf '%s\n' "${queued}" | grep -Eq '^climate-agent-(feedback-v2|full-frozen)$'; then
  exit 90
fi
LOCK="${ROOT}/envs/${RELEASE}.submission-lock"
mkdir "${LOCK}"
# Only this process: inherited settings must not override frozen resources.
while IFS= read -r name; do unset "${name}"; done < <(compgen -A variable SBATCH_)
export CLIMATE_SOURCE_GIT CLIMATE_SOURCE_SHA256 CLIMATE_SOURCE_TAR CLIMATE_FEEDBACK_RELEASE
submit() {
  sbatch "$@" --account=punim2936 --partition=gpu-a100,gpu-a100-short \
    --gres=gpu:A100:1 --nodes=1 --ntasks=1 --cpus-per-task=8 --mem=32G --tmp=30G \
    --time=00:45:00 --no-requeue --export=ALL --chdir="${ROOT}" \
    --output="${ROOT}/runs/${RELEASE}-%j.log" "${CLIMATE_WRAPPER}"
}
bash -n "${CLIMATE_WRAPPER}"
submit --test-only 2>&1 | tee "${LOCK}/test-only.log"
job=$(submit --parsable)
printf '%s\n' "${job}" > "${LOCK}/job-id"
printf 'CLIMATE_FEEDBACK_SUBMITTED %s\n' "${job}"
