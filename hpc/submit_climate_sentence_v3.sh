#!/bin/bash
# Default is a non-submitting scheduling check. --submit needs separate release.
set -euo pipefail
umask 077
MODE="${1:---test-only}"
test "${MODE}" = --test-only || test "${MODE}" = --submit
ROOT=/data/gpfs/projects/punim2936/portfolio_20260903/climate-public-retrieval-v2
RELEASE=climate-sentence-v3-20260930-pilot-r1
: "${CLIMATE_SOURCE_GIT:?}"
: "${CLIMATE_SOURCE_SHA256:?}"
: "${CLIMATE_SOURCE_TAR:?}"
: "${CLIMATE_WRAPPER:?}"
: "${CLIMATE_V3_GRAMMAR_TAR:?}"
: "${CLIMATE_V3_GRAMMAR_SHA256:?}"
: "${CLIMATE_V3_PROTOCOL_SHA256:?}"
test "${CLIMATE_V3_RELEASE:-}" = "${RELEASE}"
[[ "${CLIMATE_SOURCE_GIT}" =~ ^[0-9a-f]{40}$ ]]
for value in "${CLIMATE_SOURCE_SHA256}" "${CLIMATE_V3_GRAMMAR_SHA256}" "${CLIMATE_V3_PROTOCOL_SHA256}"; do
  [[ "${value}" =~ ^[0-9a-f]{64}$ ]]
done
test "$(readlink -f "${CLIMATE_SOURCE_TAR}")" = "${ROOT}/envs/agent-v3-source-${CLIMATE_SOURCE_GIT}.tar"
test "$(readlink -f "${CLIMATE_WRAPPER}")" = "${ROOT}/envs/agent-v3-wrapper-${CLIMATE_SOURCE_GIT}.sbatch"
test "$(readlink -f "${CLIMATE_V3_GRAMMAR_TAR}")" = "${ROOT}/envs/agent-v3-grammar-${CLIMATE_V3_GRAMMAR_SHA256}.tar"
test "$(sha256sum "${CLIMATE_SOURCE_TAR}" | cut -d' ' -f1)" = "${CLIMATE_SOURCE_SHA256}"
test "$(sha256sum "${CLIMATE_V3_GRAMMAR_TAR}" | cut -d' ' -f1)" = "${CLIMATE_V3_GRAMMAR_SHA256}"
test "$(tar -xOf "${CLIMATE_SOURCE_TAR}" SOURCE_REVISION | tr -d '\r\n')" = "${CLIMATE_SOURCE_GIT}"
test "$(tar -xOf "${CLIMATE_SOURCE_TAR}" configs/agent_sentence_v3_pilot_20260930.json | sha256sum | cut -d' ' -f1)" = "${CLIMATE_V3_PROTOCOL_SHA256}"
test "$(sha256sum "${CLIMATE_WRAPPER}" | cut -d' ' -f1)" = "$(tar -xOf "${CLIMATE_SOURCE_TAR}" hpc/agent_sentence_v3_pilot.sbatch | sha256sum | cut -d' ' -f1)"
test "$(sha256sum "$0" | cut -d' ' -f1)" = "$(tar -xOf "${CLIMATE_SOURCE_TAR}" hpc/submit_climate_sentence_v3.sh | sha256sum | cut -d' ' -f1)"
test ! -e "${ROOT}/runs/${RELEASE}"
LOCK="${ROOT}/envs/${RELEASE}.submission-lock"
test ! -e "${LOCK}"
check_queue() {
  local queued
  queued=$(squeue -u "$(id -un)" -h -o '%j')
  if printf '%s\n' "${queued}" | grep -Eq '^climate-agent-(sentence-v3|feedback-v2|full-frozen)$'; then
    return 90
  fi
}
check_queue
# Apply identical explicit resources to test-only and real submission.
while IFS= read -r name; do unset "${name}"; done < <(compgen -A variable SBATCH_)
export CLIMATE_SOURCE_GIT CLIMATE_SOURCE_SHA256 CLIMATE_SOURCE_TAR CLIMATE_V3_RELEASE
export CLIMATE_V3_GRAMMAR_TAR CLIMATE_V3_GRAMMAR_SHA256 CLIMATE_V3_PROTOCOL_SHA256
submit() {
  sbatch "$@" --qos=normal --account=punim2936 --partition=gpu-a100,gpu-a100-short \
    --gres=gpu:A100:1 --nodes=1 --ntasks=1 --cpus-per-task=8 --mem=32G --tmp=30G \
    --time=01:00:00 --no-requeue --export=ALL --chdir="${ROOT}" \
    --output="${ROOT}/runs/${RELEASE}-%j.log" "${CLIMATE_WRAPPER}"
}
bash -n "${CLIMATE_WRAPPER}"
if test "${MODE}" = --test-only; then
  submit --test-only
  exit 0
fi
test "${CLIMATE_V3_SUBMIT_AUTHORIZED:-}" = "${RELEASE}:${CLIMATE_SOURCE_GIT}"
mkdir "${LOCK}"  # atomic reservation; retain even on failure for manual diagnosis
check_queue
python - "${LOCK}/inputs.json" <<'PY'
import json, os, sys
names = ('CLIMATE_SOURCE_GIT', 'CLIMATE_SOURCE_SHA256', 'CLIMATE_V3_RELEASE',
         'CLIMATE_V3_PROTOCOL_SHA256', 'CLIMATE_V3_GRAMMAR_SHA256')
with open(sys.argv[1], 'x') as stream:
    json.dump({name: os.environ[name] for name in names}, stream, indent=2)
PY
submit --test-only 2>&1 | tee "${LOCK}/test-only.log"
job=$(submit --parsable)
printf '%s\n' "${job}" > "${LOCK}/job-id"
printf 'CLIMATE_SENTENCE_V3_SUBMITTED %s\n' "${job}"
