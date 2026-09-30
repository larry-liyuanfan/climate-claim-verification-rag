#!/bin/bash
# One release only; durable guard prevents a second submission after disconnect.
set -euo pipefail
umask 077
root=/data/gpfs/projects/punim2936/portfolio_20260903/climate-public-retrieval-v2
release=climate-full-72eaa90-20260930-r1
test ! -e "${root}/runs/${release}"
existing=$(squeue -u "$(id -un)" -h -o '%i|%Z' | awk -F'|' 'index($2,"climate") {print $1}')
test -z "${existing}"
test "$(sha256sum "${root}/envs/budget-agent-full-operator-0fa11b5.tar" | cut -d' ' -f1)" = 0965c471b2f13197ae0eaffa30ada17745734fc5cf77e4ff06f853830da73cef
test "$(sha256sum "${root}/envs/budget-agent-full-0fa11b5.sbatch" | cut -d' ' -f1)" = c266835b8dc614f221a66269e6f89224f7d7630bc550bbaa3c6f3d774bb56bf9
bash -n "${root}/envs/budget-agent-full-0fa11b5.sbatch"
guard="${root}/envs/${release}.submission-lock"
mkdir -m 700 "${guard}"
export CLIMATE_FULL_RELEASE_ID="${release}"
export CLIMATE_OPERATOR_TAR="${root}/envs/budget-agent-full-operator-0fa11b5.tar"
export CLIMATE_OPERATOR_SHA256=0965c471b2f13197ae0eaffa30ada17745734fc5cf77e4ff06f853830da73cef
export CLIMATE_OPERATOR_GIT=0fa11b580b6ac7d7af1d1861da179acda1f7d9fc
unset SBATCH_GRES SBATCH_GPUS SBATCH_GPUS_PER_NODE SBATCH_PARTITION SBATCH_MEM_PER_CPU
sbatch --test-only --chdir="${root}" --output="${root}/runs/${release}-%j.log" \
  "${root}/envs/budget-agent-full-0fa11b5.sbatch" 2>&1 | tee "${guard}/test-only.txt"
date -u +%Y-%m-%dT%H:%M:%SZ > "${guard}/submit-start.txt"
sbatch --parsable --chdir="${root}" --output="${root}/runs/${release}-%j.log" \
  "${root}/envs/budget-agent-full-0fa11b5.sbatch" | tee "${guard}/job-id.txt"
