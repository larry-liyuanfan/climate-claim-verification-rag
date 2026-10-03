#!/bin/bash
# Coordinator released one infrastructure repair rerun; never reuse r1 or this lock.
set -euo pipefail
umask 077
root=/data/gpfs/projects/punim2936/portfolio_20260903/climate-public-retrieval-v2
release=climate-full-72eaa90-20260930-r2
test "${CLIMATE_R2_RELEASE_APPROVED:-}" = "${release}"
test ! -e "${root}/runs/${release}"
existing=$(squeue -u "$(id -un)" -h -o '%i|%j|%Z' | \
  awk -F'|' -v root="${root}" '$2=="climate-agent-full-frozen" || $3==root {print $1}')
test -z "${existing}"
test "$(sha256sum "${root}/envs/budget-agent-full-operator-947645b.tar" | cut -d' ' -f1)" = f9aa535d33cd8068c9a747aa8aba431a036b56e55cf0cb7a959daa36132e4d42
test "$(sha256sum "${root}/envs/budget-agent-full-947645b.sbatch" | cut -d' ' -f1)" = c266835b8dc614f221a66269e6f89224f7d7630bc550bbaa3c6f3d774bb56bf9
bash -n "${root}/envs/budget-agent-full-947645b.sbatch"
guard="${root}/envs/${release}.submission-lock"
mkdir -m 700 "${guard}"
export CLIMATE_FULL_RELEASE_ID="${release}"
export CLIMATE_OPERATOR_TAR="${root}/envs/budget-agent-full-operator-947645b.tar"
export CLIMATE_OPERATOR_SHA256=f9aa535d33cd8068c9a747aa8aba431a036b56e55cf0cb7a959daa36132e4d42
export CLIMATE_OPERATOR_GIT=947645b3feed914e30d02dc5dbd05859c4a19667
# Clear submission overrides only in this helper process; never alter login config.
while IFS= read -r key; do unset "${key}"; done < <(compgen -v SBATCH_)
args=(--account=punim2936 --partition=gpu-a100,gpu-a100-short --gres=gpu:A100:1
      --nodes=1 --ntasks=1 --cpus-per-task=8 --mem=32G --tmp=30G --time=02:00:00
      --no-requeue --signal=B:USR1@90 --export=ALL --chdir="${root}"
      --output="${root}/runs/${release}-%j.log")
sbatch --test-only "${args[@]}" "${root}/envs/budget-agent-full-947645b.sbatch" \
  2>&1 | tee "${guard}/test-only.txt"
date -u +%Y-%m-%dT%H:%M:%SZ > "${guard}/submit-start.txt"
# If this result is lost, inspect this guard and squeue; never submit a second time.
sbatch --parsable "${args[@]}" "${root}/envs/budget-agent-full-947645b.sbatch" \
  | tee "${guard}/job-id.txt"
