#!/bin/bash
#SBATCH --job-name=climate-agent-cpu-preflight
#SBATCH --account=punim2936
#SBATCH --partition=sapphire
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --time=00:20:00
#SBATCH --tmp=24G
export CLIMATE_PREFLIGHT_ONLY=1
export CUDA_VISIBLE_DEVICES=""
set -euo pipefail
: "${SLURM_JOB_ID:?must run inside a released allocation}"
: "${CLIMATE_SOURCE_TAR:?immutable source tar}"
: "${CLIMATE_SOURCE_SHA256:?source SHA256}"
case "$(readlink -f "${CLIMATE_SOURCE_TAR}")" in
  /data/gpfs/projects/punim2936/portfolio_20260903/climate-public-retrieval-v2/*) ;;
  *) exit 90 ;;
esac
test "$(sha256sum "${CLIMATE_SOURCE_TAR}" | cut -d' ' -f1)" = "${CLIMATE_SOURCE_SHA256}"
bash <(tar -xOf "${CLIMATE_SOURCE_TAR}" hpc/budget_agent_launch.sh)
