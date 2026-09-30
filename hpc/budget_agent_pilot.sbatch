#!/bin/bash
# Preparation only: explicit resource release required before submission.
#SBATCH --job-name=climate-agent-pilot
#SBATCH --account=punim2936
#SBATCH --partition=gpu-a100
#SBATCH --gres=gpu:A100:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=00:15:00
#SBATCH --tmp=30G
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
