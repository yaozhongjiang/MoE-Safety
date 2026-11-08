#!/bin/bash
#SBATCH --job-name=moe-decoy-vllm
#SBATCH --account=YOUR_ACCOUNT
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=03:00:00
#SBATCH --output=logs/%x-%j.out
set -u
cd /scratch/users/$USER/moe
source venv/bin/activate
export HF_HOME=/scratch/users/$USER/moe/hf HF_HUB_OFFLINE=1
nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv
TAG=$(nvidia-smi --query-gpu=name --format=csv,noheader | head -1 | tr -cd "[:alnum:]")
for spec in ${SPECS}; do
  model=${spec%%:*}; nd=${spec##*:}
  out=results/21_${TAG}_$(basename $model)_nd${nd}.json
  echo "=== $model n_dummy=$nd -> $out"; date
  python 21_vllm_cuda_overhead.py $model $nd $out > ${out%.json}.log 2>&1; grep -E "^\{\"model" ${out%.json}.log | cut -c1-300; grep -nE "Error|assert|Traceback" ${out%.json}.log | head -5
done
date
