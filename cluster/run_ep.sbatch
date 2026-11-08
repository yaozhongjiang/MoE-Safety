#!/bin/bash
#SBATCH --job-name=moe-ep-cost
#SBATCH --account=YOUR_ACCOUNT
#SBATCH --partition=gpu
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=16
#SBATCH --mem=192G
#SBATCH --time=03:00:00
#SBATCH --requeue
#SBATCH --output=logs/%x-%j.out
set -u
cd /scratch/users/$USER/moe
source venv/bin/activate
export HF_HOME=/scratch/users/$USER/moe/hf HF_HUB_OFFLINE=1
nvidia-smi --query-gpu=name,memory.total,memory.free --format=csv
NEED_MIB=${NEED_MIB:-75000}
for f in $(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits); do
  if [ "$f" -lt "$NEED_MIB" ]; then echo "a GPU has only $f MiB free; requeueing"; scontrol requeue $SLURM_JOB_ID; exit 0; fi
done
NG=$(nvidia-smi -L | wc -l)
TAG=$(nvidia-smi --query-gpu=name --format=csv,noheader | head -1 | tr -cd "[:alnum:]")x${NG}
M=Qwen/Qwen3-30B-A3B-Base
for nd in 0 24 120; do
  out=results/26_${TAG}_EP_Qwen3-30B-A3B-Base_nd${nd}.json
  echo "=== EP tp=$NG n_dummy=$nd"; date
  MOE_DECOY_ND=$nd python 26_vllm_ep_overhead.py $M $NG $out > ${out%.json}.log 2>&1
  grep -E "moe_decoy_plugin\] active" ${out%.json}.log | head -2
  grep -E "^\{\"model" ${out%.json}.log | cut -c1-300; grep -nE "Error|Traceback" ${out%.json}.log | head -3
done
date
