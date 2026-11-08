#!/bin/bash
#SBATCH --job-name=moe-qwen3-cost
#SBATCH --account=YOUR_ACCOUNT
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=03:00:00
#SBATCH --requeue
#SBATCH --output=logs/%x-%j.out
set -u
cd /scratch/users/$USER/moe
source venv/bin/activate
export HF_HOME=/scratch/users/$USER/moe/hf HF_HUB_OFFLINE=1
M=Qwen/Qwen3-30B-A3B-Base
nvidia-smi --query-gpu=name,driver_version,memory.total,memory.free --format=csv
# Guard: on 2026-10-02 an H200 was handed out with ~96 GiB already in use by a stale process,
# which made every vLLM start fail. Require (almost) the whole GPU; otherwise requeue and exit.
FREE_MIB=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | head -1)
NEED_MIB=${NEED_MIB:-85000}
if [ "${FREE_MIB}" -lt "${NEED_MIB}" ]; then
  echo "only ${FREE_MIB} MiB free (< ${NEED_MIB}); requeueing"; scontrol requeue ${SLURM_JOB_ID}; exit 0
fi
TAG=$(nvidia-smi --query-gpu=name --format=csv,noheader | head -1 | tr -cd "[:alnum:]")
echo "=== coverage"; date
python 22_vllm_expert_coverage.py $M results/22_${TAG}_Qwen3-30B-A3B-Base.json > results/22_${TAG}_Qwen3-30B-A3B-Base.log 2>&1
grep -E "^B=|Error|Traceback" results/22_${TAG}_Qwen3-30B-A3B-Base.log | tail -12
for nd in 0 8 24 56 120; do
  out=results/21_${TAG}_Qwen3-30B-A3B-Base_nd${nd}.json
  echo "=== n_dummy=$nd"; date
  python 21_vllm_cuda_overhead.py $M $nd $out > ${out%.json}.log 2>&1
  grep -E "^\{\"model" ${out%.json}.log | cut -c1-300; grep -nE "Error|Traceback" ${out%.json}.log | head -3
done
date
