#!/bin/bash
#SBATCH --job-name=moe-dl
#SBATCH --account=YOUR_ACCOUNT
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --time=01:00:00
#SBATCH --output=logs/%x-%j.out
cd /scratch/users/$USER/moe
curl -sI https://huggingface.co | head -1
HF_HOME=/scratch/users/$USER/moe/hf venv/bin/python dl.py
du -sh hf
