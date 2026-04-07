#!/bin/bash
#SBATCH --job-name=test
#SBATCH --output=logs/test/%x_%j.out
#SBATCH --error=logs/test/%x_%j.err
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --time=12:00:00

eval "$(conda shell.bash hook)"
conda activate /work/vihps/vihps13/env10

# ── Diagnostics ───────────────────────────────────────────
echo "Job ID:      $SLURM_JOB_ID"
echo "Node:        $SLURMD_NODENAME"
echo "GPUs:        $CUDA_VISIBLE_DEVICES"

# ── Run your workload ─────────────────────────────────────
python3 test.py
#python3 hello.py
