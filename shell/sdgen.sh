#!/bin/bash
#SBATCH --job-name=sdgen
#SBATCH --output=logs/sdgen/%x_%j.out
#SBATCH --error=logs/sdgen/%x_%j.err
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
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
python3 /work/vihps/vihps13/python-ROI/shell/sdgen.py