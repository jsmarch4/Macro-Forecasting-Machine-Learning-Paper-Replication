#!/bin/bash
#SBATCH --job-name=linear_debug
#SBATCH --partition=shared
#SBATCH --nodes=1
#SBATCH --cpus-per-task=1
#SBATCH --time=02:00:00
#SBATCH --output=slurm_logs/linear_debug_%j.out

source ~/.bashrc
source .venv/bin/activate

python debugging/debug_linear_coefficient_drift.py