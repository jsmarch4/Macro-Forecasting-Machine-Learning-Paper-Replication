#!/bin/bash
#SBATCH --job-name=complexity_denominator
#SBATCH --output=slurm_logs/complexity_denominator_%j.out
#SBATCH --error=slurm_logs/complexity_denominator_%j.err
#SBATCH --partition=shared
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=8G
#SBATCH --time=04:00:00

cd "/home/jmarch4/Macro Forecasting and Machine Learning Paper Replication"

source .venv/bin/activate

echo "Started: $(date)"
echo "Node: $(hostname)"
echo "Working directory: $(pwd)"
echo "Python: $(which python)"

python -u debugging/debug_complexity_denominator.py

echo "Finished: $(date)"