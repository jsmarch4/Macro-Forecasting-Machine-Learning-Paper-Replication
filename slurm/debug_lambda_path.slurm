#!/bin/bash
#SBATCH --job-name=lambda_path
#SBATCH --partition=shared
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=8G
#SBATCH --time=00:20:00
#SBATCH --output=slurm_logs/lambda_path_%j.out
#SBATCH --error=slurm_logs/lambda_path_%j.err

cd "$SLURM_SUBMIT_DIR"

source .venv/bin/activate

echo "Started: $(date)"
echo "Node: $(hostname)"
echo "Working directory: $(pwd)"
echo "Python: $(which python)"

python debugging/check_lambda_path.py

echo "Finished: $(date)"