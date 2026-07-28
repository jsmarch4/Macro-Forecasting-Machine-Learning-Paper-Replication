#!/bin/bash
#SBATCH --job-name=render_tables
#SBATCH --partition=shared
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --time=01:00:00
#SBATCH --output=slurm_logs/render_tables_%j.out
#SBATCH --error=slurm_logs/render_tables_%j.err

set -euo pipefail

module reset
module load GCC/13.2.0
module load Python/3.11.5-GCCcore-13.2.0

cd "$HOME/Macro Forecasting and Machine Learning Paper Replication"
source .venv/bin/activate

echo "Job: $SLURM_JOB_ID"
echo "Node: $(hostname)"
echo "Started: $(date)"

python -u 09_render_complexity_tables_colored.py

echo "Finished: $(date)"