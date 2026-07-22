#!/bin/bash
#SBATCH --job-name=complexity_tables
#SBATCH --partition=shared
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=12G
#SBATCH --time=12:00:00
#SBATCH --output=slurm_logs/complexity_%j.out
#SBATCH --error=slurm_logs/complexity_%j.err

set -euo pipefail

module reset
module load GCC/13.2.0
module load Python/3.11.5-GCCcore-13.2.0

cd "$HOME/Macro Forecasting and Machine Learning Paper Replication"

source .venv/bin/activate

export OMP_NUM_THREADS="$SLURM_CPUS_PER_TASK"
export MKL_NUM_THREADS="$SLURM_CPUS_PER_TASK"
export OPENBLAS_NUM_THREADS="$SLURM_CPUS_PER_TASK"

echo "Job: $SLURM_JOB_ID"
echo "Node: $(hostname)"
echo "Started: $(date)"

python -u 08_complexity_tables.py

echo "Finished: $(date)"
echo "Exit code: $?"