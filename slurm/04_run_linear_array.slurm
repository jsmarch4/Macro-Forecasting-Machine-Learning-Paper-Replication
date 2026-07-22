#!/bin/bash
#SBATCH --job-name=macro_linear
#SBATCH --partition=shared
#SBATCH --array=0-6
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=12G
#SBATCH --time=12:00:00
#SBATCH --output=slurm_logs/linear_%A_%a.out
#SBATCH --error=slurm_logs/linear_%A_%a.err

set -euo pipefail

module reset
module load GCC/13.2.0
module load Python/3.11.5-GCCcore-13.2.0

cd "$HOME/Macro Forecasting and Machine Learning Paper Replication"
source .venv/bin/activate

QUANTILES=(0.05 0.10 0.25 0.50 0.75 0.90 0.95)
TAU="${QUANTILES[$SLURM_ARRAY_TASK_ID]}"

export OMP_NUM_THREADS="$SLURM_CPUS_PER_TASK"
export MKL_NUM_THREADS="$SLURM_CPUS_PER_TASK"
export OPENBLAS_NUM_THREADS="$SLURM_CPUS_PER_TASK"

echo "Job: $SLURM_JOB_ID"
echo "Task: $SLURM_ARRAY_TASK_ID"
echo "Quantile: $TAU"
echo "Node: $(hostname)"
echo "Started: $(date)"

python -u 04_run_linear_torch.py --quantile "$TAU"
echo "Finished: $(date)"
echo "Exit code: $?"