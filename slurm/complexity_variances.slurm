#!/bin/bash
#SBATCH --job-name=complexity_var
#SBATCH --partition=shared
#SBATCH --array=0-174%20
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --time=08:00:00
#SBATCH --output=slurm_logs/complexity_var_%A_%a.out
#SBATCH --error=slurm_logs/complexity_var_%A_%a.err

set -euo pipefail

module reset
module load GCC/13.2.0
module load Python/3.11.5-GCCcore-13.2.0

cd "$HOME/Macro Forecasting and Machine Learning Paper Replication"
source .venv/bin/activate

QUANTILES=(0.05 0.10 0.25 0.50 0.75 0.90 0.95)

TASK_ID="$SLURM_ARRAY_TASK_ID"

# Tasks 0-48: linear_activation
#   7 quantiles x 7 architectures = 49 tasks
#
# Tasks 49-174: dnn
#   7 quantiles x 18 architectures = 126 tasks

if (( TASK_ID < 49 )); then
    FAMILY="linear_activation"
    LOCAL_ID="$TASK_ID"

    QUANTILE_INDEX=$((LOCAL_ID / 7))
    ARCH_INDEX=$((LOCAL_ID % 7))
else
    FAMILY="dnn"
    LOCAL_ID=$((TASK_ID - 49))

    QUANTILE_INDEX=$((LOCAL_ID / 18))
    ARCH_INDEX=$((LOCAL_ID % 18))
fi

TAU="${QUANTILES[$QUANTILE_INDEX]}"

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1

echo "Job: $SLURM_JOB_ID"
echo "Task: $SLURM_ARRAY_TASK_ID"
echo "Family: $FAMILY"
echo "Quantile: $TAU"
echo "Architecture index: $ARCH_INDEX"
echo "Node: $(hostname)"
echo "Started: $(date)"

python -u complexity_variances.py \
    --family "$FAMILY" \
    --quantile "$TAU" \
    --architecture-index "$ARCH_INDEX"

echo "Finished: $(date)"
