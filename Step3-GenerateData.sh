#!/bin/bash --login
#SBATCH --job-name=PhyloGNN-Data
#SBATCH --partition=general
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=12:00:00
#SBATCH --array=0-803
#SBATCH --output=logs/Step3-%A_%a.out
#SBATCH --error=logs/Step3-%A_%a.err
# If Bunya asks for an account, add:  #SBATCH --account=a_YOURGROUP

# Step 3 data generation on Bunya: one work item per array element.
#   0-799   training/validation chunks (4 models x 200 chunks of 500 trees)
#   800-803 encoding of the authors' test sets (BD, BDEI, BDSS, BiSSE)
# The first task prepares the run; the others wait for it.
#
# Submit from the project root (the folder with phyloCNN/, phylodeep/, .envs/):
#   mkdir -p logs && sbatch Step3-GenerateData.sh
#
# The array range must match the run:  python Step3-GenerateData.py --array-range
# Override without editing this file:  sbatch --array=0-803%100 Step3-GenerateData.sh
# More than 1000 work items:           sbatch --array=0-999%100 --export=ALL,TASK_OFFSET=1000 Step3-GenerateData.sh

set -euo pipefail

cd "$SLURM_SUBMIT_DIR"

export OMP_NUM_THREADS="$SLURM_CPUS_PER_TASK"
export OPENBLAS_NUM_THREADS="$SLURM_CPUS_PER_TASK"
export MKL_NUM_THREADS="$SLURM_CPUS_PER_TASK"

TASK_OFFSET="${TASK_OFFSET:-0}"

echo "=================================="
echo "PHYLOGNN STEP 3 DATA GENERATION"
echo "=================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Array Task: $SLURM_ARRAY_TASK_ID (offset $TASK_OFFSET)"
echo "Node: $(hostname)"
echo "Started: $(date)"

PY="$PWD/.envs/phylocnn/bin/python"

if [ ! -x "$PY" ]; then
    echo "ERROR: Python environment missing: $PY"
    exit 1
fi

if [ ! -f Step3-GenerateData.py ]; then
    echo "ERROR: submit from the project root (Step3-GenerateData.py not found in $PWD)"
    exit 1
fi

srun "$PY" Step3-GenerateData.py \
    --task-id "$SLURM_ARRAY_TASK_ID" \
    --task-offset "$TASK_OFFSET"

echo "Completed: $(date)"
