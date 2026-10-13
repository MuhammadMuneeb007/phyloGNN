#!/bin/bash --login
#SBATCH --job-name=PhyloGNN-Empirical
#SBATCH --account=a_ace
#SBATCH --partition=general
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=02:00:00
#SBATCH --output=logs/Step5-Empirical-%j.out
#SBATCH --error=logs/Step5-Empirical-%j.err

# Step 5: apply every trained Step 4 model (all datasets x all 15 models) to the empirical trees
# of the PhyloCNN paper, on CPU (one tree per model: minutes, no GPU needed).
#   HIV_Zurich  200 tips, sampling 0.25          -> BDSS, BD and BDEI models
#   Primates    260 tips, sampling 260/381, BiSSE -> BiSSE models
#
# Submit from the project root (the folder with Results/, phyloCNN/, .envs/), after Step 4:
#   mkdir -p logs
#   sbatch Step5-EmpiricalValidation-CPU.sh                  # every results folder in Results/
#   sbatch Step5-EmpiricalValidation-CPU.sh full100k         # one results folder
#   sbatch Step5-EmpiricalValidation-CPU.sh small1000,full100k
#
#   $1 RESULTS   Step 4 results folder(s), comma separated (blank = all)
#
# Output: EmpiricalValidation/<RESULTS>/{REPORT.txt, estimates.csv, summary.csv, manifest.json, figures/}
# Also possible without Slurm (CPU, takes a few minutes):
#   .envs/phylocnn/bin/python Step5-EmpiricalValidation.py run                     # every results folder
#   .envs/phylocnn/bin/python Step5-EmpiricalValidation.py run --results full100k
#   .envs/phylocnn/bin/python Step5-EmpiricalValidation.py check                   # encoding checks only

set -euo pipefail

cd "$SLURM_SUBMIT_DIR"

# ===================== PARAMETERS =====================
RESULTS="${1:-}"                 # blank = every results folder
CASES="HIV_Zurich,Primates"
DATASETS="BD,BDEI,BDSS,BiSSE"
HIV_SAMPLING=0.25                # Zurich HIV tree (PhyloDeep / PhyloCNN)
PRIMATES_SAMPLING=0.68241469816273   # 260 of 381 primate species
FORMATS="pdf,png"
# ======================================================

THREADS="${SLURM_CPUS_PER_TASK:-1}"
export OMP_NUM_THREADS="$THREADS"
export OPENBLAS_NUM_THREADS="$THREADS"
export MKL_NUM_THREADS="$THREADS"
export CUDA_VISIBLE_DEVICES=""
export TF_CPP_MIN_LOG_LEVEL=2

echo "=================================="
echo "PHYLOGNN STEP 5 EMPIRICAL VALIDATION (CPU)"
echo "=================================="
echo "Job ID: $SLURM_JOB_ID   Node: $(hostname)   CPUs: $THREADS   Started: $(date)"
echo "Results: ${RESULTS:-all folders in Results/}"

PY="$PWD/.envs/phylocnn/bin/python"
if [ ! -x "$PY" ]; then
    echo "ERROR: Python environment missing: $PY"
    exit 1
fi
for f in Step5-EmpiricalValidation.py Step6-Analysis.py Step4-CheckPhyloCNN.py Graph_Encoding.py GNNArchitectures.py; do
    if [ ! -f "$f" ]; then
        echo "ERROR: $f not found in $PWD (submit from the project root)"
        exit 1
    fi
done

srun "$PY" -u Step5-EmpiricalValidation.py run \
    --results "$RESULTS" \
    --cases "$CASES" \
    --datasets "$DATASETS" \
    --hiv-sampling "$HIV_SAMPLING" \
    --primates-sampling "$PRIMATES_SAMPLING" \
    --formats "$FORMATS" \
    --device cpu \
    --threads "$THREADS"

echo "Step 5 empirical validation completed: $(date)"
