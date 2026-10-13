#!/bin/bash --login
#SBATCH --job-name=PhyloGNN-CPU
#SBATCH --account=a_ace
#SBATCH --partition=general
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=32G
#SBATCH --time=24:00:00
#SBATCH --array=0-59
#SBATCH --output=logs/Step4-CPU-%A_%a.out
#SBATCH --error=logs/Step4-CPU-%A_%a.err

# Step 4 benchmark on Bunya CPUs (general partition, no GPU): 60 jobs, one per dataset x model.
#   4 datasets x 15 models = 60 runs  ->  --array=0-59
#   task  0-14  BD     (3 PhyloCNN + 12 GNN models)
#   task 15-29  BDEI
#   task 30-44  BDSS
#   task 45-59  BiSSE
#
# Submit from the project root (the folder with Data/, phyloCNN/, .envs/):
#   mkdir -p logs
#   sbatch --array=0,3 --time=01:00:00 Step4-CheckPhyloCNN-CPU.sh small1000 10000 step3 1000 100 100   # test
#   sbatch Step4-CheckPhyloCNN-CPU.sh small1000 10000 step3 1000 100 100   # 1,000/100/100 trees
#   sbatch Step4-CheckPhyloCNN-CPU.sh results1 10000 step3                  # all 10,000 trees
#   sbatch --mem=96G --time=7-00:00:00 Step4-CheckPhyloCNN-CPU.sh results2 100000 step3   # 100,000 trees
# Options given to sbatch on the command line (--cpus-per-task, --mem, --time, --array) override the
# #SBATCH lines. CPU training is much slower than GPU training, especially for gps, pna, ecc and the
# fusion models on large tree sets: increase --time if runs are STOPPED by the time limit.
#
#   $1 RESULTS     output folder name           -> Results/<RESULTS>/
#   $2 TREES       10000 (Step 3 nested subset) or 100000 (Step 3 full set)
#   $3 VALIDATION  step3 (Step 3's split) or a fraction such as 0.2
#   $4 $5 $6       optional: number of train / validation / author-test trees (0 or omitted = all)
#
# Progress of every run (state, current epoch, losses, seconds/epoch, test score when done):
#   .envs/phylocnn/bin/python Step4-CheckPhyloCNN.py --status --results small1000
#   .envs/phylocnn/bin/python Step4-CheckPhyloCNN.py --watch 60 --results small1000
# Ranking of finished runs:
#   .envs/phylocnn/bin/python Step4-CheckPhyloCNN.py --aggregate --results small1000
#
# Fewer datasets/models: change DATASETS / MODELS below and submit with the matching range:
#   sbatch --array=$(.envs/phylocnn/bin/python Step4-CheckPhyloCNN.py --array-range \
#          --datasets BD --models gcn,gps) Step4-CheckPhyloCNN-CPU.sh results1 10000 step3
# Finished runs with the same settings are skipped, so resubmitting only redoes the rest.

set -euo pipefail

cd "$SLURM_SUBMIT_DIR"

# ===================== PARAMETERS =====================
RESULTS="${1:-results1}"
TREES="${2:-10000}"
VALIDATION="${3:-step3}"

DATASETS="${DATASETS:-BD,BDEI,BDSS,BiSSE}"
MODELS="${MODELS:-phylocnn_2gen,phylocnn_1gen,phylocnn_nocontext,gcn,graphsage,gat,gin,ecc,gine,pna,transformer,gps,trignn,gnn_fusion,graphsage_gat_fusion}"

EPOCHS=1000              # maximum epochs
PATIENCE=100             # early stopping on validation loss (0 = off)
LEARNING_RATE=0.001
CNN_BATCH_SIZE=1024      # authors' PhyloCNN batch size
GNN_BATCH_SIZE=64
HIDDEN_DIM=32
NUM_LAYERS="${NUM_LAYERS:-3}"
HEADS=4
DROPOUT=0.1
CNN_PROTOCOL=authors     # PhyloCNN as in the authors' notebooks (ELU output, MAPE / MAE loss); 'unified' = like the GNNs
# GNN ablation options (defaults = the main benchmark; Step4-Ablations.sh overrides them)
POOLING="${POOLING:-mean}"            # mean | sum | max | mean_max | mean_max_root | mean_max_sum | attention
RESIDUAL="${RESIDUAL:-0}"             # 1 = residual connections between GNN layers
GNN_LOSS="${GNN_LOSS:-huber}"         # huber | paper (MRE for BD/BDEI/BDSS, MAE for BiSSE)
NODE_FEATURES="${NODE_FEATURES:-standard}"   # standard | phylocnn | both
SEED="${SEED:-42}"
GNN_TARGETS="${GNN_TARGETS:-natural}"  # natural | rescaled (PhyloCNN-style: periods / r, rates * r)
GRAPH_EDGES="${GRAPH_EDGES:-tree}"     # tree | multiscale (also edges to ancestors 2..64 steps up)
JK="${JK:-none}"                       # none | cat (pool the concatenated embeddings of every layer)

TRAIN_LIMIT="${4:-0}"    # 0 = all training trees of the chosen set
VAL_LIMIT="${5:-0}"      # 0 = all validation trees
TEST_LIMIT="${6:-0}"     # 0 = the full authors' test set

CACHE_DIR="Data/_step4_cache"   # encoded trees, shared by all RESULTS folders
WORKERS="${SLURM_CPUS_PER_TASK:-1}"
# ======================================================

# Use the CPUs Slurm gave this job (PyTorch, TensorFlow, NumPy and tree encoding).
THREADS="${SLURM_CPUS_PER_TASK:-1}"
export OMP_NUM_THREADS="$THREADS"
export OPENBLAS_NUM_THREADS="$THREADS"
export MKL_NUM_THREADS="$THREADS"
export TF_NUM_INTRAOP_THREADS="$THREADS"
export TF_NUM_INTEROP_THREADS=2
export CUDA_VISIBLE_DEVICES=""          # CPU job: never look for a GPU
export TF_CPP_MIN_LOG_LEVEL=2

echo "=================================="
echo "PHYLOGNN STEP 4 BENCHMARK (CPU)"
echo "=================================="
echo "Job ID: $SLURM_JOB_ID   Array Task: ${SLURM_ARRAY_TASK_ID:-none}"
echo "Node: $(hostname)   CPUs: $THREADS   Started: $(date)"
echo "Results: Results/$RESULTS   Trees: $TREES   Validation: $VALIDATION"
echo "Limits train/val/test: $TRAIN_LIMIT/$VAL_LIMIT/$TEST_LIMIT (0 = all)"

PY="$PWD/.envs/phylocnn/bin/python"

if [ ! -x "$PY" ]; then
    echo "ERROR: Python environment missing: $PY"
    exit 1
fi

if [ ! -f Step4-CheckPhyloCNN.py ]; then
    echo "ERROR: submit from the project root (Step4-CheckPhyloCNN.py not found in $PWD)"
    exit 1
fi

TASK_ARGS=()
if [ -n "${SLURM_ARRAY_TASK_ID:-}" ]; then
    TASK_ARGS=(--task-id "$SLURM_ARRAY_TASK_ID")
fi
if [ "$RESIDUAL" = "1" ]; then
    TASK_ARGS+=(--residual)
fi

srun "$PY" -u Step4-CheckPhyloCNN.py \
    ${TASK_ARGS[@]+"${TASK_ARGS[@]}"} \
    --device cpu \
    --results "$RESULTS" \
    --trees "$TREES" \
    --validation "$VALIDATION" \
    --datasets "$DATASETS" \
    --models "$MODELS" \
    --epochs "$EPOCHS" \
    --patience "$PATIENCE" \
    --learning-rate "$LEARNING_RATE" \
    --cnn-batch-size "$CNN_BATCH_SIZE" \
    --gnn-batch-size "$GNN_BATCH_SIZE" \
    --hidden-dim "$HIDDEN_DIM" \
    --num-layers "$NUM_LAYERS" \
    --heads "$HEADS" \
    --dropout "$DROPOUT" \
    --cnn-protocol "$CNN_PROTOCOL" \
    --pooling "$POOLING" \
    --gnn-loss "$GNN_LOSS" \
    --node-features "$NODE_FEATURES" \
    --seed "$SEED" \
    --gnn-targets "$GNN_TARGETS" \
    --graph-edges "$GRAPH_EDGES" \
    --jk "$JK" \
    --train-limit "$TRAIN_LIMIT" \
    --val-limit "$VAL_LIMIT" \
    --test-limit "$TEST_LIMIT" \
    --cache-dir "$CACHE_DIR" \
    --workers "$WORKERS"

echo "CPU job completed: $(date)"
