#!/bin/bash
# Step4-Ablations.sh - focused GNN experiments on the 10,000-tree data (no PhyloCNN, no new simulations).
#
# Train / validation / author-test trees are the same as in the main benchmark; every experiment is
# repeated with several seeds. One change against the baseline:
#   base        mean pooling, 3 layers, Huber loss, standard node features (the main benchmark)
#   pool        mean + max + root-aware pooling            -> is structural information lost by mean pooling?
#   attention   attention pooling
#   deep        6 layers with residual connections         -> is the receptive field too small?
#   loss        the paper's measure as loss (MRE; MAE BiSSE) -> is the training objective limiting MRE?
#   phylocnn    the authors' PhyloCNN node features on the real tree   -> features or message passing?
#   both        standard + PhyloCNN node features
# Cumulative steps (A -> B -> C):
#   rescaled                    A: PhyloCNN-style targets (periods / r, rates * r), back-converted per tree
#   rescaled_pool               B: A + mean/max/sum pooling
#   rescaled_pool_multiscale    C: B + edges to ancestors 2..64 steps up + jumping knowledge (all layers pooled)
#
# Usage (from the project root; it only submits jobs, through Step4-CheckPhyloCNN-CPU.sh):
#   bash Step4-Ablations.sh                                   # the default experiments x seeds
#   bash Step4-Ablations.sh pool deep                         # only these experiments
#   MODELS=gine DATASETS=BD,BDSS bash Step4-Ablations.sh base rescaled rescaled_pool rescaled_pool_multiscale
#   DRY_RUN=1 bash Step4-Ablations.sh                         # print what would be submitted
# Results: Results/abl_<experiment>_s<seed>/
# Choose settings on the VALIDATION trees, then report them on the author test set:
#   .envs/phylocnn/bin/python Step6-Analysis.py run --results "$(ls -d Results/abl_* | xargs -n1 basename | paste -sd,)" --split validation
#   (the cross-folder section averages every experiment over its seeds)
# Against the authors' pretrained PhyloCNN on identical test trees:
#   .envs/phylocnn/bin/python Step7-ExactComparison.py run --results <folders>

set -euo pipefail

# ===================== PARAMETERS =====================
TREES=10000
VALIDATION=step3
DATASETS="${DATASETS:-BD,BDSS,BiSSE}"           # BDEI left out
MODELS="${MODELS:-gcn,graphsage,gin,gine}"      # the best GNNs of the 10K / 100K runs
SEEDS="${SEEDS:-42 43 44}"
TIME="${TIME:-24:00:00}"
EXPERIMENTS="${*:-base pool deep loss phylocnn both rescaled rescaled_pool rescaled_pool_multiscale}"
# ======================================================

if [ ! -f Step4-CheckPhyloCNN-CPU.sh ]; then
    echo "ERROR: run from the project root (Step4-CheckPhyloCNN-CPU.sh not found)"
    exit 1
fi
mkdir -p logs
PY="$PWD/.envs/phylocnn/bin/python"
RANGE=$("$PY" Step4-CheckPhyloCNN.py --array-range --datasets "$DATASETS" --models "$MODELS")

for experiment in $EXPERIMENTS; do
    POOLING=mean; RESIDUAL=0; NUM_LAYERS=3; GNN_LOSS=huber; NODE_FEATURES=standard
    GNN_TARGETS=natural; GRAPH_EDGES=tree; JK=none
    case "$experiment" in
        base) ;;
        pool) POOLING=mean_max_root ;;
        attention) POOLING=attention ;;
        deep) NUM_LAYERS=6; RESIDUAL=1 ;;
        loss) GNN_LOSS=paper ;;
        phylocnn) NODE_FEATURES=phylocnn ;;
        both) NODE_FEATURES=both ;;
        rescaled) GNN_TARGETS=rescaled ;;
        rescaled_pool) GNN_TARGETS=rescaled; POOLING=mean_max_sum ;;
        rescaled_pool_multiscale) GNN_TARGETS=rescaled; POOLING=mean_max_sum; GRAPH_EDGES=multiscale; JK=cat ;;
        *) echo "Unknown experiment: $experiment"; exit 1 ;;
    esac
    for seed in $SEEDS; do
        results="abl_${experiment}_s${seed}"
        echo "$results: POOLING=$POOLING RESIDUAL=$RESIDUAL NUM_LAYERS=$NUM_LAYERS GNN_LOSS=$GNN_LOSS" \
             "NODE_FEATURES=$NODE_FEATURES GNN_TARGETS=$GNN_TARGETS GRAPH_EDGES=$GRAPH_EDGES JK=$JK SEED=$seed" \
             " ->  sbatch --array=$RANGE --time=$TIME Step4-CheckPhyloCNN-CPU.sh $results $TREES $VALIDATION"
        if [ "${DRY_RUN:-0}" != "1" ]; then
            # sbatch passes the environment to the job (values with commas cannot go through --export=).
            (export DATASETS MODELS POOLING RESIDUAL NUM_LAYERS GNN_LOSS NODE_FEATURES GNN_TARGETS GRAPH_EDGES JK \
                    SEED="$seed"
             sbatch --array="$RANGE" --time="$TIME" --job-name="$results" --export=ALL \
                 Step4-CheckPhyloCNN-CPU.sh "$results" "$TREES" "$VALIDATION")
        fi
    done
done
echo "Progress:  $PY Step4-CheckPhyloCNN.py --status --results abl_<experiment>_s<seed> --datasets $DATASETS --models $MODELS"
