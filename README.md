# PhyloGNN

Graph neural networks for estimating phylodynamic and diversification parameters from phylogenetic
trees, benchmarked against **PhyloCNN** (Perez & Gascuel, *Systematic Biology*, 2025) on the authors'
own simulations, test sets and empirical trees.

The pipeline runs on a Slurm cluster (developed on UQ's Bunya). Every step is one Python script, with
a Slurm script where a step needs many jobs. Run every command from the **project root** (the folder
that contains these scripts).

| Step | Script | What it does |
|---|---|---|
| 1 | `Step1-DownloadRepositories.py` | Clones PhyloCNN, PhyloDeep and DeepTimeLearning (simulators, encoders, test sets, trained models) |
| 2 | `Step2-InstallPackages.py` | Creates the Python environment `.envs/phylocnn` (TensorFlow, PyTorch, PyTorch Geometric) and the R environment `.envs/phylo-r` |
| 3 | `Step3-GenerateData.py` + `.sh` | Simulates 100,000 trees per model (and a nested 10,000 subset), encodes them, encodes the authors' test sets |
| 4 | `Step4-CheckPhyloCNN.py` + `.sh` | Trains and tests 15 models × 4 models of evolution (3 PhyloCNN + 12 GNN architectures) |
| 4b | `Step4-Ablations.sh` | Focused GNN experiments (pooling, depth, loss, features, target scaling, multi-scale edges) |
| 5 | `Step5-EmpiricalValidation.py` + `.sh` | Applies every trained model to the HIV (Zurich) and primate trees |
| 6 | `Step6-Analysis.py` | Statistics, significance tests, tables (CSV / LaTeX) and journal figures |
| 7 | `Step7-ExactComparison.py` | Runs the authors' own pretrained PhyloCNN models and compares them with ours on identical trees |

Helper modules (used by the steps, not run directly): `Graph_Encoding.py` (tree → graph, PhyloCNN
inputs, caching, metrics) and `GNNArchitectures.py` (the 12 PyTorch Geometric architectures).
`PrepareStep4Data.py` is optional: it exports Step 3's complete chunks to
`phyloCNN/Training_Data/<MODEL>/` for the authors' own notebooks (Step 4 reads `Data/` directly).

---

## Contents
[Models](#models-and-datasets) · [Requirements](#requirements) · [Quick start](#quick-start-all-commands) ·
[Step 1](#step-1--download-the-repositories) · [Step 2](#step-2--install-the-environments) ·
[Step 3](#step-3--generate-the-data) · [Step 4](#step-4--train-and-test-the-models) ·
[Step 4b](#step-4b--gnn-experiments-ablations) · [Step 5](#step-5--empirical-trees-hiv-and-primates) ·
[Step 6](#step-6--analysis-statistics-and-figures) · [Step 7](#step-7--comparison-with-the-authors-pretrained-phylocnn) ·
[Metrics](#metrics) · [Methods notes](#methods-notes) · [Outputs](#where-the-results-are) ·
[Troubleshooting](#troubleshooting) · [Credits](#credits)

---

## Models and datasets

**Models of evolution (datasets):** BD, BDEI, BDSS (epidemiology) and BiSSE (trait-dependent
diversification), with the authors' simulators and priors.

| Dataset | Parameters estimated |
|---|---|
| BD | R0, infectious period |
| BDEI | R0, infectious period, incubation period |
| BDSS | R0, infectious period, transmission factor (superspreading), superspreader fraction |
| BiSSE | turnover, λ1, λ2, q01 |

**15 models:**

| Family | Models |
|---|---|
| PhyloCNN (authors' architecture, Keras) | `phylocnn_2gen`, `phylocnn_1gen`, `phylocnn_nocontext` |
| Baseline GNNs | `gcn`, `graphsage`, `gat`, `gin`, `ecc` |
| Edge-aware GNNs | `gine`, `pna`, `transformer`, `gps` |
| Fusion GNNs | `trignn`, `gnn_fusion`, `graphsage_gat_fusion` |

4 datasets × 15 models = **60 runs** (Slurm array tasks `0-59`):

| Tasks | Dataset | PhyloCNN tasks |
|---|---|---|
| 0–14 | BD | 0–2 |
| 15–29 | BDEI | 15–17 |
| 30–44 | BDSS | 30–32 |
| 45–59 | BiSSE | 45–47 |

---

## Requirements

* Linux with **Slurm**, `git`, and `conda`/`mamba` (or micromamba). Developed on UQ Bunya
  (`--account=a_ace`; change the account in the `.sh` files for your own allocation).
* The Python environment is **Python 3.8** (TensorFlow 2.13 / Keras 2.13, PyTorch, PyTorch Geometric).
  All scripts are Python-3.8 compatible.
* CPU is enough for everything; a CUDA GPU (A100/H100 on `gpu_cuda`) speeds up Step 4 on 100,000 trees.
  AMD (`gpu_rocm`) GPUs cannot be used with this environment.

Always use the project's Python: `.envs/phylocnn/bin/python`.

---

## Quick start (all commands)

```bash
# --- once ---------------------------------------------------------------------------------
git clone https://github.com/MuhammadMuneeb007/PhyloGNN.git && cd PhyloGNN
mkdir -p logs
python Step1-DownloadRepositories.py                      # Step 1: phyloCNN/, phylodeep/, deeptimelearning/
python Step2-InstallPackages.py                           # Step 2: .envs/phylocnn and .envs/phylo-r
.envs/phylocnn/bin/python GNNArchitectures.py --check     # all 12 GNN architectures build and train

# --- Step 3: data (one Slurm array, 804 tasks) -----------------------------------------------
.envs/phylocnn/bin/python Step3-GenerateData.py --pilot   # one tree per model: tests the simulators
sbatch Step3-GenerateData.sh
.envs/phylocnn/bin/python Step3-GenerateData.py --status  # progress; re-submit unfinished items with:
sbatch --array=$(.envs/phylocnn/bin/python Step3-GenerateData.py --pending-array) Step3-GenerateData.sh

# --- Step 4: train + test (60 array tasks per results folder) --------------------------------
sbatch Step4-CheckPhyloCNN-CPU.sh small1000 10000 step3 1000 100 100              # quick test
sbatch Step4-CheckPhyloCNN-CPU.sh results10k 10000 step3                          # 10,000 trees
sbatch --mem=96G --time=7-00:00:00 Step4-CheckPhyloCNN-CPU.sh full100k 100000 step3   # 100,000 trees (CPU)
sbatch --mem=64G --time=48:00:00   Step4-CheckPhyloCNN-GPU.sh full100k 100000 step3   # ... or on a GPU
.envs/phylocnn/bin/python Step4-CheckPhyloCNN.py --status --results full100k     # progress
.envs/phylocnn/bin/python Step4-CheckPhyloCNN.py --aggregate --results full100k  # ranking

# --- Step 4b: GNN experiments (optional) ------------------------------------------------------
MODELS=gine DATASETS=BD,BDSS bash Step4-Ablations.sh base rescaled rescaled_pool rescaled_pool_multiscale

# --- Steps 5-7: empirical trees, analysis, comparison with the authors' models ---------------
.envs/phylocnn/bin/python Step5-EmpiricalValidation.py run     # or: sbatch Step5-EmpiricalValidation-CPU.sh
.envs/phylocnn/bin/python Step6-Analysis.py run
.envs/phylocnn/bin/python Step7-ExactComparison.py run
```

---

## Step 1 — Download the repositories

```bash
python Step1-DownloadRepositories.py
```

Clones into the project root (already-cloned repositories are skipped):

| Folder | Source | Used for |
|---|---|---|
| `phyloCNN/` | github.com/manolofperez/phyloCNN | encoders, test sets, empirical trees, trained models, notebooks |
| `phylodeep/` | github.com/evolbioinfo/phylodeep | BD / BDEI / BDSS simulators |
| `deeptimelearning/` | github.com/JakubVoz/deeptimelearning | BiSSE simulator, primate data, HiSSE fit |

For a reproducible record, note each repository's commit: `git -C phyloCNN rev-parse HEAD` (same for
the other two).

## Step 2 — Install the environments

```bash
python Step2-InstallPackages.py
.envs/phylocnn/bin/python GNNArchitectures.py --check
```

* `.envs/phylocnn` — Python 3.8 with the original PhyloCNN packages (TensorFlow 2.13, Keras 2.13,
  NumPy 1.22, ete3, ...) plus PyTorch, PyTorch Geometric and Spektral. The requirements actually
  installed are written to `Step2-PhyloCNN-Linux-requirements.txt`.
* `.envs/phylo-r` — R with `castor` and `TreeSim` (BiSSE simulations).
* `GNNArchitectures.py --check` builds, runs and back-propagates all 12 architectures (several
  settings) — run it after every installation.

`Step2-InstallPackages.sh` is an older shell variant (named conda environments, no PyTorch); prefer
the Python installer.

## Step 3 — Generate the data

Simulates **100,000 trees per model** (BD, BDEI, BDSS, BiSSE) with the authors' simulators and priors,
in 4 × 200 chunks of 500 trees, draws a **nested 10,000-tree subset**, assigns a fixed 70/30
train/validation split, encodes every tree with the authors' PhyloCNN encoder, and encodes the
authors' independent test sets. One Slurm array (`0-803`): items 0–799 are simulation chunks,
800–803 encode the test sets. The first array task prepares the run; the others wait for it.

```bash
.envs/phylocnn/bin/python Step3-GenerateData.py --pilot          # 1 tree per model, before submitting
.envs/phylocnn/bin/python Step3-GenerateData.py --array-range    # prints 0-803
mkdir -p logs && sbatch Step3-GenerateData.sh                    # the whole array
.envs/phylocnn/bin/python Step3-GenerateData.py --status         # per-model progress, kept/excluded trees
.envs/phylocnn/bin/python Step3-GenerateData.py --pending-array  # unfinished items -> re-submit:
sbatch --array=<printed list> Step3-GenerateData.sh
.envs/phylocnn/bin/python Step3-GenerateData.py --audit-author-tests
```

* Finished chunks are verified and skipped, so re-submitting is safe.
* A parameter row that cannot be simulated after 20 seeded attempts is **excluded** and recorded in
  the chunk's `DONE.json` (at most 2% per chunk); Step 4 accounts for exclusions.
* Slurm allows at most 1,000 array elements; for more work items use
  `sbatch --array=0-999 --export=ALL,TASK_OFFSET=1000 Step3-GenerateData.sh`.

Output layout:

```
Data/<MODEL>/full_100000/chunks/NNNN/{DONE.json, raw/trees.tsv.gz, raw/parameters_final.tsv.gz, encoded/encoded.tsv.gz}
Data/<MODEL>/subset_10000/...               nested 10,000-tree subset
Data/<MODEL>/{full,subset}_trainval_assignments.tsv
Data/<MODEL>/author_test/                   the authors' test set, encoded
Data/_step3/{config.json, tasks.json}
```

## Step 4 — Train and test the models

Every run trains one model on one dataset, early-stops on the validation trees, restores the best
epoch and evaluates on train, validation and the **authors' independent test set**. One Slurm array
task = one run (`0-59`).

```bash
#                                    RESULTS    TREES   VALIDATION  [TRAIN VAL TEST limits]
sbatch Step4-CheckPhyloCNN-CPU.sh    small1000  10000   step3       1000 100 100   # quick test
sbatch Step4-CheckPhyloCNN-CPU.sh    results10k 10000   step3                      # all 10,000 trees
sbatch --mem=96G --time=7-00:00:00 Step4-CheckPhyloCNN-CPU.sh full100k 100000 step3
sbatch --mem=64G --time=48:00:00   Step4-CheckPhyloCNN-GPU.sh full100k 100000 step3   # A100 (or --gres=gpu:h100:1)
```

| Argument | Meaning |
|---|---|
| `RESULTS` | output folder `Results/<RESULTS>/` |
| `TREES` | `10000` (Step 3's nested subset) or `100000` (full set) |
| `VALIDATION` | `step3` (Step 3's 70/30 split) or a fraction such as `0.2` |
| `TRAIN VAL TEST` | optional limits on the number of trees (0 or omitted = all) |

Training settings are at the top of the `.sh` files (`EPOCHS=1000`, `PATIENCE=100`, `CNN_BATCH_SIZE=1024`,
`GNN_BATCH_SIZE=64`, `HIDDEN_DIM=32`, `NUM_LAYERS=3`, ...). `Step4-CheckPhyloCNN-CPU.sh` runs on the
`general` partition, `Step4-CheckPhyloCNN-GPU.sh` on `gpu_cuda` (it checks that PyTorch can compute on
the GPU before training).

**Monitoring and results**

```bash
.envs/phylocnn/bin/python Step4-CheckPhyloCNN.py --status --results full100k     # every run: state, epoch, losses, TEST_nMAE, TEST_PAPER
.envs/phylocnn/bin/python Step4-CheckPhyloCNN.py --watch 60 --results full100k   # refresh every 60 s
.envs/phylocnn/bin/python Step4-CheckPhyloCNN.py --aggregate --results full100k  # ranking per dataset
```

**Useful variations**

```bash
# only some runs (e.g. the 12 PhyloCNN runs of a folder; finished runs are skipped anyway)
sbatch --array=0-2,15-17,30-32,45-47 Step4-CheckPhyloCNN-CPU.sh full100k 100000 step3
# by hand, without Slurm
.envs/phylocnn/bin/python Step4-CheckPhyloCNN.py --results test --trees 10000 --epochs 2 \
    --train-limit 64 --val-limit 16 --test-limit 16 --datasets BD --models gcn,phylocnn_2gen
```

* A run whose `PASS.json` matches the current settings is **skipped** — re-submitting only redoes what
  is missing or failed (`--force` redoes everything).
* The three files `Step4-CheckPhyloCNN.py`, `Graph_Encoding.py` and `GNNArchitectures.py` are updated
  together; Step 4 refuses to start with an older helper file and says which one to copy.
* Encoded trees are cached in `Data/_step4_cache/` and shared by all results folders.
* `--device auto|cpu|cuda` (the `.sh` files set it), `--threads N`.

**PhyloCNN protocol.** By default (`CNN_PROTOCOL=authors`) PhyloCNN is trained exactly as in the
authors' notebooks: their encoder and input assembly, time-scaled targets in rescaled units
(periods ÷ r, rates × r, r = the tree's mean branch length), ELU output, MAPE loss (BD/BDEI/BDSS) or MAE
loss (BiSSE), Adam, batch 1024, ≤ 1000 epochs, patience 100, predictions converted back with each tree's
r. `CNN_PROTOCOL=unified` trains it like the GNNs instead. The GNNs use standardised targets, a linear
output and the Huber loss (see Step 4b for alternatives).

**Outputs per run** — `Results/<RESULTS>/runs/<DATASET>/<MODEL>/`:

| File | Content |
|---|---|
| `checkpoint/model_final.pt` / `.keras` | the evaluated model (best validation epoch) |
| `checkpoint/checkpoint.json` | everything needed to reuse it (scalers, feature sets, settings) |
| `metrics.json`, `summary.csv`, `per_target.csv` | metrics per split and per parameter |
| `predictions_<split>.csv` | true and predicted values of every tree |
| `history.csv`, `PASS.json` / `FAILED.json`, `fusion_weights.json` | training curve, status, fusion weights |

Plus `Results/<RESULTS>/all_results.csv`, `per_target.csv`, `failures.csv`, `RUN_CONFIG.json`.

## Step 4b — GNN experiments (ablations)

Controlled experiments on the 10,000-tree data: each changes one thing (or adds one step) against
the baseline, with the same train/validation/test trees, repeated over seeds 42, 43, 44.

| Experiment | Change | Question |
|---|---|---|
| `base` | none (mean pooling, 3 layers, Huber, standard features) | reference |
| `pool` | mean + max + root-node pooling | does mean pooling lose structure? |
| `attention` | attention pooling | |
| `deep` | 6 layers with residual connections | is the receptive field too small? |
| `loss` | the paper's measure as loss (MRE; MAE for BiSSE) | does the objective limit MRE? |
| `phylocnn` / `both` | the authors' PhyloCNN node features on the real tree (alone / added) | features or message passing? |
| `rescaled` (A) | PhyloCNN-style targets (periods ÷ r, rates × r) | does target scaling limit the GNNs? |
| `rescaled_pool` (B) | A + mean/max/sum pooling | |
| `rescaled_pool_multiscale` (C) | B + edges to ancestors 2–64 steps up + all layers pooled | is long-range information missing? |

```bash
bash Step4-Ablations.sh                                              # default experiments, 4 GNNs, BD/BDSS/BiSSE
MODELS=gine DATASETS=BD,BDSS bash Step4-Ablations.sh base rescaled rescaled_pool rescaled_pool_multiscale
DRY_RUN=1 bash Step4-Ablations.sh                                    # show what would be submitted
# choose on VALIDATION trees (averaged over seeds, change vs base):
.envs/phylocnn/bin/python Step6-Analysis.py run --split validation \
    --results "$(ls -d Results/abl_* | xargs -n1 basename | paste -sd,)"
```

Results go to `Results/abl_<experiment>_s<seed>/`. The same options exist on the Step 4 command line:
`--pooling`, `--residual`, `--num-layers`, `--gnn-loss`, `--node-features`, `--gnn-targets`,
`--graph-edges`, `--jk`, `--seed`. Only non-default values enter a run's settings, so existing runs
stay valid.

## Step 5 — Empirical trees (HIV and primates)

```bash
.envs/phylocnn/bin/python Step5-EmpiricalValidation.py check                  # encoding checks only
.envs/phylocnn/bin/python Step5-EmpiricalValidation.py run                    # every results folder
.envs/phylocnn/bin/python Step5-EmpiricalValidation.py run --results full100k
sbatch Step5-EmpiricalValidation-CPU.sh full100k                              # same, as a Slurm job
```

| Case | Tree | Sampling | Models applied |
|---|---|---|---|
| HIV Zurich | 200 tips (Rasmussen et al. 2017), years | 0.25 | BDSS, BD, BDEI |
| Primates | 260 tips, seed-dispersal trait (Fabre et al. 2009; Gomez & Verdu 2012), Myr | 260/381 | BiSSE |

For every finished model it prints:
* the model's accuracy on the simulated test trees (MRE/MAE, nMAE, R², per parameter);
* its estimates on the real tree, with flags (outside the training range, tree unlike the training trees);
* a parameter-by-parameter comparison with the **published PhyloCNN estimates and 95% CIs**
  (from the authors' notebooks). Neither tree has known parameters, so "closer to the truth" is
  reported as unknown;
* for the primates, the **BiSSE log-likelihood** of every model's estimates next to the maximum
  likelihood and the published PhyloCNN — an independent criterion. The implementation reproduces the
  published values (PhyloCNN −850.06 vs −850.1; maximum −846.93 vs −846.9 for HiSSE).

Before any estimate, the PhyloCNN encoding is checked against the authors' own numbers (rescaling
factors 5.174109 for HIV and 4.064753 for the primates, tip states, input shape). Outputs:
`EmpiricalValidation/<RESULTS>/` (`REPORT.txt`, `estimates.csv`, `summary.csv`,
`likelihood_primates_bisse.csv`, figures).

## Step 6 — Analysis, statistics and figures

```bash
.envs/phylocnn/bin/python Step6-Analysis.py run                              # every results folder
.envs/phylocnn/bin/python Step6-Analysis.py run --results full100k --datasets BD,BDSS,BiSSE
.envs/phylocnn/bin/python Step6-Analysis.py run --split validation           # for model selection
.envs/phylocnn/bin/python Step6-Analysis.py run --formats pdf,png,svg --alpha 0.01
```

* Per dataset: training statistics, accuracy (nMAE with bootstrap 95% CI, paper MRE/MAE, MAE, RMSE,
  R², Pearson, Spearman), **Friedman test** and **pairwise Wilcoxon signed-rank tests with Holm
  correction** on per-tree errors, which model is significantly best, best PhyloCNN vs best PhyloGNN.
* Per parameter: all metrics, best model and models not significantly worse.
* Overview tables (nMAE and the paper's measure), LaTeX tables, critical-difference diagrams.
* With Step 5 results: estimates next to simulated accuracy, the primate likelihood ranking, a
  scorecard per model.
* Several folders: comparison across training-set sizes; `abl_*` folders averaged over seeds.

Outputs: `Results/<RESULTS>/analysis/` (`REPORT.txt`, `tables/*.csv|*.tex`, `figures/fig1..fig10.pdf|png`)
and `Results/_analysis/` for cross-folder comparisons.

## Step 7 — Comparison with the authors' pretrained PhyloCNN

```bash
.envs/phylocnn/bin/python Step7-ExactComparison.py run                      # every results folder
.envs/phylocnn/bin/python Step7-ExactComparison.py run --results results10k,full100k
.envs/phylocnn/bin/python Step7-ExactComparison.py run --limit 2000         # quick: 2,000 test trees per dataset
```

Runs every model in `phyloCNN/Trained_Models` (no training) on the authors' test sets and on the
empirical trees, then compares each of our models with the authors' pretrained PhyloCNN **on identical
test trees** (nMAE, paper MRE/MAE, % change, share of trees won, Wilcoxon + Holm). Each pretrained
model gets a reproduction check (median R² ≥ 0.5 with the released encoder); a model that does not
reproduce is reported but never used as the reference. Also: the HIV model-selection classifiers
(BD vs BDEI vs BDSS) and the authors' HIV-prior BDSS model. Outputs: `ExactComparison/`.

---

## Metrics

| Name | Definition | Where |
|---|---|---|
| **nMAE** | mean over trees and parameters of \|predicted − true\| / SD of the parameter in the training set | ranking and all significance tests |
| **MRE** (paper) | mean relative error \|predicted − true\| / true, averaged over parameters — BD, BDEI, BDSS | the PhyloCNN paper's measure |
| **MAE** (paper) | mean absolute error — BiSSE | the PhyloCNN paper's measure |
| R², Pearson r, Spearman ρ, RMSE, bias, MAPE | per parameter and averaged | Steps 4–7 |

Significance tests use nMAE (per-tree errors); MRE/MAE are reported as the paper's measures.

## Methods notes

* **Graph encoding.** One node per tree node (tips and internal nodes), the real parent–child edges in
  both directions, branch lengths on nodes and edges, 10 tree-level features (size, height, branch
  statistics, state fractions, sampling probability). BiSSE tip states come from the Newick NHX tags.
* **PhyloCNN rescaling.** The authors' encoder rescales every tree to a mean branch length of 1, so
  time-scaled parameters are learned in rescaled units and converted back with the tree's own factor.
  PhyloCNN runs trained before this was implemented learned raw values, cannot recover time-scaled
  parameters, and must be re-run (only the 12 PhyloCNN tasks); Step 5 leaves their time-scaled
  estimates blank.
* **BDEI incubation period.** The simulator's column `incubation_rate` holds 1 / (incubation factor ×
  removal rate) — a period (time), as in the authors' notebooks; `incubation_ratio` is never used.
* **Authors' pretrained models.** With the released encoder, the BDEI and BDSS models (all variants)
  and BD no-context reproduce; BD 2-generation / 1-generation and all BiSSE models do not (Step 7).
* **HIV.** The published HIV estimates come from a model trained with HIV-specific (wider) priors; a
  transmission factor of ≈ 20 lies outside the standard BDSS priors used in Step 3.
* **Leakage checks.** Step 4 refuses overlapping train/validation tree IDs and identical trees across
  train, validation and test.

## Where the results are

```
Results/<RESULTS>/runs/<DATASET>/<MODEL>/   Step 4: checkpoints, metrics, predictions
Results/<RESULTS>/analysis/                 Step 6: report, tables, figures
Results/_analysis/                          Step 6: comparison of results folders / ablations
EmpiricalValidation/<RESULTS>/              Step 5: HIV and primate estimates, likelihoods
ExactComparison/                            Step 7: pretrained PhyloCNN vs our models
logs/                                       Slurm logs (Step3-*, Step4-*, Step5-*)
```

## Troubleshooting

| Message | Cause / fix |
|---|---|
| `... is older than Step4-CheckPhyloCNN.py` | copy the matching `Graph_Encoding.py` / `GNNArchitectures.py` together with Step 4 |
| `AttributeError: module 'Graph_Encoding' has no attribute ...` | an old `Graph_Encoding.py` on the cluster — upload the current one |
| run `STOPPED` in `--status` | time limit or killed — increase `--time` and re-submit (finished runs are skipped) |
| PhyloCNN time-scaled estimates blank in Step 5 | old PhyloCNN checkpoint — re-run tasks `0-2,15-17,30-32,45-47` |
| `PyTorch cannot compute on this GPU` | wrong GPU type for this PyTorch build — use `--gres=gpu:a100:1` |
| Step 3 `Failed to rescue` / incomplete chunks | `--status`, then re-submit `--pending-array` |

Python 3.8 environment: keep all code 3.8-compatible (no `X | Y` types, `str.removeprefix`, `match`).

## Credits

* **PhyloCNN** — Perez M.F. and Gascuel O. (2025). *PhyloCNN: Improving tree representation and neural
  network architecture for deep learning from trees in phylodynamics and diversification studies.*
  Systematic Biology. <https://github.com/manolofperez/phyloCNN> — encoders, test sets, trained models,
  empirical trees and notebooks used here.
* **PhyloDeep** — Voznica J. et al. (2022), Nature Communications.
  <https://github.com/evolbioinfo/phylodeep> — BD / BDEI / BDSS simulators.
* **DeepTimeLearning** — Lambert S. et al. (2023), Systematic Biology.
  <https://github.com/JakubVoz/deeptimelearning> — BiSSE simulator, primate data, HiSSE fit.
* Graph models use **PyTorch Geometric**; PhyloCNN uses **TensorFlow / Keras**.

These repositories are downloaded by Step 1 and keep their own licences.
