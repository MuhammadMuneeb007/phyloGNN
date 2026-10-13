#!/usr/bin/env python3
"""
Step4-CheckPhyloCNN.py — PhyloCNN vs PhyloGNN (PyTorch Geometric) benchmark

Models (15):
  PhyloCNN (authors' architecture, Keras/TensorFlow, fresh weights):
    phylocnn_2gen, phylocnn_1gen, phylocnn_nocontext
  PhyloGNN (GNNArchitectures.py, PyTorch Geometric):
    gcn, graphsage, gat, gin, ecc                       message-passing baselines
    gine, pna, transformer, gps                         edge-aware PyG models
    trignn, gnn_fusion, graphsage_gat_fusion            fusion models

Datasets: BD, BDEI, BDSS, BiSSE   ->   4 x 15 = 60 runs

Data (Graph_Encoding.py, read directly from Step 3):
  --trees 10000   Step 3's nested subset    Data/<NAME>/subset_10000/
  --trees 100000  Step 3's full set         Data/<NAME>/full_100000/
  --validation step3   Step 3's own 70/30 train/validation split (default)
  --validation 0.2     re-split that tree set: 20% validation (seeded)
  author_test          the authors' test set (Data/<NAME>/author_test), never used for fitting

Each run:
  1. trains a new model on the training partition.
     PhyloCNN (--cnn-protocol authors, default) is trained exactly as in the authors' notebooks
     (phyloCNN/Preprocessing_Training/PhyloCNN_Train_*.ipynb): authors' encoder and input assembly,
     time-scaled targets in rescaled units (periods / r, rates * r; r = the tree's mean branch length)
     and NOT standardised, ELU output layer, MAPE loss for BD/BDEI/BDSS and MAE loss for BiSSE,
     Adam (lr 0.001), batch 1024, <= 1000 epochs, early stopping on validation loss with patience 100
     restoring the best epoch (set EPOCHS/PATIENCE/CNN_BATCH_SIZE in the .sh files), 30% validation.
     Predictions are converted back with each tree's own r. --cnn-protocol unified trains PhyloCNN
     like the GNNs instead (standardised targets, linear output, Huber loss).
     PhyloGNN: targets standardised on the training trees, linear output, Huber loss.
  2. early-stops on the validation partition (--patience; 0 = off) and restores the best epoch
  3. evaluates train, validation and author-test partitions
  4. saves in Results/<results>/runs/<dataset>/<model>/:
       checkpoint/model_final.pt | model_final.keras   the evaluated model (best validation epoch
                                                         when early stopping is on), self-contained
       checkpoint/model_last_epoch.*                    weights of the last epoch (if different)
       checkpoint/checkpoint.json                       build config, label + graph-feature scalers,
                                                         feature names, epochs -> load_trained_model()
       metrics.json, summary.csv, per_target.csv        metrics per split and per target
       predictions_<split>.csv, history.csv, PASS.json, fusion_weights.json (fusion models)
     Metrics: MAE, RMSE, median AE, bias, relative bias, MAPE, median APE, R2, Pearson r,
     Spearman rho, normalised MAE/RMSE/bias; plus epochs, best epoch, training time,
     inference time per tree, peak GPU memory and device.
  5. rebuilds Results/<results>/all_results.csv and per_target.csv from every finished run

A run whose PASS.json matches the current settings is skipped (use --force to redo).

Slurm (one run per array task): Step4-CheckPhyloCNN.sh
  sbatch Step4-CheckPhyloCNN.sh results1 10000 step3

By hand:
  python Step4-CheckPhyloCNN.py --results test --trees 10000 --epochs 1 \
      --train-limit 32 --val-limit 12 --test-limit 8 --datasets BD --models gcn,phylocnn_2gen
  python Step4-CheckPhyloCNN.py --results results1 --array-range     # number of runs, e.g. 0-59
  python Step4-CheckPhyloCNN.py --results results1 --aggregate       # summary + ranking

CPU or GPU: the same file runs on both. --device auto (default) uses the GPU when PyTorch /
TensorFlow can see one, otherwise the CPU; --device cpu forces the CPU for every model and
--device cuda requires a GPU. CPU runs use the CPUs Slurm gave the job (or --threads N).
Slurm scripts: Step4-CheckPhyloCNN-CPU.sh (general partition) and Step4-CheckPhyloCNN-GPU.sh (gpu_cuda).
"""

from __future__ import annotations

import os
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import argparse
import copy
import csv
import gc
import json
import random
import socket
import sys
import time
import traceback
from pathlib import Path

import numpy as np

import Graph_Encoding as ge


ROOT = Path(__file__).resolve().parent

# The three files are updated together; refuse to start with an older copy of a helper file
# (otherwise a run could train for hours and then fail during evaluation).
REQUIRED_ENCODING_API = 7
REQUIRED_ARCHITECTURES_API = 4
if getattr(ge, "STEP4_API", 0) < REQUIRED_ENCODING_API:
    raise SystemExit(f"[ERROR] {ge.__file__} is older than Step4-CheckPhyloCNN.py "
                     f"(STEP4_API {getattr(ge, 'STEP4_API', 'missing')} < {REQUIRED_ENCODING_API}). "
                     "Copy the matching Graph_Encoding.py.")


def check_architectures_file():
    """Version check of GNNArchitectures.py without importing PyTorch."""
    import re as _re
    path = ROOT / "GNNArchitectures.py"
    found = _re.search(r"^API_VERSION\s*=\s*(\d+)", path.read_text(), _re.M) if path.is_file() else None
    if not found or int(found.group(1)) < REQUIRED_ARCHITECTURES_API:
        raise SystemExit(f"[ERROR] {path} is missing or older than Step4-CheckPhyloCNN.py "
                         f"(API_VERSION {found.group(1) if found else 'missing'} < {REQUIRED_ARCHITECTURES_API}). "
                         "Copy the matching GNNArchitectures.py.")
RESULTS_ROOT = ROOT / "Results"
DEFAULT_CACHE = ROOT / "Data" / "_step4_cache"

CNN_MODELS = (
    "phylocnn_2gen",
    "phylocnn_1gen",
    "phylocnn_nocontext",
)
GNN_MODELS = (
    "gcn", "graphsage", "gat", "gin", "ecc",
    "gine", "pna", "transformer", "gps",
    "trignn", "gnn_fusion", "graphsage_gat_fusion",
)
ALL_MODELS = CNN_MODELS + GNN_MODELS
DATASETS = ge.DATASETS
TARGETS = ge.TARGETS

SEED = 42
# GNN options of the ablation experiments; only non-default values enter a run's signature.
GNN_OPTION_DEFAULTS = {"pooling": "mean", "residual": False, "gnn_loss": "huber", "node_features": "standard",
                       "gnn_targets": "natural", "graph_edges": "tree", "jk": "none"}
PARTITIONS = ("train", "validation", "author_test")
# PhyloCNN targets as in the authors' notebooks: periods / r and rates * r, where r is the
# tree's rescaling factor (mean branch length); predictions are converted back with r.
CNN_TARGET_SCALING = "authors_rescaled"


# ============================================================
# UTILITIES
# ============================================================

def log(*args):
    print(time.strftime("%H:%M:%S"), *args, flush=True)


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    if "torch" in sys.modules:
        sys.modules["torch"].manual_seed(seed)
    if "tensorflow" in sys.modules:
        sys.modules["tensorflow"].random.set_seed(seed)


def clear_memory():
    if "tensorflow" in sys.modules:
        sys.modules["tensorflow"].keras.backend.clear_session()
    if "torch" in sys.modules and sys.modules["torch"].cuda.is_available():
        sys.modules["torch"].cuda.empty_cache()
    gc.collect()


def _publish(tmp, path):
    """Atomic replace; retried because concurrent array tasks may publish the same file."""
    for attempt in range(10):
        try:
            os.replace(tmp, path)
            return
        except PermissionError:
            time.sleep(0.2 * (attempt + 1))
    os.replace(tmp, path)


def save_json(path, obj):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + f".tmp{os.getpid()}")
    tmp.write_text(json.dumps(obj, indent=2, default=str) + "\n")
    _publish(tmp, path)


def save_csv(path, rows):
    """Atomic write, so concurrent array tasks never leave a half-written file."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.unlink(missing_ok=True)
        return
    columns = list(dict.fromkeys(key for row in rows for key in row.keys()))
    tmp = path.with_name(path.name + f".tmp{os.getpid()}")
    with open(tmp, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    _publish(tmp, path)


def refresh_summary(args):
    """Rebuild the shared summary files; a failure here never fails a finished run."""
    try:
        aggregate(args, quiet=True)
    except (OSError, ValueError) as exc:
        log(f"[WARN] summary not refreshed ({exc}); run --aggregate later")


def read_csv(path):
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def paper_metric(dataset, row):
    """The accuracy measure of the PhyloCNN paper, averaged over the parameters: mean relative error
    MRE = mean(|predicted - true| / true) for BD, BDEI and BDSS, and mean absolute error for BiSSE.
    From a summary row (or aggregate): MRE = macro MAPE / 100, MAE = macro MAE in original units."""
    def number(value):
        try:
            return float(value)
        except (TypeError, ValueError):
            return float("nan")
    if dataset == "BiSSE":
        return "MAE", number(row.get("macro_mae_original_units"))
    return "MRE", number(row.get("macro_mape_percent")) / 100.0


def cpu_threads(args=None):
    """CPU threads to use: --threads, else the CPUs Slurm gave the job, else all CPUs."""
    if args is not None and getattr(args, "threads", 0):
        return int(args.threads)
    return int(os.environ.get("SLURM_CPUS_PER_TASK") or os.cpu_count() or 1)


_TF_CONFIGURED = False


def tensorflow(device="auto", threads=0):
    """Import TensorFlow (PhyloCNN runs only) and configure it once, before it starts:
    CPU threads, the device (--device cpu hides the GPU), and GPU memory growth."""
    global _TF_CONFIGURED
    import tensorflow as tf
    if not _TF_CONFIGURED:
        try:
            if threads:
                tf.config.threading.set_intra_op_parallelism_threads(int(threads))
                tf.config.threading.set_inter_op_parallelism_threads(2)
            gpus = tf.config.list_physical_devices("GPU")
            if device == "cpu" and gpus:
                tf.config.set_visible_devices([], "GPU")
            else:
                for gpu in gpus:
                    tf.config.experimental.set_memory_growth(gpu, True)
        except RuntimeError:
            pass  # TensorFlow was already initialised
        _TF_CONFIGURED = True
    return tf


# ============================================================
# COMMAND LINE ARGUMENTS
# ============================================================

def parse_args():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--results", default="results_test",
                        help="Name of the output folder: Results/<results>/")
    parser.add_argument("--trees", type=int, default=10000,
                        help="Step 3 tree set: 10000 (nested subset) or 100000 (full)")
    parser.add_argument("--validation", default="step3",
                        help="'step3' = Step 3's train/validation split, or a fraction such as 0.2")
    parser.add_argument("--datasets", default=",".join(DATASETS))
    parser.add_argument("--models", default=",".join(ALL_MODELS))

    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--patience", type=int, default=0,
                        help="Early stopping on validation loss (restores the best epoch). 0 = off.")
    parser.add_argument("--learning-rate", type=float, default=0.001)
    parser.add_argument("--cnn-batch-size", type=int, default=16)
    parser.add_argument("--gnn-batch-size", type=int, default=4)
    parser.add_argument("--hidden-dim", type=int, default=32)
    parser.add_argument("--num-layers", type=int, default=3)
    parser.add_argument("--heads", type=int, default=4)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--pooling", default="mean",
                        choices=("mean", "sum", "max", "mean_max", "mean_max_root", "mean_max_sum", "attention"),
                        help="GNN graph readout: mean (default); concatenations mean_max, mean_max_root (+ the root "
                             "node's embedding), mean_max_sum; or attention pooling")
    parser.add_argument("--gnn-targets", default="natural", choices=("natural", "rescaled"),
                        help="GNN targets: natural units (default), or rescaled like PhyloCNN (periods / r, "
                             "rates * r, r = the tree's mean branch length), converted back per tree")
    parser.add_argument("--graph-edges", default="tree", choices=("tree", "multiscale"),
                        help="Graph connectivity: the tree (default), or multiscale = also edges to the ancestors "
                             "2, 4, 8, 16, 32, 64 steps up")
    parser.add_argument("--jk", default="none", choices=("none", "cat"),
                        help="Jumping knowledge: 'cat' pools the concatenated embeddings of every layer")
    parser.add_argument("--residual", action="store_true",
                        help="Residual connections between GNN layers (use with deeper models, e.g. --num-layers 6)")
    parser.add_argument("--gnn-loss", default="huber", choices=("huber", "paper"),
                        help="GNN training loss: huber on standardised targets (default), or 'paper' = the "
                             "PhyloCNN paper's measure in natural units (mean relative error for BD/BDEI/BDSS, "
                             "MAE for BiSSE)")
    parser.add_argument("--node-features", default="standard", choices=("standard", "phylocnn", "both"),
                        help="GNN node features: standard (16), phylocnn = the authors' PhyloCNN encoder features "
                             "of every node on the real tree (21), both (35)")
    parser.add_argument("--seed", type=int, default=SEED, help="Random seed of the model training (default 42)")
    parser.add_argument("--cnn-protocol", default="authors", choices=("authors", "unified"),
                        help="PhyloCNN training: 'authors' = as in the authors' notebooks (rescaled targets, ELU "
                             "output, MAPE loss; MAE for BiSSE); 'unified' = the GNNs' protocol (standardised "
                             "targets, linear output, Huber loss)")

    parser.add_argument("--train-limit", type=int, default=0, help="0 = all training trees")
    parser.add_argument("--val-limit", type=int, default=0, help="0 = all validation trees")
    parser.add_argument("--test-limit", type=int, default=0, help="0 = all author-test trees")

    parser.add_argument("--task-id", type=int, default=None,
                        help="Run only this dataset x model combination (SLURM_ARRAY_TASK_ID)")
    parser.add_argument("--array-range", action="store_true", help="Print the sbatch --array range and exit")
    parser.add_argument("--aggregate", action="store_true", help="Only rebuild the summary files and ranking")
    parser.add_argument("--status", action="store_true",
                        help="Show settings and live progress (state, epoch, losses) of every run")
    parser.add_argument("--watch", type=int, default=0, metavar="SECONDS",
                        help="Like --status, refreshed every SECONDS (Ctrl+C to stop)")
    parser.add_argument("--force", action="store_true", help="Re-run runs that already passed")

    parser.add_argument("--workers", type=int, default=1, help="Processes for tree encoding")
    parser.add_argument("--cache-dir", default=str(DEFAULT_CACHE),
                        help="Encoded data are cached here (shared by all results); '' disables the cache")
    parser.add_argument("--device", default="auto",
                        help="auto (GPU if available, else CPU), cpu or cuda; applies to PyTorch and TensorFlow")
    parser.add_argument("--threads", type=int, default=0,
                        help="CPU threads for training (0 = the CPUs Slurm gave the job, else all)")
    parser.add_argument("--require-gpu", action="store_true",
                        help="Fail instead of silently training on the CPU when no GPU is visible")
    parser.add_argument("--strict", action="store_true",
                        help="Stop on the first invalid tree instead of skipping and reporting it")
    parser.add_argument("--fail-fast", action="store_true")

    args = parser.parse_args()
    args.datasets = tuple(x.strip() for x in args.datasets.split(",") if x.strip())
    args.models = tuple(x.strip() for x in args.models.split(",") if x.strip())
    for dataset in args.datasets:
        if dataset not in DATASETS:
            parser.error(f"Unknown dataset: {dataset}")
    for model in args.models:
        if model not in ALL_MODELS:
            parser.error(f"Unknown model: {model}")
    if args.epochs < 1:
        parser.error("Epochs must be >= 1")
    if str(args.validation).lower() != "step3":
        try:
            if not 0 < float(args.validation) < 1:
                raise ValueError
        except ValueError:
            parser.error("--validation must be 'step3' or a fraction between 0 and 1")
    args.validation = str(args.validation).lower()
    args.out = RESULTS_ROOT / args.results
    return args


def all_runs(args):
    """Every dataset x model combination, in array-task order."""
    return [(dataset, model) for dataset in args.datasets for model in args.models]


def run_signature(args, dataset, model):
    """Settings that define a run; a PASS.json with the same signature is reused."""
    keys = ("trees", "validation", "epochs", "patience", "learning_rate", "cnn_batch_size", "gnn_batch_size",
            "hidden_dim", "num_layers", "heads", "dropout", "train_limit", "val_limit", "test_limit", "strict")
    signature = {"dataset": dataset, "model": model, "seed": args.seed, "encoding_version": ge.ENCODING_VERSION,
                 **{k: getattr(args, k) for k in keys}}
    if model in GNN_MODELS:
        for key, default in GNN_OPTION_DEFAULTS.items():
            if getattr(args, key) != default:
                signature[key] = getattr(args, key)
    if model in CNN_MODELS:
        # PhyloCNN learns time-scaled targets in the authors' rescaled units (earlier runs used
        # natural units, which the scale-free PhyloCNN input cannot identify, and are re-run).
        signature["cnn_targets"] = CNN_TARGET_SCALING
        signature["cnn_protocol"] = args.cnn_protocol
    return signature


def run_dir(args, dataset, model):
    return args.out / "runs" / dataset / model


def already_passed(args, dataset, model):
    marker = run_dir(args, dataset, model) / "PASS.json"
    if args.force or not marker.is_file():
        return False
    try:
        return json.loads(marker.read_text()).get("signature") == run_signature(args, dataset, model)
    except (OSError, ValueError):
        return False


# ============================================================
# DATASET LOADING
# ============================================================

def resolve_device(args):
    import torch
    device = args.device
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    if (args.require_gpu or device == "cuda") and not torch.cuda.is_available():
        raise RuntimeError("PyTorch sees no CUDA GPU (--require-gpu / --device cuda). Check the GPU request and "
                           "that the environment has a CUDA build of torch.")
    torch.set_num_threads(cpu_threads(args))
    if device == "cuda":
        log(f"  PyTorch device: cuda ({torch.cuda.get_device_name(0)})")
    else:
        log(f"  PyTorch device: cpu ({torch.get_num_threads()} threads)")
    return device


def load_partition(dataset, partition, limit, need_cnn, args, need_resc=False):
    log(f"Loading {dataset} / {partition} (trees={args.trees}, validation={args.validation}, "
        f"limit={limit or 'ALL'})")
    start = time.time()
    gs = ge.encode_partition(
        ROOT, dataset, partition,
        trees=args.trees,
        validation=args.validation,
        limit=limit,
        need_cnn=need_cnn,
        need_resc=need_resc,
        node_features=args.node_features,
        edges=args.graph_edges,
        skip_invalid=not args.strict,
        workers=args.workers,
        cache_dir=Path(args.cache_dir) if args.cache_dir else None,
        log=log,
    )
    log(f"  {len(gs)} trees; skipped invalid={len(gs.invalid)} ({time.time() - start:.1f}s)")
    for bad in gs.invalid[:3]:
        log(f"    skipped tree {bad['id']}: {bad['error']}")
    return gs


def load_dataset(dataset, args, need_cnn):
    limits = {"train": args.train_limit, "validation": args.val_limit, "author_test": args.test_limit}
    need_resc = args.gnn_targets == "rescaled"     # GNNs with PhyloCNN-style targets need each tree's r
    data = {p: load_partition(dataset, p, limits[p], need_cnn, args, need_resc) for p in PARTITIONS}

    # Train and validation come from the same Step 3 tree set: their trees must not overlap.
    if set(np.asarray(data["train"].ids).tolist()) & set(np.asarray(data["validation"].ids).tolist()):
        raise RuntimeError("Train/validation IDs overlap")

    # Identical trees across partitions would leak information.
    hashes = {p: set(np.asarray(data[p].tree_hashes).tolist()) for p in PARTITIONS}
    for a, b in (("train", "validation"), ("train", "author_test"), ("validation", "author_test")):
        if hashes[a] & hashes[b]:
            raise RuntimeError(f"Duplicate {a}/{b} trees")

    # Fit label and graph-feature scaling ONLY on training data.
    scaler = ge.LabelScaler.fit(data["train"].y, TARGETS[dataset])
    gf = np.asarray(data["train"].graph_features, dtype=np.float64)
    gf_mean, gf_std = gf.mean(axis=0), gf.std(axis=0)
    gf_std = np.where(gf_std < 1e-6, 1.0, gf_std)
    for p in PARTITIONS:
        data[p] = data[p].with_graph_features((np.asarray(data[p].graph_features) - gf_mean) / gf_std)
    graph_feature_scaler = {"names": list(ge.GRAPH_FEATURES), "mean": gf_mean.tolist(), "std": gf_std.tolist(),
                            "fitted_on": "TRAIN_ONLY"}

    degree_histogram = ge.in_degree_histogram(data["train"])
    return data, scaler, degree_histogram, graph_feature_scaler


# ============================================================
# PHYLOCNN (authors' build_model, Keras)
# ============================================================

def build_cnn(dataset, architecture, output_dim, output_activation="elu"):
    tf = tensorflow()
    keras, layers = tf.keras, tf.keras.layers
    columns = ge.cnn_feature_columns(dataset, architecture)
    n_features = len(columns)
    model = keras.Sequential([
        layers.Input(shape=(500, n_features, 2)),
        layers.Conv2D(32, (1, n_features), groups=2, activation="elu", use_bias=False),
        layers.BatchNormalization(),
        layers.Conv2D(32, (1, 1), activation="elu", use_bias=False),
        layers.BatchNormalization(),
        layers.Conv2D(32, (1, 1), activation="elu", use_bias=False),
        layers.BatchNormalization(),
        layers.GlobalAveragePooling2D(),
        layers.Dense(64, activation="elu"),
        layers.Dense(32, activation="elu"),
        layers.Dense(16, activation="elu"),
        layers.Dense(8, activation="elu"),
        # ELU output as in the authors' notebooks (unscaled, positive targets); linear for the
        # "unified" protocol, whose targets are standardised.
        layers.Dense(output_dim, activation=output_activation),
    ], name=architecture)
    return model, columns


class Progress:
    """Live state of one run, kept in <run>/PROGRESS.json and shown by --status."""

    def __init__(self, out_dir, dataset, model, args):
        self.path = Path(out_dir) / "PROGRESS.json"
        job = os.environ.get("SLURM_ARRAY_JOB_ID")
        job = f"{job}_{os.environ.get('SLURM_ARRAY_TASK_ID')}" if job else os.environ.get("SLURM_JOB_ID")
        self.state = {
            "dataset": dataset, "model": model, "stage": "loading data",
            "epoch": 0, "max_epochs": args.epochs, "patience": args.patience,
            "best_epoch": None, "best_val_loss": None, "train_loss": None, "val_loss": None, "val_mae": None,
            "epoch_seconds": None, "mean_epoch_seconds": None, "parameters": None, "n": None,
            "started": time.strftime("%Y-%m-%d %H:%M:%S"), "started_ts": time.time(),
            "host": socket.gethostname(), "slurm_job": job, "pid": os.getpid(),
        }
        self._seconds = []
        self.update()

    def update(self, **fields):
        self.state.update(fields)
        self.state["updated"] = time.strftime("%Y-%m-%d %H:%M:%S")
        self.state["updated_ts"] = time.time()
        try:
            save_json(self.path, self.state)
        except OSError:
            pass  # progress reporting must never stop training

    def epoch(self, epoch, train_loss, val_loss, val_mae, best_epoch, best_val_loss, seconds):
        self._seconds.append(seconds)
        self.update(stage="training", epoch=epoch, train_loss=train_loss, val_loss=val_loss, val_mae=val_mae,
                    best_epoch=best_epoch, best_val_loss=best_val_loss, epoch_seconds=seconds,
                    mean_epoch_seconds=float(np.mean(self._seconds)))


def _checkpoint_meta(dataset, architecture, scaler, graph_feature_scaler, args, info):
    """Everything needed to reuse a trained model on new trees (written to checkpoint/checkpoint.json)."""
    return {
        "dataset": dataset, "architecture": architecture,
        "family": "phylocnn" if architecture in CNN_MODELS else "phylognn",
        "targets": list(scaler.targets),
        "label_scaler": scaler.json(),
        "graph_feature_scaler": graph_feature_scaler,
        "node_features": list(ge.node_feature_names(args.node_features)), "edge_features": list(ge.EDGE_FEATURES),
        "node_feature_set": args.node_features,
        "graph_features": list(ge.GRAPH_FEATURES),
        "encoding_version": ge.ENCODING_VERSION,
        "settings": run_signature(args, dataset, architecture),
        "best_epoch": info.get("best_epoch"), "epochs_run": info.get("epochs_run"),
        "evaluated_weights": info.get("evaluated_weights"),
        "saved": time.strftime("%Y-%m-%d %H:%M:%S"),
    }


def train_cnn(dataset, architecture, data, scaler, graph_feature_scaler, args, out_dir, progress):
    tf = tensorflow(args.device, cpu_threads(args))
    keras = tf.keras
    gpus = tf.config.get_visible_devices("GPU")
    log(f"  TensorFlow {tf.__version__} device: "
        f"{', '.join(g.name for g in gpus) if gpus else f'CPU ({cpu_threads(args)} threads)'}")
    if (args.require_gpu or args.device == "cuda") and not gpus:
        raise RuntimeError("TensorFlow sees no GPU (--require-gpu / --device cuda). TensorFlow needs CUDA and "
                           "cuDNN libraries matching its version.")
    if gpus:
        try:
            tf.config.experimental.reset_memory_stats("GPU:0")
        except (ValueError, AttributeError):
            pass
    set_seed(args.seed + DATASETS.index(dataset) * 409 + ALL_MODELS.index(architecture))
    targets = scaler.targets
    for split, gs in data.items():
        if gs.resc is None or len(gs.resc) != len(gs):
            raise RuntimeError(f"PhyloCNN rescaling factors missing for {split}; copy the matching Graph_Encoding.py")
    # Targets in the authors' rescaled units. "authors": used as they are (no standardisation), ELU
    # output, MAPE loss (BD/BDEI/BDSS) or MAE loss (BiSSE), as in the authors' notebooks.
    # "unified": standardised targets, linear output and Huber loss (the GNNs' protocol).
    if args.cnn_protocol == "authors":
        output_scaler = ge.LabelScaler(np.zeros(len(targets), np.float32), np.ones(len(targets), np.float32),
                                       tuple(targets))
        if dataset == "BiSSE":
            loss, metrics = keras.losses.MeanAbsoluteError(), [keras.metrics.MeanAbsolutePercentageError(name="mape")]
        else:
            loss, metrics = keras.losses.MeanAbsolutePercentageError(), [keras.metrics.MeanAbsoluteError(name="mae")]
        output_activation = "elu"
    else:
        output_scaler = ge.LabelScaler.fit(ge.to_rescaled(data["train"].y, data["train"].resc, targets), targets)
        loss, metrics, output_activation = keras.losses.Huber(), ["mae"], "linear"

    def fit_targets(gs):
        return output_scaler.transform(ge.to_rescaled(gs.y, gs.resc, targets))

    model, columns = build_cnn(dataset, architecture, len(scaler.targets), output_activation)
    model.compile(optimizer=keras.optimizers.Adam(args.learning_rate), loss=loss, metrics=metrics)

    def features(gs):
        return np.ascontiguousarray(np.asarray(gs.cnn)[:, :, columns, :])

    class TrackBest(keras.callbacks.Callback):
        """Live progress + a copy of the weights from the best validation epoch."""
        best_epoch, best_loss, best_weights, start, seconds = 0, None, None, 0.0, []

        def on_epoch_begin(self, epoch, logs=None):
            self.start = time.time()

        def on_epoch_end(self, epoch, logs=None):
            logs = logs or {}
            val = logs.get("val_loss")
            if val is not None and (self.best_loss is None or val < self.best_loss):
                self.best_loss, self.best_epoch = float(val), epoch + 1
                self.best_weights = self.model.get_weights()
            self.seconds.append(time.time() - self.start)
            progress.epoch(epoch + 1, logs.get("loss"), val, logs.get("val_mae"), self.best_epoch,
                           self.best_loss, self.seconds[-1])

    tracker = TrackBest()
    tracker.seconds = []
    progress.update(stage="training", parameters=int(model.count_params()))
    callbacks = [tracker]
    if args.patience > 0:
        # Stopping only; the best weights are restored below from TrackBest.
        callbacks.append(keras.callbacks.EarlyStopping(monitor="val_loss", patience=args.patience,
                                                       restore_best_weights=False))
    start = time.time()
    history = model.fit(
        features(data["train"]), fit_targets(data["train"]),
        validation_data=(features(data["validation"]), fit_targets(data["validation"])),
        epochs=args.epochs, batch_size=args.cnn_batch_size, shuffle=True, verbose=2,
        callbacks=callbacks,
    )
    train_seconds = time.time() - start
    epochs_run = len(next(iter(history.history.values())))
    progress.update(stage="evaluating")
    save_csv(out_dir / "history.csv", [
        {"epoch": epoch + 1, **{k: float(v[epoch]) for k, v in history.history.items()}}
        for epoch in range(epochs_run)
    ])

    # Checkpoints: the weights of the last epoch, and the evaluated (final) model.
    ckpt = out_dir / "checkpoint"
    ckpt.mkdir(parents=True, exist_ok=True)
    restore_best = args.patience > 0 and tracker.best_weights is not None and tracker.best_epoch != epochs_run
    if restore_best:
        model.save_weights(str(ckpt / "model_last_epoch.weights.h5"))
        model.set_weights(tracker.best_weights)
    model.save(str(ckpt / "model_final.keras"))

    timed = {}
    predictions = {}
    for split, gs in data.items():
        t0 = time.time()
        output = model.predict(features(gs), batch_size=args.cnn_batch_size, verbose=0)
        predictions[split] = ge.from_rescaled(output_scaler.inverse(output), gs.resc, targets).astype(np.float32)
        timed[split] = time.time() - t0

    peak = None
    if gpus:
        try:
            peak = tf.config.experimental.get_memory_info("GPU:0")["peak"] / 2 ** 20
        except (ValueError, AttributeError, KeyError):
            pass
    info = {
        "parameters": int(model.count_params()), "epochs_run": epochs_run,
        "best_epoch": tracker.best_epoch, "best_val_loss": tracker.best_loss,
        "early_stopped": epochs_run < args.epochs,
        "evaluated_weights": f"best validation epoch {tracker.best_epoch}" if (restore_best or (
            args.patience > 0 and tracker.best_epoch == epochs_run)) else f"last epoch {epochs_run}",
        "train_seconds": round(train_seconds, 2),
        "mean_epoch_seconds": round(float(np.mean(tracker.seconds)), 3) if tracker.seconds else None,
        "inference_ms_per_tree": round(1000 * timed["author_test"] / max(len(data["author_test"]), 1), 4),
        "peak_gpu_memory_mb": round(peak, 1) if peak is not None else None,
        "device": gpus[0].name if gpus else f"cpu ({cpu_threads(args)} threads)",
        "framework": f"tensorflow {tf.__version__}",
        "phylocnn_columns": columns,
        "checkpoint_files": sorted(p.name for p in ckpt.iterdir()),
    }
    save_json(ckpt / "checkpoint.json", {
        **_checkpoint_meta(dataset, architecture, scaler, graph_feature_scaler, args, info),
        "phylocnn_columns": columns,
        "cnn_protocol": args.cnn_protocol,
        "loss": type(loss).__name__ if not isinstance(loss, str) else loss,
        "output_activation": output_activation,
        "target_scaling": CNN_TARGET_SCALING,
        "time_scaled_targets": {t: ge.TIME_SCALED[t] for t in targets if t in ge.TIME_SCALED},
        "output_scaler": output_scaler.json(),
        "load": "tf.keras.models.load_model('checkpoint/model_final.keras'); "
                "inputs = PhyloCNN encoding[:, :, phylocnn_columns, :]; "
                "z = output * output_scaler.std + output_scaler.mean (authors' rescaled units); "
                "natural units with the tree's rescaling factor r: period = z * r, rate = z / r "
                "(Graph_Encoding.from_rescaled). label_scaler = natural-unit training scale (metrics)",
    })
    return predictions, info


# ============================================================
# PHYLOGNN (PyTorch Geometric)
# ============================================================

def gnn_build_config(architecture, output_dim, degree_histogram, args):
    """Keyword arguments of GNNArchitectures.build_graph_model (saved in the checkpoint)."""
    return {
        "architecture": architecture,
        "input_dim": len(ge.node_feature_names(args.node_features)),
        "output_dim": int(output_dim),
        "num_graph_features": len(ge.GRAPH_FEATURES),
        "edge_dim": len(ge.EDGE_FEATURES),
        "hidden_dim": args.hidden_dim,
        "num_layers": args.num_layers,
        "dropout": args.dropout,
        "pooling": args.pooling,
        "residual": bool(args.residual),
        "jk": args.jk,
        "norm": "batch",
        "heads": args.heads,
        "degree_histogram": [int(v) for v in np.asarray(degree_histogram).tolist()],
        "output_activation": "linear",
    }


def build_gnn(architecture, output_dim, degree_histogram, args):
    from GNNArchitectures import build_graph_model
    return build_graph_model(**gnn_build_config(architecture, output_dim, degree_histogram, args))


def predict_gnn(model, gs, scaler, batch_size, device, rescaled=False):
    """Predictions in original units, in the order of gs (rescaled: outputs are PhyloCNN-style
    rescaled targets, converted back with each tree's r)."""
    import torch
    model.eval()
    out = np.zeros_like(np.asarray(gs.y), dtype=np.float32)
    with torch.no_grad():
        for start in range(0, len(gs), batch_size):
            idx = np.arange(start, min(start + batch_size, len(gs)))
            pred = model(gs.pyg_batch(idx, device)).cpu().numpy()
            if pred.shape != (len(idx), out.shape[1]):
                raise RuntimeError(f"Unexpected prediction shape: {pred.shape}")
            out[idx] = scaler.inverse(pred)
    if rescaled:
        out = ge.from_rescaled(out, gs.resc, scaler.targets).astype(np.float32)
    return out


def train_gnn(dataset, architecture, data, scaler, graph_feature_scaler, degree_histogram, args, out_dir,
              progress):
    import torch
    import torch_geometric
    from GNNArchitectures import count_params

    seed = args.seed + DATASETS.index(dataset) * 409 + ALL_MODELS.index(architecture)
    set_seed(seed)
    device = torch.device(resolve_device(args))
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats()
    build_config = gnn_build_config(architecture, len(scaler.targets), degree_histogram, args)
    model = build_gnn(architecture, len(scaler.targets), degree_histogram, args).to(device)
    progress.update(stage="training", parameters=count_params(model)["trainable"])
    train_start = time.time()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
    huber = torch.nn.HuberLoss(delta=1.0)
    targets = scaler.targets
    rescaled = args.gnn_targets == "rescaled"
    if rescaled:
        for split, gs in data.items():
            if gs.resc is None or len(gs.resc) != len(gs):
                raise RuntimeError(f"rescaling factors missing for {split}; copy the matching Graph_Encoding.py")
        # PhyloCNN-style targets (periods / r, rates * r), standardised on the training trees.
        target_scaler = ge.LabelScaler.fit(ge.to_rescaled(data["train"].y, data["train"].resc, targets), targets)
    else:
        target_scaler = scaler

    def fit_space(y, gs):
        return target_scaler.transform(ge.to_rescaled(y, gs.resc, targets) if rescaled else y)

    # 'paper' loss: relative errors are unchanged by the rescaling (prediction and truth share r); for
    # BiSSE the MAE of the rescaled rates is exactly the authors' PhyloCNN loss.
    label_mean = torch.tensor(np.asarray(target_scaler.mean), dtype=torch.float32)
    label_std = torch.tensor(np.asarray(target_scaler.std), dtype=torch.float32)

    def loss_fn(pred, target):
        """Both arguments standardised. 'paper': the paper's measure in natural units (MRE; MAE for BiSSE)."""
        if args.gnn_loss == "huber":
            return huber(pred, target)
        mean, std = label_mean.to(pred.device), label_std.to(pred.device)
        natural_pred, natural_true = pred * std + mean, target * std + mean
        if dataset == "BiSSE":
            return (natural_pred - natural_true).abs().mean()
        return ((natural_pred - natural_true).abs() / natural_true.abs().clamp_min(1e-8)).mean()

    train, validation = data["train"], data["validation"]
    y_train = torch.from_numpy(fit_space(train.y, train)).to(device)
    y_val_scaled = fit_space(validation.y, validation)
    rng = np.random.default_rng(seed)

    history = []
    best_loss, best_epoch, best_state, wait = np.inf, 0, None, 0
    for epoch in range(1, args.epochs + 1):
        start = time.time()
        model.train()
        losses = []
        for chunk in np.array_split(rng.permutation(len(train)),
                                    max(1, int(np.ceil(len(train) / args.gnn_batch_size)))):
            pred = model(train.pyg_batch(chunk, device))
            loss = loss_fn(pred, y_train[torch.from_numpy(chunk).to(device)])
            if not torch.isfinite(loss):
                raise FloatingPointError(f"Non-finite training loss at epoch {epoch}")
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            losses.append(float(loss.detach().cpu()))

        val_pred = predict_gnn(model, validation, target_scaler, args.gnn_batch_size, device, rescaled)
        val_scaled = fit_space(val_pred, validation)
        val_loss = float(loss_fn(torch.from_numpy(val_scaled), torch.from_numpy(y_val_scaled)))
        val_mae = float(np.mean(np.abs(val_scaled - y_val_scaled)))
        if val_loss < best_loss:
            best_loss, best_epoch, wait = val_loss, epoch, 0
            best_state = copy.deepcopy(model.state_dict())
        else:
            wait += 1

        elapsed = time.time() - start
        history.append({"epoch": epoch, "train_huber": float(np.mean(losses)), "validation_huber": val_loss,
                        "validation_normalized_mae": val_mae, "best_epoch": best_epoch, "seconds": elapsed})
        log(f"  Epoch {epoch}/{args.epochs} train_loss={np.mean(losses):.5f} val_loss={val_loss:.5f} "
            f"val_MAE={val_mae:.5f} best={best_epoch} time={elapsed:.1f}s")
        progress.epoch(epoch, float(np.mean(losses)), val_loss, val_mae, best_epoch, float(best_loss), elapsed)
        if args.patience > 0 and wait >= args.patience:
            log(f"  Early stopping at epoch {epoch} (best epoch {best_epoch})")
            break

    train_seconds = time.time() - train_start
    epochs_run = len(history)
    progress.update(stage="evaluating")
    save_csv(out_dir / "history.csv", history)

    # Checkpoints: the weights of the last epoch, and the evaluated (final) model.
    ckpt = out_dir / "checkpoint"
    ckpt.mkdir(parents=True, exist_ok=True)
    restore_best = args.patience > 0 and best_state is not None and best_epoch != epochs_run
    if restore_best:
        torch.save({"state_dict": model.state_dict(), "epoch": epochs_run, "build_config": build_config},
                   ckpt / "model_last_epoch.pt")
        model.load_state_dict(best_state)
    evaluated = (f"best validation epoch {best_epoch}"
                 if args.patience > 0 and best_state is not None else f"last epoch {epochs_run}")
    info = {
        "parameters": count_params(model)["trainable"], "epochs_run": epochs_run,
        "best_epoch": best_epoch, "best_val_loss": float(best_loss) if np.isfinite(best_loss) else None,
        "early_stopped": epochs_run < args.epochs, "evaluated_weights": evaluated,
        "train_seconds": round(train_seconds, 2),
        "mean_epoch_seconds": round(float(np.mean([h["seconds"] for h in history])), 3),
        "device": torch.cuda.get_device_name(0) if device.type == "cuda" else "cpu",
        "framework": f"torch {torch.__version__}, torch_geometric {torch_geometric.__version__}",
    }
    meta = _checkpoint_meta(dataset, architecture, scaler, graph_feature_scaler, args, info)
    meta.update({"gnn_targets": args.gnn_targets, "output_scaler": target_scaler.json(),
                 "graph_edges": args.graph_edges})
    torch.save({"state_dict": model.state_dict(), "build_config": build_config, **meta},
               ckpt / "model_final.pt")

    if hasattr(model, "fusion_weights"):
        save_json(out_dir / "fusion_weights.json", {
            "architecture": architecture,
            "branches": list(getattr(model, "BRANCHES", ())),
            "weights": model.fusion_weights().tolist(),
            "targets": list(scaler.targets),
        })

    timed = {}
    predictions = {}
    for split, gs in data.items():
        t0 = time.time()
        predictions[split] = predict_gnn(model, gs, target_scaler, args.gnn_batch_size, device, rescaled)
        timed[split] = time.time() - t0
    info.update({
        "inference_ms_per_tree": round(1000 * timed["author_test"] / max(len(data["author_test"]), 1), 4),
        "peak_gpu_memory_mb": (round(torch.cuda.max_memory_allocated() / 2 ** 20, 1)
                               if device.type == "cuda" else None),
        "checkpoint_files": sorted(p.name for p in ckpt.iterdir()) + ["checkpoint.json"],
    })
    save_json(ckpt / "checkpoint.json", {
        **meta, "build_config": build_config,
        "load": "python Step4-CheckPhyloCNN.py load_trained_model(run_dir) -> (model, meta); "
                "graph features are standardised with graph_feature_scaler; "
                "outputs: z = output * output_scaler.std + output_scaler.mean; gnn_targets 'natural': y = z, "
                "'rescaled': periods = z * r, rates = z / r (Graph_Encoding.from_rescaled)",
    })
    return predictions, info


def load_trained_model(folder, device="cpu"):
    """Reload a finished run for new predictions: (model, checkpoint metadata).

    folder: Results/<results>/runs/<dataset>/<model>. GNNs come back as torch modules
    (in eval mode), PhyloCNN models as Keras models.
    """
    folder = Path(folder)
    meta = json.loads((folder / "checkpoint" / "checkpoint.json").read_text())
    if meta["family"] == "phylocnn":
        model = tensorflow().keras.models.load_model(str(folder / "checkpoint" / "model_final.keras"))
        return model, meta
    import torch
    from GNNArchitectures import build_graph_model
    saved = torch.load(folder / "checkpoint" / "model_final.pt", map_location=device)
    model = build_graph_model(**saved["build_config"]).to(device)
    model.load_state_dict(saved["state_dict"])
    model.eval()
    return model, meta


# ============================================================
# RESULTS
# ============================================================

RUN_INFO_COLUMNS = ("parameters", "epochs_run", "best_epoch", "best_val_loss", "early_stopped",
                    "evaluated_weights", "train_seconds", "mean_epoch_seconds", "inference_ms_per_tree",
                    "peak_gpu_memory_mb", "device")


def save_experiment_results(dataset, architecture, data, scaler, predictions, info, out_dir, args):
    summary_rows, target_rows = [], []
    metrics = {"dataset": dataset, "architecture": architecture, "results": args.results,
               "trees": args.trees, "validation": args.validation, "run": info, "splits": {}}
    for split in PARTITIONS:
        gs = data[split]
        actual, predicted = np.asarray(gs.y), predictions[split]
        if not np.isfinite(predicted).all():
            raise ValueError("Non-finite predictions")
        aggregate, per_target = ge.summarize_metrics(actual, predicted, scaler)
        metrics["splits"][split] = {"aggregate": aggregate, "per_target": per_target,
                                    "skipped_invalid_trees": len(gs.invalid)}
        common = {"results": args.results, "trees": args.trees, "validation": args.validation,
                  "dataset": dataset, "architecture": architecture, "split": split}
        run_columns = {k: info.get(k) for k in RUN_INFO_COLUMNS}
        summary_rows.append({**common, "status": "PASS", **run_columns, **aggregate})
        target_rows.extend({**common, **score} for score in per_target)

        prediction_rows = []
        ids = np.asarray(gs.ids)
        for i in range(len(actual)):
            row = {"tree_id": int(ids[i])}
            for j, target in enumerate(scaler.targets):
                row[f"true_{target}"] = float(actual[i, j])
                row[f"pred_{target}"] = float(predicted[i, j])
            prediction_rows.append(row)
        save_csv(out_dir / f"predictions_{split}.csv", prediction_rows)

        paper_name, paper_value = paper_metric(dataset, aggregate)
        log(f"  {split:12s} n={len(actual):6d} {paper_name}(paper)={paper_value:.5f} "
            f"MAE={aggregate['macro_mae_original_units']:.5f} "
            f"normalized_MAE={aggregate['macro_normalized_mae']:.5f} R2={aggregate.get('macro_r2', float('nan')):.4f}")

    save_csv(out_dir / "summary.csv", summary_rows)
    save_csv(out_dir / "per_target.csv", target_rows)
    save_json(out_dir / "metrics.json", metrics)


def aggregate(args, quiet=False):
    """Rebuild Results/<results>/all_results.csv, per_target.csv and failures.csv from every run."""
    rows, targets, failures = [], [], []
    for marker in sorted(args.out.glob("runs/*/*/PASS.json")):
        folder = marker.parent
        if (folder / "summary.csv").is_file():
            rows.extend(read_csv(folder / "summary.csv"))
        if (folder / "per_target.csv").is_file():
            targets.extend(read_csv(folder / "per_target.csv"))
    for marker in sorted(args.out.glob("runs/*/*/FAILED.json")):
        if not (marker.parent / "PASS.json").is_file():
            info = json.loads(marker.read_text())
            failures.append({k: info.get(k) for k in ("dataset", "architecture", "status", "error")})
    save_csv(args.out / "all_results.csv", rows)
    save_csv(args.out / "per_target.csv", targets)
    save_csv(args.out / "failures.csv", failures)
    if quiet:
        return rows, failures

    log("=" * 75)
    log(f"RESULTS: {args.out}")
    passed = {(r["dataset"], r["architecture"]) for r in rows if r["split"] == "author_test"}
    log(f"Finished runs: {len(passed)} | failed: {len(failures)}")
    for dataset in DATASETS:
        ranking = sorted((r for r in rows if r["dataset"] == dataset and r["split"] == "author_test"),
                         key=lambda r: float(r["macro_normalized_mae"]))
        if ranking:
            log(f"{dataset} author-test ranking (trees={ranking[0]['trees']}, "
                f"validation={ranking[0]['validation']}):")
            for r in ranking:
                paper_name, paper_value = paper_metric(dataset, r)
                log(f"  {r['architecture']:25s} normalized_MAE={float(r['macro_normalized_mae']):.5f}"
                    f" {paper_name}(paper)={paper_value:.5f} MAE={float(r['macro_mae_original_units']):.5f}")
    for f in failures:
        log(f"FAILED {f['dataset']}/{f['architecture']}: {str(f['error'])[:150]}")
    return rows, failures


# ============================================================
# MAIN
# ============================================================

# ============================================================
# STATUS
# ============================================================

def _hms(seconds):
    if seconds is None:
        return "-"
    seconds = int(seconds)
    return f"{seconds // 3600}:{seconds % 3600 // 60:02d}:{seconds % 60:02d}"


def _num(value, digits=4):
    try:
        return f"{float(value):.{digits}f}"
    except (TypeError, ValueError):
        return "-"


def _read_json(path):
    try:
        return json.loads(Path(path).read_text())
    except (OSError, ValueError):
        return None


def _slurm_queue():
    """{'<job>_<task>' or '<job>': state} for the current user, or None without Slurm."""
    import shutil
    import subprocess
    if not shutil.which("squeue"):
        return None
    try:
        out = subprocess.run(["squeue", "-h", "-r", "-u", os.environ.get("USER", ""), "-o", "%i %T"],
                             capture_output=True, text=True, timeout=30).stdout
    except (OSError, subprocess.SubprocessError):
        return None
    queue = {}
    for line in out.splitlines():
        parts = line.split()
        if len(parts) == 2:
            queue[parts[0]] = parts[1]
    return queue


def show_status(args):
    runs = all_runs(args)
    now = time.time()
    queue = _slurm_queue()

    print("=" * 128)
    print(f"STEP 4 STATUS — {args.out}   ({time.strftime('%Y-%m-%d %H:%M:%S')})")
    print("=" * 128)
    if not args.out.is_dir():
        print("No results folder yet. Submitted jobs create it when they start.")
        return

    config = _read_json(args.out / "RUN_CONFIG.json") or {}
    if config:
        print(f"Settings: trees={config.get('trees')}  validation={config.get('validation')}  "
              f"epochs<={config.get('epochs')}  patience={config.get('patience')}  lr={config.get('learning_rate')}  "
              f"batch CNN/GNN={config.get('cnn_batch_size')}/{config.get('gnn_batch_size')}")
        print(f"          GNN: pooling={config.get('pooling', 'mean')}  residual={config.get('residual', False)}  "
              f"loss={config.get('gnn_loss', 'huber')}  node features={config.get('node_features', 'standard')}  "
              f"targets={config.get('gnn_targets', 'natural')}  edges={config.get('graph_edges', 'tree')}  "
              f"jk={config.get('jk', 'none')}  "
              f"seed={config.get('seed', SEED)}  PhyloCNN protocol={config.get('cnn_protocol', 'authors')}")
        print(f"          hidden={config.get('hidden_dim')}  layers={config.get('num_layers')}  "
              f"heads={config.get('heads')}  dropout={config.get('dropout')}  "
              f"limits train/val/test={config.get('train_limit') or 'all'}/{config.get('val_limit') or 'all'}/"
              f"{config.get('test_limit') or 'all'}")

    # Array jobs that belong to this results folder (from the runs that already started).
    folder_jobs = set()
    for marker in args.out.glob("runs/*/*/PROGRESS.json"):
        job = (_read_json(marker) or {}).get("slurm_job") or ""
        if "_" in job:
            folder_jobs.add(job.split("_")[0])

    header = (f"{'TASK':>4} {'DATASET':7} {'MODEL':21} {'STATE':10} {'STAGE':12} {'EPOCH':>10} {'BEST':>5} "
              f"{'TRAIN_LOSS':>10} {'VAL_LOSS':>9} {'S/EPOCH':>8} {'RUNTIME':>9} {'PARAMS':>8} "
              f"{'TEST_nMAE':>9} {'TEST_PAPER':>11}  JOB")
    print()
    print(header)
    print("-" * len(header))

    counts, notes, finished = {}, [], []
    for task, (dataset, model) in enumerate(runs):
        folder = run_dir(args, dataset, model)
        progress = _read_json(folder / "PROGRESS.json") or {}
        passed = _read_json(folder / "PASS.json")
        failed = _read_json(folder / "FAILED.json")
        job = progress.get("slurm_job") or ""
        stage = progress.get("stage", "-")
        test_nmae, test_paper = "-", "-"

        if passed:
            state = "DONE"
            summary = folder / "summary.csv"
            if summary.is_file():
                for row in read_csv(summary):
                    if row.get("split") == "author_test":
                        test_nmae = _num(row.get("macro_normalized_mae"))
                        paper_name, paper_value = paper_metric(dataset, row)
                        test_paper = f"{paper_name} {_num(paper_value)}"
                        finished.append((dataset, model, folder))
        elif failed:
            state = "FAILED"
            notes.append(f"  task {task} {dataset}/{model}: {str(failed.get('error'))[:110]}")
        elif progress:
            age = now - progress.get("updated_ts", now)
            if queue is not None and job:
                state = "RUNNING" if job in queue else "STOPPED"
                if state == "STOPPED":
                    notes.append(f"  task {task} {dataset}/{model}: job {job} is no longer in the queue "
                                 f"(time limit or killed?) — last update {_hms(age)} ago")
            else:
                state = "RUNNING" if age < 1800 else "STALE?"
        else:
            pending = [j for j in folder_jobs if f"{j}_{task}" in (queue or {})]
            job = f"{pending[0]}_{task}" if pending else ""
            if not pending:
                state = "WAITING"
            else:
                state = "STARTING" if queue.get(job) == "RUNNING" else "QUEUED"

        counts[state] = counts.get(state, 0) + 1
        epoch = progress.get("epoch")
        epoch_text = f"{epoch}/{progress.get('max_epochs')}" if epoch else "-"
        started = progress.get("started_ts")
        end = progress.get("updated_ts") if state in ("DONE", "FAILED", "STOPPED", "STALE?") else now
        runtime = (end - started) if started and end else None
        print(f"{task:>4} {dataset:7} {model:21} {state:10} {str(stage)[:12]:12} {epoch_text:>10} "
              f"{str(progress.get('best_epoch') or '-'):>5} {_num(progress.get('train_loss')):>10} "
              f"{_num(progress.get('val_loss')):>9} {_num(progress.get('mean_epoch_seconds'), 1):>8} "
              f"{_hms(runtime):>9} {str(progress.get('parameters') or '-'):>8} {test_nmae:>9} {test_paper:>11}  {job}")

    print("-" * len(header))
    order = ("DONE", "RUNNING", "STARTING", "QUEUED", "WAITING", "FAILED", "STOPPED", "STALE?")
    print("  ".join(f"{k}={counts[k]}" for k in order if k in counts) + f"   (total {len(runs)})")
    if queue is None:
        print("Slurm queue not available here: RUNNING means updated in the last 30 minutes.")
    if notes:
        print("\nProblems:")
        print("\n".join(notes))
    if finished:
        print("\nAuthor-test accuracy per parameter as reported in the PhyloCNN paper "
              "(MRE = mean |predicted - true| / true for BD, BDEI, BDSS; MAE for BiSSE):")
        for dataset in dict.fromkeys(d for d, _, _ in finished):
            print(f"  {dataset}  ({'MAE' if dataset == 'BiSSE' else 'MRE'})")
            targets = TARGETS[dataset]
            print("    " + f"{'MODEL':21} " + " ".join(f"{t[:19]:>19}" for t in targets))
            for d, model, folder in finished:
                if d != dataset or not (folder / "per_target.csv").is_file():
                    continue
                values = {r.get("target"): r for r in read_csv(folder / "per_target.csv")
                          if r.get("split") == "author_test"}
                cells = []
                for t in targets:
                    row = values.get(t) or {}
                    if dataset == "BiSSE":
                        value = row.get("mae")
                    else:
                        value = float(row["mape_nonzero_percent"]) / 100 if row.get("mape_nonzero_percent") else None
                    cells.append(f"{_num(value):>19}")
                print("    " + f"{model:21} " + " ".join(cells))
    print("\nWAITING = not started yet (queued or not submitted).  STAGE: loading data -> training -> evaluating.")
    print("TEST_nMAE = mean |error| / training SD (this benchmark); TEST_PAPER = the paper's measure "
          "(MRE for BD/BDEI/BDSS, MAE for BiSSE), averaged over the parameters.")
    print(f"Ranking of finished runs:  python Step4-CheckPhyloCNN.py --results {args.results} --aggregate")


def main():
    args = parse_args()
    runs = all_runs(args)

    if args.array_range:
        print(f"0-{len(runs) - 1}")
        return 0
    if args.aggregate:
        aggregate(args)
        return 0
    if args.status or args.watch:
        if not args.watch:
            show_status(args)
            return 0
        try:
            while True:
                print("\033[2J\033[H", end="")
                show_status(args)
                print(f"\nRefreshing every {args.watch}s — Ctrl+C to stop.")
                time.sleep(args.watch)
        except KeyboardInterrupt:
            return 0

    if args.task_id is not None:
        if not 0 <= args.task_id < len(runs):
            log(f"[NOTHING TO DO] task {args.task_id}; there are {len(runs)} runs (array 0-{len(runs) - 1})")
            return 0
        runs = [runs[args.task_id]]
    if any(model in GNN_MODELS for _, model in runs):
        check_architectures_file()

    args.out.mkdir(parents=True, exist_ok=True)
    config = {**{k: (str(v) if isinstance(v, Path) else v) for k, v in vars(args).items() if k != "task_id"},
              "python": sys.executable}
    config_file = args.out / "RUN_CONFIG.json"
    try:
        if not config_file.is_file():
            save_json(config_file, config)
        elif {k: v for k, v in json.loads(config_file.read_text()).items() if k != "python"} != \
                {k: v for k, v in config.items() if k != "python"}:
            log(f"[WARN] {config_file} was written with different settings; "
                "each run's own settings are in its PASS.json")
    except (OSError, ValueError) as exc:
        log(f"[WARN] could not write {config_file}: {exc}")

    log("=" * 75)
    log("STEP 4 — PHYLOCNN + PHYLOGNN BENCHMARK")
    log("=" * 75)
    log(f"Results: {args.out} | trees={args.trees} | validation={args.validation}")
    log(f"Runs this job: {len(runs)} | epochs={args.epochs} patience={args.patience or 'off'}")
    log(f"Python: {sys.executable}")

    failed_here = 0
    for dataset in dict.fromkeys(d for d, _ in runs):
        models = [m for d, m in runs if d == dataset]
        todo = [m for m in models if not already_passed(args, dataset, m)]
        for m in models:
            if m not in todo:
                log(f"[SKIP] {dataset}/{m} already passed with the same settings (--force to redo)")
        if not todo:
            continue

        log("\n" + "=" * 75)
        log("DATASET:", dataset)
        log("=" * 75)
        progress = {}
        for architecture in todo:
            out_dir = run_dir(args, dataset, architecture)
            out_dir.mkdir(parents=True, exist_ok=True)
            for stale in ("PASS.json", "FAILED.json"):
                (out_dir / stale).unlink(missing_ok=True)
            progress[architecture] = Progress(out_dir, dataset, architecture, args)
        try:
            data, scaler, degree_histogram, graph_feature_scaler = load_dataset(
                dataset, args, need_cnn=any(m in CNN_MODELS for m in todo))
        except Exception as exc:
            log("DATA ERROR:", dataset, str(exc))
            for architecture in todo:
                progress[architecture].update(stage="failed", error=str(exc)[:300])
                save_json(run_dir(args, dataset, architecture) / "FAILED.json", {
                    "dataset": dataset, "architecture": architecture, "status": "DATA_ERROR",
                    "error": str(exc), "traceback": traceback.format_exc()})
            failed_here += len(todo)
            refresh_summary(args)
            if args.fail_fast:
                raise
            continue

        log("Loaded:", "train=", len(data["train"]), "validation=", len(data["validation"]),
            "author_test=", len(data["author_test"]), "| targets:", TARGETS[dataset])

        for architecture in todo:
            out_dir = run_dir(args, dataset, architecture)
            prog = progress[architecture]
            prog.update(stage="starting", n={p: len(data[p]) for p in PARTITIONS})
            log(f"\n[RUN] {dataset} / {architecture}")
            clear_memory()
            start = time.time()
            try:
                if architecture in CNN_MODELS:
                    predictions, info = train_cnn(
                        dataset, architecture, data, scaler, graph_feature_scaler, args, out_dir, prog)
                else:
                    predictions, info = train_gnn(
                        dataset, architecture, data, scaler, graph_feature_scaler, degree_histogram, args,
                        out_dir, prog)
                save_experiment_results(dataset, architecture, data, scaler, predictions, info, out_dir, args)
                save_json(out_dir / "PASS.json", {
                    "signature": run_signature(args, dataset, architecture),
                    **info, "seconds": time.time() - start,
                    "label_scaler": scaler.json(),
                    "graph_feature_scaler": graph_feature_scaler,
                    "n": {p: len(data[p]) for p in PARTITIONS},
                    "skipped_invalid": {p: len(data[p].invalid) for p in PARTITIONS},
                })
                prog.update(stage="done", epochs_run=info["epochs_run"], parameters=info["parameters"])
                log(f"PASS: {dataset}/{architecture} ({time.time() - start:.1f}s)")
            except Exception as exc:
                failed_here += 1
                prog.update(stage="failed", error=str(exc)[:300])
                log("FAILED:", dataset, architecture, str(exc))
                save_json(out_dir / "FAILED.json", {
                    "dataset": dataset, "architecture": architecture, "status": "FAILED",
                    "error": str(exc), "traceback": traceback.format_exc()})
                if args.fail_fast:
                    raise
            finally:
                clear_memory()
                refresh_summary(args)

        del data
        gc.collect()

    if args.task_id is not None:
        refresh_summary(args)
    else:
        aggregate(args)
    return 1 if failed_here else 0


if __name__ == "__main__":
    raise SystemExit(main())
