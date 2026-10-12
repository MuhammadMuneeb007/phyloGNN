#!/usr/bin/env python3
"""
Step4-CheckPhyloCNN.py — PhyloCNN vs PhyloGNN (PyTorch Geometric) benchmark

15 architectures x 4 datasets = 60 experiments.

Models:
  PhyloCNN (authors' architecture, Keras/TensorFlow, fresh weights):
    phylocnn_2gen, phylocnn_1gen, phylocnn_nocontext

  PhyloGNN (GNNArchitectures.py, PyTorch Geometric):
    gcn, graphsage, gat, gin, ecc                       message-passing baselines
    gine, pna, transformer, gps                         edge-aware PyG models
    trignn, gnn_fusion, graphsage_gat_fusion            fusion models

Datasets: BD, BDEI, BDSS, BiSSE

Data (Graph_Encoding.py):
  train / validation : Step 1 simulations in PhyloGNN/Training_Data/<NAME>/
                       (python Step4_Experiment2_FullDataset.py --assemble), split 70/30
                       with the same seeded rule as Step 4
  author_test        : PhyloGNN/Test_Sets/<NAME>/ — the authors' test set, never used for fitting

Each experiment:
  1. trains a new model on the training partition (targets standardised on train only)
  2. early-stops on the validation partition (--patience; 0 = off) and restores the best epoch
  3. evaluates train, validation and author-test partitions
  4. saves predictions, metrics, history and weights

Dependencies:
  GNNArchitectures.py, Graph_Encoding.py, numpy, pandas, ete3,
  torch + torch_geometric (GNNs), tensorflow (PhyloCNN only; imported only when needed)

Quick test:
  python Step4-CheckPhyloCNN.py --epochs 1 --train-limit 32 --val-limit 12 --test-limit 8 --fail-fast

Full available data:
  python Step4-CheckPhyloCNN.py --train-limit 0 --val-limit 0 --test-limit 0 \
      --epochs 1000 --patience 100 --workers 8

Outputs:
  Results/Step4_AllModels/
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
import sys
import time
import traceback
from pathlib import Path

import numpy as np
import torch

import Graph_Encoding as ge
from GNNArchitectures import (
    OUR_ARCHITECTURES,
    build_graph_model,
    count_params,
)


ROOT = Path(__file__).resolve().parent
OUT = ROOT / "Results" / "Step4_AllModels"

CNN_MODELS = (
    "phylocnn_2gen",
    "phylocnn_1gen",
    "phylocnn_nocontext",
)
GNN_MODELS = tuple(OUR_ARCHITECTURES)
ALL_MODELS = CNN_MODELS + GNN_MODELS
DATASETS = ge.DATASETS
TARGETS = ge.TARGETS

SEED = 42
PARTITIONS = ("train", "validation", "author_test")


# ============================================================
# UTILITIES
# ============================================================

def log(*args):
    print(*args, flush=True)


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if "tensorflow" in sys.modules:
        sys.modules["tensorflow"].random.set_seed(seed)


def clear_memory():
    if "tensorflow" in sys.modules:
        sys.modules["tensorflow"].keras.backend.clear_session()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    gc.collect()


def save_json(path, obj):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(obj, f, indent=2, default=str)


def save_csv(path, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    columns = list(dict.fromkeys(key for row in rows for key in row.keys()))
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def tensorflow():
    """Import TensorFlow only for PhyloCNN runs; never let it reserve all GPU memory."""
    import tensorflow as tf
    for gpu in tf.config.list_physical_devices("GPU"):
        try:
            tf.config.experimental.set_memory_growth(gpu, True)
        except RuntimeError:
            pass
    return tf


# ============================================================
# COMMAND LINE ARGUMENTS
# ============================================================

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--patience", type=int, default=0,
                        help="Early stopping on validation loss (restores the best epoch). 0 = off.")
    parser.add_argument("--train-limit", type=int, default=200, help="0 = all")
    parser.add_argument("--val-limit", type=int, default=80, help="0 = all")
    parser.add_argument("--test-limit", type=int, default=32, help="0 = all")
    parser.add_argument("--cnn-batch-size", type=int, default=16)
    parser.add_argument("--gnn-batch-size", type=int, default=4)
    parser.add_argument("--hidden-dim", type=int, default=32)
    parser.add_argument("--num-layers", type=int, default=3)
    parser.add_argument("--heads", type=int, default=4)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--learning-rate", type=float, default=0.001)
    parser.add_argument("--datasets", default=",".join(DATASETS))
    parser.add_argument("--models", default=",".join(ALL_MODELS))
    parser.add_argument("--workers", type=int, default=1, help="Processes for tree encoding")
    parser.add_argument("--cache-dir", default=str(OUT / "encoding_cache"),
                        help="Encoded partitions are cached here; '' disables the cache")
    parser.add_argument("--device", default="auto", help="auto, cpu or cuda")
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
    if args.device == "auto":
        args.device = "cuda" if torch.cuda.is_available() else "cpu"
    return args


# ============================================================
# DATASET LOADING
# ============================================================

def load_partition(dataset, partition, limit, need_cnn, args):
    log(f"Loading {dataset} / {partition} (limit={limit or 'ALL'})")
    start = time.time()
    gs = ge.encode_partition(
        ROOT, dataset, partition,
        limit=limit,
        need_cnn=need_cnn,
        skip_invalid=not args.strict,
        workers=args.workers,
        cache_dir=Path(args.cache_dir) if args.cache_dir else None,
    )
    log(f"  {len(gs)} trees; skipped invalid={len(gs.invalid)} ({time.time() - start:.1f}s)")
    for bad in gs.invalid[:3]:
        log(f"    skipped tree {bad['id']}: {bad['error']}")
    return gs


def load_dataset(dataset, args):
    need_cnn = any(name in CNN_MODELS for name in args.models)
    limits = {"train": args.train_limit, "validation": args.val_limit, "author_test": args.test_limit}
    data = {p: load_partition(dataset, p, limits[p], need_cnn, args) for p in PARTITIONS}

    # Train and validation come from the same simulation file: their rows must not overlap.
    if set(data["train"].ids) & set(data["validation"].ids):
        raise RuntimeError("Train/validation IDs overlap")

    # Identical trees across partitions would leak information.
    hashes = {p: set(data[p].tree_hashes.tolist()) for p in PARTITIONS}
    for a, b in (("train", "validation"), ("train", "author_test"), ("validation", "author_test")):
        if hashes[a] & hashes[b]:
            raise RuntimeError(f"Duplicate {a}/{b} trees")

    # Fit label and graph-feature scaling ONLY on training data.
    scaler = ge.LabelScaler.fit(data["train"].y, TARGETS[dataset])
    gf = data["train"].graph_features.astype(np.float64)
    gf_mean, gf_std = gf.mean(axis=0), gf.std(axis=0)
    gf_std = np.where(gf_std < 1e-6, 1.0, gf_std)
    for p in PARTITIONS:
        data[p] = data[p].with_graph_features((data[p].graph_features - gf_mean) / gf_std)

    degree_histogram = ge.in_degree_histogram(data["train"])
    return data, scaler, degree_histogram


# ============================================================
# PHYLOCNN ARCHITECTURE (authors' build_model, Keras)
# ============================================================

def build_cnn(dataset, architecture, output_dim):
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
        # Linear output: targets are standardised (the authors' ELU output assumes positive targets).
        layers.Dense(output_dim, activation="linear"),
    ], name=architecture)
    return model, columns


# ============================================================
# TRAIN PHYLOCNN
# ============================================================

def train_cnn(dataset, architecture, data, scaler, args, run_dir):
    tf = tensorflow()
    keras = tf.keras
    set_seed(SEED + DATASETS.index(dataset) * 409 + ALL_MODELS.index(architecture))
    model, columns = build_cnn(dataset, architecture, len(scaler.targets))
    model.compile(optimizer=keras.optimizers.Adam(args.learning_rate),
                  loss=keras.losses.Huber(), metrics=["mae"])

    def features(gs):
        return np.ascontiguousarray(gs.cnn[:, :, columns, :])

    callbacks = []
    if args.patience > 0:
        callbacks.append(keras.callbacks.EarlyStopping(monitor="val_loss", patience=args.patience,
                                                       restore_best_weights=True))
    history = model.fit(
        features(data["train"]), scaler.transform(data["train"].y),
        validation_data=(features(data["validation"]), scaler.transform(data["validation"].y)),
        epochs=args.epochs, batch_size=args.cnn_batch_size, shuffle=True, verbose=2,
        callbacks=callbacks,
    )
    epochs_run = len(next(iter(history.history.values())))
    save_csv(run_dir / "history.csv", [
        {"epoch": epoch + 1, **{k: float(v[epoch]) for k, v in history.history.items()}}
        for epoch in range(epochs_run)
    ])

    predictions = {
        split: scaler.inverse(model.predict(features(gs), batch_size=args.cnn_batch_size, verbose=0))
        for split, gs in data.items()
    }
    model.save_weights(str(run_dir / "final.weights.h5"))
    return predictions, int(model.count_params()), epochs_run


# ============================================================
# BUILD / TRAIN PHYLOGNN (PyTorch Geometric)
# ============================================================

def build_gnn(architecture, output_dim, degree_histogram, args):
    return build_graph_model(
        architecture,
        input_dim=len(ge.NODE_FEATURES),
        output_dim=output_dim,
        num_graph_features=len(ge.GRAPH_FEATURES),
        edge_dim=len(ge.EDGE_FEATURES),
        hidden_dim=args.hidden_dim,
        num_layers=args.num_layers,
        dropout=args.dropout,
        pooling="mean",
        norm="batch",
        heads=args.heads,
        degree_histogram=degree_histogram,
        output_activation="linear",
    )


def predict_gnn(model, gs, scaler, batch_size, device):
    """Predictions in original units, in the order of gs."""
    model.eval()
    out = np.zeros_like(gs.y, dtype=np.float32)
    with torch.no_grad():
        for start in range(0, len(gs), batch_size):
            idx = np.arange(start, min(start + batch_size, len(gs)))
            pred = model(gs.pyg_batch(idx, device)).cpu().numpy()
            if pred.shape != (len(idx), out.shape[1]):
                raise RuntimeError(f"Unexpected prediction shape: {pred.shape}")
            out[idx] = scaler.inverse(pred)
    return out


def train_gnn(dataset, architecture, data, scaler, degree_histogram, args, run_dir):
    seed = SEED + DATASETS.index(dataset) * 409 + ALL_MODELS.index(architecture)
    set_seed(seed)
    device = torch.device(args.device)
    model = build_gnn(architecture, len(scaler.targets), degree_histogram, args).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
    loss_fn = torch.nn.HuberLoss(delta=1.0)

    train, validation = data["train"], data["validation"]
    y_train = torch.from_numpy(scaler.transform(train.y)).to(device)
    y_val_scaled = scaler.transform(validation.y)
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

        val_pred = predict_gnn(model, validation, scaler, args.gnn_batch_size, device)
        val_scaled = scaler.transform(val_pred)
        val_loss = float(loss_fn(torch.from_numpy(val_scaled), torch.from_numpy(y_val_scaled)))
        val_mae = float(np.mean(np.abs(val_scaled - y_val_scaled)))
        improved = val_loss < best_loss
        if improved:
            best_loss, best_epoch, wait = val_loss, epoch, 0
            best_state = copy.deepcopy(model.state_dict())
        else:
            wait += 1

        elapsed = time.time() - start
        history.append({"epoch": epoch, "train_huber": float(np.mean(losses)), "validation_huber": val_loss,
                        "validation_normalized_mae": val_mae, "best_epoch": best_epoch, "seconds": elapsed})
        log(f"  Epoch {epoch}/{args.epochs} train_loss={np.mean(losses):.5f} val_loss={val_loss:.5f} "
            f"val_MAE={val_mae:.5f} best={best_epoch} time={elapsed:.1f}s")
        if args.patience > 0 and wait >= args.patience:
            log(f"  Early stopping at epoch {epoch} (best epoch {best_epoch})")
            break

    if args.patience > 0 and best_state is not None:
        model.load_state_dict(best_state)

    save_csv(run_dir / "history.csv", history)
    torch.save(model.state_dict(), run_dir / "final_state_dict.pt")
    save_json(run_dir / "model_config.json", model.phylognn_meta)

    if hasattr(model, "fusion_weights"):
        save_json(run_dir / "fusion_weights.json", {
            "architecture": architecture,
            "branches": list(getattr(model, "BRANCHES", ())),
            "weights": model.fusion_weights().tolist(),
            "targets": list(scaler.targets),
        })

    predictions = {split: predict_gnn(model, gs, scaler, args.gnn_batch_size, device)
                   for split, gs in data.items()}
    return predictions, count_params(model)["trainable"], len(history)


# ============================================================
# SAVE EXPERIMENT RESULTS
# ============================================================

def save_experiment_results(dataset, architecture, data, scaler, predictions, parameters, run_dir, epochs):
    summary_rows, target_rows = [], []
    for split in PARTITIONS:
        gs = data[split]
        actual, predicted = gs.y, predictions[split]
        if not np.isfinite(predicted).all():
            raise ValueError("Non-finite predictions")
        aggregate, per_target = ge.summarize_metrics(actual, predicted, scaler)
        summary_rows.append({"dataset": dataset, "architecture": architecture, "split": split,
                             "status": "PASS", "epochs": epochs, "parameters": parameters, **aggregate})
        for score in per_target:
            target_rows.append({"dataset": dataset, "architecture": architecture, "split": split, **score})

        prediction_rows = []
        for i in range(len(actual)):
            row = {"tree_id": int(gs.ids[i])}
            for j, target in enumerate(scaler.targets):
                row[f"true_{target}"] = float(actual[i, j])
                row[f"pred_{target}"] = float(predicted[i, j])
            prediction_rows.append(row)
        save_csv(run_dir / f"predictions_{split}.csv", prediction_rows)

        log(f"  {split:12s} n={len(actual):5d} MAE={aggregate['macro_mae_original_units']:.5f} "
            f"normalized_MAE={aggregate['macro_normalized_mae']:.5f}")
    return summary_rows, target_rows


# ============================================================
# MAIN BENCHMARK
# ============================================================

def main():
    args = parse_args()
    set_seed(SEED)
    OUT.mkdir(parents=True, exist_ok=True)

    import torch_geometric

    log("=" * 75)
    log("STEP 4 — PHYLOCNN + PHYLOGNN (PyTorch Geometric) BENCHMARK")
    log("=" * 75)
    log("Datasets:", args.datasets)
    log("Architectures:", args.models)
    planned = len(args.datasets) * len(args.models)
    log("Planned configurations:", planned)
    log("Epochs:", args.epochs, "| patience:", args.patience or "off", "| device:", args.device)
    log("Python:", sys.executable)

    all_results, all_targets, failures = [], [], []
    run_number = 0

    save_json(OUT / "RUN_CONFIG.json", {
        **{k: v for k, v in vars(args).items()},
        "planned": planned, "seed": SEED,
        "torch": torch.__version__, "torch_geometric": torch_geometric.__version__,
        "node_features": ge.NODE_FEATURES, "edge_features": ge.EDGE_FEATURES,
        "graph_features": ge.GRAPH_FEATURES,
    })

    for dataset in args.datasets:
        log("\n" + "=" * 75)
        log("DATASET:", dataset)
        log("=" * 75)

        try:
            data, scaler, degree_histogram = load_dataset(dataset, args)
        except Exception as exc:
            log("DATA ERROR:", dataset, str(exc))
            for architecture in args.models:
                failures.append({"dataset": dataset, "architecture": architecture, "status": "DATA_ERROR",
                                 "error": str(exc), "traceback": traceback.format_exc()})
            save_csv(OUT / "failures.csv", failures)
            if args.fail_fast:
                raise
            run_number += len(args.models)
            continue

        log("Loaded:", "train=", len(data["train"]), "validation=", len(data["validation"]),
            "author_test=", len(data["author_test"]))
        log("Targets:", TARGETS[dataset])

        for architecture in args.models:
            run_number += 1
            log(f"\n[{run_number}/{planned}] {dataset} / {architecture}")
            run_dir = OUT / "runs" / dataset / architecture
            run_dir.mkdir(parents=True, exist_ok=True)
            clear_memory()
            start = time.time()

            try:
                if architecture in CNN_MODELS:
                    predictions, parameters, epochs_run = train_cnn(
                        dataset, architecture, data, scaler, args, run_dir)
                else:
                    predictions, parameters, epochs_run = train_gnn(
                        dataset, architecture, data, scaler, degree_histogram, args, run_dir)

                summary, targets = save_experiment_results(
                    dataset, architecture, data, scaler, predictions, parameters, run_dir, epochs_run)
                all_results.extend(summary)
                all_targets.extend(targets)
                save_csv(OUT / "all_results.csv", all_results)
                save_csv(OUT / "per_target.csv", all_targets)
                save_json(run_dir / "PASS.json", {"dataset": dataset, "architecture": architecture,
                                                  "epochs": epochs_run, "parameters": parameters,
                                                  "seconds": time.time() - start,
                                                  "label_scaler": scaler.json()})
                (run_dir / "FAILED.json").unlink(missing_ok=True)
                log(f"PASS: {dataset}/{architecture} ({time.time() - start:.1f}s)")

            except Exception as exc:
                log("FAILED:", dataset, architecture, str(exc))
                failure = {"dataset": dataset, "architecture": architecture, "status": "FAILED",
                           "error": str(exc), "traceback": traceback.format_exc()}
                failures.append(failure)
                save_json(run_dir / "FAILED.json", failure)
                save_csv(OUT / "failures.csv", failures)
                if args.fail_fast:
                    raise
            finally:
                clear_memory()

        del data
        gc.collect()

    # --------------------------------------------------------
    # FINAL SUMMARY
    # --------------------------------------------------------

    log("\n" + "=" * 75)
    log("FINAL SUMMARY")
    log("=" * 75)
    completed = {(row["dataset"], row["architecture"]) for row in all_results
                 if row["status"] == "PASS" and row["split"] == "author_test"}
    log(f"Completed: {len(completed)}/{planned}")
    log("Results:", OUT / "all_results.csv")
    log("Per-target:", OUT / "per_target.csv")
    if failures:
        log("Failures:", OUT / "failures.csv")

    for dataset in args.datasets:
        rows = sorted((row for row in all_results if row["dataset"] == dataset and row["split"] == "author_test"),
                      key=lambda row: row["macro_normalized_mae"])
        if rows:
            log(f"\n{dataset} author-test ranking:")
            for row in rows:
                log(f"  {row['architecture']:25s} normalized_MAE={row['macro_normalized_mae']:.5f}"
                    f" MAE={row['macro_mae_original_units']:.5f}")

    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
