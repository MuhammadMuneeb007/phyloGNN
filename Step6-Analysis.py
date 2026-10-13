#!/usr/bin/env python3
"""
Step6-Analysis.py — statistics, significance tests and journal figures for the Step 4 benchmark

Reads every finished Step 4 run (Results/<results>/runs/<dataset>/<model>/) and
  1. prints the statistics of each model: size, epochs, training time, speed, and the
     train / validation / author-test accuracy (normalised MAE, MAE, RMSE, R2, Pearson r,
     Spearman rho, MAPE) with bootstrap 95% confidence intervals
  2. tests, for each dataset, which model is statistically the best on the authors' test trees
  3. prints the metrics of every predicted parameter (R0, infectious period, ...) for each model,
     with the best model per parameter and the models that are not significantly worse
  4. draws journal figures (vector PDF with embedded fonts + 600 dpi PNG; matplotlib only)
  5. saves every table as CSV (and the main ones as LaTeX), and the printout as REPORT.txt
  6. adds the empirical results of Step 5 (EmpiricalValidation/<results>/): every model's estimates
     on the HIV and primate trees next to its simulated accuracy, the BiSSE likelihood ranking of the
     primate estimates, whether simulated accuracy goes with the fit to the real tree (Spearman), and
     a scorecard per model

Usage (from the project root, after Step 4 and Step 5; all datasets and models by default):
  python Step6-Analysis.py run                              # every results folder in Results/
  python Step6-Analysis.py run --results small1000          # one folder (or a comma list)
  python Step6-Analysis.py run --datasets BDSS,BiSSE        # only these datasets
  python Step6-Analysis.py --results small1000,full100k --formats pdf,png,svg
  python Step6-Analysis.py --no-figures                     # tables and tests only
On Bunya:  .envs/phylocnn/bin/python Step6-Analysis.py

Statistics (per dataset, on the test trees that every compared model predicted):
  per-tree error   mean over the parameters of |predicted - true| / SD(parameter in training set);
                   its mean over the trees is the normalised MAE (nMAE) that Step 4 ranks by
  Friedman test    do the models differ at all? (trees = blocks, models = treatments)
  Wilcoxon signed-rank test for every pair of models, Holm-corrected over all pairs
  effect size      matched-pairs rank-biserial correlation, win rate (share of trees with the lower error)
  95% CI           percentile bootstrap over trees (paired for differences between models)
  A model is "significantly best" when its Holm-adjusted p < alpha against every other model.
  Per parameter: the best model is compared with each other model (Wilcoxon, Holm over k-1 tests).
Runs that are missing (not finished or failed) are left blank in every table.

Outputs:
  Results/<results>/analysis/REPORT.txt
  Results/<results>/analysis/tables/   model_statistics.csv, per_parameter_metrics.csv, pairwise_tests.csv,
                                       significance_summary.csv, overview.csv, *.tex (needs booktabs)
  Results/<results>/analysis/figures/  fig1_model_comparison, fig2_overview_heatmap, fig3_critical_difference,
                                       fig4_per_parameter_nmae, fig5_per_parameter_r2, fig6_error_distributions,
                                       fig7_efficiency, fig8_training_curves, fig9_pred_vs_true_<dataset>,
                                       fig10_empirical_vs_simulated (with Step 5 results)
  Results/<results>/analysis/tables/   empirical_estimates.csv, empirical_likelihood.csv, empirical_scorecard.csv,
                                       table_empirical_likelihood.tex (with Step 5 results)
  Results/_analysis/                   comparison of results folders (when there are several)
"""

from __future__ import annotations

import argparse
import copy
import csv
import json
import math
import sys
import warnings
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent
RESULTS_ROOT = ROOT / "Results"

DATASETS = ("BD", "BDEI", "BDSS", "BiSSE")
FAMILIES = {
    "PhyloCNN": ("phylocnn_2gen", "phylocnn_1gen", "phylocnn_nocontext"),
    "Baseline GNN": ("gcn", "graphsage", "gat", "gin", "ecc"),
    "Edge-aware GNN": ("gine", "pna", "transformer", "gps"),
    "Fusion GNN": ("trignn", "gnn_fusion", "graphsage_gat_fusion"),
}
MODELS = tuple(m for members in FAMILIES.values() for m in members)
FAMILY_OF = {m: f for f, members in FAMILIES.items() for m in members}

LABELS = {
    "phylocnn_2gen": "PhyloCNN (2 gen.)", "phylocnn_1gen": "PhyloCNN (1 gen.)",
    "phylocnn_nocontext": "PhyloCNN (no context)",
    "gcn": "GCN", "graphsage": "GraphSAGE", "gat": "GAT", "gin": "GIN", "ecc": "ECC",
    "gine": "GINE", "pna": "PNA", "transformer": "Graph Transformer", "gps": "GPS",
    "trignn": "TriGNN", "gnn_fusion": "GNN Fusion", "graphsage_gat_fusion": "SAGE-GAT Fusion",
}
# One fixed colour per model, grouped by family (blues, oranges, greens, purples).
COLORS = {
    "phylocnn_2gen": "#08519c", "phylocnn_1gen": "#3182bd", "phylocnn_nocontext": "#9ecae1",
    "gcn": "#8c2d04", "graphsage": "#d94801", "gat": "#f16913", "gin": "#fd8d3c", "ecc": "#fdae6b",
    "gine": "#00441b", "pna": "#238b45", "transformer": "#41ab5d", "gps": "#a1d99b",
    "trignn": "#3f007d", "gnn_fusion": "#6a51a3", "graphsage_gat_fusion": "#9e9ac8",
}
MARKERS = {"PhyloCNN": "s", "Baseline GNN": "o", "Edge-aware GNN": "^", "Fusion GNN": "D", "Other": "x"}
TARGET_LABELS = {
    "r_nought": "$R_0$", "infectious_period": "Infectious period", "incubation_period": "Incubation period",
    "transmission_factor": "Transmission factor", "fraction": "Superspreader fraction",
    "turnover": "Turnover", "lambda_1": "$\\lambda_1$", "lambda_2": "$\\lambda_2$", "q_01": "$q_{01}$",
}
SPLIT_NAMES = {"train": "training", "validation": "validation", "author_test": "author test"}


# ============================================================
# SMALL HELPERS
# ============================================================

def label(model):
    return LABELS.get(model, model)


def family(model):
    return FAMILY_OF.get(model, "Other")


def color(model):
    return COLORS.get(model, "#666666")


def model_order(names):
    names = set(names)
    return [m for m in MODELS if m in names] + sorted(n for n in names if n not in MODELS)


def dataset_order(names):
    names = set(names)
    return [d for d in DATASETS if d in names] + sorted(n for n in names if n not in DATASETS)


def to_float(value):
    try:
        return float(value)
    except (TypeError, ValueError):
        return float("nan")


def finite(value):
    return value is not None and isinstance(value, (int, float, np.floating)) and math.isfinite(value)


def fnum(value, digits=4):
    """Number for a table; blank when missing."""
    value = to_float(value)
    if not math.isfinite(value):
        return ""
    if value != 0 and (abs(value) >= 1e5 or abs(value) < 10 ** -digits):
        return f"{value:.{max(digits - 2, 1)}e}"
    return f"{value:.{digits}f}"


def fint(value):
    value = to_float(value)
    return "" if not math.isfinite(value) else f"{int(round(value)):,}"


def fpval(p):
    p = to_float(p)
    if not math.isfinite(p):
        return ""
    return f"{p:.1e}" if p < 1e-4 else f"{p:.4f}"


def fhms(seconds):
    seconds = to_float(seconds)
    if not math.isfinite(seconds):
        return ""
    seconds = int(round(seconds))
    return f"{seconds // 3600}:{seconds % 3600 // 60:02d}:{seconds % 60:02d}"


def stars(p, alpha):
    p = to_float(p)
    if not math.isfinite(p):
        return ""
    if p < min(alpha, 0.001):
        return "***"
    if p < min(alpha, 0.01):
        return "**"
    if p < alpha:
        return "*"
    return "n.s."


def read_json(path):
    try:
        return json.loads(Path(path).read_text())
    except (OSError, ValueError):
        return None


def read_csv(path):
    try:
        with open(path, newline="") as f:
            return list(csv.DictReader(f))
    except OSError:
        return []


def write_csv(path, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    columns = list(dict.fromkeys(key for row in rows for key in row))
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=columns)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: ("" if (isinstance(v, float) and not math.isfinite(v)) else v)
                             for k, v in row.items()})


def column_min(values):
    """Smallest finite value of each column (NaN for an empty column), shaped (1, columns)."""
    values = np.where(np.isfinite(values), values, np.nan)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return np.nanmin(values, axis=0, keepdims=True)


def latex_escape(text):
    text = str(text)
    if text.startswith("$") and text.endswith("$"):
        return text
    for a, b in (("\\", "\\textbackslash{}"), ("&", "\\&"), ("%", "\\%"), ("_", "\\_"), ("#", "\\#")):
        text = text.replace(a, b)
    return text


def write_latex(path, headers, rows, caption, tag, bold=(), align=None):
    """booktabs table; `bold` holds (row, column) cells to print in bold."""
    bold = set(bold)
    align = align or ("l" + "r" * (len(headers) - 1))
    lines = [r"\begin{table}[ht]", r"\centering", r"\footnotesize",
             f"\\caption{{{latex_escape(caption)}}}", f"\\label{{{tag}}}",
             f"\\begin{{tabular}}{{{align}}}", r"\toprule",
             " & ".join(latex_escape(h) for h in headers) + r" \\", r"\midrule"]
    for r, row in enumerate(rows):
        cells = []
        for c, cell in enumerate(row):
            cell = latex_escape(cell)
            cells.append(f"\\textbf{{{cell}}}" if (r, c) in bold and cell else cell)
        lines.append(" & ".join(cells) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text("\n".join(lines) + "\n")


class Report:
    """Prints and keeps every line for REPORT.txt."""

    def __init__(self):
        self.lines = []

    def __call__(self, *parts):
        text = " ".join(str(p) for p in parts)
        print(text, flush=True)
        self.lines.append(text)

    def table(self, headers, rows, left=1, indent="  "):
        widths = [max([len(str(h))] + [len(str(r[i])) for r in rows]) for i, h in enumerate(headers)]

        def line(cells):
            return indent + "  ".join(str(c).ljust(w) if i < left else str(c).rjust(w)
                                      for i, (c, w) in enumerate(zip(cells, widths))).rstrip()

        self(line(headers))
        self(indent + "-" * (sum(widths) + 2 * (len(widths) - 1)))
        for row in rows:
            self(line(row))

    def heading(self, text, char="="):
        self("")
        self(char * 100)
        self(text)
        self(char * 100)

    def save(self, path):
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        Path(path).write_text("\n".join(self.lines) + "\n", encoding="utf-8")


# ============================================================
# READING STEP 4 RESULTS
# ============================================================

class Run:
    """One finished Step 4 run: Results/<results>/runs/<dataset>/<model>/."""

    def __init__(self, folder):
        self.folder = Path(folder)
        self.dataset = self.folder.parent.name
        self.model = self.folder.name
        self.passed = read_json(self.folder / "PASS.json") or {}
        self.summary = {row.get("split"): row for row in read_csv(self.folder / "summary.csv")}
        self.per_target = {}
        for row in read_csv(self.folder / "per_target.csv"):
            self.per_target.setdefault(row.get("split"), {})[row.get("target")] = row
        scaler = self.passed.get("label_scaler") or {}
        targets = scaler.get("targets") or list((self.per_target.get("author_test") or {}).keys())
        self.targets = tuple(targets)
        self.std = np.asarray(scaler.get("std") or [np.nan] * len(self.targets), dtype=np.float64)
        self._predictions = {}

    def metric(self, split, key):
        return to_float((self.summary.get(split) or {}).get(key))

    def target_metric(self, split, target, key):
        return to_float(((self.per_target.get(split) or {}).get(target) or {}).get(key))

    def info(self, key):
        return self.passed.get(key)

    def n(self, split):
        return (self.passed.get("n") or {}).get(split)

    def predictions(self, split):
        """(tree ids, true values, predictions) of one split, or None if the file is missing."""
        if split not in self._predictions:
            rows = read_csv(self.folder / f"predictions_{split}.csv")
            if not rows or not self.targets:
                self._predictions[split] = None
            else:
                try:
                    ids = np.array([int(float(r["tree_id"])) for r in rows])
                    true = np.array([[float(r["true_" + t]) for t in self.targets] for r in rows])
                    pred = np.array([[float(r["pred_" + t]) for t in self.targets] for r in rows])
                    self._predictions[split] = (ids, true, pred)
                except (KeyError, ValueError):
                    self._predictions[split] = None
        return self._predictions[split]

    def history(self):
        """(epochs, train loss, validation loss) from history.csv (Keras or PyTorch column names)."""
        rows = read_csv(self.folder / "history.csv")
        if not rows:
            return None
        val_key = "val_loss" if "val_loss" in rows[0] else "validation_huber"
        train_key = "loss" if "loss" in rows[0] else "train_huber"
        if val_key not in rows[0]:
            return None
        epochs = np.array([to_float(r.get("epoch")) for r in rows])
        return epochs, np.array([to_float(r.get(train_key)) for r in rows]), \
            np.array([to_float(r.get(val_key)) for r in rows])


def load_folder(folder):
    """{(dataset, model): Run} of finished runs, and {(dataset, model): error} of failed runs."""
    runs, failed = {}, {}
    for run_folder in sorted((folder / "runs").glob("*/*")):
        if not run_folder.is_dir():
            continue
        key = (run_folder.parent.name, run_folder.name)
        if (run_folder / "PASS.json").is_file() and (run_folder / "summary.csv").is_file():
            runs[key] = Run(run_folder)
        elif (run_folder / "FAILED.json").is_file():
            failed[key] = str((read_json(run_folder / "FAILED.json") or {}).get("error", ""))
    return runs, failed


# ============================================================
# STATISTICS
# ============================================================

def holm(pvalues):
    """Holm-Bonferroni adjusted p-values (same order as the input)."""
    p = np.asarray(pvalues, dtype=np.float64)
    adjusted = np.full(len(p), np.nan)
    valid = np.where(np.isfinite(p))[0]
    m = len(valid)
    running = 0.0
    for rank, idx in enumerate(valid[np.argsort(p[valid], kind="stable")]):
        running = max(running, min(1.0, (m - rank) * p[idx]))
        adjusted[idx] = running
    return adjusted


def wilcoxon(a, b):
    """Two-sided Wilcoxon signed-rank test of paired errors: (p, rank-biserial r).
    r > 0 means `a` has the larger errors."""
    from scipy import stats
    d = np.asarray(a, np.float64) - np.asarray(b, np.float64)
    nonzero = d[d != 0]
    if len(nonzero) == 0:
        return 1.0, 0.0
    ranks = stats.rankdata(np.abs(nonzero))
    effect = float((ranks[nonzero > 0].sum() - ranks[nonzero < 0].sum()) / ranks.sum())
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        try:
            p = float(stats.wilcoxon(a, b, zero_method="wilcox", alternative="two-sided").pvalue)
        except ValueError:
            p = float("nan")
    return p, effect


def bootstrap_means(errors, n_boot, seed):
    """(n_boot, k) bootstrap means of the columns of errors (N trees x k models), resampling trees."""
    n, k = errors.shape
    if n == 0 or n_boot <= 0:
        return np.full((max(n_boot, 1), k), np.nan)
    rng = np.random.default_rng(seed)
    out = np.empty((n_boot, k))
    step = max(1, int(4_000_000 // (n * k)))
    for start in range(0, n_boot, step):
        size = min(step, n_boot - start)
        out[start:start + size] = errors[rng.integers(0, n, size=(size, n))].mean(axis=1)
    return out


def analyse_dataset(runs, dataset, split, alpha, n_boot, seed, report):
    """Matched per-tree errors and all tests for one dataset (None if no model has predictions)."""
    from scipy import stats

    models = [m for m in model_order(m for d, m in runs if d == dataset)
              if runs[(dataset, m)].predictions(split) is not None]
    if not models:
        return None
    targets = runs[(dataset, models[0])].targets
    skipped = [m for m in models if runs[(dataset, m)].targets != targets]
    if skipped:
        report(f"  [WARN] {dataset}: different targets, left out of the tests: {', '.join(skipped)}")
    models = [m for m in models if m not in skipped]

    # One common scale per parameter (every run of a dataset uses the same training trees).
    stds = np.array([runs[(dataset, m)].std for m in models], dtype=np.float64)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        std = np.nanmedian(stds, axis=0)
        std = np.where(np.isfinite(std) & (std > 1e-12), std, 1.0)
        spread = np.nanmax(np.abs(stds - std) / std) if np.isfinite(stds).any() else 0.0
    if spread > 1e-3:
        report(f"  [WARN] {dataset}: the runs were trained on different training sets (different label scales)")

    common = None
    for m in models:
        ids = set(runs[(dataset, m)].predictions(split)[0].tolist())
        common = ids if common is None else common & ids
    ids = np.array(sorted(common), dtype=np.int64)
    n_each = {m: len(runs[(dataset, m)].predictions(split)[0]) for m in models}
    if len(ids) < max(n_each.values()):
        report(f"  [NOTE] {dataset}: tests use the {len(ids)} trees predicted by every model "
               f"(per model: {min(n_each.values())}-{max(n_each.values())})")
    if len(ids) == 0:
        report(f"  [WARN] {dataset}: the models share no test trees; no tests")
        return None

    errors, true_ref = {}, None
    for m in models:
        tree_ids, true, pred = runs[(dataset, m)].predictions(split)
        position = {int(t): i for i, t in enumerate(tree_ids)}
        rows = np.array([position[int(t)] for t in ids])
        errors[m] = np.abs(pred[rows] - true[rows]) / std
        if true_ref is None:
            true_ref = true[rows]
        elif not np.allclose(true_ref, true[rows], rtol=1e-4, atol=1e-6):
            report(f"  [WARN] {dataset}/{m}: true values differ from the other models for the same tree ids")

    k = len(models)
    tree_error = np.column_stack([errors[m].mean(axis=1) for m in models])   # N x k
    mean = tree_error.mean(axis=0)
    order = [int(i) for i in np.argsort(mean, kind="stable")]
    best = order[0]
    ranks = stats.rankdata(tree_error, axis=1).mean(axis=0) if k > 1 else np.ones(1)

    friedman = (float("nan"), float("nan"))
    if k >= 3 and len(ids) >= 2:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            try:
                result = stats.friedmanchisquare(*[tree_error[:, i] for i in range(k)])
                friedman = (float(result.statistic), float(result.pvalue))
            except ValueError:
                pass

    pairs = []
    for i in range(k):
        for j in range(i + 1, k):
            p, effect = wilcoxon(tree_error[:, i], tree_error[:, j])
            pairs.append({"i": i, "j": j, "p": p, "effect": effect,
                          "win_i": float(np.mean(tree_error[:, i] < tree_error[:, j])),
                          "win_j": float(np.mean(tree_error[:, j] < tree_error[:, i]))})
    adjusted = holm([pair["p"] for pair in pairs])
    p_holm = np.ones((k, k))
    p_raw = np.ones((k, k))
    win = np.full((k, k), np.nan)
    for pair, p_adj in zip(pairs, adjusted):
        pair["p_holm"] = p_adj
        i, j = pair["i"], pair["j"]
        p_holm[i, j] = p_holm[j, i] = p_adj if math.isfinite(p_adj) else 1.0
        p_raw[i, j] = p_raw[j, i] = pair["p"]
        win[i, j], win[j, i] = pair["win_i"], pair["win_j"]

    boots = bootstrap_means(tree_error, n_boot, seed)
    ci = np.nanpercentile(boots, [2.5, 97.5], axis=0)
    diff_ci = np.nanpercentile(boots - boots[:, [best]], [2.5, 97.5], axis=0)
    tied = [models[i] for i in order[1:] if p_holm[best, i] >= alpha]

    per_target = {}
    for t, target in enumerate(targets):
        column = np.column_stack([errors[m][:, t] for m in models])
        target_mean = column.mean(axis=0)
        target_best = int(np.argmin(target_mean))
        others = [i for i in range(k) if i != target_best]
        p_adj = holm([wilcoxon(column[:, i], column[:, target_best])[0] for i in others])
        status, padj = {models[target_best]: "BEST"}, {models[target_best]: float("nan")}
        for i, p in zip(others, p_adj):
            status[models[i]] = "n.s." if (not math.isfinite(p) or p >= alpha) else stars(p, alpha)
            padj[models[i]] = p
        per_target[target] = {"best": models[target_best], "status": status, "p_holm": padj,
                              "nmae": dict(zip(models, target_mean.tolist()))}

    return {"dataset": dataset, "models": models, "targets": targets, "ids": ids, "std": std,
            "errors": errors, "tree_error": tree_error, "mean": mean, "order": order, "best": best,
            "ranks": ranks, "friedman": friedman, "pairs": pairs, "p_holm": p_holm, "p_raw": p_raw,
            "win": win, "ci": ci, "diff_ci": diff_ci, "tied": tied, "per_target": per_target}


def family_duel(a, alpha):
    """Best PhyloCNN against the best PhyloGNN of one dataset (None if a family is missing)."""
    cnn = [i for i in a["order"] if family(a["models"][i]) == "PhyloCNN"]
    gnn = [i for i in a["order"] if family(a["models"][i]) != "PhyloCNN"]
    if not cnn or not gnn:
        return None
    c, g = cnn[0], gnn[0]
    p = a["p_holm"][c, g]
    winner = g if a["mean"][g] < a["mean"][c] else c
    return {"cnn": a["models"][c], "gnn": a["models"][g], "cnn_nmae": a["mean"][c], "gnn_nmae": a["mean"][g],
            "winner": a["models"][winner], "p_holm": p, "significant": bool(p < alpha),
            "gnn_wins_trees": a["win"][g, c]}


# ============================================================
# ONE RESULTS FOLDER: TABLES AND REPORT
# ============================================================

def analyse_folder(name, args):
    folder = RESULTS_ROOT / name
    out = Path(args.out) / name if args.out else folder / "analysis"
    tables, figures = out / "tables", out / "figures"
    report = Report()
    runs, failed = load_folder(folder)
    config = read_json(folder / "RUN_CONFIG.json") or {}

    datasets = dataset_order(set(config.get("datasets") or []) | {d for d, _ in runs} | {d for d, _ in failed})
    if args.datasets:
        datasets = [d for d in datasets if d in args.datasets]
        runs = {k: v for k, v in runs.items() if k[0] in args.datasets}
        failed = {k: v for k, v in failed.items() if k[0] in args.datasets}
    models = model_order(set(config.get("models") or []) | {m for _, m in runs} | {m for _, m in failed})
    split = args.split

    report.heading(f"STEP 6 ANALYSIS - {name}   ({folder})")
    if config:
        limits = "/".join(str(config.get(k) or "all") for k in ("train_limit", "val_limit", "test_limit"))
        report(f"Settings: trees={config.get('trees')}  validation={config.get('validation')}  "
               f"epochs<={config.get('epochs')}  patience={config.get('patience')}  "
               f"limits train/val/test={limits}")
    expected = len(datasets) * len(models)
    report(f"Finished runs: {len(runs)} of {expected}   failed: {len(failed)}   "
           f"not finished: {max(expected - len(runs) - len(failed), 0)}")
    report(f"Evaluated split: {split} ({SPLIT_NAMES.get(split, split)})   alpha={args.alpha}   "
           f"bootstrap={args.bootstrap}")
    report("Blank cells = no finished result for that model.")
    for (d, m), error in sorted(failed.items()):
        report(f"  FAILED {d}/{m}: {error[:140]}")
    if not runs:
        report("Nothing to analyse yet.")
        report.save(out / "REPORT.txt")
        return {"name": name, "runs": runs, "config": config}

    analyses = {}
    stats_rows, target_rows, pair_rows, summary_rows = [], [], [], []
    for dataset in datasets:
        present = [m for m in models if (dataset, m) in runs]
        report.heading(f"DATASET {dataset}   ({len(present)} of {len(models)} models finished)", "-")
        if not present:
            report("  No finished runs.")
            continue
        a = analyse_dataset(runs, dataset, split, args.alpha, args.bootstrap, args.seed, report)
        analyses[dataset] = a
        print_dataset(report, dataset, models, runs, a, split, args.alpha, stats_rows, target_rows, pair_rows)
        summary_rows.append(significance_row(name, dataset, runs, a, args.alpha))

    overview = print_overview(report, name, datasets, models, runs, analyses, split, args.alpha, tables)
    print_paper_overview(report, name, datasets, models, runs, split, tables)
    print_significance_summary(report, summary_rows, args.alpha)
    empirical = None
    if not args.no_empirical:
        empirical = analyse_empirical(report, name, models, runs, analyses, split, args.alpha, tables, figures, args)

    write_csv(tables / "model_statistics.csv", stats_rows)
    write_csv(tables / "per_parameter_metrics.csv", target_rows)
    write_csv(tables / "pairwise_tests.csv", pair_rows)
    write_csv(tables / "significance_summary.csv", summary_rows)
    write_csv(tables / "overview.csv", overview)
    write_per_parameter_latex(tables, name, datasets, models, runs, analyses, split)

    if not args.no_figures:
        try:
            make_figures(name, figures, datasets, models, runs, analyses, split, args)
            report("")
            report(f"Figures: {figures}")
        except ImportError as exc:
            report(f"[WARN] figures skipped ({exc}); install matplotlib")
    report(f"Tables:  {tables}")
    report(f"Report:  {out / 'REPORT.txt'}")
    report.save(out / "REPORT.txt")
    return {"name": name, "runs": runs, "config": config, "analyses": analyses, "empirical": empirical}


def print_dataset(report, dataset, models, runs, a, split, alpha, stats_rows, target_rows, pair_rows):
    present = [m for m in models if (dataset, m) in runs]
    tested = a["models"] if a else []
    # Display order: tested models by error, then finished models without predictions, then missing models.
    shown = ([tested[i] for i in a["order"]] if a else []) + \
        [m for m in present if m not in tested] + [m for m in models if m not in present]
    index = {m: i for i, m in enumerate(tested)}
    best = tested[a["best"]] if a else None

    # ---------- training statistics ----------
    report("")
    report("Model statistics (training)")
    rows = []
    for m in shown:
        run = runs.get((dataset, m))
        if run is None:
            rows.append([label(m), family(m)] + [""] * 10)
            continue
        rows.append([label(m), family(m), fint(run.info("parameters")), fint(run.info("epochs_run")),
                     fint(run.info("best_epoch")), "yes" if run.info("early_stopped") else "no",
                     fhms(run.info("train_seconds")), fnum(run.info("mean_epoch_seconds"), 2),
                     fnum(run.info("inference_ms_per_tree"), 3), str(run.info("device") or "")[:24],
                     fint(run.n("train")), fint(run.n(split))])
    report.table(["Model", "Family", "Params", "Epochs", "Best ep.", "Early stop", "Train time", "s/epoch",
                  "ms/tree", "Device", "n train", f"n {split}"], rows, left=2)

    # ---------- accuracy and significance ----------
    report("")
    report(f"Accuracy on the {SPLIT_NAMES.get(split, split)} trees (sorted by normalised MAE; "
           f"Sig = Holm-adjusted Wilcoxon against the best model)")
    rows = []
    for m in shown:
        run = runs.get((dataset, m))
        i = index.get(m)
        record = {"results": run.folder.parents[2].name if run else "", "dataset": dataset, "model": m,
                  "label": label(m), "family": family(m), "finished": run is not None}
        if run is None:
            rows.append([label(m)] + [""] * 16)
            stats_rows.append(record)
            continue
        ci = f"[{fnum(a['ci'][0, i])}, {fnum(a['ci'][1, i])}]" if i is not None else ""
        p = a["p_holm"][a["best"], i] if i is not None and m != best else float("nan")
        sig = "BEST" if m == best else (stars(p, alpha) if i is not None else "")
        win = a["win"][i, a["best"]] if i is not None and m != best else float("nan")
        rows.append([label(m),
                     fnum(run.metric("train", "macro_normalized_mae")),
                     fnum(run.metric("validation", "macro_normalized_mae")),
                     fnum(run.metric(split, "macro_normalized_mae")), ci,
                     fnum(paper_value(run, split, dataset)),
                     fnum(run.metric(split, "macro_mae_original_units")),
                     fnum(run.metric(split, "macro_rmse_original_units")),
                     fnum(run.metric(split, "macro_r2")), fnum(run.metric(split, "macro_pearson_r")),
                     fnum(run.metric(split, "macro_spearman_rho")),
                     fnum(run.metric(split, "macro_mape_percent"), 2),
                     fnum(a["ranks"][i], 2) if i is not None else "",
                     fnum(100 * win, 1) if finite(win) else "", fpval(p), sig, ""])
        record.update({
            "parameters": run.info("parameters"), "epochs_run": run.info("epochs_run"),
            "best_epoch": run.info("best_epoch"), "early_stopped": run.info("early_stopped"),
            "train_seconds": run.info("train_seconds"), "mean_epoch_seconds": run.info("mean_epoch_seconds"),
            "inference_ms_per_tree": run.info("inference_ms_per_tree"), "device": run.info("device"),
            "n_train": run.n("train"), "n_validation": run.n("validation"), f"n_{split}": run.n(split)})
        for s in ("train", "validation", split):
            for key in ("macro_normalized_mae", "macro_mae_original_units", "macro_rmse_original_units",
                        "macro_r2", "macro_pearson_r", "macro_spearman_rho", "macro_mape_percent",
                        "macro_median_ape_percent", "macro_abs_normalized_bias"):
                record[f"{s}_{key}"] = run.metric(s, key)
            record[f"{s}_paper_{paper_name(dataset).lower()}"] = paper_value(run, s, dataset)
        if i is not None:
            record.update({"tested_trees": len(a["ids"]), "tested_nmae": float(a["mean"][i]),
                           "nmae_ci_low": float(a["ci"][0, i]), "nmae_ci_high": float(a["ci"][1, i]),
                           "mean_rank": float(a["ranks"][i]),
                           "diff_to_best_ci_low": float(a["diff_ci"][0, i]),
                           "diff_to_best_ci_high": float(a["diff_ci"][1, i]),
                           "p_holm_vs_best": p, "win_rate_vs_best": win, "significance": sig})
        stats_rows.append(record)
    report.table(["Model", "Train nMAE", "Val nMAE", "Test nMAE", "95% CI", f"Test {paper_name(dataset)} (paper)",
                  "MAE", "RMSE", "R2", "Pearson r", "Spearman", "MAPE %", "Mean rank", "Win % vs best", "p (Holm)",
                  "Sig", ""], rows)
    report(f"  {paper_name(dataset)} (paper) = the PhyloCNN paper's measure: "
           + ("mean absolute error" if dataset == "BiSSE" else "mean relative error |predicted - true| / true")
           + ", averaged over the parameters.")

    if not a:
        report("  No prediction files: significance tests skipped.")
        return
    no_predictions = [m for m in present if m not in tested]
    if no_predictions:
        report(f"  [NOTE] no predictions_{split}.csv (not in the tests): {', '.join(label(m) for m in no_predictions)}")

    # ---------- verdict ----------
    report("")
    k = len(tested)
    n_trees = len(a["ids"])
    chi2, p_f = a["friedman"]
    if k >= 3:
        report(f"Friedman test ({k} models, {n_trees} trees): chi2={fnum(chi2, 2)}, p={fpval(p_f)} -> "
               + ("the models differ significantly." if p_f < alpha else "no significant difference between the models."))
    report(f"Best model on {dataset}: {label(best)} (nMAE {fnum(a['mean'][a['best']])}, 95% CI "
           f"[{fnum(a['ci'][0, a['best']])}, {fnum(a['ci'][1, a['best']])}])")
    if k == 1:
        report("  Only one model has predictions: nothing to compare.")
    elif not a["tied"]:
        report(f"  -> {label(best)} is SIGNIFICANTLY better than every other model "
               f"(Holm-adjusted Wilcoxon p < {alpha}).")
    else:
        report(f"  -> {label(best)} is NOT significantly better than: "
               f"{', '.join(label(m) for m in a['tied'])} (Holm-adjusted p >= {alpha}).")
        report(f"     It is significantly better than the other {k - 1 - len(a['tied'])} models.")
    duel = family_duel(a, alpha)
    if duel:
        report(f"  Best PhyloCNN {label(duel['cnn'])} ({fnum(duel['cnn_nmae'])}) vs best PhyloGNN "
               f"{label(duel['gnn'])} ({fnum(duel['gnn_nmae'])}): "
               + (f"{label(duel['winner'])} is significantly better" if duel["significant"] else "no significant difference")
               + f" (p Holm = {fpval(duel['p_holm'])}; the GNN has the lower error on "
               f"{fnum(100 * duel['gnn_wins_trees'], 1)}% of the trees)")

    for pair in a["pairs"]:
        mi, mj = tested[pair["i"]], tested[pair["j"]]
        pair_rows.append({"dataset": dataset, "model_a": mi, "model_b": mj,
                          "nmae_a": float(a["mean"][pair["i"]]), "nmae_b": float(a["mean"][pair["j"]]),
                          "p_wilcoxon": pair["p"], "p_holm": pair["p_holm"],
                          "significant": bool(pair["p_holm"] < alpha),
                          "rank_biserial_a_worse": pair["effect"], "win_rate_a": pair["win_i"],
                          "win_rate_b": pair["win_j"], "trees": n_trees})

    # ---------- per parameter ----------
    for target in a["targets"]:
        info = a["per_target"][target]
        report("")
        report(f"Parameter {target} ({dataset}): best = {label(info['best'])}; "
               f"Sig = Holm-adjusted Wilcoxon against the best model for this parameter")
        ordered = sorted(tested, key=lambda m: info["nmae"][m]) + [m for m in shown if m not in tested]
        rows = []
        for m in ordered:
            run = runs.get((dataset, m))
            if run is None:
                rows.append([label(m)] + [""] * 14)
                continue

            def g(key, digits=4):
                return fnum(run.target_metric(split, target, key), digits)

            rows.append([label(m), fnum(paper_target(run, split, dataset, target)),
                         g("mae"), g("normalized_mae"), g("rmse"), g("median_ae"), g("bias"),
                         g("relative_bias_percent", 2), g("mape_nonzero_percent", 2), g("median_ape_percent", 2),
                         g("r2"), g("pearson_r"), g("spearman_rho"), fpval(info["p_holm"].get(m)),
                         info["status"].get(m, "")])
            row = {"dataset": dataset, "target": target, "model": m, "label": label(m), "family": family(m),
                   "split": split}
            for key in ("n", "mae", "normalized_mae", "rmse", "normalized_rmse", "median_ae", "bias",
                        "normalized_bias", "relative_bias_percent", "mape_nonzero_percent", "median_ape_percent",
                        "r2", "pearson_r", "spearman_rho", "true_mean", "pred_mean"):
                row[key] = run.target_metric(split, target, key)
            row[f"paper_{paper_name(dataset).lower()}"] = paper_target(run, split, dataset, target)
            row.update({"tested_nmae": info["nmae"].get(m, float("nan")),
                        "p_holm_vs_best": info["p_holm"].get(m, float("nan")),
                        "significance": info["status"].get(m, "")})
            target_rows.append(row)
        report.table(["Model", f"{paper_name(dataset)} (paper)", "MAE", "nMAE", "RMSE", "Median AE", "Bias",
                      "Rel. bias %", "MAPE %",
                      "Median APE %", "R2", "Pearson r", "Spearman", "p (Holm)", "Sig"], rows)


def significance_row(name, dataset, runs, a, alpha):
    row = {"results": name, "dataset": dataset}
    if not a:
        return row
    best = a["models"][a["best"]]
    row.update({"tested_trees": len(a["ids"]), "models_tested": len(a["models"]),
                "friedman_chi2": a["friedman"][0], "friedman_p": a["friedman"][1],
                "best_model": best, "best_nmae": float(a["mean"][a["best"]]),
                "best_ci_low": float(a["ci"][0, a["best"]]), "best_ci_high": float(a["ci"][1, a["best"]]),
                "significantly_best": bool(len(a["models"]) > 1 and not a["tied"]),
                "not_significantly_worse": ";".join(a["tied"])})
    duel = family_duel(a, alpha)
    if duel:
        row.update({"best_phylocnn": duel["cnn"], "best_phylocnn_nmae": duel["cnn_nmae"],
                    "best_phylognn": duel["gnn"], "best_phylognn_nmae": duel["gnn_nmae"],
                    "cnn_vs_gnn_winner": duel["winner"], "cnn_vs_gnn_p_holm": duel["p_holm"],
                    "cnn_vs_gnn_significant": duel["significant"]})
    for target, info in a["per_target"].items():
        row[f"best_{target}"] = info["best"]
    return row


def paper_name(dataset):
    """The PhyloCNN paper's accuracy measure: mean relative error (BD, BDEI, BDSS) or MAE (BiSSE)."""
    return "MAE" if dataset == "BiSSE" else "MRE"


def paper_value(run, split, dataset):
    """Paper measure averaged over the parameters (MRE = macro MAPE / 100)."""
    if dataset == "BiSSE":
        return run.metric(split, "macro_mae_original_units")
    return run.metric(split, "macro_mape_percent") / 100.0


def paper_target(run, split, dataset, target):
    if dataset == "BiSSE":
        return run.target_metric(split, target, "mae")
    return run.target_metric(split, target, "mape_nonzero_percent") / 100.0


def print_paper_overview(report, name, datasets, models, runs, split, tables):
    """The models in the units of the PhyloCNN paper: MRE for BD/BDEI/BDSS, MAE for BiSSE (lower = better)."""
    report.heading(f"OVERVIEW IN THE PAPER'S MEASURE - {SPLIT_NAMES.get(split, split)} trees "
                   "(MRE for BD, BDEI, BDSS; MAE for BiSSE; rank in brackets; blank = no result)")
    ranks = {}
    for d in datasets:
        values = sorted((paper_value(runs[(d, m)], split, d), m) for m in models if (d, m) in runs)
        for r, (_, m) in enumerate([x for x in values if math.isfinite(x[0])], 1):
            ranks[(d, m)] = r
    rows, csv_rows = [], []
    for m in models:
        cells, record = [label(m), family(m)], {"model": m, "label": label(m), "family": family(m)}
        for d in datasets:
            run = runs.get((d, m))
            value = paper_value(run, split, d) if run else float("nan")
            cells.append(f"{fnum(value)} ({ranks[(d, m)]})" if (d, m) in ranks else "")
            record[f"{d}_{paper_name(d)}"] = value
        rows.append(cells)
        csv_rows.append(record)
    report.table(["Model", "Family"] + [f"{d} {paper_name(d)}" for d in datasets], rows, left=2)

    columns = [(d, t) for d in datasets
               for t in next((runs[(d, m)].targets for m in models if (d, m) in runs), ())]
    if not columns:
        return
    report("")
    report("Per parameter (as in the paper's tables)")
    rows, per_csv, latex_rows, bold = [], [], [], []
    best = {}
    for d, t in columns:
        values = [(paper_target(runs[(d, m)], split, d, t), m) for m in models if (d, m) in runs]
        values = [v for v in values if math.isfinite(v[0])]
        if values:
            best[(d, t)] = min(values)[1]
    for r, m in enumerate(models):
        cells, latex = [label(m)], [label(m)]
        for c, (d, t) in enumerate(columns, 1):
            run = runs.get((d, m))
            value = paper_target(run, split, d, t) if run else float("nan")
            cells.append(fnum(value))
            latex.append(fnum(value))
            if run:
                per_csv.append({"model": m, "label": label(m), "dataset": d, "parameter": t,
                                "measure": paper_name(d), "value": value})
            if best.get((d, t)) == m:
                bold.append((r, c))
        rows.append(cells)
        latex_rows.append(latex)
    report.table(["Model"] + [f"{d} {t} ({paper_name(d)})" for d, t in columns], rows)
    write_csv(tables / "paper_metric_overview.csv", csv_rows)
    write_csv(tables / "paper_metric_per_parameter.csv", per_csv)
    write_latex(tables / "table_paper_metric.tex",
                ["Model"] + [f"{d} {TARGET_LABELS.get(t, t)}" for d, t in columns], latex_rows,
                f"Accuracy on the {SPLIT_NAMES.get(split, split)} trees in the measures of the PhyloCNN paper "
                f"(MRE for BD, BDEI, BDSS; MAE for BiSSE; {name}); best per parameter in bold.",
                f"tab:{name}_paper_metric", bold=bold)


def print_overview(report, name, datasets, models, runs, analyses, split, alpha, tables):
    """Models x datasets nMAE, rank per dataset and mean rank (blank = no result)."""
    report.heading(f"OVERVIEW - normalised MAE on the {SPLIT_NAMES.get(split, split)} trees "
                   "(rank in brackets; blank = no result)")
    ranks = {}
    for d in datasets:
        values = sorted((runs[(d, m)].metric(split, "macro_normalized_mae"), m) for m in models if (d, m) in runs)
        values = [(v, m) for v, m in values if math.isfinite(v)]
        for r, (_, m) in enumerate(values, 1):
            ranks[(d, m)] = r
    rows, csv_rows, latex_rows, bold = [], [], [], []
    for m in models:
        cells, record, latex = [label(m), family(m)], {"model": m, "label": label(m), "family": family(m)}, [label(m)]
        own_ranks, best_in, top_in = [], 0, 0
        for d in datasets:
            run = runs.get((d, m))
            value = run.metric(split, "macro_normalized_mae") if run else float("nan")
            cells.append(f"{fnum(value)} ({ranks[(d, m)]})" if (d, m) in ranks else "")
            latex.append(fnum(value))
            record[d] = value
            if (d, m) in ranks:
                own_ranks.append(ranks[(d, m)])
                if ranks[(d, m)] == 1:
                    bold.append((len(latex_rows), len(latex) - 1))
            a = analyses.get(d)
            if a and m in a["models"]:
                if a["models"][a["best"]] == m:
                    best_in += 1
                    top_in += 1
                elif m in a["tied"]:
                    top_in += 1
        mean_rank = float(np.mean(own_ranks)) if own_ranks else float("nan")
        cells += [fnum(mean_rank, 2), str(len(own_ranks)) if own_ranks else "", str(best_in) if own_ranks else "",
                  str(top_in) if own_ranks else ""]
        latex.append(fnum(mean_rank, 2))
        record.update({"mean_rank": mean_rank, "datasets": len(own_ranks), "best_in": best_in,
                       "best_or_tied_in": top_in})
        rows.append(cells)
        csv_rows.append(record)
        latex_rows.append(latex)
    report.table(["Model", "Family"] + list(datasets) + ["Mean rank", "Datasets", "Best in", "Best/tied in"],
                 rows, left=2)
    report("Best/tied in = datasets where the model is best or not significantly worse than the best.")
    write_latex(tables / "table_overview_nmae.tex", ["Model"] + list(datasets) + ["Mean rank"], latex_rows,
                f"Normalised MAE on the {SPLIT_NAMES.get(split, split)} trees ({name}); best per dataset in bold.",
                f"tab:{name}_nmae", bold=bold)
    return csv_rows


def print_significance_summary(report, rows, alpha):
    report.heading(f"WHICH MODEL IS STATISTICALLY SIGNIFICANT? (alpha = {alpha}, Holm-adjusted Wilcoxon)")
    table = []
    for r in rows:
        if "best_model" not in r:
            table.append([r["dataset"], "", "", "", "", "", "", ""])
            continue
        tied = r["not_significantly_worse"].split(";") if r["not_significantly_worse"] else []
        duel = ""
        if "best_phylocnn" in r:
            duel = (f"{label(r['cnn_vs_gnn_winner'])} " +
                    ("(significant)" if r["cnn_vs_gnn_significant"] else "(n.s.)"))
        table.append([r["dataset"], str(r["tested_trees"]), fpval(r["friedman_p"]), label(r["best_model"]),
                      fnum(r["best_nmae"]),
                      "YES" if r["significantly_best"] else ("only model" if r["models_tested"] == 1 else "no"),
                      ", ".join(label(m) for m in tied) if tied else "-", duel])
    report.table(["Dataset", "Trees", "Friedman p", "Best model", "nMAE", "Significantly best?",
                  "Not significantly worse than the best", "Best PhyloCNN vs best PhyloGNN"], table)

    report("")
    report("Best model per parameter:")
    for r in rows:
        best = [(k[5:], v) for k, v in r.items() if k.startswith("best_") and k[5:] not in
                ("model", "nmae", "ci_low", "ci_high", "phylocnn", "phylocnn_nmae", "phylognn", "phylognn_nmae")]
        if best:
            report(f"  {r['dataset']:6s} " + "   ".join(f"{t}: {label(m)}" for t, m in best))


def write_per_parameter_latex(tables, name, datasets, models, runs, analyses, split):
    for metric, title in (("normalized_mae", "normalised MAE"), ("r2", "$R^2$")):
        columns = [(d, t) for d in datasets if analyses.get(d) for t in analyses[d]["targets"]]
        if not columns:
            continue
        headers = ["Model"] + [f"{d} {TARGET_LABELS.get(t, t)}" for d, t in columns]
        rows, bold = [], []
        for r, m in enumerate(models):
            row = [label(m)]
            for c, (d, t) in enumerate(columns, 1):
                run = runs.get((d, m))
                row.append(fnum(run.target_metric(split, t, metric)) if run else "")
            rows.append(row)
        for c, (d, t) in enumerate(columns, 1):
            values = [(to_float(rows[r][c]) if rows[r][c] else float("nan")) for r in range(len(models))]
            if any(math.isfinite(v) for v in values):
                pick = np.nanargmin(values) if metric != "r2" else np.nanargmax(values)
                bold.append((int(pick), c))
        write_latex(tables / f"table_per_parameter_{metric}.tex", headers, rows,
                    f"Per-parameter {title} on the {SPLIT_NAMES.get(split, split)} trees ({name}); "
                    "best per parameter in bold.", f"tab:{name}_{metric}", bold=bold)


# ============================================================
# FIGURES (journal style)
# ============================================================

def pyplot():
    import logging
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    logging.getLogger("matplotlib.font_manager").setLevel(logging.ERROR)
    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica", "Liberation Sans", "DejaVu Sans"],
        "font.size": 7, "axes.titlesize": 7.5, "axes.labelsize": 7, "xtick.labelsize": 6.5,
        "ytick.labelsize": 6.5, "legend.fontsize": 6.5, "axes.linewidth": 0.6,
        "xtick.major.width": 0.6, "ytick.major.width": 0.6, "xtick.minor.width": 0.4, "ytick.minor.width": 0.4,
        "xtick.major.size": 2.5, "ytick.major.size": 2.5, "xtick.minor.size": 1.5, "ytick.minor.size": 1.5,
        "axes.spines.top": False, "axes.spines.right": False, "legend.frameon": False,
        "lines.linewidth": 1.0, "mathtext.default": "regular",
        "pdf.fonttype": 42, "ps.fonttype": 42, "svg.fonttype": "none",   # editable text for journals
        "savefig.dpi": 600, "figure.dpi": 150, "savefig.bbox": "tight", "savefig.pad_inches": 0.02,
    })
    return plt


def colormap(name):
    import matplotlib
    cmap = matplotlib.colormaps[name] if hasattr(matplotlib, "colormaps") else matplotlib.cm.get_cmap(name)
    cmap = copy.copy(cmap)
    cmap.set_bad("white")
    return cmap


def save_figure(plt, fig, folder, name, formats):
    folder.mkdir(parents=True, exist_ok=True)
    for ext in formats:
        fig.savefig(folder / f"{name}.{ext}")
    plt.close(fig)


def panel_label(ax, letter, x=-0.02, y=1.02):
    ax.text(x, y, letter, transform=ax.transAxes, fontsize=9, fontweight="bold", ha="right", va="bottom")


def family_legend(fig, plt, families, y=0.0, ncol=4):
    from matplotlib.lines import Line2D
    handles = [Line2D([], [], marker=MARKERS[f], ls="", color="0.35", ms=4, label=f) for f in families]
    fig.legend(handles=handles, loc="lower center", ncol=ncol, bbox_to_anchor=(0.5, y))


def model_legend(fig, models, y=0.0, ncol=5, lines=True):
    from matplotlib.lines import Line2D
    handles = [Line2D([], [], color=color(m), marker=MARKERS[family(m)], ms=3.5, lw=1.0 if lines else 0,
                      mec="black", mew=0.3, label=label(m))
               for m in models]
    fig.legend(handles=handles, loc="lower center", ncol=ncol, bbox_to_anchor=(0.5, y), columnspacing=1.2,
               handlelength=1.8)


def grid(plt, n, width=7.2, row_height=2.4, ncols=2):
    ncols = min(ncols, n)
    nrows = int(math.ceil(n / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(width if ncols > 1 else width / 2 + 0.4, row_height * nrows),
                             squeeze=False)
    for ax in axes.flat[n:]:
        ax.set_visible(False)
    return fig, list(axes.flat[:n])


def heat_text_color(cmap, norm, value):
    r, g, b, _ = cmap(norm(value))
    return "black" if 0.299 * r + 0.587 * g + 0.114 * b > 0.5 else "white"


def make_figures(name, folder, datasets, models, runs, analyses, split, args):
    plt = pyplot()
    formats = args.formats
    ds = [d for d in datasets if analyses.get(d)]
    split_text = SPLIT_NAMES.get(split, split)
    letters = "abcdefghijklmnopqrstuvwxyz"

    # ---------- Fig 1: nMAE with 95% CI and significance against the best model ----------
    if ds:
        kmax = max(len(analyses[d]["models"]) for d in ds)
        fig, axes = grid(plt, len(ds), row_height=0.55 + 0.16 * kmax)
        for n, (ax, d) in enumerate(zip(axes, ds)):
            a = analyses[d]
            ys = np.arange(len(a["order"]))[::-1]
            for y, i in zip(ys, a["order"]):
                m = a["models"][i]
                mu, lo, hi = a["mean"][i], a["ci"][0, i], a["ci"][1, i]
                ax.errorbar(mu, y, xerr=[[max(mu - lo, 0)], [max(hi - mu, 0)]], fmt=MARKERS[family(m)],
                            color=color(m), mec="black", mew=0.3, ms=4, elinewidth=0.8, capsize=1.5)
                text = "best" if i == a["best"] else stars(a["p_holm"][a["best"], i], args.alpha)
                ax.text(1.01, y, text, transform=ax.get_yaxis_transform(), va="center", fontsize=6,
                        fontweight="bold" if i == a["best"] else "normal")
            ax.axvline(a["mean"][a["best"]], color="0.6", lw=0.6, ls=":", zorder=0)
            ax.set_yticks(ys)
            ax.set_yticklabels([label(a["models"][i]) for i in a["order"]])
            ax.get_yticklabels()[0].set_fontweight("bold")
            ax.set_ylim(-0.7, len(ys) - 0.3)
            ax.set_xlabel(f"Normalised MAE ({split_text}, 95% CI)")
            ax.set_title(f"{d}  (n = {len(a['ids']):,} trees)", loc="left")
            panel_label(ax, letters[n], x=-0.38 if n % 2 == 0 else -0.35)
        family_legend(fig, plt, [f for f in list(FAMILIES) if any(family(m) == f for d in ds
                                                                   for m in analyses[d]["models"])], y=-0.01)
        fig.tight_layout(rect=(0, 0.035, 1, 1))
        save_figure(plt, fig, folder, "fig1_model_comparison", formats)

    # ---------- Fig 2: overview heatmap (models x datasets) ----------
    matrix = np.array([[runs[(d, m)].metric(split, "macro_normalized_mae") if (d, m) in runs else np.nan
                        for d in datasets] for m in models], dtype=np.float64)
    if np.isfinite(matrix).any():
        with np.errstate(invalid="ignore", divide="ignore"):
            relative = matrix / column_min(matrix)
        fig, ax = plt.subplots(figsize=(1.6 + 0.75 * len(datasets), 0.55 + 0.2 * len(models)))
        cmap = colormap("viridis_r")
        vmax = float(np.nanpercentile(relative, 95)) if np.isfinite(relative).sum() > 1 else 2.0
        import matplotlib.colors as mcolors
        norm = mcolors.Normalize(vmin=1.0, vmax=max(vmax, 1.01))
        image = ax.imshow(np.ma.masked_invalid(relative), cmap=cmap, norm=norm, aspect="auto")
        for r in range(len(models)):
            for c in range(len(datasets)):
                if math.isfinite(matrix[r, c]):
                    ax.text(c, r, f"{matrix[r, c]:.3f}", ha="center", va="center", fontsize=5.8,
                            color=heat_text_color(cmap, norm, relative[r, c]),
                            fontweight="bold" if relative[r, c] <= 1.0 + 1e-12 else "normal")
        ax.set_xticks(range(len(datasets)))
        ax.set_xticklabels(datasets)
        ax.xaxis.tick_top()
        ax.set_yticks(range(len(models)))
        ax.set_yticklabels([label(m) for m in models])
        ax.tick_params(length=0)
        for spine in ax.spines.values():
            spine.set_visible(False)
        boundaries = np.cumsum([sum(1 for m in models if family(m) == f) for f in FAMILIES])[:-1]
        for b in boundaries:
            if 0 < b < len(models):
                ax.axhline(b - 0.5, color="white", lw=1.5)
        bar = fig.colorbar(image, ax=ax, fraction=0.05, pad=0.03)
        bar.set_label("nMAE / best nMAE of the dataset")
        bar.outline.set_linewidth(0.5)
        ax.set_title(f"Normalised MAE ({split_text}); bold = best", fontsize=7, pad=14)
        save_figure(plt, fig, folder, "fig2_overview_heatmap", formats)

    # ---------- Fig 3: critical difference diagrams ----------
    cd = [d for d in ds if len(analyses[d]["models"]) >= 2]
    if cd:
        heights = []
        layouts = {}
        for d in cd:
            a = analyses[d]
            order = list(np.argsort(a["ranks"], kind="stable"))
            k = len(order)
            sig = [[a["p_holm"][order[i], order[j]] < args.alpha for j in range(k)] for i in range(k)]
            cliques = []
            for i in range(k):
                j = i
                while j + 1 < k and all(not sig[x][j + 1] for x in range(i, j + 1)):
                    j += 1
                if j > i and not any(c0 <= i and j <= c1 for c0, c1 in cliques):
                    cliques.append((i, j))
            half = int(math.ceil(k / 2))
            height = 0.42 + 0.07 * len(cliques) + 0.12 + 0.13 * half
            layouts[d] = (order, cliques, half, height)
            heights.append(height)
        fig = plt.figure(figsize=(7.2, sum(heights)))
        spec = fig.add_gridspec(len(cd), 1, height_ratios=heights, hspace=0.0)
        for n, d in enumerate(cd):
            a = analyses[d]
            order, cliques, half, height = layouts[d]
            ax = fig.add_subplot(spec[n])
            draw_cd(ax, a, order, cliques, half, height)
            ax.text(0.0, -0.06, f"{letters[n]}   {d}", fontsize=8, fontweight="bold", va="top")
        save_figure(plt, fig, folder, "fig3_critical_difference", formats)

    # ---------- Fig 4 / 5: per-parameter heatmaps ----------
    if ds:
        per_parameter_heatmap(plt, folder, ds, models, runs, analyses, split, "normalized_mae", formats,
                              "fig4_per_parameter_nmae")
        per_parameter_heatmap(plt, folder, ds, models, runs, analyses, split, "r2", formats,
                              "fig5_per_parameter_r2")

    # ---------- Fig 6: per-tree error distributions ----------
    if ds:
        kmax = max(len(analyses[d]["models"]) for d in ds)
        fig, axes = grid(plt, len(ds), row_height=0.55 + 0.16 * kmax)
        for n, (ax, d) in enumerate(zip(axes, ds)):
            a = analyses[d]
            order = a["order"][::-1]
            data = [a["tree_error"][:, i] for i in order]
            box = ax.boxplot(data, vert=False, showfliers=False, patch_artist=True, widths=0.6, showmeans=True,
                             meanprops={"marker": "D", "markerfacecolor": "white", "markeredgecolor": "black",
                                        "markersize": 2.5, "markeredgewidth": 0.4},
                             medianprops={"color": "black", "lw": 0.8},
                             whiskerprops={"lw": 0.6}, capprops={"lw": 0.6}, boxprops={"lw": 0.6})
            for patch, i in zip(box["boxes"], order):
                patch.set_facecolor(color(a["models"][i]))
                patch.set_alpha(0.85)
            ax.set_yticks(range(1, len(order) + 1))
            ax.set_yticklabels([label(a["models"][i]) for i in order])
            ax.set_xlabel("Per-tree normalised absolute error")
            ax.set_title(f"{d}  (diamond = mean = nMAE)", loc="left")
            panel_label(ax, letters[n], x=-0.38 if n % 2 == 0 else -0.35)
        fig.tight_layout()
        save_figure(plt, fig, folder, "fig6_error_distributions", formats)

    # ---------- Fig 7: accuracy vs model size and training time ----------
    from matplotlib.ticker import FuncFormatter, NullFormatter
    present = [d for d in datasets if any((d, m) in runs for m in models)]
    if present:
        fig, axes = plt.subplots(2, len(present), figsize=(min(7.2, 1.9 * len(present) + 0.4), 3.6),
                                 squeeze=False)
        used = []
        for c, d in enumerate(present):
            for r, (key, xlabel, scale) in enumerate((("parameters", "Trainable parameters", 1.0),
                                                       ("train_seconds", "Training time (h)", 1 / 3600))):
                ax = axes[r, c]
                for m in models:
                    run = runs.get((d, m))
                    if run is None:
                        continue
                    x = to_float(run.info(key)) * scale
                    y = run.metric(split, "macro_normalized_mae")
                    if math.isfinite(x) and x > 0 and math.isfinite(y):
                        ax.scatter(x, y, s=14, marker=MARKERS[family(m)], color=color(m), edgecolors="black",
                                   linewidths=0.3, zorder=3)
                        if m not in used:
                            used.append(m)
                ax.set_xscale("log")
                ax.xaxis.set_minor_formatter(NullFormatter())
                ax.set_xlabel(xlabel)
                if c == 0:
                    ax.set_ylabel(f"nMAE ({split_text})")
                if r == 0:
                    ax.set_title(d, fontweight="bold")
                panel_label(ax, letters[r * len(present) + c], x=-0.08)
        model_legend(fig, model_order(used), y=-0.02, ncol=5, lines=False)
        rows_legend = int(math.ceil(len(used) / 5))
        fig.tight_layout(rect=(0, 0.035 * rows_legend + 0.01, 1, 1))
        save_figure(plt, fig, folder, "fig7_efficiency", formats)

    # ---------- Fig 8: validation-loss curves ----------
    with_history = [d for d in datasets if any((d, m) in runs and runs[(d, m)].history() for m in models)]
    if with_history:
        fig, axes = grid(plt, len(with_history), row_height=2.1)
        used = []
        for n, (ax, d) in enumerate(zip(axes, with_history)):
            for m in models:
                run = runs.get((d, m))
                history = run.history() if run else None
                if not history:
                    continue
                epochs, _, val = history
                ok = np.isfinite(val) & (val > 0)
                if not ok.any():
                    continue
                # Relative to epoch 1: PhyloCNN (authors' MAPE / MAE loss) and the GNNs (Huber) use
                # different losses, so only the shape of the curves is comparable.
                val = val / val[ok][0]
                ax.plot(epochs[ok], val[ok], color=color(m), lw=0.8, alpha=0.9)
                best = to_float(run.info("best_epoch"))
                if math.isfinite(best) and best >= 1 and int(best) <= len(val) and val[int(best) - 1] > 0:
                    ax.plot(best, val[int(best) - 1], marker=MARKERS[family(m)], color=color(m), ms=3,
                            mec="black", mew=0.3, ls="")
                if m not in used:
                    used.append(m)
            ax.set_yscale("log")
            low, high = ax.get_ylim()
            if low > 0 and high / low < 30:     # less than ~1.5 decades: label the minor ticks too
                ax.yaxis.set_minor_formatter(FuncFormatter(lambda v, _: f"{v:g}"))
                ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g}"))
            ax.set_xlabel("Epoch")
            ax.set_ylabel("Validation loss / epoch 1")
            ax.set_title(d, loc="left", fontweight="bold")
            panel_label(ax, letters[n], x=-0.12)
        model_legend(fig, model_order(used), y=-0.02, ncol=5)
        rows_legend = int(math.ceil(len(used) / 5))
        fig.tight_layout(rect=(0, 0.03 * rows_legend + 0.01, 1, 1))
        save_figure(plt, fig, folder, "fig8_training_curves", formats)

    # ---------- Fig 9: predicted vs true (best PhyloCNN and best PhyloGNN) ----------
    for d in ds:
        a = analyses[d]
        picks = []
        duel = family_duel(a, args.alpha)
        if duel:
            picks = [duel["cnn"], duel["gnn"]]
        else:
            picks = [a["models"][a["best"]]]
        targets = a["targets"]
        fig, axes = plt.subplots(len(picks), len(targets), squeeze=False,
                                 figsize=(min(7.2, 1.75 * len(targets) + 0.3), 1.75 * len(picks) + 0.15))
        for r, m in enumerate(picks):
            run = runs[(d, m)]
            _, true, pred = run.predictions(split)
            for c, t in enumerate(targets):
                ax = axes[r, c]
                x, y = true[:, c], pred[:, c]
                both = np.concatenate([x, y])
                lo, hi = np.percentile(both, [0.5, 99.5])
                pad = 0.04 * (hi - lo if hi > lo else 1.0)
                lo, hi = lo - pad, hi + pad
                if len(x) > 2000:
                    ax.hexbin(x, y, gridsize=45, extent=(lo, hi, lo, hi), cmap="Blues", bins="log", mincnt=1,
                              linewidths=0)
                else:
                    ax.scatter(x, y, s=3, color=color(m), alpha=0.5, linewidths=0)
                ax.plot([lo, hi], [lo, hi], color="black", lw=0.6, ls="--")
                ax.set_xlim(lo, hi)
                ax.set_ylim(lo, hi)
                ax.set_aspect("equal", adjustable="box")
                ax.text(0.04, 0.96, f"R² = {fnum(run.target_metric(split, t, 'r2'), 3)}\n"
                                    f"nMAE = {fnum(run.target_metric(split, t, 'normalized_mae'), 3)}",
                        transform=ax.transAxes, va="top", fontsize=6)
                if r == 0:
                    ax.set_title(TARGET_LABELS.get(t, t))
                if r == len(picks) - 1:
                    ax.set_xlabel("True")
                ax.set_ylabel(f"{label(m)}\nPredicted" if c == 0 else "Predicted")
                panel_label(ax, letters[r * len(targets) + c], x=-0.08)
        fig.tight_layout()
        save_figure(plt, fig, folder, f"fig9_pred_vs_true_{d}", formats)


def draw_cd(ax, a, order, cliques, half, height):
    """Critical difference diagram: average ranks over trees; bars join models that are NOT
    significantly different (Holm-adjusted Wilcoxon)."""
    k = len(order)
    ranks = [a["ranks"][i] for i in order]
    names = [a["models"][i] for i in order]
    lo, hi = 1, max(k, 2)

    def x(rank):
        return 0.24 + 0.52 * (rank - lo) / (hi - lo)

    ax.set_xlim(0, 1)
    ax.set_ylim(-height, 0)
    ax.axis("off")
    axis_y = -0.32
    ax.plot([x(lo), x(hi)], [axis_y, axis_y], color="black", lw=0.7)
    for t in range(lo, hi + 1):
        ax.plot([x(t), x(t)], [axis_y, axis_y + 0.04], color="black", lw=0.7)
        ax.text(x(t), axis_y + 0.06, str(t), ha="center", va="bottom", fontsize=6)
    ax.text(0.5, axis_y + 0.2, "Average rank over trees (1 = lowest error)", ha="center", va="bottom", fontsize=6.5)
    for c, (i, j) in enumerate(cliques):
        y = axis_y - 0.08 - 0.07 * c
        ax.plot([x(ranks[i]) - 0.006, x(ranks[j]) + 0.006], [y, y], color="black", lw=2.2,
                solid_capstyle="round", zorder=3)
    top = axis_y - 0.08 - 0.07 * len(cliques) - 0.06
    for idx in range(k):
        xr = x(ranks[idx])
        if idx < half:
            y, xt, ha = top - 0.13 * idx, 0.22, "right"
        else:
            y, xt, ha = top - 0.13 * (k - 1 - idx), 0.78, "left"
        ax.plot([xr, xr, xt], [axis_y, y, y], color=color(names[idx]), lw=0.7, zorder=2)
        ax.text(xt + (-0.008 if ha == "right" else 0.008), y, f"{label(names[idx])} ({ranks[idx]:.2f})",
                ha=ha, va="center", fontsize=6.5,
                fontweight="bold" if names[idx] == a["models"][a["best"]] else "normal")


def per_parameter_heatmap(plt, folder, ds, models, runs, analyses, split, metric, formats, name):
    import matplotlib.colors as mcolors
    rows = [m for m in models if any((d, m) in runs for d in ds)]
    if not rows:
        return
    widths = [len(analyses[d]["targets"]) for d in ds]
    fig = plt.figure(figsize=(min(7.2, 1.4 + 0.62 * sum(widths) + 0.25 * len(ds)), 0.75 + 0.19 * len(rows)))
    spec = fig.add_gridspec(1, len(ds) + 1, width_ratios=widths + [0.25], wspace=0.08)
    if metric == "r2":
        cmap, norm = colormap("viridis"), mcolors.Normalize(vmin=0.0, vmax=1.0)
        bar_label = "$R^2$"
    else:
        cmap = colormap("viridis_r")
        norm = None
        bar_label = "nMAE / best nMAE of the parameter"
    images = []
    all_relative = []
    panels = []
    for d in ds:
        targets = analyses[d]["targets"]
        values = np.array([[runs[(d, m)].target_metric(split, t, metric) if (d, m) in runs else np.nan
                            for t in targets] for m in rows], dtype=np.float64)
        if metric == "r2":
            shown = values
        else:
            with np.errstate(invalid="ignore", divide="ignore"):
                shown = values / column_min(values)
            all_relative.append(shown)
        panels.append((d, targets, values, shown))
    if norm is None:
        stacked = np.concatenate([s.ravel() for s in all_relative])
        vmax = float(np.nanpercentile(stacked, 95)) if np.isfinite(stacked).sum() > 1 else 2.0
        norm = mcolors.Normalize(vmin=1.0, vmax=max(vmax, 1.01))
    for n, (d, targets, values, shown) in enumerate(panels):
        ax = fig.add_subplot(spec[0, n])
        images.append(ax.imshow(np.ma.masked_invalid(shown), cmap=cmap, norm=norm, aspect="auto"))
        for c in range(len(targets)):
            column = values[:, c]
            if not np.isfinite(column).any():
                continue
            best = int(np.nanargmax(column)) if metric == "r2" else int(np.nanargmin(column))
            for r in range(len(rows)):
                if math.isfinite(values[r, c]):
                    ax.text(c, r, f"{values[r, c]:.2f}" if metric == "r2" else f"{values[r, c]:.3f}",
                            ha="center", va="center", fontsize=5.3,
                            color=heat_text_color(cmap, norm, shown[r, c]) if math.isfinite(shown[r, c]) else "black",
                            fontweight="bold" if r == best else "normal")
        ax.set_xticks(range(len(targets)))
        ax.set_xticklabels([TARGET_LABELS.get(t, t) for t in targets], rotation=35, ha="left")
        ax.xaxis.tick_top()
        ax.set_yticks(range(len(rows)))
        ax.set_yticklabels([label(m) for m in rows] if n == 0 else [])
        ax.tick_params(length=0)
        for spine in ax.spines.values():
            spine.set_visible(False)
        ax.set_xlabel(d, fontweight="bold")
    cax = fig.add_subplot(spec[0, -1])
    bar = fig.colorbar(images[-1], cax=cax)
    bar.set_label(bar_label)
    bar.outline.set_linewidth(0.5)
    save_figure(plt, fig, folder, name, formats)


# ============================================================
# EMPIRICAL VALIDATION (results of Step 5)
# ============================================================

EMPIRICAL_ROOT = ROOT / "EmpiricalValidation"
CASE_NAMES = {"HIV_Zurich": "HIV Zurich", "Primates": "Primates"}


def truthy(value):
    return str(value).strip().lower() in ("true", "1", "yes")


def simulated_status(a, model, alpha):
    """Step 6 significance of a model on simulated trees: BEST / n.s. / stars, or blank."""
    if not a or model not in a["models"]:
        return ""
    i = a["models"].index(model)
    if i == a["best"]:
        return "BEST"
    return stars(a["p_holm"][a["best"], i], alpha)


def simulated_ranks(runs, dataset, models, split):
    values = sorted((runs[(dataset, m)].metric(split, "macro_normalized_mae"), m)
                    for m in models if (dataset, m) in runs)
    return {m: r for r, (v, m) in enumerate((x for x in values if math.isfinite(x[0])), 1)}


def spearman(x, y):
    """(rho, p) over the pairs where both values are finite; nan when fewer than 4 pairs."""
    from scipy import stats
    pairs = [(a, b) for a, b in zip(x, y) if math.isfinite(a) and math.isfinite(b)]
    if len(pairs) < 4:
        return float("nan"), float("nan"), len(pairs)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        result = stats.spearmanr([a for a, _ in pairs], [b for _, b in pairs])
    return float(result[0]), float(result[1]), len(pairs)


def analyse_empirical(report, name, models, runs, analyses, split, alpha, tables, figures, args):
    """Join Step 5's estimates on the empirical trees with the simulated benchmark of this folder."""
    folder = EMPIRICAL_ROOT / name
    estimates = [r for r in read_csv(folder / "estimates.csv") if r.get("target")]
    likelihood = read_csv(folder / "likelihood_primates_bisse.csv")
    report.heading(f"EMPIRICAL VALIDATION (Step 5) - {name}   ({folder})")
    if not estimates:
        report(f"No Step 5 results for {name}. Run:  python Step5-EmpiricalValidation.py run --results {name}")
        return None
    report("Empirical trees have no known parameters: the published PhyloCNN estimates are a reference, and for the "
           "primates the BiSSE likelihood is an independent criterion. Neither is the truth.")

    joined, scorecard = [], {m: {"model": m, "label": label(m), "family": family(m)} for m in models}
    cases = list(dict.fromkeys((r["case"], r["dataset"]) for r in estimates))
    for case, dataset in cases:
        rows = [r for r in estimates if r["case"] == case and r["dataset"] == dataset]
        targets = list(dict.fromkeys(r["target"] for r in rows))
        a = analyses.get(dataset)
        ranks = simulated_ranks(runs, dataset, models, split)
        published = {r["target"]: (to_float(r.get("published")), to_float(r.get("published_ci_low")),
                                   to_float(r.get("published_ci_high"))) for r in rows}
        has_published = any(math.isfinite(v[0]) for v in published.values())
        report("")
        report(f"{CASE_NAMES.get(case, case)} under {dataset}: estimates next to the simulated benchmark "
               f"({SPLIT_NAMES.get(split, split)} nMAE; Sig = Step 6 test against the best model)")
        table = []
        order = sorted(models, key=lambda m: (m not in ranks, ranks.get(m, 0)))
        for m in order:
            mine = {r["target"]: r for r in rows if r["model"] == m}
            run = runs.get((dataset, m))
            nmae = run.metric(split, "macro_normalized_mae") if run else float("nan")
            inside = [truthy(r.get("within_published_ci")) for r in mine.values() if r.get("within_published_ci") != ""]
            diffs = [abs(to_float(r["estimate"]) - published[t][0]) / abs(published[t][0]) * 100
                     for t, r in mine.items() if math.isfinite(to_float(r["estimate"]))
                     and math.isfinite(published[t][0]) and published[t][0] != 0]
            flags = sum(1 for r in mine.values() if r.get("flags"))
            mean_diff = float(np.mean(diffs)) if diffs else float("nan")
            row = [label(m), fnum(nmae), str(ranks.get(m, "")), simulated_status(a, m, alpha)] + \
                  [fnum(to_float(mine[t]["estimate"])) if t in mine else "" for t in targets] + \
                  ([f"{sum(inside)}/{len(inside)}" if inside else "", fnum(mean_diff, 1)] if has_published else []) + \
                  [str(flags) if mine else ""]
            table.append(row)
            if mine:
                joined.append({"results": name, "case": case, "dataset": dataset, "model": m, "family": family(m),
                               "simulated_nmae": nmae, "simulated_rank": ranks.get(m),
                               "simulated_significance": simulated_status(a, m, alpha),
                               **{f"estimate_{t}": to_float(mine[t]["estimate"]) for t in mine},
                               "in_published_ci": sum(inside) if inside else "",
                               "mean_abs_pct_diff_vs_published": mean_diff, "flagged_estimates": flags})
                if case == "HIV_Zurich" and dataset == "BDSS":
                    scorecard[m]["hiv_bdss_in_published_ci"] = f"{sum(inside)}/{len(inside)}" if inside else ""
                    scorecard[m]["hiv_bdss_mean_pct_diff"] = mean_diff
        if has_published:
            table.append(["Original PhyloCNN (published)", "", "", "reference"] +
                         [fnum(published[t][0]) for t in targets] + ["", "", ""])
        report.table(["Model", "Sim. nMAE", "Rank", "Sig"] + targets +
                     (["In pub. CI", "Mean |%diff|"] if has_published else []) + ["Flagged"], table)
        if has_published:
            sim = [runs[(dataset, m)].metric(split, "macro_normalized_mae") for m in order if (dataset, m) in runs]
            agree = [next((j["mean_abs_pct_diff_vs_published"] for j in joined
                           if j["model"] == m and j["case"] == case and j["dataset"] == dataset), float("nan"))
                     for m in order if (dataset, m) in runs]
            rho, p, n = spearman(sim, agree)
            if math.isfinite(rho):
                report(f"  Spearman (simulated nMAE vs |%diff| from published PhyloCNN), {n} models: rho={rho:+.2f}, "
                       f"p={fpval(p)}  (agreement with PhyloCNN, not accuracy)")

    lik_rows = []
    if likelihood:
        a = analyses.get("BiSSE")
        ranks = simulated_ranks(runs, "BiSSE", models, split)
        ml = next((to_float(r["loglik"]) for r in likelihood if r["method"].startswith("Maximum")), float("nan"))
        pub = next((to_float(r["loglik"]) for r in likelihood if r["method"].startswith("Original")), float("nan"))
        report("")
        report("Primates: BiSSE log-likelihood of each model's estimates (Step 5) next to its simulated BiSSE accuracy")
        table = []
        for r in sorted(likelihood, key=lambda r: -to_float(r["loglik"]) if math.isfinite(to_float(r["loglik"])) else 1e300):
            m = r["method"]
            is_model = r.get("kind") != "reference"
            run = runs.get(("BiSSE", m)) if is_model else None
            nmae = run.metric(split, "macro_normalized_mae") if run else float("nan")
            loglik = to_float(r["loglik"])
            entry = {"results": name, "method": m, "kind": r.get("kind"), "loglik": loglik,
                     "delta_vs_ml": loglik - ml, "delta_vs_published_phylocnn": loglik - pub,
                     "inside_95_region": truthy(r.get("inside_95_region")),
                     "likelihood_rank": int(to_float(r.get("rank"))) if math.isfinite(to_float(r.get("rank"))) else "",
                     "simulated_nmae": nmae, "simulated_rank": ranks.get(m, "") if is_model else "",
                     "simulated_significance": simulated_status(a, m, alpha) if is_model else "",
                     **{t: to_float(r.get(t)) for t in ("turnover", "lambda_1", "lambda_2", "q_01")}}
            lik_rows.append(entry)
            if is_model and m in scorecard:
                scorecard[m].update({"primates_loglik": loglik, "primates_likelihood_rank": entry["likelihood_rank"],
                                     "primates_in_95_region": entry["inside_95_region"],
                                     "primates_beats_published_phylocnn": bool(loglik > pub)})
            table.append([str(entry["likelihood_rank"]), label(m) if is_model else m.split(" (")[0],
                          fnum(nmae), str(entry["simulated_rank"]), entry["simulated_significance"],
                          f"{loglik:.2f}" if math.isfinite(loglik) else "n/a",
                          f"{loglik - ml:+.2f}" if math.isfinite(loglik) else "",
                          f"{loglik - pub:+.2f}" if math.isfinite(loglik) and is_model else "",
                          "yes" if entry["inside_95_region"] else "no"])
        report.table(["LL rank", "Method", "Sim. nMAE", "Sim. rank", "Sig", "logL", "dlogL vs ML",
                      "vs PhyloCNN", "In 95% region"], table, left=2)
        models_only = [e for e in lik_rows if e["kind"] != "reference"]
        rho, p, n = spearman([e["simulated_nmae"] for e in models_only], [e["loglik"] for e in models_only])
        if math.isfinite(rho):
            report(f"  Spearman (simulated BiSSE nMAE vs primate logL), {n} models: rho={rho:+.2f}, p={fpval(p)} "
                   "(negative = models that are more accurate on simulations also fit the real tree better)")
        better = [label(e["method"]) for e in models_only if math.isfinite(e["loglik"]) and e["loglik"] > pub]
        report(f"  Models whose estimates fit the primate tree better than the published PhyloCNN estimates: "
               f"{', '.join(better) if better else 'none'}")
        write_latex(tables / "table_empirical_likelihood.tex",
                    ["Method", "turnover", "$\\lambda_1$", "$\\lambda_2$", "$q_{01}$", "logL", "$\\Delta$logL"],
                    [[label(e["method"]) if e["kind"] != "reference" else e["method"].split(" (")[0],
                      fnum(e["turnover"]), fnum(e["lambda_1"]), fnum(e["lambda_2"]), fnum(e["q_01"]),
                      f"{e['loglik']:.2f}" if math.isfinite(e["loglik"]) else "",
                      f"{e['delta_vs_ml']:+.2f}" if math.isfinite(e["loglik"]) else ""] for e in lik_rows],
                    f"BiSSE log-likelihood of the primate tree under each set of estimates ({name}).",
                    f"tab:{name}_empirical_likelihood")

    # ---------- scorecard: simulated benchmark + empirical checks per model ----------
    report("")
    report("Scorecard per model (simulated benchmark from this analysis, empirical checks from Step 5; blank = no result)")
    table, card_rows = [], []
    for m in models:
        card = scorecard[m]
        best, tied, ranks = 0, 0, []
        for d, a in analyses.items():
            if a and m in a["models"]:
                status = simulated_status(a, m, alpha)
                best += status == "BEST"
                tied += status in ("BEST", "n.s.")
            r = simulated_ranks(runs, d, models, split).get(m)
            if r:
                ranks.append(r)
        card.update({"datasets_best": best, "datasets_best_or_tied": tied,
                     "mean_simulated_rank": float(np.mean(ranks)) if ranks else float("nan")})
        card_rows.append(card)
        if not ranks and "primates_loglik" not in card:
            table.append([label(m)] + [""] * 7)
            continue
        table.append([label(m), fnum(card["mean_simulated_rank"], 2), str(best), str(tied),
                      card.get("hiv_bdss_in_published_ci", ""), fnum(card.get("hiv_bdss_mean_pct_diff"), 1),
                      f"{card['primates_loglik']:.2f}" if math.isfinite(to_float(card.get("primates_loglik"))) else "",
                      ("yes" if card.get("primates_in_95_region") else "no") if "primates_loglik" in card else ""])
    report.table(["Model", "Mean sim. rank", "Best in", "Best/tied in", "HIV in pub. CI", "HIV |%diff|",
                  "Primates logL", "In 95% region"], table)

    write_csv(tables / "empirical_estimates.csv", joined)
    write_csv(tables / "empirical_likelihood.csv", lik_rows)
    write_csv(tables / "empirical_scorecard.csv", card_rows)
    if not args.no_figures:
        try:
            empirical_figure(figures, joined, lik_rows, args.formats)
        except ImportError as exc:
            report(f"[WARN] empirical figure skipped ({exc})")
    return {"estimates": joined, "likelihood": lik_rows}


def empirical_figure(folder, joined, lik_rows, formats):
    """fig10: does accuracy on simulated trees go with the fit to the real trees?"""
    hiv = [j for j in joined if j["case"] == "HIV_Zurich" and j["dataset"] == "BDSS"
           and math.isfinite(j["simulated_nmae"]) and math.isfinite(to_float(j["mean_abs_pct_diff_vs_published"]))]
    prim = [e for e in lik_rows if e["kind"] != "reference" and math.isfinite(e["simulated_nmae"])
            and math.isfinite(e["loglik"])]
    panels = [p for p in (("primates", prim), ("hiv", hiv)) if p[1]]
    if not panels:
        return
    plt = pyplot()
    fig, axes = plt.subplots(1, len(panels), figsize=(3.5 * len(panels), 2.6), squeeze=False)
    for k, (ax, (kind, points)) in enumerate(zip(axes[0], panels)):
        if kind == "primates":
            pub = next((e["delta_vs_ml"] for e in lik_rows if e["method"].startswith("Original")), float("nan"))
            for e in points:
                ax.scatter(e["simulated_nmae"], e["delta_vs_ml"], s=18, marker=MARKERS[family(e["method"])],
                           color=color(e["method"]), edgecolors="black", linewidths=0.3, zorder=3)
            ax.axhline(0, color="black", lw=0.7)
            if math.isfinite(pub):
                ax.axhline(pub, color="black", lw=0.7, ls="--")
                ax.text(ax.get_xlim()[1], pub, " published PhyloCNN", va="center", ha="left", fontsize=6)
            ax.set_ylabel("Primates: logL - max. (BiSSE)")
            ax.set_xlabel("Simulated BiSSE nMAE (author test)")
            ax.set_title("Fit of the estimates to the real tree", loc="right")
        else:
            for e in points:
                ax.scatter(e["simulated_nmae"], e["mean_abs_pct_diff_vs_published"], s=18,
                           marker=MARKERS[family(e["model"])], color=color(e["model"]), edgecolors="black",
                           linewidths=0.3, zorder=3)
            ax.set_ylabel("HIV BDSS: mean |%diff| vs PhyloCNN")
            ax.set_xlabel("Simulated BDSS nMAE (author test)")
            ax.set_title("Agreement with the published PhyloCNN", loc="right")
        panel_label(ax, "ab"[k], x=-0.2, y=1.06)
    used = model_order({e["method"] for e in prim} | {j["model"] for j in hiv})
    model_legend(fig, used, y=-0.02, ncol=5, lines=False)
    fig.tight_layout(rect=(0, 0.06 * int(math.ceil(len(used) / 5)) + 0.02, 1, 1))
    save_figure(plt, fig, folder, "fig10_empirical_vs_simulated", formats)


# ============================================================
# SEVERAL RESULTS FOLDERS
# ============================================================

def ablation_summary(report, rows, split, out):
    """Folders named <experiment>_s<seed> (Step4-Ablations.sh): mean +- SD over seeds per experiment,
    and the change against the 'base' experiment (lower = better)."""
    import re
    groups = {}
    for r in rows:
        match = re.match(r"^(.*)_s(\d+)$", r["results"])
        if match:
            groups.setdefault((match.group(1), r["dataset"], r["model"]), []).append(r)
    experiments = list(dict.fromkeys(k[0] for k in groups))
    if len(experiments) < 2:
        return
    base = next((e for e in experiments if e.endswith("base")), None)
    report.heading(f"EXPERIMENTS AVERAGED OVER SEEDS - {SPLIT_NAMES.get(split, split)} trees "
                   "(mean +- SD; change vs base; lower = better)")
    if split == "author_test":
        report("Choose settings with --split validation; report the chosen ones on the author test trees.")
    summary = []
    for d in dataset_order({k[1] for k in groups}):
        table = []
        for m in model_order({k[2] for k in groups if k[1] == d}):
            for e in experiments:
                runs = groups.get((e, d, m), [])
                if not runs:
                    continue
                nmae = np.array([r["nmae"] for r in runs], float)
                paper = np.array([r["paper_value"] for r in runs], float)
                ref = groups.get((base, d, m), []) if base else []
                ref_nmae = float(np.mean([r["nmae"] for r in ref])) if ref else float("nan")
                ref_paper = float(np.mean([r["paper_value"] for r in ref])) if ref else float("nan")
                change = 100 * (nmae.mean() - ref_nmae) / ref_nmae if ref and e != base else float("nan")
                change_paper = 100 * (paper.mean() - ref_paper) / ref_paper if ref and e != base else float("nan")
                table.append([label(m), e, str(len(runs)), f"{fnum(nmae.mean())} +- {fnum(nmae.std(), 3)}",
                              f"{change:+.1f}%" if math.isfinite(change) else "",
                              f"{fnum(paper.mean())} +- {fnum(paper.std(), 3)}",
                              f"{change_paper:+.1f}%" if math.isfinite(change_paper) else ""])
                summary.append({"dataset": d, "model": m, "experiment": e, "seeds": len(runs),
                                "nmae_mean": nmae.mean(), "nmae_sd": nmae.std(), "nmae_change_vs_base_percent": change,
                                "paper_measure": paper_name(d), "paper_mean": paper.mean(), "paper_sd": paper.std(),
                                "paper_change_vs_base_percent": change_paper})
        report("")
        report(f"{d}")
        report.table(["Model", "Experiment", "Seeds", "nMAE", "nMAE vs base", f"{paper_name(d)} (paper)",
                      f"{paper_name(d)} vs base"], table, left=2)
    write_csv(out / "experiments_over_seeds.csv", summary)


def compare_folders(collected, args):
    """nMAE of every model across results folders (e.g. 10,000 vs 100,000 training trees)."""
    out = Path(args.out) / "_analysis" if args.out else RESULTS_ROOT / "_analysis"
    report = Report()
    names = [n for n in collected if collected[n]["runs"]]
    if len(names) < 2:
        return
    split = args.split
    report.heading(f"COMPARISON OF RESULTS FOLDERS - nMAE on the {SPLIT_NAMES.get(split, split)} trees "
                   "(blank = no result)")
    datasets = dataset_order({d for n in names for d, _ in collected[n]["runs"]})
    models = model_order({m for n in names for _, m in collected[n]["runs"]})
    rows_csv = []
    for d in datasets:
        report("")
        sizes = []
        for n in names:
            trains = [r.n("train") for (dd, _), r in collected[n]["runs"].items() if dd == d and r.n("train")]
            sizes.append(f"{n} (train {max(trains):,})" if trains else n)
        report(f"{d}: " + " | ".join(sizes))
        rows = []
        for m in models:
            cells = [label(m)]
            for n in names:
                run = collected[n]["runs"].get((d, m))
                value = run.metric(split, "macro_normalized_mae") if run else float("nan")
                cells.append(fnum(value))
                if run:
                    rows_csv.append({"results": n, "dataset": d, "model": m, "family": family(m),
                                     "n_train": run.n("train"), "nmae": value,
                                     "paper_measure": paper_name(d), "paper_value": paper_value(run, split, d),
                                     "r2": run.metric(split, "macro_r2"),
                                     "train_seconds": run.info("train_seconds")})
            rows.append(cells)
        report.table(["Model"] + names, rows)
    write_csv(out / "cross_results.csv", rows_csv)
    ablation_summary(report, rows_csv, split, out)

    # Empirical (Step 5): primate log-likelihood of every model in every folder.
    lik = {n: {e["method"]: e for e in ((collected[n].get("empirical") or {}).get("likelihood") or [])}
           for n in names}
    if any(lik.values()):
        report("")
        report("Primates (Step 5): BiSSE log-likelihood of each model's estimates per results folder (blank = no result)")
        rows, lik_csv = [], []
        for n in names:
            for e in lik[n].values():
                lik_csv.append({"results": n, **{k: v for k, v in e.items() if k != "results"}})
        for ref in [e for e in (lik[names[0]] or lik[next(n for n in names if lik[n])]).values()
                    if e["kind"] == "reference"]:
            rows.append([ref["method"].split(" (")[0]] + [f"{ref['loglik']:.2f}"] * len(names))
        for m in models:
            cells = [label(m)]
            for n in names:
                e = lik[n].get(m)
                cells.append(f"{e['loglik']:.2f}" if e and math.isfinite(e["loglik"]) else "")
            if any(cells[1:]):
                rows.append(cells)
        report.table(["Method"] + names, rows)
        write_csv(out / "cross_results_empirical_likelihood.csv", lik_csv)

    sizes = {(r["dataset"], r["model"]): set() for r in rows_csv}
    for r in rows_csv:
        if r["n_train"]:
            sizes[(r["dataset"], r["model"])].add(int(r["n_train"]))
    if not args.no_figures and any(len(v) >= 2 for v in sizes.values()):
        try:
            plt = pyplot()
        except ImportError as exc:
            report(f"[WARN] figure skipped ({exc}); install matplotlib")
            report.save(out / "REPORT.txt")
            return
        ds = [d for d in datasets if any(len(sizes.get((d, m), ())) >= 2 for m in models)]
        fig, axes = grid(plt, len(ds), row_height=2.1)
        used = []
        for n, (ax, d) in enumerate(zip(axes, ds)):
            for m in models:
                points = sorted((int(r["n_train"]), r["nmae"]) for r in rows_csv
                                if r["dataset"] == d and r["model"] == m and r["n_train"] and math.isfinite(r["nmae"]))
                if len({p[0] for p in points}) < 2:
                    continue
                ax.plot([p[0] for p in points], [p[1] for p in points], color=color(m),
                        marker=MARKERS[family(m)], ms=3, mec="black", mew=0.3, lw=0.9)
                if m not in used:
                    used.append(m)
            ax.set_xscale("log")
            ax.set_xlabel("Training trees")
            ax.set_ylabel(f"nMAE ({SPLIT_NAMES.get(split, split)})")
            ax.set_title(d, loc="left", fontweight="bold")
            panel_label(ax, "abcdefgh"[n], x=-0.12)
        model_legend(fig, model_order(used), y=-0.02, ncol=5)
        fig.tight_layout(rect=(0, 0.03 * int(math.ceil(len(used) / 5)) + 0.01, 1, 1))
        save_figure(plt, fig, out, "fig_training_set_size", args.formats)
        report("")
        report(f"Figure: {out / 'fig_training_set_size'}")
    report(f"Table:  {out / 'cross_results.csv'}")
    report.save(out / "REPORT.txt")


# ============================================================
# MAIN
# ============================================================

def parse_args():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("command", nargs="?", default="run", choices=("run",),
                        help="run (default): the full analysis")
    parser.add_argument("--results", default="",
                        help="Results folder(s), comma separated; blank (default) = every folder in Results/")
    parser.add_argument("--datasets", default="",
                        help="Datasets to analyse, comma separated; blank (default) = all (BD, BDEI, BDSS, BiSSE)")
    parser.add_argument("--no-empirical", action="store_true",
                        help="Leave out the Step 5 empirical results")
    parser.add_argument("--split", default="author_test", choices=("author_test", "validation", "train"),
                        help="Split used for the tests and figures (default: the authors' test set)")
    parser.add_argument("--alpha", type=float, default=0.05, help="Significance level (default 0.05)")
    parser.add_argument("--bootstrap", type=int, default=2000, help="Bootstrap resamples for the 95%% CIs")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--formats", default="pdf,png", help="Figure formats, e.g. pdf,png,svg,tiff")
    parser.add_argument("--no-figures", action="store_true", help="Only tables and tests")
    parser.add_argument("--out", default="",
                        help="Output folder (default: Results/<results>/analysis and Results/_analysis)")
    args = parser.parse_args()
    args.formats = [f.strip().lstrip(".") for f in args.formats.split(",") if f.strip()]
    args.datasets = [d.strip() for d in args.datasets.split(",") if d.strip()]
    if not 0 < args.alpha < 1:
        parser.error("--alpha must be between 0 and 1")
    return args


def results_folders(text):
    if text.strip():
        names = [x.strip() for x in text.split(",") if x.strip()]
        missing = [n for n in names if not (RESULTS_ROOT / n / "runs").is_dir()]
        if missing:
            raise SystemExit(f"[ERROR] no Step 4 results for: {', '.join(missing)} (looked in {RESULTS_ROOT})")
        return names
    if not RESULTS_ROOT.is_dir():
        return []
    return sorted(p.name for p in RESULTS_ROOT.iterdir()
                  if p.is_dir() and not p.name.startswith("_") and (p / "runs").is_dir())


def main():
    args = parse_args()
    try:
        import scipy  # noqa: F401
    except ImportError:
        raise SystemExit("[ERROR] scipy is needed for the statistical tests (pip install scipy)")
    names = results_folders(args.results)
    if not names:
        print(f"No Step 4 results found in {RESULTS_ROOT}")
        return 1
    print(f"Results folders: {', '.join(names)}")
    collected = {name: analyse_folder(name, args) for name in names}
    if len(names) > 1:
        compare_folders(collected, args)
    return 0


if __name__ == "__main__":
    sys.exit(main())
