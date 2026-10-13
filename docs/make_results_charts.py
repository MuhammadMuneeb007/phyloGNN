#!/usr/bin/env python3
"""Draw the README result charts of the 100,000-tree experiment from docs/results_100k.csv.

  python docs/make_results_charts.py        ->  docs/figures/*.png  (+ the summary table, printed)

Sources in the CSV: "Paper CNN" = values published in the PhyloCNN paper (100K training set),
"Pretrained CNN" = the authors' released models run by Step 7 ("valid" = passed the reproduction
check), "GNN 100K" = our graph models trained on 100,000 simulated trees (finished runs only).
Errors: MRE for BD / BDEI / BDSS, MAE for BiSSE (the paper's measures); lower is better.
"""

import csv
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

HERE = Path(__file__).resolve().parent
OUT = HERE / "figures"
PAPER, PRETRAINED, GNN = "#8c1d18", "#e8817b", "#067d53"
ORDER = ("BD", "BDSS", "BiSSE", "BDEI")
PARAMETERS = {
    "BD": ("R0", "infectious period"),
    "BDSS": ("R0", "infectious period", "transmission factor", "superspreader fraction"),
    "BiSSE": ("lambda_1", "lambda_2", "q_01"),
    "BDEI": ("R0", "infectious period", "incubation period"),
}
LABELS = {"R0": "$R_0$", "infectious period": "infectious\nperiod", "incubation period": "incubation\nperiod",
          "transmission factor": "transmission\nfactor", "superspreader fraction": "superspreader\nfraction",
          "lambda_1": r"$\lambda_1$", "lambda_2": r"$\lambda_2$", "q_01": "$q_{01}$"}

plt.rcParams.update({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False,
                     "font.family": "sans-serif", "savefig.dpi": 200, "savefig.bbox": "tight"})


def load():
    with open(HERE / "results_100k.csv", newline="") as f:
        return list(csv.DictReader(f))


def best(rows, dataset, parameter, source):
    """(value, model) with the lowest error among the valid rows, or (None, None)."""
    cands = [(float(r["value"]), r["model"]) for r in rows
             if r["kind"] == "per_parameter" and r["dataset"] == dataset and r["parameter"] == parameter
             and r["source"] == source and r["valid"] == "yes"]
    return min(cands) if cands else (None, None)


def best_mean(rows, dataset, source):
    """Best variant / model by the error averaged over the dataset's compared parameters."""
    values = defaultdict(dict)
    for r in rows:
        if (r["kind"] == "per_parameter" and r["dataset"] == dataset and r["source"] == source
                and r["valid"] == "yes" and r["parameter"] in PARAMETERS[dataset]):
            values[r["model"]][r["parameter"]] = float(r["value"])
    complete = {m: sum(v.values()) / len(v) for m, v in values.items() if len(v) == len(PARAMETERS[dataset])}
    return min(((v, m) for m, v in complete.items()), default=(None, None))


def per_parameter_figure(rows):
    fig, axes = plt.subplots(2, 2, figsize=(11, 7.2))
    for ax, dataset in zip(axes.flat, ORDER):
        params = PARAMETERS[dataset]
        width = 0.26
        for k, (source, colour, name) in enumerate(((("Paper CNN", PAPER, "Paper CNN (best variant)")),
                                                    ("Pretrained CNN", PRETRAINED, "Pretrained CNN (best valid)"),
                                                    ("GNN 100K", GNN, "Best GNN, 100K"))):
            xs, ys, tags = [], [], []
            for i, p in enumerate(params):
                value, model = best(rows, dataset, p, source)
                if value is not None:
                    xs.append(i + (k - 1) * width)
                    ys.append(value)
                    tags.append(model)
            bars = ax.bar(xs, ys, width, color=colour, label=name)
            if source == "GNN 100K":
                for bar, tag in zip(bars, tags):
                    ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(), tag, rotation=90,
                            ha="center", va="bottom", fontsize=6.5, color=GNN)
        ax.set_xticks(range(len(params)))
        ax.set_xticklabels([LABELS[p] for p in params])
        metric = "MAE" if dataset == "BiSSE" else "MRE"
        ax.set_ylabel(f"{metric} (lower is better)")
        title = dataset + ("   (no valid pretrained CNN)" if dataset == "BiSSE" else "")
        ax.set_title(title, loc="left", fontweight="bold")
        ax.margins(y=0.25)
    axes[0][0].legend(frameon=False, fontsize=8, loc="upper left")
    fig.suptitle("100K experiment: error per parameter on the authors' test sets", x=0.01, ha="left",
                 fontweight="bold")
    fig.tight_layout()
    fig.savefig(OUT / "fig_100k_per_parameter.png")
    plt.close(fig)


def architecture_figure(rows):
    fig, axes = plt.subplots(1, 4, figsize=(13, 3.4))
    for ax, dataset in zip(axes, ORDER):
        items = sorted(((float(r["value"]), r["model"]) for r in rows
                        if r["kind"] == "nmae" and r["dataset"] == dataset), reverse=True)
        values = [v for v, _ in items]
        colours = [GNN if v == min(values) else "#9fd3bd" for v in values]
        ax.barh([m for _, m in items], values, color=colours)
        for y, v in enumerate(values):
            ax.text(v, y, f" {v:.4f}", va="center", fontsize=7)
        ax.set_title(f"{dataset} ({len(items)}/12 finished)", loc="left", fontweight="bold", fontsize=9)
        ax.set_xlabel("nMAE (lower is better)")
        ax.set_xlim(min(values) * 0.97, max(values) * 1.02)
        ax.xaxis.set_major_locator(plt.MaxNLocator(3))
    fig.suptitle("100K experiment: GNN architectures (finished runs, author test sets)", x=0.01, ha="left",
                 fontweight="bold")
    fig.tight_layout()
    fig.savefig(OUT / "fig_100k_architectures.png")
    plt.close(fig)


def primates_figure(rows):
    items = [(float(r["value"]), r["source"], r["model"]) for r in rows if r["kind"] == "primates"]
    items.sort()
    fig, ax = plt.subplots(figsize=(6.4, 2.6))
    colours = {"Maximum likelihood": "#555555", "Paper CNN": PAPER, "GNN 100K": GNN}
    labels = [f"{s}: {m}" if s != "Maximum likelihood" else "Maximum likelihood" for _, s, m in items]
    ax.barh(labels, [v + 860 for v, _, _ in items], left=-860, color=[colours[s] for _, s, _ in items])
    for y, (v, _, _) in enumerate(items):
        ax.text(v, y, f" {v:.2f}", va="center", fontsize=8)
    ax.set_xlim(-855, -845.5)
    ax.set_xlabel("BiSSE log-likelihood of the primate tree (higher = better fit)")
    ax.set_title("Primates: fit of each set of estimates (independent criterion, not the truth)",
                 loc="left", fontsize=9, fontweight="bold")
    fig.tight_layout()
    fig.savefig(OUT / "fig_100k_primates_likelihood.png")
    plt.close(fig)


def summary(rows):
    print("| Dataset | Measure | Paper CNN (best variant) | Pretrained CNN (best valid) | Best GNN 100K | GNN vs paper |")
    print("|---|---|---|---|---|---|")
    for dataset in ORDER:
        metric = "MAE" if dataset == "BiSSE" else "MRE"
        paper, paper_m = best_mean(rows, dataset, "Paper CNN")
        pre, pre_m = best_mean(rows, dataset, "Pretrained CNN")
        gnn, gnn_m = best_mean(rows, dataset, "GNN 100K")
        change = 100 * (gnn - paper) / paper
        print(f"| {dataset} | {metric} | {paper:.4f} ({paper_m}) | "
              + (f"{pre:.4f} ({pre_m})" if pre is not None else "none valid")
              + f" | {gnn:.4f} ({gnn_m}) | {change:+.1f}% |")


if __name__ == "__main__":
    OUT.mkdir(parents=True, exist_ok=True)
    data = load()
    per_parameter_figure(data)
    architecture_figure(data)
    primates_figure(data)
    summary(data)
    print(f"Figures: {OUT}")
