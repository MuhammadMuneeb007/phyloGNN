#!/usr/bin/env python3
"""
Step5-EmpiricalValidation.py — apply every trained Step 4 model to the empirical phylogenies of the
PhyloCNN paper (Perez & Gascuel 2025, Systematic Biology) and compare with the published estimates.

Case studies (trees shipped in phyloCNN/Empirical_Datasets/):
  HIV_Zurich  200 HIV-1 sequences from Zurich (Rasmussen et al. 2017), sampling probability 0.25
              (as in PhyloDeep / PhyloCNN). Models: BDSS (main) and also BD, BDEI.
              Units: years.
  Primates    260 primate species with seed-dispersal trait (Fabre et al. 2009; Gomez & Verdu 2012),
              sampling fraction 0.68241469816273 (260 of 381 species). Model: BiSSE.
              Units: per million years. Tip states from the NHX tags [&&NHX-t_s=1|2].

For every results folder and every finished run (dataset x model) the tree is encoded exactly as the
model saw its training trees:
  PhyloCNN  the authors' encoder (phyloCNN/Encoding/PhyloCNN_Encoding_*.py, process_tree), the
            notebook assembly (rescaling row removed, sampling column added, leaves / internal nodes
            sorted by distance to root, padded to 500) and the model's own feature columns.
            Time-scaled outputs are converted back with the tree's rescaling factor r
            (periods * r, rates / r), as in the authors' notebooks. Checked against the factors the
            authors used for these trees (HIV 5.174108653956917, primates 4.064753).
  PhyloGNN  Graph_Encoding.tree_to_graph, graph features standardised with the scaler saved in the
            checkpoint, outputs de-standardised with the checkpoint's label scaler.
There is no ground truth: the script reports every model's estimate, flags estimates outside the
range of parameters the model was trained on (and trees unlike the training trees), and compares
the estimates with the published PhyloCNN values and 95% CIs, model by model.

Independent check (primates): the BiSSE log-likelihood of the tree and tip states under each set of
estimates, next to the maximum likelihood and the published PhyloCNN estimates (same model as
Lambert et al.'s HiSSE fit: lambda_1, lambda_2, mu_i = turnover * lambda_i, q01 = q10). Validated:
the published PhyloCNN estimates give logL -850.06 and the maximum is -846.93, as reported in the
paper (-850.1 and -846.9). Estimates with a higher likelihood explain the data better under BiSSE;
this is a criterion, not the truth. For HIV no independent likelihood is computed (BDSS would need
BEAST2/bdmm).

PhyloCNN checkpoints trained before Step 4 used the authors' rescaled targets cannot recover
time-scaled parameters (infectious / incubation period, lambda, q) from the scale-free input; those
values are left blank with a note. Re-run Step 4 (only the PhyloCNN runs are redone).

Usage (from the project root, after Step 4; runs on the CPU by default):
  python Step5-EmpiricalValidation.py run                      # all models of every results folder
  python Step5-EmpiricalValidation.py run --results full100k   # one folder (comma list allowed)
  python Step5-EmpiricalValidation.py run --cases HIV_Zurich --hiv-sampling 0.25
  python Step5-EmpiricalValidation.py check                    # only the PhyloCNN encoding checks
  sbatch Step5-EmpiricalValidation-CPU.sh full100k             # Bunya
For every case and model it prints the model's performance on the authors' simulated test trees
(Step 4: normalised MAE, R2, MAE, R2 per parameter), its inference time on the empirical tree, and
its estimates.

Outputs: EmpiricalValidation/<results>/
  REPORT.txt           everything printed
  estimates.csv        one row per case x dataset x model x parameter (estimate, flags, Step 4 test nMAE)
  estimates_wide.csv   one row per model
  summary.csv          per parameter: median / IQR / range over models, best models, published value
  likelihood_primates_bisse.csv   BiSSE logL of every model, the published PhyloCNN and the ML estimates
  manifest.json        trees (SHA-256, tips, height, rescaling factor, states), settings, versions
  figures/             fig_empirical_<case>_<dataset>.pdf/.png
EmpiricalValidation/all_estimates.csv   every results folder together
"""

from __future__ import annotations

import os
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import argparse
import csv
import hashlib
import importlib.util
import json
import math
import platform
import re
import sys
import time
from pathlib import Path

import warnings

import numpy as np

import Graph_Encoding as ge

# torch.load of our own checkpoints (newer PyTorch warns about the weights_only default).
warnings.filterwarnings("ignore", category=FutureWarning, message=".*weights_only.*")

ROOT = Path(__file__).resolve().parent
RESULTS_ROOT = ROOT / "Results"
OUT_ROOT = ROOT / "EmpiricalValidation"

CASES = {
    "HIV_Zurich": {
        "tree": "Empirical_Datasets/HIV/ZurichHIV-tree.trees",
        "sampling": 0.25,
        "datasets": ("BDSS", "BD", "BDEI"),
        "units": "years",
        "description": "HIV-1 Zurich, 200 tips (Rasmussen et al. 2017); sampling probability 0.25",
        "authors_rescale_factor": 5.174108653956917,   # phyloCNN/CI/CI_HIV.ipynb
    },
    "Primates": {
        "tree": "Empirical_Datasets/primates/Primates-traits-ultrametric-tree.nwk",
        "sampling": 0.68241469816273,
        "datasets": ("BiSSE",),
        "units": "per million years",
        "description": "Primates, 260 tips, seed-dispersal trait (Fabre et al. 2009; Gomez & Verdu 2012); "
                       "sampling fraction 260/381",
        "authors_rescale_factor": 4.064753,            # phyloCNN/CI/CI_primates.ipynb
    },
}

# Published PhyloCNN estimates (point, 2.5%, 97.5%) from the notebook outputs in the authors' repository.
PUBLISHED = {
    ("HIV_Zurich", "BDSS"): {
        "label": "PhyloCNN 2-gen., HIV priors (published)",
        "source": "phyloCNN/CI/CI_HIV.ipynb (Trained_2Generation_BDSS_HIV; CI from 1,000 nearest simulations)",
        "note": "trained on simulations with HIV-specific priors, which are wider than the standard BDSS "
                "priors used in Step 3",
        "values": {"r_nought": (1.412436, 1.156663, 1.739923),
                   "infectious_period": (11.32054, 8.761571, 14.303697),
                   "transmission_factor": (20.390572, 10.561785, 26.227327),
                   "fraction": (0.129327, 0.078163, 0.168898)},
    },
    ("Primates", "BiSSE"): {
        "label": "PhyloCNN 2-gen. (published)",
        "source": "phyloCNN/CI/CI_primates.ipynb (Trained_2Generation_BiSSE; CI from 1,000 nearest simulations)",
        "note": "",
        "values": {"turnover": (0.610441, 0.391435, 0.733746),
                   "lambda_1": (0.378217, 0.310225, 0.440335),
                   "lambda_2": (0.166157, 0.10892, 0.275925),
                   "q_01": (0.010635, 0.005794, 0.020707)},
    },
}

CNN_MODELS = ("phylocnn_2gen", "phylocnn_1gen", "phylocnn_nocontext")
GNN_MODELS = ("gcn", "graphsage", "gat", "gin", "ecc", "gine", "pna", "transformer", "gps",
              "trignn", "gnn_fusion", "graphsage_gat_fusion")
MODELS = CNN_MODELS + GNN_MODELS
FAMILY = {**{m: "PhyloCNN" for m in CNN_MODELS},
          **{m: "Baseline GNN" for m in ("gcn", "graphsage", "gat", "gin", "ecc")},
          **{m: "Edge-aware GNN" for m in ("gine", "pna", "transformer", "gps")},
          **{m: "Fusion GNN" for m in ("trignn", "gnn_fusion", "graphsage_gat_fusion")}}
OOD_Z = 4.0          # |z| of a graph feature beyond which the tree is unlike the training trees
BOUNDARY = 0.02      # estimate within 2% of the training range from its edge = "at the prior boundary"
NO_TRUTH = "unknown (no ground truth)"


# ============================================================
# BISSE LIKELIHOOD (independent check for the primates)
# ============================================================
# The empirical trees have no known parameters. For BiSSE an independent criterion exists: the
# likelihood of the tree and tip states under each set of estimates (as the authors compare
# PhyloCNN with maximum likelihood / HiSSE). Model as in the simulations and in Lambert et al.'s
# HiSSE fit (deeptimelearning/estimation/MLE/empirical/Bisse_primates_MLE_CI.R): speciation
# lambda_1 (tip state t_s=1) and lambda_2 (t_s=2), extinction mu_i = turnover * lambda_i (shared
# turnover = mu/lambda), symmetric transitions q01 = q10, sampling fraction f for both states.
# Likelihood: FitzJohn (2009) ODEs; root weights from the root likelihoods ("madfitz", the HiSSE
# default), conditioned on survival as in HiSSE.

class BiSSELikelihood:
    """BiSSE log-likelihood of one ultrametric tree with tip states."""

    def __init__(self, newick, sampling):
        root = ge.parse_tree(newick)
        self.f = float(sampling)
        nodes, parent, dist, stack = [], [], [], [(root, -1, 0.0)]
        while stack:
            node, p, d = stack.pop()
            index = len(nodes)
            nodes.append(node)
            parent.append(p)
            dist.append(d)
            stack.extend((child, index, d + child.branch) for child in node.children)
        n = len(nodes)
        self.children = [[] for _ in range(n)]
        for i, p in enumerate(parent):
            if p >= 0:
                self.children[p].append(i)
        self.leaves = [i for i in range(n) if not nodes[i].children]
        self.n_tips = len(self.leaves)
        height = max(dist[i] for i in self.leaves)
        ages = [height - d for d in dist]
        # Extant species only: tip ages must agree (numerical rounding is snapped to the present).
        self.ultrametric_error = max(ages[i] for i in self.leaves) / height
        if self.ultrametric_error > 1e-3:
            raise ValueError(f"BiSSE likelihood needs an ultrametric tree (tip ages differ by "
                             f"{100 * self.ultrametric_error:.2f}% of the tree height)")
        for i in self.leaves:
            ages[i] = 0.0
        self.ages = ages
        self.internal = sorted((i for i in range(n) if nodes[i].children), key=lambda i: ages[i])
        states = [nodes[i].state for i in self.leaves]
        if not any(s in (1, 2) for s in states):
            raise ValueError("BiSSE likelihood needs tip states (t_s=1/2)")
        f = self.f
        self.tip_d = np.array([[f if s in (0, 1) else 0.0, f if s in (0, 2) else 0.0] for s in states])

    @staticmethod
    def _propagate(e1, e2, dt, lam, mu, q, h_max):
        """RK4 over dt for E (shared by all lineages of an ultrametric tree) and the 2x2 propagator
        P of the linear D equations: D(t + dt) = P @ D(t)."""
        l1, l2 = lam
        m1, m2 = mu
        q1, q2 = q
        t1, t2 = l1 + m1 + q1, l2 + m2 + q2
        steps = max(1, int(math.ceil(dt / h_max)))
        h = dt / steps

        def f(y):
            a, b, p11, p12, p21, p22 = y
            da = m1 - t1 * a + q1 * b + l1 * a * a
            db = m2 - t2 * b + q2 * a + l2 * b * b
            k1 = -t1 + 2.0 * l1 * a
            k2 = -t2 + 2.0 * l2 * b
            return (da, db, k1 * p11 + q1 * p21, k1 * p12 + q1 * p22, q2 * p11 + k2 * p21, q2 * p12 + k2 * p22)

        y = (e1, e2, 1.0, 0.0, 0.0, 1.0)
        for _ in range(steps):
            k1 = f(y)
            k2 = f(tuple(v + 0.5 * h * k for v, k in zip(y, k1)))
            k3 = f(tuple(v + 0.5 * h * k for v, k in zip(y, k2)))
            k4 = f(tuple(v + h * k for v, k in zip(y, k3)))
            y = tuple(v + h / 6.0 * (a + 2.0 * b + 2.0 * c + d) for v, a, b, c, d in zip(y, k1, k2, k3, k4))
        return y[0], y[1], np.array([[y[2], y[3]], [y[4], y[5]]])

    def loglik(self, lam1, lam2, mu1, mu2, q01, q10):
        lam, mu, q = (float(lam1), float(lam2)), (float(mu1), float(mu2)), (float(q01), float(q10))
        values = lam + mu + q
        if not all(math.isfinite(v) for v in values) or min(lam) <= 0 or min(mu) < 0 or min(q) < 0:
            return float("-inf")
        h_max = 0.05 / (max(lam) + max(mu) + max(q))
        d = self.tip_d.copy()                     # one row per lineage, rows reused at merges
        logs = np.zeros(len(d))
        alive = np.ones(len(d), dtype=bool)
        row = {leaf: k for k, leaf in enumerate(self.leaves)}
        e1 = e2 = 1.0 - self.f
        t = 0.0
        lam_vec = np.array(lam)
        for node in self.internal:                # from the present to the root
            age = self.ages[node]
            if age > t:
                e1, e2, p = self._propagate(e1, e2, age - t, lam, mu, q, h_max)
                live = d[alive] @ p.T
                scale = live.sum(axis=1)
                if not np.all(scale > 0) or not np.all(np.isfinite(scale)):
                    return float("-inf")
                d[alive] = live / scale[:, None]
                logs[alive] += np.log(scale)
                t = age
            rows = [row[c] for c in self.children[node]]
            merged = np.prod(d[rows], axis=0) * lam_vec ** (len(rows) - 1)
            total = merged.sum()
            if not (total > 0 and math.isfinite(total)):
                return float("-inf")
            alive[rows] = False
            r = rows[0]
            d[r] = merged / total
            logs[r] = logs[rows].sum() + math.log(total)
            alive[r] = True
            row[node] = r
        r = row[0]
        root_d, e = d[r], np.array([e1, e2])
        root_p = root_d / root_d.sum()
        condition = float(np.sum(root_p * lam_vec * (1.0 - e) ** 2))   # survival (HiSSE)
        if not condition > 0:
            return float("-inf")
        return float(math.log(np.sum(root_p * root_d) / condition) + logs[r])

    def loglik_targets(self, values):
        """Log-likelihood of the four BiSSE targets (turnover, lambda_1, lambda_2, q_01)."""
        try:
            tau, l1, l2, q = (float(values[k]) for k in ("turnover", "lambda_1", "lambda_2", "q_01"))
        except (KeyError, TypeError, ValueError):
            return float("nan")
        if not all(math.isfinite(v) for v in (tau, l1, l2, q)):
            return float("nan")
        if l1 <= 0 or l2 <= 0 or q < 0 or not 0 <= tau < 1:
            return float("-inf")
        return self.loglik(l1, l2, tau * l1, tau * l2, q, q)

    def maximum_likelihood(self, starts):
        """ML estimates of (turnover, lambda_1, lambda_2, q_01) with Nelder-Mead from several starts."""
        from scipy.optimize import minimize

        def unpack(x):
            return {"turnover": 1.0 / (1.0 + math.exp(-x[0])), "lambda_1": math.exp(x[1]),
                    "lambda_2": math.exp(x[2]), "q_01": math.exp(x[3])}

        def objective(x):
            value = self.loglik_targets(unpack(x))
            return 1e10 if not math.isfinite(value) else -value

        best = None
        for start in starts:
            tau = min(max(float(start["turnover"]), 1e-4), 1 - 1e-4)
            x0 = [math.log(tau / (1 - tau)), math.log(max(start["lambda_1"], 1e-6)),
                  math.log(max(start["lambda_2"], 1e-6)), math.log(max(start["q_01"], 1e-8))]
            result = minimize(objective, x0, method="Nelder-Mead",
                              options={"maxiter": 3000, "xatol": 1e-6, "fatol": 1e-7})
            if best is None or result.fun < best.fun:
                best = result
        values = unpack(best.x)
        return values, -float(best.fun)


# ============================================================
# HELPERS
# ============================================================

class Report:
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


def fnum(value, digits=4):
    try:
        value = float(value)
    except (TypeError, ValueError):
        return ""
    if not math.isfinite(value):
        return ""
    if value != 0 and (abs(value) >= 1e5 or abs(value) < 10 ** -digits):
        return f"{value:.{max(digits - 2, 1)}e}"
    return f"{value:.{digits}f}"


def to_float(value):
    try:
        return float(value)
    except (TypeError, ValueError):
        return float("nan")


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
    columns = list(dict.fromkeys(k for r in rows for k in r))
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=columns)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: ("" if isinstance(v, float) and not math.isfinite(v) else v) for k, v in row.items()})


def load_module(name, filename):
    path = ROOT / filename
    if not path.is_file():
        raise FileNotFoundError(f"{path} not found (keep the Step files together)")
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


# ============================================================
# EMPIRICAL TREES
# ============================================================

def read_tree_text(path):
    """First tree of a Newick or NEXUS file, as one Newick string."""
    text = Path(path).read_text(encoding="utf-8", errors="replace").strip()
    if text.upper().startswith("#NEXUS"):
        if re.search(r"^\s*translate\b", text, re.I | re.M):
            raise ValueError(f"{path}: NEXUS translate blocks are not supported; export the tree as Newick")
        match = re.search(r"^\s*tree\s+[^=]+=\s*(?:\[&[RU]\]\s*)?(.+?;)", text, re.I | re.M | re.S)
        if not match:
            raise ValueError(f"{path}: no tree found in the NEXUS file")
        text = match.group(1)
    else:
        text = text[:text.index(";") + 1] if ";" in text else text
    text = re.sub(r"\s+", "", text)
    if not text.endswith(";") or "(" not in text:
        raise ValueError(f"{path}: not a Newick tree")
    return text


class EmpiricalTree:
    """One empirical phylogeny, encoded for PhyloCNN (per encoder family) and PhyloGNN."""

    def __init__(self, case, path, sampling, report):
        self.case = case
        self.path = Path(path)
        self.sampling = float(sampling)
        if not 0 < self.sampling <= 1:
            raise ValueError(f"{case}: sampling probability must be in (0, 1], got {sampling}")
        self.newick = read_tree_text(self.path)
        self.sha256 = hashlib.sha256(self.path.read_bytes()).hexdigest()
        self.encoding_dir = ge.find_repo(ROOT) / "Encoding"
        self._raw, self._graph = {}, {}
        graph = self.graph("BiSSE" if case == "Primates" else "BD")
        self.n_tips = int(graph.n_tips)
        self.height = float(np.expm1(graph.graph_features[ge.GRAPH_FEATURES.index("height_log")]))
        self.state_fraction = (float(graph.graph_features[ge.GRAPH_FEATURES.index("state_one_fraction")]),
                               float(graph.graph_features[ge.GRAPH_FEATURES.index("state_two_fraction")]))
        self.report = report

    def raw(self, dataset):
        family = "BiSSE" if dataset == "BiSSE" else "PhyloDyn"
        if family not in self._raw:
            self._raw[family] = ge.phylocnn_raw_encoding(self.newick, dataset, self.encoding_dir)
        return self._raw[family]

    def rescale_factor(self, dataset):
        return ge.phylocnn_rescale_factor(self.raw(dataset))

    def cnn_input(self, dataset):
        x = ge.build_phylocnn_input(self.raw(dataset), self.sampling)
        leaves = int((x[:, 3, 0] == 1).sum())
        if leaves != self.n_tips:
            raise ValueError(f"PhyloCNN input has {leaves} leaf rows, tree has {self.n_tips} tips")
        return x

    def graph(self, dataset):
        if dataset not in self._graph:
            self._graph[dataset] = ge.tree_to_graph(self.newick, self.sampling, dataset)
        return self._graph[dataset]


def check_encoding(tree, case, report):
    """PhyloCNN encoding checks against the authors' own numbers for these trees."""
    dataset = CASES[case]["datasets"][0]
    r = tree.rescale_factor(dataset)
    expected = CASES[case].get("authors_rescale_factor")
    x = tree.cnn_input(dataset)
    checks = [
        ("CNN input shape", x.shape == (ge.MAX_TIPS, ge.RAW_CNN_WIDTH[dataset] + 1, 2), str(x.shape)),
        ("leaf rows = tips", int((x[:, 3, 0] == 1).sum()) == tree.n_tips, str(tree.n_tips)),
        ("sampling column", bool(np.allclose(x[:tree.n_tips, -1, 0], tree.sampling)), fnum(tree.sampling, 6)),
    ]
    if expected and tree.path.name == Path(CASES[case]["tree"]).name:   # the authors' tree
        checks.append(("rescaling factor = authors'", abs(r - expected) <= 1e-5 * expected,
                       f"{r:.6f} vs {expected:.6f}"))
    if case == "Primates":
        checks.append(("tip states read (1 and 2)", min(tree.state_fraction) > 0,
                       f"state 1 {tree.state_fraction[0]:.3f}, state 2 {tree.state_fraction[1]:.3f}"))
    for name, ok, detail in checks:
        report(f"  [{'PASS' if ok else 'FAIL'}] {name}: {detail}")
    if not all(ok for _, ok, _ in checks):
        raise RuntimeError(f"{case}: the PhyloCNN encoding check failed; no estimates are made")
    return r


# ============================================================
# MODELS
# ============================================================

def training_ranges(run_folder, targets):
    """(min, max) of every target over the run's training trees (predictions_train.csv)."""
    rows = read_csv(Path(run_folder) / "predictions_train.csv")
    ranges = {}
    for t in targets:
        values = np.array([to_float(r.get("true_" + t)) for r in rows])
        values = values[np.isfinite(values)]
        ranges[t] = (float(values.min()), float(values.max())) if len(values) else (float("nan"), float("nan"))
    return ranges


def range_flag(value, low, high):
    if not (math.isfinite(value) and math.isfinite(low) and math.isfinite(high)):
        return ""
    span = high - low
    if value < low or value > high:
        return "outside training range"
    if span > 0 and (value - low < BOUNDARY * span or high - value < BOUNDARY * span):
        return "at training-range edge"
    return ""


def graph_feature_z(tree, dataset, scaler):
    gf = np.asarray(tree.graph(dataset).graph_features, dtype=np.float64)
    mean = np.asarray(scaler["mean"], dtype=np.float64)
    std = np.asarray(scaler["std"], dtype=np.float64)
    return dict(zip(scaler.get("names", ge.GRAPH_FEATURES), ((gf - mean) / std).tolist()))


def predict_run(step4, tree, dataset, model_name, folder, device, threads):
    """Estimates in natural units for one trained run: (values {target: v}, notes, meta)."""
    meta = read_json(folder / "checkpoint" / "checkpoint.json")
    if meta is None:
        raise FileNotFoundError("checkpoint/checkpoint.json missing")
    targets = list(meta["targets"])
    notes = []
    if meta["family"] == "phylocnn":
        model, meta = step4.load_trained_model(folder)
        columns = meta["phylocnn_columns"]
        x = tree.cnn_input(dataset)[None][:, :, columns, :]
        if tuple(model.input_shape[1:]) != x.shape[1:]:
            raise ValueError(f"model input {model.input_shape} vs encoding {x.shape}")
        output = np.asarray(model.predict(x, verbose=0), dtype=np.float64)
        if meta.get("target_scaling") == "authors_rescaled":
            scaler = meta["output_scaler"]
            z = output * np.asarray(scaler["std"]) + np.asarray(scaler["mean"])
            values = ge.from_rescaled(z, [tree.rescale_factor(dataset)], targets)[0]
        else:
            scaler = meta["label_scaler"]
            values = (output * np.asarray(scaler["std"]) + np.asarray(scaler["mean"]))[0]
            for j, t in enumerate(targets):
                if t in ge.TIME_SCALED:
                    values[j] = float("nan")
            if any(t in ge.TIME_SCALED for t in targets):
                notes.append("old PhyloCNN checkpoint (natural-unit targets): time-scaled parameters not "
                             "identifiable from the scale-free input; re-run Step 4")
        step4.clear_memory()
    else:
        import torch
        model, meta = step4.load_trained_model(folder, device=device)
        scaler = meta["graph_feature_scaler"]
        node_features = meta.get("node_feature_set", "standard")
        edges = meta.get("graph_edges", "tree")
        graph = tree.graph(dataset) if (node_features, edges) == ("standard", "tree") else             ge.tree_to_graph(tree.newick, tree.sampling, dataset, node_features=node_features, edges=edges)
        gf = (np.asarray(graph.graph_features, np.float64) - np.asarray(scaler["mean"])) / np.asarray(scaler["std"])
        gs = ge.GraphSet.pack([(0, np.zeros(len(targets), np.float32), graph)], [])
        gs = gs.with_graph_features(gf[None].astype(np.float32))
        with torch.no_grad():
            output = model(gs.pyg_batch([0], device)).cpu().numpy().astype(np.float64)
        label = meta.get("output_scaler") or meta["label_scaler"]
        values = output * np.asarray(label["std"]) + np.asarray(label["mean"])
        if meta.get("gnn_targets") == "rescaled":     # PhyloCNN-style targets: back with the tree's r
            values = ge.from_rescaled(values, [tree.rescale_factor(dataset)], targets)
        values = values[0]
    return dict(zip(targets, (float(v) for v in values))), notes, meta


# ============================================================
# ONE RESULTS FOLDER
# ============================================================

def run_folder(name, trees, args, step4, device, likelihood=None):
    folder = RESULTS_ROOT / name
    out = OUT_ROOT / name
    report = Report()
    report.heading(f"STEP 5 EMPIRICAL VALIDATION - {name}   ({folder})")
    config = read_json(folder / "RUN_CONFIG.json") or {}
    if config:
        report(f"Step 4 settings: trees={config.get('trees')}  validation={config.get('validation')}  "
               f"limits train/val/test={config.get('train_limit') or 'all'}/{config.get('val_limit') or 'all'}/"
               f"{config.get('test_limit') or 'all'}")
        if config.get("train_limit"):
            report("[NOTE] these models were trained on a limited number of trees: exploratory estimates only.")

    rows, manifest_runs = [], []
    for case, tree in trees.items():
        for dataset in CASES[case]["datasets"]:
            if dataset not in args.datasets:
                continue
            report.heading(f"{case} under {dataset}   ({CASES[case]['units']}; sampling {tree.sampling:g})", "-")
            published = PUBLISHED.get((case, dataset))
            ood_reported = False
            for model_name in MODELS:
                run = folder / "runs" / dataset / model_name
                passed = read_json(run / "PASS.json")
                record = {"results": name, "case": case, "dataset": dataset, "model": model_name,
                          "family": FAMILY[model_name]}
                if passed is None:
                    rows.append({**record, "status": "no finished run"})
                    continue
                summary = {r.get("split"): r for r in read_csv(run / "summary.csv")}
                test = summary.get("author_test") or {}
                test_nmae = to_float(test.get("macro_normalized_mae"))
                per_target = {r.get("target"): r for r in read_csv(run / "per_target.csv")
                              if r.get("split") == "author_test"}
                paper_name, paper_value = step4.paper_metric(dataset, test)
                performance = {"step4_test_r2": to_float(test.get("macro_r2")),
                               "step4_test_mae": to_float(test.get("macro_mae_original_units")),
                               "paper_metric": paper_name, "step4_test_paper_metric": paper_value,
                               "parameters": passed.get("parameters"),
                               "train_seconds": passed.get("train_seconds")}
                try:
                    start = time.time()
                    values, notes, meta = predict_run(step4, tree, dataset, model_name, run, device, args.threads)
                    seconds = time.time() - start
                except Exception as exc:   # one broken checkpoint must not stop the others
                    rows.append({**record, "status": f"FAILED: {type(exc).__name__}: {str(exc)[:200]}"})
                    report(f"  [FAILED] {model_name}: {type(exc).__name__}: {str(exc)[:200]}")
                    continue
                ranges = training_ranges(run, values.keys())
                z = graph_feature_z(tree, dataset, meta["graph_feature_scaler"]) if meta.get("graph_feature_scaler") else {}
                ood = {k: v for k, v in z.items() if abs(v) > OOD_Z}
                if ood and not ood_reported:
                    report(f"  [NOTE] the tree is unlike the training trees in: "
                           + ", ".join(f"{k} (z={v:+.1f})" for k, v in ood.items()))
                    ood_reported = True
                for target, value in values.items():
                    low, high = ranges.get(target, (float("nan"), float("nan")))
                    flags = [f for f in (range_flag(value, low, high),) if f] + \
                        (notes if target in ge.TIME_SCALED else [])
                    ref = (published or {}).get("values", {}).get(target)
                    rows.append({**record, "status": "ok" if math.isfinite(value) else "not identifiable",
                                 "target": target, "estimate": value, "units": CASES[case]["units"]
                                 if target in ge.TIME_SCALED else "dimensionless",
                                 "train_min": low, "train_max": high, "flags": "; ".join(flags),
                                 "step4_test_nmae": test_nmae, **performance,
                                 "step4_target_test_r2": to_float((per_target.get(target) or {}).get("r2")),
                                 "step4_target_test_paper_metric": (
                                     to_float((per_target.get(target) or {}).get("mae")) if dataset == "BiSSE" else
                                     to_float((per_target.get(target) or {}).get("mape_nonzero_percent")) / 100),
                                 "step4_target_test_nmae": to_float(
                                     (per_target.get(target) or {}).get("normalized_mae")),
                                 "inference_ms": round(1000 * seconds, 1),
                                 "published": ref[0] if ref else float("nan"),
                                 "published_ci_low": ref[1] if ref else float("nan"),
                                 "published_ci_high": ref[2] if ref else float("nan"),
                                 "within_published_ci": (bool(ref[1] <= value <= ref[2])
                                                         if ref and math.isfinite(value) else ""),
                                 "max_abs_graph_feature_z": max((abs(v) for v in z.values()), default=float("nan")),
                                 "rescale_factor": tree.rescale_factor(dataset),
                                 "seconds": round(seconds, 3)})
                if likelihood and case == "Primates" and dataset == "BiSSE":
                    add_likelihood(rows, record, values, likelihood)
                manifest_runs.append({"case": case, "dataset": dataset, "model": model_name,
                                      "checkpoint": str(run / "checkpoint"),
                                      "target_scaling": meta.get("target_scaling", "natural")})
            subset = [r for r in rows if r["case"] == case and r["dataset"] == dataset]
            print_case(report, case, dataset, subset, published)
            bisse = likelihood if (case == "Primates" and dataset == "BiSSE") else None
            print_model_comparisons(report, case, dataset, subset, published, bisse)
            if bisse:
                ranking = print_likelihood_ranking(report, subset, bisse)
                write_csv(out / "likelihood_primates_bisse.csv", ranking)
                if not args.no_figures:
                    try:
                        likelihood_figure(ranking, bisse, out / "figures", args.formats)
                    except ImportError as exc:
                        report(f"[WARN] likelihood figure skipped ({exc})")

    summary = summarise(rows)
    print_summary(report, summary)
    write_csv(out / "estimates.csv", rows)
    write_csv(out / "estimates_wide.csv", wide(rows))
    write_csv(out / "summary.csv", summary)
    manifest = {
        "generated": time.strftime("%Y-%m-%d %H:%M:%S"), "results": name, "results_folder": str(folder),
        "step4_settings": config, "device": device,
        "trees": {case: {"path": str(t.path), "sha256": t.sha256, "tips": t.n_tips, "height": t.height,
                         "sampling": t.sampling, "units": CASES[case]["units"],
                         "rescale_factor": {d: t.rescale_factor(d) for d in CASES[case]["datasets"]},
                         "tip_state_fractions": t.state_fraction if case == "Primates" else None}
                  for case, t in trees.items()},
        "runs": manifest_runs, "published_reference": {f"{c}/{d}": v for (c, d), v in PUBLISHED.items()},
        "versions": versions(),
    }
    out.mkdir(parents=True, exist_ok=True)
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2, default=str) + "\n")
    if not args.no_figures:
        try:
            make_figures(rows, out / "figures", args.formats)
            report(f"Figures:  {out / 'figures'}")
        except ImportError as exc:
            report(f"[WARN] figures skipped ({exc})")
    report(f"Tables:   {out / 'estimates.csv'}, {out / 'summary.csv'}")
    report(f"Manifest: {out / 'manifest.json'}")
    report.save(out / "REPORT.txt")
    return rows


def bisse_reference(tree, report):
    """BiSSE likelihood of the primate tree: maximum likelihood and the published PhyloCNN estimates."""
    from scipy.stats import chi2
    report.heading("PRIMATES - BiSSE likelihood (independent check of the estimates)")
    model = BiSSELikelihood(tree.newick, tree.sampling)
    published = PUBLISHED.get(("Primates", "BiSSE")) if tree.path.name == Path(CASES["Primates"]["tree"]).name else None
    point = {t: v[0] for t, v in published["values"].items()} if published else None
    starts = ([point] if point else []) + [{"turnover": 0.5, "lambda_1": 0.2, "lambda_2": 0.2, "q_01": 0.01},
                                           {"turnover": 0.2, "lambda_1": 0.5, "lambda_2": 0.1, "q_01": 0.05}]
    start = time.time()
    ml, ml_loglik = model.maximum_likelihood(starts)
    reference = {"model": model, "ml": ml, "ml_loglik": ml_loglik,
                 "published": point, "published_loglik": model.loglik_targets(point) if point else float("nan"),
                 "region": float(chi2.ppf(0.95, 4) / 2.0)}       # 95% likelihood-ratio region, 4 parameters
    report(f"Model: lambda_1 (state t_s=1), lambda_2 (t_s=2), mu_i = turnover * lambda_i, q01 = q10, "
           f"sampling {tree.sampling:g}; root weights as in HiSSE ('madfitz'), conditioned on survival")
    report(f"  Maximum likelihood ({time.time() - start:.0f}s): logL = {ml_loglik:.2f}   "
           + "  ".join(f"{t}={fnum(v)}" for t, v in ml.items()))
    if point:
        report(f"  Published PhyloCNN estimates: logL = {reference['published_loglik']:.2f}   "
               + "  ".join(f"{t}={fnum(v)}" for t, v in point.items()))
    report(f"  95% likelihood region: logL within {reference['region']:.2f} of the maximum "
           "(likelihood-ratio test, 4 parameters)")
    return reference


def add_likelihood(rows, record, values, likelihood):
    """BiSSE log-likelihood of one model's estimates, added to that model's rows."""
    loglik = likelihood["model"].loglik_targets(values)
    extra = {"bisse_loglik": loglik,
             "delta_loglik_vs_ml": loglik - likelihood["ml_loglik"],
             "delta_loglik_vs_published_phylocnn": loglik - likelihood["published_loglik"],
             "inside_95_likelihood_region": (bool(loglik - likelihood["ml_loglik"] >= -likelihood["region"])
                                             if math.isfinite(loglik) else "")}
    for row in rows:
        if all(row.get(k) == v for k, v in record.items()) and "target" in row:
            row.update(extra)


def print_model_comparisons(report, case, dataset, rows, published, likelihood):
    """For every model: its estimates next to the original PhyloCNN's, parameter by parameter."""
    by_model = {}
    for r in rows:
        if "target" in r:
            by_model.setdefault(r["model"], []).append(r)
    if not by_model:
        return
    order = sorted(by_model, key=lambda m: (not math.isfinite(by_model[m][0]["step4_test_nmae"]),
                                            by_model[m][0]["step4_test_nmae"]))
    report("")
    report(f"Per-model comparison with the original PhyloCNN - {case} / {dataset}")
    if published:
        report(f"  Original PhyloCNN = {published['label']} ({published['source']})")
    else:
        report("  No published PhyloCNN estimate for this model; only this pipeline's estimates are shown.")
    report("  Neither empirical tree has known parameters: which estimate is closer to the truth cannot be "
           "decided from the estimates themselves.")
    for m in order:
        first = by_model[m][0]
        report("")
        report(f"  {m}  ({FAMILY[m]}; Step 4 test {first.get('paper_metric', 'MRE')} "
               f"{fnum(first.get('step4_test_paper_metric'))}, nMAE {fnum(first['step4_test_nmae'])}, "
               f"R2 {fnum(first['step4_test_r2'])})")
        table = []
        for r in by_model[m]:
            ref = (published or {}).get("values", {}).get(r["target"])
            value = r["estimate"]
            if ref and math.isfinite(value):
                diff = value - ref[0]
                pct = f"{100 * diff / ref[0]:+.1f}%" if ref[0] else ""
                inside = "yes" if ref[1] <= value <= ref[2] else "no"
                original = f"{fnum(ref[0])} [{fnum(ref[1], 3)}, {fnum(ref[2], 3)}]"
                diff = f"{diff:+.4f}"
            else:
                original, diff, pct, inside = (f"{fnum(ref[0])} [{fnum(ref[1], 3)}, {fnum(ref[2], 3)}]"
                                               if ref else "-"), "", "", ""
            table.append([r["target"], original, fnum(value) or "n/a", diff, pct, inside,
                          (r["flags"] or "")[:40], NO_TRUTH])
        report.table(["Parameter", "Original PhyloCNN [95% CI]", "This model", "Difference", "% diff",
                      "In PhyloCNN CI", "Flags", "Closer to truth"], table, indent="    ")
        if likelihood:
            loglik = first.get("bisse_loglik", float("nan"))
            if not math.isfinite(to_float(loglik)):
                report("    BiSSE likelihood: not computable for these estimates (invalid or missing values)")
                continue
            verdict = ("this model's estimates fit the tree BETTER than the published PhyloCNN estimates"
                       if loglik > likelihood["published_loglik"] else
                       "the published PhyloCNN estimates fit the tree better than this model's")
            report(f"    BiSSE log-likelihood: this model {loglik:.2f} | published PhyloCNN "
                   f"{likelihood['published_loglik']:.2f} | maximum likelihood {likelihood['ml_loglik']:.2f}")
            report(f"    -> under the likelihood criterion {verdict} "
                   f"({'inside' if first['inside_95_likelihood_region'] else 'outside'} the 95% likelihood region)")
    if case == "HIV_Zurich":
        report("")
        report("  Independent check for HIV: not computed here. It needs the BDSS (multi-type birth-death) "
               "likelihood, e.g. BEAST2/bdmm; the published PhyloCNN values are the only reference in this script.")


def print_likelihood_ranking(report, rows, likelihood):
    """Every model's BiSSE log-likelihood next to the maximum and the published PhyloCNN."""
    entries = [{"method": "Maximum likelihood (this script; same model as the HiSSE fit)", "kind": "reference",
                **likelihood["ml"], "loglik": likelihood["ml_loglik"]}]
    if likelihood["published"]:
        entries.append({"method": "Original PhyloCNN 2-gen. (published estimates)", "kind": "reference",
                        **likelihood["published"], "loglik": likelihood["published_loglik"]})
    seen = set()
    for r in rows:
        if "bisse_loglik" in r and r["model"] not in seen:
            seen.add(r["model"])
            values = {x["target"]: x["estimate"] for x in rows if x["model"] == r["model"] and "target" in x}
            entries.append({"method": r["model"], "kind": r["family"], **values, "loglik": r["bisse_loglik"],
                            "step4_test_nmae": r["step4_test_nmae"]})
    for e in entries:
        e["delta_vs_ml"] = e["loglik"] - likelihood["ml_loglik"]
        e["inside_95_region"] = bool(math.isfinite(e["loglik"]) and e["delta_vs_ml"] >= -likelihood["region"])
    entries.sort(key=lambda e: -e["loglik"] if math.isfinite(e["loglik"]) else float("inf"))
    best = next((e for e in entries if not e["method"].startswith("Maximum") and math.isfinite(e["loglik"])), None)
    table = []
    for rank, e in enumerate(entries, 1):
        mark = "*" if e is best else ""
        table.append([str(rank), f"{mark} {e['method']}".strip(), e["kind"], fnum(e.get("turnover")),
                      fnum(e.get("lambda_1")), fnum(e.get("lambda_2")), fnum(e.get("q_01")),
                      f"{e['loglik']:.2f}" if math.isfinite(e["loglik"]) else "n/a",
                      f"{e['delta_vs_ml']:+.2f}" if math.isfinite(e["delta_vs_ml"]) else "",
                      "yes" if e["inside_95_region"] else "no"])
        e["rank"], e["best_estimate_method"] = rank, e is best
    report("")
    report("BiSSE likelihood ranking - primate tree (higher logL = the estimates explain the tree and traits better)")
    report.table(["Rank", "Method", "Family", "turnover", "lambda_1", "lambda_2", "q_01", "logL",
                  "dlogL vs ML", "In 95% region"], table, left=3)
    report(f"  * = best estimate other than maximum likelihood itself. 95% region: dlogL >= "
           f"-{likelihood['region']:.2f}. The likelihood is an independent criterion, not the truth.")
    return entries


def likelihood_figure(entries, likelihood, folder, formats):
    analysis = load_module("step6_analysis", "Step6-Analysis.py")
    plt = analysis.pyplot()
    shown = [e for e in entries if math.isfinite(e["loglik"]) and not e["method"].startswith("Maximum")]
    if not shown:
        return
    fig, ax = plt.subplots(figsize=(3.5, 0.6 + 0.17 * len(shown)))
    ys = np.arange(len(shown))[::-1]
    ax.axvspan(-likelihood["region"], 0, color="#e5e5e5", lw=0, zorder=0)
    ax.axvline(0, color="black", lw=0.8)
    for y, e in zip(ys, shown):
        if e["kind"] == "reference":
            ax.plot(e["delta_vs_ml"], y, marker="*", color="black", ms=7, ls="", zorder=3)
        else:
            ax.plot(e["delta_vs_ml"], y, marker=analysis.MARKERS.get(e["kind"], "o"), color=analysis.color(e["method"]),
                    mec="black", mew=0.3, ms=4, ls="", zorder=3)
    ax.set_yticks(ys)
    ax.set_yticklabels(["Original PhyloCNN" if e["kind"] == "reference" else analysis.label(e["method"])
                        for e in shown])
    ax.set_xlabel("log-likelihood - maximum (primate tree, BiSSE)")
    ax.set_title("grey: 95% likelihood region; line: maximum likelihood", fontsize=6.5, loc="left")
    fig.tight_layout()
    analysis.save_figure(plt, fig, folder, "fig_likelihood_Primates_BiSSE", formats)


def print_case(report, case, dataset, rows, published):
    ok = [r for r in rows if "target" in r]
    targets = list(dict.fromkeys(r["target"] for r in ok))
    missing = [r["model"] for r in rows if "target" not in r]
    if not targets:
        report("  No finished runs for this dataset.")
        return
    by_model = {}
    for r in ok:
        by_model.setdefault(r["model"], {})[r["target"]] = r
    order = sorted(by_model, key=lambda m: (not math.isfinite(next(iter(by_model[m].values()))["step4_test_nmae"]),
                                            next(iter(by_model[m].values()))["step4_test_nmae"]))

    # Performance of each model (Step 4, on the authors' simulated test trees with known parameters).
    table = []
    for m in order:
        cells = by_model[m]
        first = next(iter(cells.values()))
        table.append([m, FAMILY[m], f"{int(first['parameters']):,}" if first.get("parameters") else "",
                      fnum(first["step4_test_paper_metric"]), fnum(first["step4_test_nmae"]),
                      fnum(first["step4_test_r2"]), fnum(first["step4_test_mae"])] +
                     [fnum(cells[t]["step4_target_test_paper_metric"]) if t in cells else "" for t in targets] +
                     [fnum(cells[t]["step4_target_test_r2"], 3) if t in cells else "" for t in targets] +
                     [fnum(first["inference_ms"], 1)])
    paper = "MAE" if dataset == "BiSSE" else "MRE"
    report("Model performance on the authors' simulated test trees (Step 4; known parameters) and "
           "time to load the model and predict this tree")
    report(f"  {paper} = the PhyloCNN paper's measure ("
           + ("mean absolute error" if paper == "MAE" else "mean relative error |predicted - true| / true") + ")")
    report.table(["Model", "Family", "Params", f"Test {paper} (paper)", "Test nMAE", "Test R2", "Test MAE"] +
                 [f"{paper} {t}" for t in targets] + [f"R2 {t}" for t in targets] + ["Load+predict ms"],
                 table, left=2)
    report("")

    table = []
    for m in order:
        cells = by_model[m]
        flags = sorted({f for r in cells.values() for f in str(r["flags"]).split("; ") if f})
        table.append([m, FAMILY[m], fnum(next(iter(cells.values()))["step4_test_nmae"])] +
                     [fnum(cells[t]["estimate"]) if t in cells else "" for t in targets] +
                     ["; ".join(f.split(" (")[0] for f in flags)[:70]])
    if published:
        table.append([published["label"], "reference", ""] +
                     [f"{fnum(published['values'][t][0])} [{fnum(published['values'][t][1], 3)}, "
                      f"{fnum(published['values'][t][2], 3)}]" if t in published["values"] else "" for t in targets] +
                     [""])
    report("Estimates (models sorted by their Step 4 author-test normalised MAE; blank = not available)")
    report.table(["Model", "Family", "Test nMAE"] + targets + ["Flags"], table, left=2)
    ranges = {t: next((r for r in ok if r["target"] == t), {}) for t in targets}
    report("  Training range: " + "   ".join(f"{t} [{fnum(ranges[t].get('train_min'), 3)}, "
                                              f"{fnum(ranges[t].get('train_max'), 3)}]" for t in targets))
    if missing:
        report(f"  No result: {', '.join(missing)}")
    if published:
        report(f"  Published: {published['source']}")
        if published.get("note"):
            report(f"  Note: {published['note']}")


def summarise(rows):
    out = []
    keys = list(dict.fromkeys((r["results"], r["case"], r["dataset"], r["target"]) for r in rows if "target" in r))
    for results, case, dataset, target in keys:
        group = [r for r in rows if (r.get("results"), r.get("case"), r.get("dataset"), r.get("target")) ==
                 (results, case, dataset, target) and math.isfinite(to_float(r.get("estimate")))]
        if not group:
            continue
        values = np.array([r["estimate"] for r in group])
        ranked = sorted((r for r in group if math.isfinite(r["step4_test_nmae"])), key=lambda r: r["step4_test_nmae"])
        best = ranked[0] if ranked else None
        best_cnn = next((r for r in ranked if r["family"] == "PhyloCNN"), None)
        best_gnn = next((r for r in ranked if r["family"] != "PhyloCNN"), None)
        top3 = [r["estimate"] for r in ranked[:3]]
        ref = PUBLISHED.get((case, dataset), {}).get("values", {}).get(target)
        out.append({
            "results": results, "case": case, "dataset": dataset, "target": target,
            "units": group[0]["units"], "models": len(group),
            "median": float(np.median(values)), "q25": float(np.percentile(values, 25)),
            "q75": float(np.percentile(values, 75)), "min": float(values.min()), "max": float(values.max()),
            "best_model": best["model"] if best else "", "best_model_estimate": best["estimate"] if best else float("nan"),
            "top3_mean": float(np.mean(top3)) if top3 else float("nan"),
            "best_phylocnn": best_cnn["model"] if best_cnn else "",
            "best_phylocnn_estimate": best_cnn["estimate"] if best_cnn else float("nan"),
            "best_phylognn": best_gnn["model"] if best_gnn else "",
            "best_phylognn_estimate": best_gnn["estimate"] if best_gnn else float("nan"),
            "flagged_models": sum(1 for r in group if r["flags"]),
            "published": ref[0] if ref else float("nan"),
            "published_ci_low": ref[1] if ref else float("nan"), "published_ci_high": ref[2] if ref else float("nan"),
            "models_within_published_ci": sum(1 for r in group if r["within_published_ci"] is True) if ref else "",
        })
    return out


def print_summary(report, summary):
    report.heading("SUMMARY - estimates across models (best = lowest Step 4 author-test nMAE)")
    table = []
    for s in summary:
        table.append([s["case"], s["dataset"], s["target"], str(s["models"]), fnum(s["median"]),
                      f"[{fnum(s['q25'], 3)}, {fnum(s['q75'], 3)}]",
                      f"{s['best_model']} {fnum(s['best_model_estimate'])}".strip(),
                      f"{s['best_phylocnn']} {fnum(s['best_phylocnn_estimate'])}".strip(),
                      f"{s['best_phylognn']} {fnum(s['best_phylognn_estimate'])}".strip(),
                      (f"{fnum(s['published'])} [{fnum(s['published_ci_low'], 3)}, {fnum(s['published_ci_high'], 3)}]"
                       if math.isfinite(s["published"]) else ""),
                      (f"{s['models_within_published_ci']}/{s['models']}" if s["models_within_published_ci"] != "" else ""),
                      str(s["flagged_models"])])
    if table:
        report.table(["Case", "Dataset", "Parameter", "Models", "Median", "IQR", "Best model", "Best PhyloCNN",
                      "Best PhyloGNN", "Published (95% CI)", "In pub. CI", "Flagged"], table, left=3)
    report("Flagged = estimates outside / at the edge of the training range, or not identifiable.")


def wide(rows):
    out = {}
    for r in rows:
        if "target" not in r:
            continue
        key = (r["results"], r["case"], r["dataset"], r["model"])
        row = out.setdefault(key, {"results": r["results"], "case": r["case"], "dataset": r["dataset"],
                                   "model": r["model"], "family": r["family"],
                                   "step4_test_nmae": r["step4_test_nmae"]})
        row[r["target"]] = r["estimate"]
        if r["flags"]:
            row[f"{r['target']}_flags"] = r["flags"]
    return list(out.values())


def versions():
    out = {"python": sys.version.split()[0], "platform": platform.platform(), "numpy": np.__version__,
           "encoding_version": ge.ENCODING_VERSION}
    for module in ("tensorflow", "torch", "torch_geometric"):
        if module in sys.modules:
            out[module] = getattr(sys.modules[module], "__version__", "")
    return out


# ============================================================
# FIGURES
# ============================================================

def make_figures(rows, folder, formats):
    analysis = load_module("step6_analysis", "Step6-Analysis.py")
    plt = analysis.pyplot()
    groups = list(dict.fromkeys((r["case"], r["dataset"]) for r in rows if "target" in r))
    for case, dataset in groups:
        sub = [r for r in rows if r.get("case") == case and r.get("dataset") == dataset and "target" in r]
        targets = list(dict.fromkeys(r["target"] for r in sub))
        models = sorted({r["model"] for r in sub},
                        key=lambda m: next(r["step4_test_nmae"] for r in sub if r["model"] == m))
        published = PUBLISHED.get((case, dataset))
        ncols = min(len(targets), 4)
        fig, axes = plt.subplots(1, ncols, figsize=(min(7.2, 1.15 + 1.6 * ncols), 0.6 + 0.17 * len(models)),
                                 squeeze=False, sharey=True)
        ys = np.arange(len(models))[::-1]
        for k, (ax, target) in enumerate(zip(axes[0], targets)):
            for y, m in zip(ys, models):
                r = next((x for x in sub if x["model"] == m and x["target"] == target), None)
                if r is None or not math.isfinite(r["estimate"]):
                    continue
                ax.plot(r["estimate"], y, ls="", marker=analysis.MARKERS.get(FAMILY[m], "o"),
                        color=analysis.color(m), mec="black", mew=0.3, ms=4,
                        alpha=0.45 if r["flags"] else 1.0, zorder=3)
            low = next((x["train_min"] for x in sub if x["target"] == target), float("nan"))
            high = next((x["train_max"] for x in sub if x["target"] == target), float("nan"))
            if math.isfinite(low) and math.isfinite(high):
                for edge in (low, high):
                    ax.axvline(edge, color="0.55", lw=0.6, ls=":", zorder=1)
            if published and target in published["values"]:
                point, lo, hi = published["values"][target]
                ax.axvspan(lo, hi, color="#d9d9d9", alpha=0.6, lw=0, zorder=0)
                ax.axvline(point, color="black", lw=0.8, zorder=2)
            unit = "" if target not in ge.TIME_SCALED else f" ({CASES[case]['units']})"
            ax.set_xlabel(analysis.TARGET_LABELS.get(target, target) + unit)
            analysis.panel_label(ax, "abcdefgh"[k], x=-0.05 if k else -0.55)
        axes[0][0].set_yticks(ys)
        axes[0][0].set_yticklabels([analysis.label(m) for m in models])
        axes[0][0].set_ylim(-0.7, len(models) - 0.3)
        title = f"{case.replace('_', ' ')} - {dataset}   (dotted: training range; faded: flagged estimate"
        title += "; black line and grey band: published PhyloCNN estimate and 95% CI)" if published else ")"
        fig.suptitle(title, fontsize=7, x=0.01, ha="left")
        fig.tight_layout()
        analysis.save_figure(plt, fig, folder, f"fig_empirical_{case}_{dataset}", formats)


# ============================================================
# MAIN
# ============================================================

def parse_args():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("command", nargs="?", default="run", choices=("run", "check"),
                        help="run (default): estimates of every model; check: only the encoding checks")
    parser.add_argument("--results", default="",
                        help="Step 4 results folder(s), comma separated; blank (default) = every folder in Results/")
    parser.add_argument("--cases", default=",".join(CASES), help="HIV_Zurich,Primates")
    parser.add_argument("--datasets", default="BD,BDEI,BDSS,BiSSE",
                        help="Models to apply (HIV: BD, BDEI, BDSS; Primates: BiSSE)")
    parser.add_argument("--hiv-tree", default="", help="Other HIV tree (Newick/NEXUS) instead of the Zurich tree")
    parser.add_argument("--hiv-sampling", type=float, default=CASES["HIV_Zurich"]["sampling"])
    parser.add_argument("--primates-tree", default="", help="Other primate tree with [&&NHX-t_s=1|2] tip states")
    parser.add_argument("--primates-sampling", type=float, default=CASES["Primates"]["sampling"])
    parser.add_argument("--device", default="cpu", help="cpu (default), auto or cuda (PyTorch models)")
    parser.add_argument("--threads", type=int, default=0, help="CPU threads (0 = Slurm CPUs or all)")
    parser.add_argument("--formats", default="pdf,png")
    parser.add_argument("--no-figures", action="store_true")
    parser.add_argument("--no-likelihood", action="store_true",
                        help="Skip the BiSSE likelihood check of the primate estimates")
    args = parser.parse_args()
    args.cases = [c.strip() for c in args.cases.split(",") if c.strip()]
    args.datasets = [d.strip() for d in args.datasets.split(",") if d.strip()]
    args.formats = [f.strip().lstrip(".") for f in args.formats.split(",") if f.strip()]
    for case in args.cases:
        if case not in CASES:
            parser.error(f"Unknown case {case}; choose from {', '.join(CASES)}")
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


def check_trees(args):
    """Read, validate and encode the empirical trees (PhyloCNN encoding checks)."""
    report = Report()
    report.heading("EMPIRICAL TREES - PhyloCNN encoding checks")
    repo = ge.find_repo(ROOT)
    custom = {"HIV_Zurich": (args.hiv_tree, args.hiv_sampling), "Primates": (args.primates_tree, args.primates_sampling)}
    trees = {}
    for case in args.cases:
        path = Path(custom[case][0]) if custom[case][0] else repo / CASES[case]["tree"]
        if not path.is_file():
            raise SystemExit(f"[ERROR] {case}: tree not found: {path}")
        tree = EmpiricalTree(case, path, custom[case][1], report)
        report(f"{case}: {path}")
        report(f"  {tree.n_tips} tips, height {tree.height:.4g} {CASES[case]['units'].replace('per ', '')}, "
               f"sampling {tree.sampling:g}, SHA-256 {tree.sha256[:16]}...")
        check_encoding(tree, case, report)
        trees[case] = tree
    report.save(OUT_ROOT / "ENCODING_CHECKS.txt")
    return trees


def main():
    args = parse_args()
    if args.command == "check":
        check_trees(args)
        print(f"\nEncoding checks passed: {OUT_ROOT / 'ENCODING_CHECKS.txt'}")
        return 0
    names = results_folders(args.results)
    if not names:
        print(f"No Step 4 results found in {RESULTS_ROOT}")
        return 1
    print(f"Results folders: {', '.join(names)}")
    step4 = load_module("step4", "Step4-CheckPhyloCNN.py")
    if any((RESULTS_ROOT / n / "runs" / d / m).is_dir() for n in names for d in args.datasets for m in GNN_MODELS):
        step4.check_architectures_file()
        import torch
        torch.set_num_threads(step4.cpu_threads(args))
        device = args.device
        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        if device == "cuda" and not torch.cuda.is_available():
            raise SystemExit("[ERROR] --device cuda but PyTorch sees no GPU")
    else:
        device = "cpu"
    step4.tensorflow("cpu" if device == "cpu" else "auto", step4.cpu_threads(args))
    print(f"Device: {device} ({step4.cpu_threads(args)} CPU threads)")
    trees = check_trees(args)

    likelihood = None
    if "Primates" in trees and "BiSSE" in args.datasets and not args.no_likelihood:
        reference_report = Report()
        likelihood = bisse_reference(trees["Primates"], reference_report)
        reference_report.save(OUT_ROOT / "LIKELIHOOD_REFERENCE.txt")

    every = []
    for name in names:
        every.extend(run_folder(name, trees, args, step4, device, likelihood))
    write_csv(OUT_ROOT / "all_estimates.csv", every)
    print(f"\nAll results folders: {OUT_ROOT / 'all_estimates.csv'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
