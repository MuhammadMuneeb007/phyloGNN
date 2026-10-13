#!/usr/bin/env python3
"""
Step7-ExactComparison.py — the authors' own pretrained PhyloCNN models (phyloCNN/Trained_Models)
against our models, on identical test trees and on the empirical trees.

Nothing is trained. For every pretrained model shipped with the PhyloCNN repository:
  1. Simulated test sets (the authors' Test_Sets, the same trees and tree ids as Step 4's author test):
     inference with the authors' encoder and input assembly, outputs converted back with each tree's
     rescaling factor exactly as in the authors' notebooks (periods * r, rates / r), metrics as in
     Step 4 (MAE, RMSE, R2, normalised MAE, ...). A reproduction check marks every pretrained model:
       REPRODUCED      median R2 over its parameters >= 0.5 (the released weights work with the
                       released encoder)
       NOT REPRODUCED  the released weights do not fit the released encoder / inputs; such a model
                       is reported but never used as the reference
  2. Comparison with every Step 4 model of every results folder on the SAME trees: normalised MAE,
     difference to the authors' pretrained PhyloCNN (the best reproduced variant of the dataset),
     paired Wilcoxon test (Holm-corrected), share of trees where our model has the lower error.
  3. Empirical trees: HIV Zurich (sampling 0.25) with the authors' HIV-prior BDSS model
     (Trained_2Generation_BDSS_HIV, the model behind the published HIV estimates), their simulated
     BD/BDEI/BDSS models and their model-selection classifiers; primates (sampling 260/381) with
     their BiSSE models, plus the BiSSE log-likelihood of every estimate (Step 5's validated
     likelihood). Side by side with the published notebook values and with our models' Step 5
     estimates (when Step 5 has been run).

Usage (from the project root, after Step 3; Step 4 / Step 5 results are added when present):
  python Step7-ExactComparison.py run                        # everything, every results folder
  python Step7-ExactComparison.py run --results full100k     # compare with one results folder
  python Step7-ExactComparison.py run --limit 2000           # first 2000 test trees per dataset (quick)
  python Step7-ExactComparison.py run --no-empirical         # simulated test sets only
On Bunya (CPU is enough):  .envs/phylocnn/bin/python Step7-ExactComparison.py run

Outputs: ExactComparison/
  REPORT.txt                       everything printed
  pretrained_test_metrics.csv      every pretrained model x dataset: metrics + reproduction status
  pretrained_per_target.csv        per parameter
  predictions/<dataset>_<variant>.csv   tree_id, true and predicted values
  comparison_<results>.csv         every Step 4 model vs the authors' pretrained PhyloCNN, same trees
  empirical_pretrained.csv         HIV and primate estimates of every pretrained model (+ BiSSE logL)
  hiv_model_selection.csv          probabilities of BD / BDEI / BDSS for the HIV tree
  figures/fig_pretrained_vs_ours.pdf/.png
"""

from __future__ import annotations

import os
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import argparse
import importlib.util
import math
import sys
import time
import warnings
from pathlib import Path

import numpy as np

import Graph_Encoding as ge

warnings.filterwarnings("ignore", category=FutureWarning, message=".*weights_only.*")

ROOT = Path(__file__).resolve().parent
RESULTS_ROOT = ROOT / "Results"
OUT = ROOT / "ExactComparison"
DATASETS = ge.DATASETS
VARIANTS = {"phylocnn_2gen": "2Generation", "phylocnn_1gen": "1Generation", "phylocnn_nocontext": "NoContext"}
VARIANT_LABEL = {"phylocnn_2gen": "2-gen.", "phylocnn_1gen": "1-gen.", "phylocnn_nocontext": "no context"}
FOLDER = {"BD": "Phylodynamics_Simulated", "BDEI": "Phylodynamics_Simulated",
          "BDSS": "Phylodynamics_Simulated", "BiSSE": "BiSSE"}
SELECTION_MODELS = (  # (label, folder, file stem, variant) — output order BD, BDEI, BDSS (notebooks)
    ("PhyDyn 2-gen., HIV priors", "Phylodynamics_HIV", "Trained_2Generation_PhyDyn_HIV", "phylocnn_2gen"),
    ("PhyDyn 2-gen.", "Phylodynamics_Simulated", "Trained_2Generation_PhyDyn", "phylocnn_2gen"),
    ("PhyDyn 1-gen.", "Phylodynamics_Simulated", "Trained_1Generation_PhyDyn", "phylocnn_1gen"),
    ("PhyDyn no context", "Phylodynamics_Simulated", "Trained_NoContext_PhyDyn", "phylocnn_nocontext"),
)
REPRODUCED_R2 = 0.5


def load_module(name, filename):
    path = ROOT / filename
    if not path.is_file():
        raise FileNotFoundError(f"{path} not found (keep the Step files together)")
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


step4 = load_module("step4", "Step4-CheckPhyloCNN.py")
step5 = load_module("step5", "Step5-EmpiricalValidation.py")
step6 = load_module("step6", "Step6-Analysis.py")
Report, fnum, fpval = step6.Report, step6.fnum, step6.fpval
read_csv, read_json, write_csv, to_float = step6.read_csv, step6.read_json, step6.write_csv, step6.to_float


def log(*parts):
    print(time.strftime("%H:%M:%S"), *parts, flush=True)


# ============================================================
# PRETRAINED MODELS
# ============================================================

def _keras2_config(obj):
    """A Keras 3 model JSON made readable by Keras 2 (InputLayer batch_shape, dtype policies)."""
    if isinstance(obj, dict):
        obj = dict(obj)
        if obj.get("class_name") == "InputLayer" and "batch_shape" in obj.get("config", {}):
            config = dict(obj["config"])
            config["batch_input_shape"] = config.pop("batch_shape")
            obj["config"] = config
        if isinstance(obj.get("dtype"), dict):
            obj["dtype"] = obj["dtype"].get("config", {}).get("name", "float32")
        for key in ("module", "registered_name", "build_config", "compile_config"):
            obj.pop(key, None)
        return {k: _keras2_config(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_keras2_config(v) for v in obj]
    return obj


def _load_keras3_weights(model, path):
    """Weights saved by Keras 3 (layers/<name>/vars/<i>). Keras renumbers layer names between
    sessions (conv2d_3 vs conv2d), so layers are matched by type and order: the k-th Conv2D of the
    model gets the k-th saved conv2d, and so on; shapes are checked."""
    import re
    import h5py

    def kind_and_index(name):
        match = re.match(r"^(.*?)(?:_(\d+))?$", name)
        return match.group(1), int(match.group(2) or 0)

    with h5py.File(path, "r") as f:
        saved = {}
        for name in f["layers"].keys():
            kind, index = kind_and_index(name)
            if "vars" in f["layers"][name] and len(f["layers"][name]["vars"]):
                saved.setdefault(kind, []).append((index, name))
        queues = {kind: [n for _, n in sorted(items)] for kind, items in saved.items()}
        for layer in model.layers:
            if not layer.weights:
                continue
            kind = kind_and_index(layer.name)[0]
            if not queues.get(kind):
                raise KeyError(f"no saved weights for layer {layer.name} in {path}")
            group = f["layers"][queues[kind].pop(0)]["vars"]
            values = [group[k][()] for k in sorted(group.keys(), key=int)]
            if [tuple(v.shape) for v in values] != [tuple(w.shape) for w in layer.weights]:
                raise ValueError(f"weight shapes differ for layer {layer.name}")
            layer.set_weights(values)
        left = [n for q in queues.values() for n in q]
        if left:
            raise ValueError(f"unused saved layers: {left}")


def load_pretrained(folder, stem, dataset, variant, n_out):
    """The authors' model: their JSON architecture + H5 weights. Fallbacks for files written by
    another Keras version: (1) Keras 3 JSON/weights converted, (2) the same architecture rebuilt
    (Step 4's build_cnn, ELU output) with the H5 weights."""
    import json
    tf = step4.tensorflow()
    base = ge.find_repo(ROOT) / "Trained_Models" / folder / stem
    h5, js = str(base) + ".h5", Path(str(base) + ".json")
    if not Path(h5).is_file():
        raise FileNotFoundError(f"{h5} not found")
    try:
        model = tf.keras.models.model_from_json(js.read_text())
        model.load_weights(h5)
        return model, "json + h5"
    except Exception as first:
        try:
            model = tf.keras.models.model_from_json(json.dumps(_keras2_config(json.loads(js.read_text()))))
            _load_keras3_weights(model, h5)
            return model, "Keras 3 files converted"
        except Exception:
            if dataset is None:
                raise first
            model, _ = step4.build_cnn(dataset, variant, n_out, "elu")
            model.load_weights(h5)
            return model, f"rebuilt + h5 ({type(first).__name__})"


def predict_batches(model, cnn, columns, batch=512):
    out = []
    for start in range(0, len(cnn), batch):
        x = np.ascontiguousarray(np.asarray(cnn[start:start + batch])[:, :, columns, :], dtype=np.float32)
        out.append(model.predict(x, verbose=0))
    return np.concatenate(out).astype(np.float64)


def reference_scale(dataset, names, y):
    """Normalisation of the errors: the training SD of a Step 4 run (as in Step 4 / Step 6), else the test SD."""
    for name in names:
        for run in sorted((RESULTS_ROOT / name / "runs" / dataset).glob("*/PASS.json")):
            scaler = (read_json(run) or {}).get("label_scaler") or {}
            if scaler.get("std"):
                return np.asarray(scaler["std"], np.float64), f"training SD of {name}/{dataset}"
    std = np.asarray(y, np.float64).std(axis=0)
    return np.where(std > 0, std, 1.0), "SD of the test trees"


def evaluate_pretrained(report, args, names):
    """Every pretrained regression model on the authors' test sets."""
    report.heading("1. AUTHORS' PRETRAINED PhyloCNN ON THE AUTHORS' TEST SETS")
    report(f"Reproduced = median R2 over the parameters >= {REPRODUCED_R2}. Outputs are converted back with each "
           "tree's rescaling factor as in the authors' notebooks.")
    metric_rows, target_rows, results = [], [], {}
    for dataset in args.datasets:
        log(f"Encoding the {dataset} author test set (cached; shared with Step 4)")
        gs = ge.encode_partition(ROOT, dataset, "author_test", limit=args.limit, need_cnn=True,
                                 skip_invalid=True, workers=args.workers,
                                 cache_dir=Path(args.cache_dir) if args.cache_dir else None, log=log)
        targets = ge.TARGETS[dataset]
        y = np.asarray(gs.y, np.float64)
        std, scale_source = reference_scale(dataset, names, y)
        scaler = ge.LabelScaler(np.zeros(len(targets), np.float32), std.astype(np.float32), tuple(targets))
        results[dataset] = {"ids": np.asarray(gs.ids), "y": y, "std": std, "variants": {}}
        report("")
        report(f"{dataset}: {len(gs)} test trees; errors normalised by the {scale_source}")
        table = []
        for variant, tag in VARIANTS.items():
            stem = f"Trained_{tag}_{dataset}"
            columns = ge.cnn_feature_columns(dataset, variant)
            try:
                model, how = load_pretrained(FOLDER[dataset], stem, dataset, variant, len(targets))
                start = time.time()
                output = predict_batches(model, gs.cnn, columns)
                seconds = time.time() - start
            except Exception as exc:
                report(f"  [FAILED] {stem}: {type(exc).__name__}: {str(exc)[:200]}")
                continue
            if output.shape != y.shape:
                report(f"  [FAILED] {stem}: output shape {output.shape}, expected {y.shape}")
                continue
            pred = ge.from_rescaled(output, gs.resc, targets)
            aggregate, per_target = ge.summarize_metrics(y, pred, scaler)
            r2 = [t["r2"] for t in per_target]
            status = "REPRODUCED" if float(np.median(r2)) >= REPRODUCED_R2 else "NOT REPRODUCED"
            results[dataset]["variants"][variant] = {"pred": pred, "status": status,
                                                     "nmae": aggregate["macro_normalized_mae"], "stem": stem}
            paper_label, paper_value = step4.paper_metric(dataset, aggregate)
            per_paper = [t["mae"] if dataset == "BiSSE" else t["mape_nonzero_percent"] / 100 for t in per_target]
            table.append([f"PhyloCNN {VARIANT_LABEL[variant]} (pretrained)", status, fnum(paper_value),
                          fnum(aggregate["macro_normalized_mae"]),
                          fnum(aggregate["macro_mae_original_units"]), fnum(aggregate["macro_r2"])] +
                         [fnum(v) for v in per_paper] + [fnum(v, 3) for v in r2] +
                         [how, f"{1000 * seconds / max(len(y), 1):.2f}"])
            metric_rows.append({"dataset": dataset, "model": f"pretrained_{variant}", "file": stem, "status": status,
                                "trees": len(y), "loaded_as": how, **aggregate})
            target_rows.extend({"dataset": dataset, "model": f"pretrained_{variant}", **t} for t in per_target)
            rows = [{"tree_id": int(i), **{f"true_{t}": float(y[k, j]) for j, t in enumerate(targets)},
                     **{f"pred_{t}": float(pred[k, j]) for j, t in enumerate(targets)}}
                    for k, i in enumerate(np.asarray(gs.ids))]
            write_csv(OUT / "predictions" / f"{dataset}_{variant}.csv", rows)
            step4.clear_memory()
        if table:
            paper = "MAE" if dataset == "BiSSE" else "MRE"
            report.table(["Model", "Status", f"{paper} (paper)", "nMAE", "MAE", "R2"] +
                         [f"{paper} {t}" for t in targets] + [f"R2 {t}" for t in targets] +
                         ["Loaded", "ms/tree"], table, left=2)
        reproduced = {v: d for v, d in results[dataset]["variants"].items() if d["status"] == "REPRODUCED"}
        if reproduced:
            best = min(reproduced, key=lambda v: reproduced[v]["nmae"])
            results[dataset]["reference"] = best
            report(f"  Reference for {dataset}: PhyloCNN {VARIANT_LABEL[best]} (pretrained), the best reproduced variant")
        else:
            results[dataset]["reference"] = None
            report(f"  [NOTE] none of the pretrained {dataset} models reproduces with the released encoder: they are "
                   "not used as a reference. Use the PhyloCNN retrained by Step 4 (authors' protocol) instead.")
    write_csv(OUT / "pretrained_test_metrics.csv", metric_rows)
    write_csv(OUT / "pretrained_per_target.csv", target_rows)
    return results


# ============================================================
# OUR MODELS VS THE PRETRAINED PHYLOCNN (same trees)
# ============================================================

def compare_with_ours(report, names, pretrained, args):
    figure_rows = []
    for name in names:
        report.heading(f"2. {name}: OUR MODELS VS THE AUTHORS' PRETRAINED PhyloCNN (identical test trees)")
        rows_csv = []
        for dataset in args.datasets:
            info = pretrained.get(dataset)
            if not info:
                continue
            ref_variant = info.get("reference")
            ids, y, std = info["ids"], info["y"], info["std"]
            position = {int(i): k for k, i in enumerate(ids)}
            candidates = {}
            for v, d in info["variants"].items():
                candidates[f"pretrained_{v}"] = ("PhyloCNN " + VARIANT_LABEL[v] + " (authors' pretrained)",
                                                 "pretrained", ids, d["pred"], d["status"])
            for run in sorted((RESULTS_ROOT / name / "runs" / dataset).glob("*/PASS.json")):
                folder = run.parent
                rows = read_csv(folder / "predictions_author_test.csv")
                if not rows:
                    continue
                targets = ge.TARGETS[dataset]
                try:
                    tid = np.array([int(float(r["tree_id"])) for r in rows])
                    pred = np.array([[float(r[f"pred_{t}"]) for t in targets] for r in rows])
                except (KeyError, ValueError):
                    continue
                family = step6.family(folder.name)
                note = ""
                if folder.name in step4.CNN_MODELS:
                    sig = (read_json(run) or {}).get("signature") or {}
                    note = f"retrained, {sig.get('cnn_protocol', 'old protocol')}"
                candidates[folder.name] = (step6.label(folder.name), family, tid, pred, note)
            if len(candidates) <= len(info["variants"]):
                report(f"{dataset}: no Step 4 predictions in {name}")
                continue
            common = set(int(i) for i in ids)
            for _, _, tid, _, _ in candidates.values():
                common &= set(int(i) for i in tid)
            common = sorted(common)
            if not common:
                report(f"{dataset}: our models and the pretrained models share no test trees")
                continue
            truth = y[[position[i] for i in common]]
            errors, paper = {}, {}
            for key, (_, _, tid, pred, _) in candidates.items():
                where = {int(i): k for k, i in enumerate(tid)}
                p = pred[[where[i] for i in common]]
                errors[key] = (np.abs(p - truth) / std).mean(axis=1)
                if dataset == "BiSSE":
                    paper[key] = float(np.abs(p - truth).mean())
                else:   # mean relative error per parameter (non-zero true values), averaged over parameters
                    paper[key] = float(np.mean([np.mean(np.abs(p[:, j] - truth[:, j])[truth[:, j] != 0] /
                                                        np.abs(truth[:, j][truth[:, j] != 0]))
                                                for j in range(truth.shape[1])]))
            ref_key = f"pretrained_{ref_variant}" if ref_variant else None
            pairs = [k for k in errors if k != ref_key] if ref_key else []
            p_raw = [step6.wilcoxon(errors[k], errors[ref_key])[0] for k in pairs]
            p_holm = dict(zip(pairs, step6.holm(p_raw))) if pairs else {}
            report("")
            report(f"{dataset}: {len(common)} identical test trees. Reference = "
                   + (f"PhyloCNN {VARIANT_LABEL[ref_variant]} (authors' pretrained)" if ref_key else
                      "none (no pretrained model reproduces)"))
            table = []
            for key in sorted(errors, key=lambda k: errors[k].mean()):
                label, family, _, _, note = candidates[key]
                nmae = float(errors[key].mean())
                if ref_key and key != ref_key:
                    ref = float(errors[ref_key].mean())
                    change = 100 * (nmae - ref) / ref
                    win = float(np.mean(errors[key] < errors[ref_key]))
                    p = p_holm.get(key, float("nan"))
                    verdict = ("better" if change < 0 else "worse") if p < args.alpha else "n.s."
                else:
                    change, win, p, verdict = float("nan"), float("nan"), float("nan"), "reference" if key == ref_key else ""
                table.append([label, family, note, fnum(paper[key]), fnum(nmae),
                              f"{change:+.1f}%" if math.isfinite(change) else "",
                              fnum(100 * win, 1) if math.isfinite(win) else "", fpval(p), verdict])
                rows_csv.append({"results": name, "dataset": dataset, "model": key, "label": label, "family": family,
                                 "note": note, "trees": len(common), "nmae": nmae,
                                 ("mae_paper" if dataset == "BiSSE" else "mre_paper"): paper[key],
                                 "change_vs_pretrained_percent": change, "win_rate_vs_pretrained": win,
                                 "p_holm_vs_pretrained": p, "verdict": verdict})
                figure_rows.append({"results": name, "dataset": dataset, "model": key, "label": label,
                                    "family": family, "nmae": nmae, "pretrained": key.startswith("pretrained_"),
                                    "status": note})
            report.table(["Model", "Family", "Note", f"{'MAE' if dataset == 'BiSSE' else 'MRE'} (paper)", "nMAE",
                          "nMAE vs pretrained", "Win %", "p (Holm)", "Verdict"], table, left=3)
        write_csv(OUT / f"comparison_{name}.csv", rows_csv)
    return figure_rows


# ============================================================
# EMPIRICAL TREES
# ============================================================

def empirical(report, args, names):
    report.heading("3. EMPIRICAL TREES WITH THE AUTHORS' PRETRAINED MODELS")
    repo = ge.find_repo(ROOT)
    trees = {}
    for case in ("HIV_Zurich", "Primates"):
        path = repo / step5.CASES[case]["tree"]
        if path.is_file():
            trees[case] = step5.EmpiricalTree(case, path, step5.CASES[case]["sampling"], report)
            report(f"{case}: {trees[case].n_tips} tips, sampling {trees[case].sampling:g}, rescaling factor "
                   f"{trees[case].rescale_factor(step5.CASES[case]['datasets'][0]):.6f}")
    rows = []

    def estimate(case, dataset, variant, folder, stem, label):
        tree = trees[case]
        targets = ge.TARGETS[dataset]
        try:
            model, how = load_pretrained(folder, stem, dataset, variant, len(targets))
            x = tree.cnn_input(dataset)[None][:, :, ge.cnn_feature_columns(dataset, variant), :]
            output = model.predict(x, verbose=0).astype(np.float64)
        except Exception as exc:
            report(f"  [FAILED] {stem}: {type(exc).__name__}: {str(exc)[:160]}")
            return None
        values = dict(zip(targets, ge.from_rescaled(output, [tree.rescale_factor(dataset)], targets)[0].tolist()))
        rows.append({"case": case, "dataset": dataset, "model": label, "file": stem, "loaded_as": how, **values})
        return values

    if "HIV_Zurich" in trees:
        report("")
        report("HIV Zurich - parameter estimates (years); published = the authors' notebook output (CI_HIV.ipynb)")
        published = step5.PUBLISHED[("HIV_Zurich", "BDSS")]["values"]
        table = []
        value = estimate("HIV_Zurich", "BDSS", "phylocnn_2gen", "Phylodynamics_HIV", "Trained_2Generation_BDSS_HIV",
                         "PhyloCNN 2-gen., HIV priors (pretrained)")
        if value:
            table.append(["PhyloCNN 2-gen., HIV priors (pretrained)", "BDSS"] +
                         [fnum(value.get(t)) for t in ge.TARGETS["BDSS"]])
        table.append(["Published notebook output (same model)", "BDSS"] +
                     [f"{fnum(published[t][0])} [{fnum(published[t][1], 2)}, {fnum(published[t][2], 2)}]"
                      for t in ge.TARGETS["BDSS"]])
        report.table(["Model", "Dataset"] + list(ge.TARGETS["BDSS"]), table, left=2)
        if value:
            diffs = ", ".join(f"{t} {100 * (value[t] - published[t][0]) / published[t][0]:+.1f}%" for t in published)
            report(f"  Reproduction of the published HIV estimates with the released encoder: {diffs}")
        for dataset in ("BD", "BDEI", "BDSS"):
            table = []
            for variant, tag in VARIANTS.items():
                value = estimate("HIV_Zurich", dataset, variant, FOLDER[dataset], f"Trained_{tag}_{dataset}",
                                 f"PhyloCNN {VARIANT_LABEL[variant]} (pretrained, standard priors)")
                if value:
                    table.append([f"PhyloCNN {VARIANT_LABEL[variant]} (pretrained, standard priors)"] +
                                 [fnum(value[t]) for t in ge.TARGETS[dataset]])
            if table:
                report("")
                report(f"HIV Zurich under {dataset} - authors' pretrained models trained on the standard priors")
                report.table(["Model"] + list(ge.TARGETS[dataset]), table)
        selection = []
        for label, folder, stem, variant in SELECTION_MODELS:
            try:
                model, how = load_pretrained(folder, stem, None, variant, 3)
                x = trees["HIV_Zurich"].cnn_input("BDSS")[None][:, :, ge.cnn_feature_columns("BDSS", variant), :]
                p = model.predict(x, verbose=0)[0].astype(np.float64)
            except Exception as exc:
                report(f"  [FAILED] {stem}: {type(exc).__name__}: {str(exc)[:160]}")
                continue
            if p.min() < 0 or abs(p.sum() - 1) > 1e-3:
                p = np.exp(p - p.max()) / np.exp(p - p.max()).sum()
            selection.append({"model": label, "file": stem, "prob_BD": p[0], "prob_BDEI": p[1], "prob_BDSS": p[2],
                              "selected": ("BD", "BDEI", "BDSS")[int(np.argmax(p))]})
        if selection:
            report("")
            report("HIV Zurich - model selection with the authors' pretrained classifiers")
            report.table(["Classifier", "P(BD)", "P(BDEI)", "P(BDSS)", "Selected"],
                         [[s["model"], fnum(s["prob_BD"]), fnum(s["prob_BDEI"]), fnum(s["prob_BDSS"]), s["selected"]]
                          for s in selection])
            write_csv(OUT / "hiv_model_selection.csv", selection)

    if "Primates" in trees:
        report("")
        report("Primates under BiSSE (per million years) - authors' pretrained models and the BiSSE log-likelihood")
        published = step5.PUBLISHED[("Primates", "BiSSE")]["values"]
        likelihood = step5.BiSSELikelihood(trees["Primates"].newick, trees["Primates"].sampling)
        point = {t: v[0] for t, v in published.items()}
        ml, ml_loglik = likelihood.maximum_likelihood([point, {"turnover": 0.5, "lambda_1": 0.2, "lambda_2": 0.2,
                                                                "q_01": 0.01}])
        table = [["Maximum likelihood (BiSSE)"] + [fnum(ml[t]) for t in ge.TARGETS["BiSSE"]] + [f"{ml_loglik:.2f}"],
                 ["Published notebook output (2-gen.)"] + [fnum(point[t]) for t in ge.TARGETS["BiSSE"]] +
                 [f"{likelihood.loglik_targets(point):.2f}"]]
        for variant, tag in VARIANTS.items():
            value = estimate("Primates", "BiSSE", variant, "BiSSE", f"Trained_{tag}_BiSSE",
                             f"PhyloCNN {VARIANT_LABEL[variant]} (pretrained)")
            if value:
                loglik = likelihood.loglik_targets(value)
                rows[-1]["bisse_loglik"] = loglik
                table.append([f"PhyloCNN {VARIANT_LABEL[variant]} (pretrained)"] +
                             [fnum(value[t]) for t in ge.TARGETS["BiSSE"]] +
                             [f"{loglik:.2f}" if math.isfinite(loglik) else "n/a"])
        report.table(["Model"] + list(ge.TARGETS["BiSSE"]) + ["logL"], table)
        report("  n/a = invalid estimates (negative rate or turnover outside [0, 1)).")

    # Our models (Step 5) next to the authors' pretrained ones.
    for name in names:
        estimates = [r for r in read_csv(ROOT / "EmpiricalValidation" / name / "estimates.csv") if r.get("target")]
        if not estimates:
            continue
        report("")
        report(f"Our models ({name}, Step 5): best model by simulated test error per case")
        table = []
        for case, dataset in dict.fromkeys((r["case"], r["dataset"]) for r in estimates):
            sub = [r for r in estimates if r["case"] == case and r["dataset"] == dataset]
            for family_filter, title in ((lambda f: f != "PhyloCNN", "best PhyloGNN"),
                                         (lambda f: f == "PhyloCNN", "best retrained PhyloCNN")):
                models = sorted({r["model"] for r in sub if family_filter(r["family"])},
                                key=lambda m: to_float(next(r["step4_test_nmae"] for r in sub if r["model"] == m)))
                if not models:
                    continue
                best = models[0]
                values = {r["target"]: to_float(r["estimate"]) for r in sub if r["model"] == best}
                loglik = next((to_float(r.get("bisse_loglik")) for r in sub if r["model"] == best), float("nan"))
                table.append([case, dataset, f"{title}: {step6.label(best)}"] +
                             ["  ".join(f"{t}={fnum(v)}" for t, v in values.items()),
                              f"{loglik:.2f}" if math.isfinite(loglik) else ""])
        report.table(["Case", "Dataset", "Model", "Estimates", "BiSSE logL"], table, left=4)
    write_csv(OUT / "empirical_pretrained.csv", rows)


# ============================================================
# FIGURE
# ============================================================

def figure(rows, formats):
    if not rows:
        return
    plt = step6.pyplot()
    for name in dict.fromkeys(r["results"] for r in rows):
        sub = [r for r in rows if r["results"] == name]
        datasets = [d for d in DATASETS if any(r["dataset"] == d for r in sub)]
        kmax = max(sum(1 for r in sub if r["dataset"] == d) for d in datasets)
        fig, axes = step6.grid(plt, len(datasets), row_height=0.55 + 0.17 * kmax)
        for n, (ax, d) in enumerate(zip(axes, datasets)):
            items = sorted((r for r in sub if r["dataset"] == d), key=lambda r: r["nmae"])[::-1]
            ys = np.arange(len(items))
            for y, r in zip(ys, items):
                if r["pretrained"]:
                    ax.plot(r["nmae"], y, marker="*", ms=8, color="black", ls="", zorder=3)
                else:
                    ax.plot(r["nmae"], y, marker=step6.MARKERS.get(r["family"], "o"),
                            color=step6.color(r["model"]), mec="black", mew=0.3, ms=4, ls="", zorder=3)
            ax.set_yticks(ys)
            ax.set_yticklabels([r["label"].replace(" (authors' pretrained)", " *") for r in items])
            ax.set_xlabel("Normalised MAE (identical author-test trees)")
            ax.set_title(d, loc="left", fontweight="bold")
            step6.panel_label(ax, "abcd"[n], x=-0.45 if n % 2 == 0 else -0.42)
        fig.text(0.01, 0.0, "* = the authors' pretrained PhyloCNN (black stars)", fontsize=6)
        fig.tight_layout(rect=(0, 0.02, 1, 1))
        step6.save_figure(plt, fig, OUT / "figures", f"fig_pretrained_vs_ours_{name}", formats)


# ============================================================
# MAIN
# ============================================================

def parse_args():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("command", nargs="?", default="run", choices=("run",))
    parser.add_argument("--results", default="",
                        help="Step 4 results folder(s) to compare with; blank (default) = every folder in Results/")
    parser.add_argument("--datasets", default=",".join(DATASETS))
    parser.add_argument("--limit", type=int, default=0, help="Test trees per dataset (0 = all, as in Step 4)")
    parser.add_argument("--alpha", type=float, default=0.05)
    parser.add_argument("--workers", type=int, default=0, help="Encoding processes (0 = the CPUs Slurm gave the job)")
    parser.add_argument("--threads", type=int, default=0, help="CPU threads (0 = Slurm CPUs or all)")
    parser.add_argument("--cache-dir", default=str(ROOT / "Data" / "_step4_cache"),
                        help="Encoded test trees (shared with Step 4); '' = no cache")
    parser.add_argument("--no-empirical", action="store_true", help="Only the simulated test sets")
    parser.add_argument("--formats", default="pdf,png")
    parser.add_argument("--no-figures", action="store_true")
    args = parser.parse_args()
    args.datasets = [d.strip() for d in args.datasets.split(",") if d.strip()]
    args.formats = [f.strip().lstrip(".") for f in args.formats.split(",") if f.strip()]
    args.workers = args.workers or step4.cpu_threads(args)
    for d in args.datasets:
        if d not in DATASETS:
            parser.error(f"Unknown dataset {d}")
    return args


def main():
    args = parse_args()
    names = step6.results_folders(args.results)
    step4.tensorflow("cpu", step4.cpu_threads(args))
    report = Report()
    report.heading("STEP 7 - EXACT COMPARISON WITH THE AUTHORS' PRETRAINED PhyloCNN")
    report(f"Pretrained models: {ge.find_repo(ROOT) / 'Trained_Models'}")
    report(f"Results folders compared: {', '.join(names) if names else 'none (Step 4 not run yet)'}")
    report(f"Datasets: {', '.join(args.datasets)}   test trees per dataset: {args.limit or 'all'}   device: cpu")
    pretrained = evaluate_pretrained(report, args, names)
    rows = compare_with_ours(report, names, pretrained, args) if names else []
    if not args.no_empirical:
        empirical(report, args, names)
    if not args.no_figures:
        try:
            figure(rows, args.formats)
        except ImportError as exc:
            report(f"[WARN] figure skipped ({exc})")
    report("")
    report(f"Outputs: {OUT}")
    report.save(OUT / "REPORT.txt")
    return 0


if __name__ == "__main__":
    sys.exit(main())
