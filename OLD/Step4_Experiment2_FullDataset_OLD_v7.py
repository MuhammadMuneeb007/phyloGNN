#!/usr/bin/env python3
"""STEP 4 — PhyloGNN Experiment 2, ALL AVAILABLE VALID simulated trees.

Intended scientific protocol:
    Four newly simulated ~100,000-tree datasets (BD/BDEI/BDSS/BiSSE).
    SKIP malformed/missing records. Split 70%/30% of each valid dataset.
    Identical frozen record IDs/indices for all 11 architectures.
    The authors' complete shipped test resources are NEVER read, staged or
    included in optimisation. AFTER all 44 model checkpoints are frozen,
    evaluate all models exactly once on all usable author-provided test trees.
    Three PhyloCNN architectures + eight graph models => 44 tasks.

The project-specific PhyloGNN training backend is invoked through the same
run_experiment/config/model-building functions used by Step 3, with a custom
`dataset_config` pointing to explicit training/validation data. The private
`phylognn` loader implementation was not available for local inspection:
this script FAILS CLOSED if the backend doesn't support this configuration or
cannot PROVE it honoured the exact split. Never treat mere run success as proof.

Public commands (run inside PhyloGNN_FINAL):
    python Step4_Experiment2_FullDataset.py --prepare
    python Step4_Experiment2_FullDataset.py --preflight
    python Step4_Experiment2_FullDataset.py --submit

Internal advanced commands (retained for Slurm compatibility):
    python Step4_Experiment2_FullDataset.py --plan
    python Step4_Experiment2_FullDataset.py --discover
    python Step4_Experiment2_FullDataset.py --prepare
    python Step4_Experiment2_FullDataset.py --preflight
    python Step4_Experiment2_FullDataset.py --backend-probe
    python Step4_Experiment2_FullDataset.py --smoke-submit  # two architecture families
    python Step4_Experiment2_FullDataset.py --submit        # remaining tasks after smoke pass
    python Step4_Experiment2_FullDataset.py --status  # includes GNN recovery audit
    python Step4_Experiment2_FullDataset.py --recover-only --only-family graph
    python Step4_Experiment2_FullDataset.py --resume --only-family graph --ids 3,4,5,6
    python Step4_Experiment2_FullDataset.py --resume    # auto submits CPU external tests for completed author CNNs
    python Step4_Experiment2_FullDataset.py --test-only --ids 1 --test-limit 100  # preview original test
    python Step4_Experiment2_FullDataset.py --test-one 1 --test-limit 100  # login node single model, no training
    python Step4_Experiment2_FullDataset.py --resume --only-family graph
    python Step4_Experiment2_FullDataset.py --resume --only-family graph --ids 3,4,5,6 --force
    python Step4_Experiment2_FullDataset.py --aggregate
    python Step4_Experiment2_FullDataset.py --author-test-audit
    # Original authors external test: auto inference for frozen author CNNs via --resume.
    # General GNN test evaluation still requires a verified inference loader:
    python Step4_Experiment2_FullDataset.py --evaluate-submit \
      --eval-command 'python my_readonly_evaluator.py --checkpoint {checkpoint} --manifest {test_manifest} --output {output}'
    python Step4_Experiment2_FullDataset.py --aggregate-test

Additional switches:
    --source-map ./Step4_sources.json (explicit glob patterns; see README)
    --deep-qc                    parse every Newick using ETE3
    --author-python /.../phylocnn-author/bin/python
    --gnn-python /.../leopard-sdm/bin/python
    --backend-command 'python my_verified_full_data_trainer.py --config {config} ...'

Python stdlib + NumPy + pandas are required for preparation and aggregation.
TensorFlow/Spektral/ETE3 required only in their existing model environments.
"""
from __future__ import annotations

import argparse
import ast
import platform
import csv
import datetime as dt
import glob
import gzip
import hashlib
import inspect
import json
import os
from pathlib import Path
import re
import shlex
import shutil
import subprocess
import sys
import traceback
import tempfile
import runpy
from collections import Counter

try:
    import numpy as np
except ImportError:
    np = None

PROJECT = Path(__file__).resolve().parent
OUTPUT = PROJECT / "results" / "Experiment2_FullDataset"
STATE = PROJECT / ".step4_experiment2"
LOGS = STATE / "logs"
SOURCE_DOC = STATE / "source_spec_frozen.json"
SEED, TOTAL, TRAIN, VAL = 42, 100000, None, None
DATASETS = ("BD", "BDEI", "BDSS", "BiSSE")
AUTHOR = ("author_phylocnn_2gen", "author_phylocnn_1gen", "author_phylocnn_nocontext")
GRAPHS = ("gcn", "graphsage", "gat", "gin", "ecc", "trignn", "gnn_fusion", "graphsage_gat_fusion")
MODELS = AUTHOR + GRAPHS
TARGETS = {
    "BD": ("r_nought", "infectious_period"),
    "BDEI": ("r_nought", "infectious_period", "incubation_period"),
    "BDSS": ("r_nought", "infectious_period", "transmission_factor", "fraction"),
    "BiSSE": ("turnover", "lambda_1", "lambda_2", "q_01"),
}
ALIASES = {
    "r_nought": {"r_nought", "r0", "r_0", "basic_reproduction_number"},
    "infectious_period": {"infectious_period", "infectious_time", "inf_period", "gamma_inverse", "1_gamma"},
    "incubation_period": {"incubation_period", "incubation_factor", "incubation_ratio", "incubation_time", "exposed_period", "e_over_gamma", "epsilon_gamma"},
    "transmission_factor": {"transmission_factor", "x_ss", "xss", "x_transmission", "superspreading_factor"},
    "fraction": {"fraction", "f_ss", "fss", "fraction_1", "superspreader_fraction"},
    "turnover": {"turnover", "tau", "turnover_rate"},
    "lambda_1": {"lambda_1", "lambda1"},
    "lambda_2": {"lambda_2", "lambda2"},
    "q_01": {"q_01", "q01", "q", "transition_rate"},
}
EXCLUDED_PATH_TOKENS = {"test_sets", "test_set", "testset", "test", "heldout", "held_out", "holdout", "experiment1_limitedtest", "experiment2_fulldataset", "trained_models", "__pycache__", ".git", "original_test"}
TREE_ENDINGS = (".nwk", ".nwk.gz", ".newick", ".newick.gz", ".trees", ".trees.gz")
PARAM_ENDINGS = (".csv", ".csv.gz", ".tsv", ".tsv.gz", ".txt", ".txt.gz")
SHARD_PATTERN = re.compile(r"(?:chunk|shard|part|batch|block|job)[_\-. ]*0*(\d+)", re.I)

# Headerless, POST-SIMULATION state-aligned BiSSE parameter order is fixed by
# deeptimelearning/simulators/BiSSE/BiSSE_simulator.R (NOT prior draws).
BISSE_FINAL_COLUMNS = (
    "index", "lambda1", "lambda2", "turnover", "sampling_frac", "tree_size",
    "mu1", "mu2", "net_rate1", "net_rate2", "q01", "q10",
    "lambda2_ratio", "q01_ratio",
)
AUTHOR_TEST_USABLE = {"BD": 10000, "BDEI": 9998, "BDSS": 10000, "BiSSE": 1000}

# Exact archive names verified against manolofperez/phyloCNN/Test_Sets.
# These are EXTERNAL evaluation resources, NEVER training/validation sources.
ORIGINAL_TEST_ARCHIVES = {
    "BD": ("BD_1.nwk.gz", "BD_2.nwk.gz"),
    "BDEI": ("BDEI_1.nwk.gz", "BDEI_2.nwk.gz"),
    "BDSS": ("BDSS_1.nwk.gz", "BDSS_2.nwk.gz"),
    "BiSSE": ("BiSSE.nwk.gz",),
}
ORIGINAL_TEST_ROOT = PROJECT / "PhyloGNN" / "Test_Sets"



def say(*parts):
    print(*parts, flush=True)


def stamp():
    return dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds")


def sha(path: Path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for part in iter(lambda: f.read(4 * 1024 * 1024), b""):
            h.update(part)
    return h.hexdigest()


def save_json(path, obj):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + f".tmp.{os.getpid()}")
    tmp.write_text(json.dumps(obj, indent=2, sort_keys=True, default=str) + "\n")
    os.replace(tmp, path)


def read_json(path):
    return json.loads(Path(path).read_text())


def op(path, mode="rt"):
    path = Path(path)
    return gzip.open(path, mode, newline="") if str(path).endswith(".gz") else open(path, mode, newline="")


def key_name(x):
    return re.sub(r"[^a-z0-9]+", "_", str(x).strip().lower()).strip("_")


def normalized_simulation_id(value, dataset, row):
    """Compare numeric Step1 identifiers without confusing display formats.

    Pandas writes the same ID as `0.0` in simulator trees.tsv and
    `0.000000` in parameter TSV. Verify numerical *integer* equality,
    but never silently round, truncate, or skip a genuinely different ID.
    """
    from decimal import Decimal, InvalidOperation
    try:
        ident = Decimal(str(value).strip())
    except (InvalidOperation, ValueError) as exc:
        raise ValueError(f"{dataset} row {row}: invalid simulation ID {value!r}") from exc
    if not ident.is_finite() or ident != ident.to_integral_value():
        raise ValueError(f"{dataset} row {row}: simulation ID must be a finite integer; observed {value!r}")
    return int(ident)


def belongs(path, dataset):
    parts = [key_name(x) for x in path.parts]
    if dataset.lower() in parts:
        return True
    name = path.name.lower()
    return bool(re.search(r"(?:^|[_\-.])" + dataset.lower() + r"(?:[_\-.]|$)", name))


def excluded(p, root):
    rel = p.relative_to(root)
    return any(key_name(s) in EXCLUDED_PATH_TOKENS for s in rel.parts) or any(s.startswith(".") for s in rel.parts)


def shard_id(p):
    for part in (p.stem, p.name, p.parent.name, p.parent.parent.name):
        m = SHARD_PATTERN.search(part)
        if m:
            return int(m.group(1))
    # Conventional BD_trees_00001.nwk / params_00001.csv names.
    m = re.search(r"(?:^|[_\-.])0*(\d{1,5})(?:\.[^.]+)*(?:\.gz)?$", p.name, re.I)
    return int(m.group(1)) if m else None


def classify_file(p):
    name = p.name.lower()
    if name.endswith(TREE_ENDINGS):
        return "trees"
    if name.endswith(PARAM_ENDINGS):
        if any(s in name for s in ("param", "target", "label")) and not any(s in name for s in ("initial", "design", "draw", "input")):
            return "parameters"
    return None


def all_tasks():
    return [{"id": i*len(MODELS)+j, "dataset": d, "model": m,
             "family": "author" if m in AUTHOR else "graph"}
            for i,d in enumerate(DATASETS) for j,m in enumerate(MODELS)]


def task(idx):
    tasks = all_tasks()
    if not 0 <= idx < len(tasks):
        raise ValueError(f"Task ID out of range: {idx}")
    return tasks[idx]


def task_dir(t):
    return OUTPUT / "models" / t["dataset"] / t["model"]


def marker(t):
    return task_dir(t) / "DONE.json"


def failfile(t):
    return task_dir(t) / "FAILED.json"


def data_dir(d):
    return OUTPUT / "data" / d


def stage_config(d):
    return data_dir(d) / "dataset_config.json"


def split_info(d):
    return data_dir(d) / "split.json"


def actual_step1_chunk_root(root: Path):
    """Discover Step1's ACTUAL output location (not ./results)."""
    checks = [root / ".step1_generate_data" / "chunks", root / "chunks", root]
    for c in checks:
        if all((c / name).is_dir() for name in DATASETS):
            return c
    return None


def step1_pairs(chunk_root: Path, d: str):
    """Require every one of Step1's 100 verified 1000-record chunks.

    BD/BDEI/BDSS: out_00000/trees.tsv (index,tree); params_00000.tsv.
    BiSSE: out_00000/trees.nwk and state-aligned params_final.txt.
    The *original* BiSSE prior draws are not the regression ground truth
    following state relabelling; use simulator-produced params_final.txt.
    """
    pairs = []
    base = chunk_root / d
    existing = sorted(base.glob("out_*"))
    if len(existing) != 100:
        raise RuntimeError(f"{d}: found {len(existing)} Step1 output directories, expected exactly 100")
    for chunk in range(100):
        out = base / f"out_{chunk:05d}"
        t = out / ("trees.nwk" if d == "BiSSE" else "trees.tsv")
        p = (out / "params_final.txt") if d == "BiSSE" else (base / f"params_{chunk:05d}.tsv")
        done = out / "DONE.json"
        for q in (t, p, done):
            if not q.is_file() or q.stat().st_size == 0:
                raise FileNotFoundError(f"{d} chunk {chunk}: missing/empty {q}")
        evidence = read_json(done)
        if evidence.get("model") != d or evidence.get("chunk") != chunk or evidence.get("verified_rows") != 1000:
            raise RuntimeError(f"{d} chunk {chunk}: invalid Step1 DONE verification {done}")
        pairs.append((t.resolve(), p.resolve(), chunk))
    return pairs


def prepared_rows(th, ph, tree_path: Path, param_path: Path, d: str):
    """Read real Step1 formats and verify row-wise simulation IDs."""
    from itertools import zip_longest
    if d == "BiSSE" and param_path.name == "params_final.txt":
        cols = list(BISSE_FINAL_COLUMNS)
        param_rows = csv.reader(ph, delimiter="\t")
        def stream():
            for index, (tree, values) in enumerate(zip_longest(th, param_rows)):
                if tree is None or values is None:
                    raise ValueError(f"{d}: tree/final-parameter count mismatch at row {index}")
                if len(values) != len(cols):
                    raise ValueError(f"{d}: expected {len(cols)} aligned parameter columns in {param_path}, got {len(values)}")
                yield tree.strip(), dict(zip(cols, values))
        return cols, stream()
    if tree_path.name == "trees.tsv":
        cols, param_rows = csv_reader(ph, param_path)
        if "index" not in cols:
            raise ValueError(f"{d}: original Step1 parameter file must retain index: {param_path}")
        trees = csv.DictReader(th, delimiter="\t")
        if not trees.fieldnames or not {"index", "tree"}.issubset(set(trees.fieldnames)):
            raise ValueError(f"{d}: expected original Step1 TSV with index/tree columns: {tree_path}, header={trees.fieldnames}")
        def stream():
            for index, (tr, row) in enumerate(zip_longest(trees, param_rows)):
                if tr is None or row is None:
                    raise ValueError(f"{d}: tree/parameter record counts differ at {index}")
                if len(row) != len(cols):
                    raise ValueError(f"{d}: invalid parameter row at {index}")
                record = dict(zip(cols, row))
                tree_id = normalized_simulation_id(tr["index"], d, index)
                param_id = normalized_simulation_id(record["index"], d, index)
                if tree_id != param_id:
                    raise ValueError(
                        f"{d}: actual simulator ID {tree_id} != parameter ID {param_id} "
                        f"at row {index} ({tr['index']!r} versus {record['index']!r})"
                    )
                yield tr["tree"].strip(), record
        return cols, stream()
    # Explicitly configured flat Newick + CSV files are supported too.
    cols, records = csv_reader(ph, param_path)
    def stream():
        for index, (tree, row) in enumerate(zip_longest(th, records)):
            if tree is None or row is None or len(row) != len(cols):
                raise ValueError(f"{d}: supplied tree/parameter files misaligned at {index}")
            yield tree.strip(), dict(zip(cols, row))
    return cols, stream()


def get_pairs(root: Path, d: str, explicit: dict | None):
    """Return exact tree-parameter file pairs, never guessing row associations."""
    chunk_root = actual_step1_chunk_root(root) if not explicit else None
    if chunk_root is not None:
        return step1_pairs(chunk_root, d)
    if explicit:
        base = explicit.get(d)
        if not isinstance(base, dict):
            raise ValueError(f"source map missing dataset {d}")
        def lookup(field, pattern_field):
            if field in base:
                return [Path(x).expanduser().resolve() if Path(x).is_absolute() else (root / x).resolve() for x in base[field]]
            pattern = base.get(pattern_field)
            if not pattern:
                raise ValueError(f"{d}: specify {field} or {pattern_field} in source map")
            return [Path(x).resolve() for x in sorted(glob.glob(str(root / pattern), recursive=True))]
        trees = lookup("tree_files", "tree_glob")
        params = lookup("parameter_files", "parameter_glob")
    else:
        candidates = [p for p in root.rglob("*") if p.is_file() and not excluded(p, root) and belongs(p.relative_to(root), d)]
        trees = sorted(p for p in candidates if classify_file(p) == "trees")
        params = sorted(p for p in candidates if classify_file(p) == "parameters")
    if not trees or not params:
        raise ValueError(f"{d}: found {len(trees)} tree files and {len(params)} parameter files. Use --source-map to specify exact chunk pairs.")
    if any(not p.is_file() for p in trees + params):
        raise FileNotFoundError(f"{d}: a configured source file is missing")
    if any(
        (any(part in EXCLUDED_PATH_TOKENS for part in map(key_name, p.parts))
         or re.search(r"(?:^|[/_.\-])(?:testsets?|heldout|holdout|test_set)(?:[/_.\-]|$)", str(p).lower()))
        for p in trees + params
    ):
        raise ValueError("SAFETY: source map references a test/held-out resource")
    t_map = {shard_id(p): p for p in trees}
    p_map = {shard_id(p): p for p in params}
    if explicit and all(shard_id(p) is None for p in trees + params) and len(trees)==len(params):
        # Explicit arrays in the source map establish paired file order.
        return [(t,p,i) for i,(t,p) in enumerate(zip(trees,params))]
    if len(t_map) != len(trees) or len(p_map) != len(params) or set(t_map) != set(p_map):
        raise ValueError(f"{d}: ambiguous or unmatched source shard IDs; trees={[p.name for p in trees[:6]]}, parameters={[p.name for p in params[:6]]}. Use a source-map with matching chunk numbers or explicitly ordered arrays without chunk numbers.")
    return [(t_map[i], p_map[i], i) for i in sorted(t_map, key=lambda x: (-1 if x is None else x))]


def csv_reader(handle, filename):
    # Prefer explicit TSV extension; otherwise inspect the header.
    header = handle.readline()
    if not header:
        raise ValueError(f"Empty parameter file: {filename}")
    delim = "\t" if (".tsv" in str(filename) or header.count("\t") > header.count(",")) else ","
    rest = csv.reader(handle, delimiter=delim)
    columns = next(csv.reader([header], delimiter=delim))
    columns = [c.strip().lstrip("\ufeff") for c in columns]
    if len(columns) < 2:
        raise ValueError(f"Parameter file appears headerless/misdelimited: {filename}; header={header[:100]!r}")
    return columns, rest


def match_targets(columns, d):
    original = {key_name(c): c for c in columns}
    target_columns = {}
    for t in TARGETS[d]:
        candidates = [original[a] for a in ALIASES[t] if a in original]
        if len(candidates) != 1:
            raise ValueError(f"{d}: expected exactly one column for {t}, available={columns}, matched={candidates}. Explicit target header mapping is needed.")
        target_columns[t] = candidates[0]
    return target_columns


def tree_digest(tree):
    return hashlib.blake2b(tree.strip().encode("utf-8"), digest_size=16).hexdigest()


def check_newick(s, deep=False, min_tips=0, max_tips=0):
    """Check a *candidate* Newick. Do not require 100k good trees.

    Default policy: use all structurally plausible trees (including shorter
    trees); optional --min-tips/--max-tips enables study-specific tip QC.
    """
    t = (s or "").strip()
    if not t or t.upper() in {"NA", "NAN", "NONE", "NULL"}:
        raise ValueError("NA/empty tree")
    if not t.endswith(";") or not t.startswith("(") or ")" not in t:
        raise ValueError("Incomplete/non-Newick tree")
    if t.count("(") != t.count(")"):
        raise ValueError("Unbalanced Newick parentheses")
    # This is valid for bifurcating and multifurcating Newick with no quoted
    # commas in taxon labels; deep QC may be requested explicitly.
    tips = t.count(",") + 1
    if tips < 2 or (min_tips and tips < min_tips) or (max_tips and tips > max_tips):
        raise ValueError(f"Unexpected tip count {tips}; allowed {min_tips or 2}..{max_tips or 'unbounded'}")
    if deep:
        from ete3 import Tree
        parsed = Tree(t, format=1)
        if len(parsed.get_leaves()) != tips:
            raise ValueError("Parsed tip count mismatch")
        for node in parsed.traverse():
            if not np.isfinite(float(node.dist)) or float(node.dist) < 0:
                raise ValueError("Invalid branch length")
    return tips


def discover(args):
    root = Path(args.data_root).expanduser().resolve()
    if not root.is_dir():
        raise FileNotFoundError(f"--data-root does not exist: {root}")
    explicit = read_json(args.source_map) if args.source_map else None
    spec = {}
    say("SOURCE DISCOVERY — no author test resources permitted")
    for d in DATASETS:
        pairs = get_pairs(root, d, explicit)
        spec[d] = [{"trees": str(t), "parameters": str(p), "shard": k} for t,p,k in pairs]
        say(f"  {d:5s} {len(pairs):4d} paired shards | e.g. {pairs[0][0].name} + {pairs[0][1].name}")
    return root, spec



# -----------------------------------------------------------------------------
# Step1 valid-row QC and targeted resimulation. Never modify original shards.
# A verified 1000-row shard can legitimately contain a literal NA tree;
# this is NOT a usable simulation. Re-run only that parameter vector.
# -----------------------------------------------------------------------------
def audit_simulation_inputs(args):
    """Non-destructive inventory of malformed trees, never silently skip them."""
    _, spec = discover(args)
    out = OUTPUT / 'qc'
    out.mkdir(parents=True, exist_ok=True)
    bad = 0
    for d in DATASETS:
        problems = out / f'{d}_invalid_records.csv'
        seen = set()
        counts = Counter()
        with problems.open('w', newline='') as handle:
            writer = csv.writer(handle)
            writer.writerow(['dataset','shard','row','reason','preview','source'])
            for item in spec[d]:
                t,p,k = Path(item['trees']), Path(item['parameters']), item['shard']
                with op(t) as th, op(p) as ph:
                    cols, records = prepared_rows(th, ph, t, p, d)
                    targets = match_targets(cols,d)
                    for local,(tree,params) in enumerate(records):
                        counts['raw'] += 1
                        try:
                            check_newick(tree, deep=args.deep_qc,min_tips=args.min_tips,max_tips=args.max_tips)
                            for label, column in targets.items():
                                if not np.isfinite(float(params[column])):
                                    raise ValueError(f'Nonfinite {label}')
                            digest = tree_digest(tree)
                            if digest in seen:
                                raise ValueError('Duplicate tree')
                            seen.add(digest)
                            counts['valid'] += 1
                        except (ValueError, TypeError) as exc:
                            counts['invalid'] += 1
                            writer.writerow([d,k,local,str(exc),str(tree)[:100],str(t)])
        say(f'[AUDIT] {d}: raw={counts["raw"]:,} valid={counts["valid"]:,} invalid={counts["invalid"]:,} report={problems}')
        bad += counts['invalid']
    say(f'[AUDIT TOTAL] {bad:,} invalid records. Plain --prepare SKIPS them; it does NOT regenerate any data.')
    return bad


def write_stage(d, pairs, args):
    """Two-pass stream: inventory valid trees, then write shared 70/30 split.

    Invalid tree records (NA, malformed, invalid target or exact duplicates)
    are logged and SKIPPED rather than repaired or padded. All original Step 1
    files stay immutable. Only newly simulated Step 1 paths are opened.
    """
    out = data_dir(d)
    out.mkdir(parents=True, exist_ok=True)
    previous_path = split_info(d)
    if previous_path.is_file():
        previous = read_json(previous_path)
        if previous.get("completed"):
            if previous.get("protocol") != "available_70_30_v2":
                raise RuntimeError(f"{d}: old incompatible frozen split at {previous_path}; archive only Experiment2 data/{d} before rerunning")
            recorded = {(v["trees"],v["parameters"],v["shard"]) for v in previous["source_files"]}
            selected = {(str(t),str(p),k) for t,p,k in pairs}
            needed = [out / f"{part}.{kind}" for part in ("train", "validation")
                      for kind in ("trees.nwk.gz", "parameters.csv.gz", "targets.csv.gz")]
            if recorded != selected:
                raise RuntimeError(f"{d}: source paths changed since split was frozen")
            needed += [out / "all.trees.nwk.gz", out / "all.targets.csv.gz"]
            if all(x.is_file() and x.stat().st_size for x in needed):
                say(f"[SKIP] {d}: previously prepared {previous['total']:,} usable trees "
                    f"({previous['train']:,} train / {previous['validation']:,} val)")
                return
            say(f"[REBUILD STAGING] {d}: existing staged files incomplete; rebuilding from unchanged verified Step1 sources")

    # A single source row is identified by its shard/offset. A row is only
    # excluded after a documented, per-row validity check. Identical trees
    # are excluded to prevent possible train/validation identity leakage.
    accepted = bytearray()
    reasons = Counter()
    source_files = []
    seen = set()
    target_columns = None
    columns = None
    invalid_records = []
    per_shard_counts = Counter()
    low_tips = 0
    raw = 0
    say(f"[SCAN] {d}: identifying available usable trees across {len(pairs)} shards")
    for tree_path, param_path, shard in pairs:
        source_files.append({"trees":str(tree_path),"parameters":str(param_path),
                             "shard":shard,"trees_sha256":sha(tree_path),
                             "parameters_sha256":sha(param_path)})
        n = 0
        with op(tree_path) as th, op(param_path) as ph:
            these_columns, rows = prepared_rows(th,ph,tree_path,param_path,d)
            these_targets = match_targets(these_columns,d)
            if columns is None:
                columns, target_columns = these_columns, these_targets
            if columns != these_columns:
                raise ValueError(f"{d} shard {shard}: inconsistent parameter columns")
            for local, (tree, rec) in enumerate(rows):
                raw += 1
                n += 1
                reason = None
                try:
                    tips = check_newick(tree,deep=args.deep_qc,
                                        min_tips=args.min_tips,max_tips=args.max_tips)
                    if tips < 200:
                        low_tips += 1
                    for target, col in target_columns.items():
                        if not np.isfinite(float(rec[col])):
                            raise ValueError(f"Non-finite target {target}")
                    h = tree_digest(tree)
                    if h in seen:
                        raise ValueError("Exact duplicate tree")
                    seen.add(h)
                except (ValueError,TypeError,KeyError,OverflowError) as exc:
                    reason = str(exc)
                valid = reason is None
                accepted.append(1 if valid else 0)
                if valid:
                    per_shard_counts[shard] += 1
                else:
                    reasons[reason] += 1
                    invalid_records.append((d,shard,local,raw-1,reason,str(tree)[:100],str(tree_path)))
        if n != 1000:
            raise RuntimeError(f"{d} shard={shard}: {n} rows instead of Step1's expected 1,000")
        say(f"  [{d}] shard {shard:03d}/099 raw=1,000 usable={per_shard_counts[shard]:,} "
            f"excluded={1000-per_shard_counts[shard]}")
    if raw != TOTAL:
        raise RuntimeError(f"{d}: expected {TOTAL:,} Step1 records; observed {raw:,}")

    valid_total = int(sum(accepted))
    if valid_total < 2:
        raise RuntimeError(f"{d}: not enough usable data ({valid_total})")
    n_train = int(valid_total * 0.70)
    n_val = valid_total-n_train
    rng = np.random.default_rng(SEED + DATASETS.index(d)*100003)
    perm = rng.permutation(valid_total)
    train_mask = np.zeros(valid_total,dtype=bool)
    train_mask[perm[:n_train]] = True
    train_ids = np.flatnonzero(train_mask).astype(np.int64)
    validation_ids = np.flatnonzero(~train_mask).astype(np.int64)
    assert len(train_ids) == n_train and len(validation_ids) == n_val
    say(f"[SPLIT] {d}: generated={raw:,} accepted={valid_total:,} "
        f"excluded={raw-valid_total:,} train={n_train:,} validation={n_val:,} "
        f"small_trees_lt_200={low_tips:,}")
    for reason,n in reasons.most_common(15):
        say(f"  [EXCLUDE] {n:5,}  {reason}")

    # Stage using temporary files. We publish split.json only *after* all
    # six final data files, split indices, and manifests are ready.
    from contextlib import ExitStack
    tmp_pairs = []
    split_path = out / "split_indices.npz"
    split_temp = out / "split_indices.npz.part"
    invalid_final = out / "excluded_records.csv"
    manifest_final = out / "source_records.csv"
    seen_second = 0
    valid_cursor = 0
    counts = Counter()
    try:
        with ExitStack() as stack:
            handles = {}
            target_writers = {}
            param_writers = {}
            for part in ("train", "validation"):
                for kind in ("trees.nwk.gz", "parameters.csv.gz", "targets.csv.gz"):
                    destination = out/f"{part}.{kind}"
                    temp=out/f"{part}.{kind}.part"
                    tmp_pairs.append((temp,destination))
                    handles[(part,kind)] = stack.enter_context(gzip.open(temp,"wt",newline=""))
                param_writers[part] = csv.DictWriter(handles[(part,"parameters.csv.gz")],fieldnames=columns)
                target_writers[part] = csv.DictWriter(handles[(part,"targets.csv.gz")],fieldnames=list(TARGETS[d]))
                param_writers[part].writeheader()
                target_writers[part].writeheader()
            manifest_tmp = manifest_final.with_suffix(".csv.part")
            excluded_tmp = invalid_final.with_suffix(".csv.part")
            mh = stack.enter_context(open(manifest_tmp,"w",newline=""))
            ih = stack.enter_context(open(excluded_tmp,"w",newline=""))
            mw,iw = csv.writer(mh),csv.writer(ih)
            mw.writerow(["valid_index","original_index","dataset","shard","row_in_shard","partition","tree_blake2b"])
            iw.writerow(["dataset","shard","row_in_shard","original_index","reason","tree_preview","source_file"])
            iw.writerows(invalid_records)
            for tree_path,param_path,shard in pairs:
                with op(tree_path) as th, op(param_path) as ph:
                    _,rows = prepared_rows(th,ph,tree_path,param_path,d)
                    for local,(tree,rec) in enumerate(rows):
                        if not accepted[seen_second]:
                            seen_second += 1
                            continue
                        partition = "train" if train_mask[valid_cursor] else "validation"
                        handles[(partition,"trees.nwk.gz")].write(tree.rstrip("\r\n")+"\n")
                        param_writers[partition].writerow(rec)
                        target_writers[partition].writerow({target:float(rec[col])
                            for target,col in target_columns.items()})
                        mw.writerow([valid_cursor,seen_second,d,shard,local,partition,tree_digest(tree)])
                        counts[partition] += 1
                        valid_cursor += 1
                        seen_second += 1
        if seen_second != raw or valid_cursor != valid_total:
            raise RuntimeError(f"{d}: scan/write counts changed: {seen_second}/{raw}, {valid_cursor}/{valid_total}")
        if counts["train"] != n_train or counts["validation"] != n_val:
            raise RuntimeError(f"{d}: partition count mismatch: {dict(counts)}")
        # Save frozen *valid ordinal* indices; original source row IDs and
        # their correspondence are available in source_records.csv.
        with open(split_temp,"wb") as f:
            np.savez_compressed(f,train=train_ids,validation=validation_ids,
                                original_valid_rows=np.flatnonzero(np.frombuffer(accepted,dtype=np.uint8)))
        for tmp,dest in tmp_pairs:
            os.replace(tmp,dest)
        os.replace(manifest_tmp,manifest_final)
        os.replace(excluded_tmp,invalid_final)
        os.replace(split_temp,split_path)
        # Local phylognn DatasetSpec accepts an all pool with a deterministic
        # tail validation split. The pool order is TRAIN followed by VALIDATION.
        # Hence the last n_val records reproduce the already frozen partitions.
        merged_trees = out / "all.trees.nwk.gz"
        merged_targets = out / "all.targets.csv.gz"
        for kind, output in (("trees.nwk.gz", merged_trees),
                             ("targets.csv.gz", merged_targets)):
            temp = output.with_name(output.name + ".part")
            with gzip.open(temp, "wt", newline="") as fout:
                for part in ("train", "validation"):
                    with gzip.open(out / f"{part}.{kind}", "rt", newline="") as fin:
                        for line_number, line in enumerate(fin):
                            if kind == "targets.csv.gz" and part == "validation" and line_number == 0:
                                continue  # never repeat the CSV header
                            fout.write(line)
            os.replace(temp, output)
        cfg = {
            "name": f"{d}_Experiment2",
            "task": "regression",
            "root": str(out.resolve()),
            "trees": {"all": merged_trees.name},
            "targets": {"all": merged_targets.name},
            "tree_format": "newick",
            "target_columns": list(TARGETS[d]),
            "split": {"strategy": "tail", "test_fraction": 0.0,
                      "validation_fraction": n_val / valid_total},
            "notes": "Step1 newly simulated trees only; train then validation; tail split. No authors test trees.",
        }
        # Validate against the ACTUAL local repository, not a guessed schema.
        sys.path.insert(0, str(PROJECT / "PhyloGNN" / "src"))
        from phylognn.data.spec import DatasetSpec
        DatasetSpec(**cfg)
        save_json(stage_config(d),cfg)
        meta={"completed":True,"protocol":"available_70_30_v2","dataset":d,
              "requested_seed":SEED,"created_utc":stamp(),"raw":raw,
              "total":valid_total,"train":n_train,"validation":n_val,
              "excluded":raw-valid_total,"excluded_by_reason":dict(reasons),
              "below_200_tips_accepted":low_tips,
              "effective_split_seed":SEED+DATASETS.index(d)*100003,
              "source_files":source_files,"split_file_sha256":sha(split_path),
              "dataset_config_sha256":sha(stage_config(d)),
              "manifest_sha256":sha(manifest_final),"excluded_sha256":sha(invalid_final),
              "external_test":"NEVER_READ_IN_PREPARATION"}
        save_json(split_info(d),meta)
        say(f"[PREPARED] {d}: {n_train:,} TRAIN / {n_val:,} VALIDATION; skipped {meta['excluded']:,}")
    except BaseException:
        for temp,_ in tmp_pairs:
            temp.unlink(missing_ok=True)
        (out/"source_records.csv.part").unlink(missing_ok=True)
        (out/"excluded_records.csv.part").unlink(missing_ok=True)
        split_temp.unlink(missing_ok=True)
        raise


def prepare(args):
    if np is None:
        raise RuntimeError("NumPy is required for --prepare")
    root, spec = discover(args)
    STATE.mkdir(parents=True, exist_ok=True)
    # Freeze the exact source-file ordering. Re-running prepare must not silently
    # change source selection while keeping a historical split/results manifest.
    source_spec = {"source_root": str(root), "datasets": spec}
    if SOURCE_DOC.exists():
        previous = read_json(SOURCE_DOC)
        if {k: previous.get(k) for k in source_spec} != source_spec:
            raise RuntimeError("The Step 4 source file list/order differs from the frozen manifest. "
                               "Do NOT mix old and new data; archive .step4_experiment2 "
                               "and Experiment2_FullDataset before intentionally restarting.")
    else:
        save_json(SOURCE_DOC, {**source_spec, "created_utc": stamp()})
    for d in DATASETS:
        pair = [(Path(x["trees"]), Path(x["parameters"]), x["shard"]) for x in spec[d]]
        write_stage(d, pair, args)
    say("[READY] Experiment 2 staged datasets are frozen. Execute --preflight, then --submit.")


def preflight():
    say("="*105)
    say("EXPERIMENT 2 PREFLIGHT — USE ALL VALID SIMULATIONS; EXTERNAL TEST EXCLUDED")
    say("="*105)
    failures=[]
    if not SOURCE_DOC.is_file():
        failures.append("No frozen source manifest: run --prepare")
    for d in DATASETS:
        p=split_info(d)
        if not p.is_file():
            failures.append(f"{d}: data not yet prepared — run --prepare")
            continue
        m=read_json(p)
        n,ntr,nval=m.get("total"),m.get("train"),m.get("validation")
        if (m.get("protocol") != "available_70_30_v2" or
            m.get("completed") is not True or not isinstance(n,int) or
            ntr != int(n*0.7) or nval!=n-ntr or m.get("raw") != TOTAL):
            failures.append(f"{d}: incorrect/old split format")
            continue
        sp=data_dir(d)/"split_indices.npz"
        if not sp.is_file() or sha(sp)!=m.get("split_file_sha256"):
            failures.append(f"{d}: split file missing/changed")
            continue
        with np.load(sp) as z:
            tr=np.asarray(z["train"])
            va=np.asarray(z["validation"])
            originals=np.asarray(z["original_valid_rows"])
            if (len(tr)!=ntr or len(va)!=nval or len(originals)!=n or
                len(np.unique(np.r_[tr,va]))!=n or
                not np.array_equal(np.sort(np.r_[tr,va]),np.arange(n)) or
                len(np.unique(originals))!=n or
                (originals.min()<0 or originals.max()>=TOTAL)):
                failures.append(f"{d}: invalid/overlapping split indices")
        for part in ("train","validation"):
            for kind in ("trees.nwk.gz","parameters.csv.gz","targets.csv.gz"):
                f=data_dir(d)/f"{part}.{kind}"
                if not f.is_file() or f.stat().st_size==0:
                    failures.append(f"{d}: missing {f}")
        for kind in ("trees.nwk.gz", "targets.csv.gz"):
            merged = data_dir(d) / f"all.{kind}"
            if not merged.is_file() or merged.stat().st_size == 0:
                failures.append(f"{d}: missing merged training/validation pool: {merged}")
        if not stage_config(d).is_file() or sha(stage_config(d))!=m.get("dataset_config_sha256"):
            failures.append(f"{d}: changed custom dataset config")
        else:
            sys.path.insert(0, str(PROJECT / "PhyloGNN" / "src"))
            from phylognn.data.spec import DatasetSpec
            validated = DatasetSpec(**read_json(stage_config(d)))
            if (validated.split.strategy != "tail" or
                validated.split.test_fraction != 0 or
                (Path(read_json(stage_config(d))["root"]) / Path(validated.trees["all"][0] if isinstance(validated.trees.get("all"), (list, tuple)) and len(validated.trees["all"]) == 1 else "__INVALID_FROZEN_TREE_LIST__")).resolve() != (data_dir(d) / "all.trees.nwk.gz").resolve()):
                failures.append(f"{d}: invalid non-leaking staged manifest")
        say(f"{d:5s} raw={m['raw']:,} usable={n:,} excluded={m['excluded']:,} "
            f"train={ntr:,} validation={nval:,}")
    for needed in ("Step3_Experiment1_LimitedTest.py","Step2_GNNArchitectures.py"):
        if not (PROJECT/needed).is_file():
            failures.append(f"Missing {needed}")
    if not (PROJECT/"PhyloGNN"/"src").is_dir():
        failures.append("Missing PhyloGNN/src")
    if failures:
        for failure in failures:say("[FAIL]",failure)
        raise RuntimeError(f"Preflight failed ({len(failures)} issues)")
    say("[PASS] 4 independent frozen 70/30 usable-data splits; no original authors' test data")



def worker_preflight(t):
    """
    Lightweight validation for ONE GPU worker.

    This deliberately does NOT call the global controller preflight and does
    not inspect the other software family. It validates only the frozen
    Experiment-2 data required by the current task.

    IMPORTANT:
      split.json stores the usable count under key "total", not "usable".
    """
    d = t["dataset"]

    if np is None:
        raise RuntimeError("NumPy is required inside Experiment 2 workers")

    p = split_info(d)
    if not p.is_file():
        raise RuntimeError(f"{d}: missing frozen split metadata: {p}")

    info = read_json(p)

    n = info.get("total")
    n_train = info.get("train")
    n_val = info.get("validation")

    if info.get("completed") is not True:
        raise RuntimeError(f"{d}: frozen preparation is not marked completed")

    if info.get("protocol") != "available_70_30_v2":
        raise RuntimeError(
            f"{d}: wrong split protocol {info.get('protocol')!r}"
        )

    if not isinstance(n, int) or n <= 0:
        raise RuntimeError(
            f"{d}: invalid usable total in split metadata: {n!r}"
        )

    expected_train = int(n * 0.70)
    expected_val = n - expected_train

    if n_train != expected_train or n_val != expected_val:
        raise RuntimeError(
            f"{d}: frozen 70/30 counts changed: "
            f"total={n}, train={n_train}, validation={n_val}, "
            f"expected={expected_train}/{expected_val}"
        )

    if info.get("raw") != TOTAL:
        raise RuntimeError(
            f"{d}: raw count changed: {info.get('raw')} != {TOTAL}"
        )

    split_file = data_dir(d) / "split_indices.npz"

    if not split_file.is_file() or split_file.stat().st_size == 0:
        raise RuntimeError(f"{d}: missing/empty frozen split file: {split_file}")

    if sha(split_file) != info.get("split_file_sha256"):
        raise RuntimeError(f"{d}: frozen split file hash changed")

    with np.load(split_file) as z:
        if not {"train", "validation", "original_valid_rows"}.issubset(z.files):
            raise RuntimeError(
                f"{d}: split file is missing required arrays: {z.files}"
            )

        tr = np.asarray(z["train"], dtype=np.int64).ravel()
        va = np.asarray(z["validation"], dtype=np.int64).ravel()
        originals = np.asarray(z["original_valid_rows"], dtype=np.int64).ravel()

        if len(tr) != n_train:
            raise RuntimeError(
                f"{d}: train split length {len(tr)} != {n_train}"
            )

        if len(va) != n_val:
            raise RuntimeError(
                f"{d}: validation split length {len(va)} != {n_val}"
            )

        if len(originals) != n:
            raise RuntimeError(
                f"{d}: original_valid_rows length {len(originals)} != {n}"
            )

        combined = np.r_[tr, va]

        if len(np.unique(combined)) != n:
            raise RuntimeError(f"{d}: train/validation split overlaps")

        if not np.array_equal(
            np.sort(combined),
            np.arange(n, dtype=np.int64),
        ):
            raise RuntimeError(
                f"{d}: train/validation indices do not cover 0..{n-1}"
            )

    config = stage_config(d)

    if not config.is_file() or config.stat().st_size == 0:
        raise RuntimeError(f"{d}: missing staged dataset config: {config}")

    if sha(config) != info.get("dataset_config_sha256"):
        raise RuntimeError(f"{d}: staged dataset config hash changed")

    for part in ("train", "validation"):
        for kind in (
            "trees.nwk.gz",
            "parameters.csv.gz",
            "targets.csv.gz",
        ):
            f = data_dir(d) / f"{part}.{kind}"

            if not f.is_file() or f.stat().st_size == 0:
                raise RuntimeError(
                    f"{d}: missing/empty staged file: {f}"
                )

    # Family-specific CODE presence only.
    # Never import the other family in this preflight.
    if t["family"] == "graph":
        if not (PROJECT / "Step2_GNNArchitectures.py").is_file():
            raise RuntimeError("Missing Step2_GNNArchitectures.py")

    if not (PROJECT / "Step3_Experiment1_LimitedTest.py").is_file():
        raise RuntimeError("Missing Step3_Experiment1_LimitedTest.py")

    if not (PROJECT / "PhyloGNN" / "src").is_dir():
        raise RuntimeError("Missing PhyloGNN/src")

    say(
        "[WORKER PREFLIGHT PASS]",
        d,
        t["model"],
        f"usable={n:,}",
        f"train={n_train:,}",
        f"validation={n_val:,}",
        f"family={t['family']}",
    )




def _e2_nonempty_test_partition(npz, key):
    """Reject test records, allowing a scalar zero to mean no test records."""
    name = str(key).lower()

    if "seed" in name or "random_state" in name:
        return False

    value = np.asarray(npz[key])

    if value.ndim > 0:
        return value.size > 0

    scalar = value.item()

    if isinstance(scalar, (int, float, bool, np.number)):
        return not (np.isfinite(scalar) and scalar == 0)

    return True


def verify_backend_indices(run_dir,d):
    """Only accept proof of the staged split; no internal test cases permitted."""
    with np.load(data_dir(d)/"split_indices.npz") as z:
        # The backend sees the concatenated pool: train records first, then val.
        n_train = len(z["train"])
        n_val = len(z["validation"])
        expected_train = np.arange(n_train, dtype=np.int64)
        expected_val = np.arange(n_train, n_train+n_val, dtype=np.int64)
    choices=list(Path(run_dir).rglob("split_indices.npz"))
    # A custom backend can use a *different* field name, but must save the
    # ORIGINAL split IDs from the frozen manifest exactly, not new random IDs.
    for path in choices:
        with np.load(path) as z:
            lower={x.lower():x for x in z.files}
            tr_key=next((v for k,v in lower.items() if "train" in k),None)
            val_key=next((v for k,v in lower.items() if "val" in k),None)
            tests=[v for k,v in lower.items() if k in ("test", "test_idx", "test_indices", "test_ids", "test_sample_ids")]
            if not tr_key or not val_key or any(_e2_nonempty_test_partition(z, k) for k in tests):
                continue
            tr=np.asarray(z[tr_key],dtype=np.int64).ravel()
            va=np.asarray(z[val_key],dtype=np.int64).ravel()
            if np.array_equal(np.sort(tr),expected_train) and np.array_equal(np.sort(va),expected_val):
                return str(path)
    raise RuntimeError(f"{d}: training backend did not prove use of the exact "
                       "frozen valid-record train/validation IDs; refused success")


def backend_probe():
    """Validate local training API/config import, WITHOUT touching any test data.

    WARNING: this does not prove that the underlying loader actually obeys the
    custom dataset_config schema. That requires a real trainer integration test.
    """
    if not stage_config(DATASETS[0]).is_file():
        raise RuntimeError("Run --prepare first")
    from importlib.util import module_from_spec,spec_from_file_location
    script=PROJECT/"Step3_Experiment1_LimitedTest.py"
    if not script.is_file():
        raise FileNotFoundError(script)
    spec=spec_from_file_location("e2_step3_probe",script)
    mod=module_from_spec(spec)
    spec.loader.exec_module(mod)
    mod.apply_runtime_environment()
    mod.activate_repo_source()
    from phylognn.config import cli_overrides, build_config, validate_experiment
    for d in DATASETS:
        overrides=cli_overrides(dataset=d+"-shipped",dataset_config=str(stage_config(d)),
                                model=mod.config_model_name("gcn"),
                                feature_mode=mod.FEATURE_MODE,edge_mode=None,
                                epochs=2,batch_size=64,learning_rate=.001,
                                seed=SEED,hidden_dim=32,num_layers=3,
                                dropout=.1,pooling="mean",attention_heads=4,
                                sage_aggregation="mean",residual=False,
                                phylocnn_variant=None,optimizer="adam",patience=2,
                                loss=None,max_trees=None,validation_fraction=.3,
                                output_dir=str(OUTPUT/"probe"/d),device="auto",
                                workers=None,log_every=1)
        config=build_config(None,overrides)
        errors,warnings=validate_experiment(config)
        for w in warnings: say("[WARN]",d,w)
        if errors: raise RuntimeError(f"{d}: training config rejected: {errors}")
        say("[CONFIG OK]",d,"dataset_config",stage_config(d))
    say("[API PROBE PASS] Model training APIs accept four custom configuration paths")
    say("[IMPORTANT] Actual loader support and exact model split usage have NOT been verified here.")
    say("Run --worker 0 as a single job before --submit; a complete training result must pass split-proof checks.")


def existing_valid(t):
    p = marker(t)
    if not p.exists():return False
    try:
        m=read_json(p)
        if m.get("status") != "SUCCESS" or m.get("dataset") != t["dataset"] or m.get("model") != t["model"]: return False
        if m.get("dataset_config_sha256") != sha(stage_config(t["dataset"])):return False
        if m.get("split_file_sha256") != sha(data_dir(t["dataset"])/"split_indices.npz"):return False
        for key in ("package_metrics", "predictions", "backend_split_proof"):
            f=Path(m.get(key,""))
            if not f.is_file() or f.stat().st_size == 0:return False
        return True
    except Exception:return False


def train_existing_backend(t, command_template=""):
    """
    Train exactly one Experiment-2 task.

    FAMILY ISOLATION
    ----------------
    - Graph workers import Step2_GNNArchitectures.py.
    - Author PhyloCNN workers NEVER import Step2_GNNArchitectures.py.
    - Both families reuse the same PhyloGNN config/training backend.
    """
    # Preserve the existing fast family routes already used on HPC.
    if not command_template and (PROJECT / "Step4_E2_FastFamily.py").is_file():
        if t["family"] == "author":
            from Step4_E2_FastFamily import author_fast_train
            return author_fast_train(t, PROJECT)

    dest = task_dir(t)
    dest.mkdir(parents=True, exist_ok=True)

    conf = stage_config(t["dataset"])

    if command_template:
        argv = shlex.split(
            command_template.format(
                config=str(conf),
                dataset=t["dataset"],
                model=t["model"],
                output=str(dest),
                seed=SEED,
                split=str(
                    data_dir(t["dataset"])
                    /
                    "split_indices.npz"
                ),
            )
        )

        say("[BACKEND]", shlex.join(argv))

        subprocess.run(
            argv,
            check=True,
            cwd=PROJECT,
        )

        return dest

    # ------------------------------------------------------------------
    # Reuse Step3's tested utility functions without using its mixed
    # install_experiment_model_factory() path.
    # ------------------------------------------------------------------
    from importlib.util import (
        module_from_spec,
        spec_from_file_location,
    )

    step3_path = PROJECT / "Step3_Experiment1_LimitedTest.py"

    spec = spec_from_file_location(
        "phylognn_step3_e2_helpers",
        step3_path,
    )

    if spec is None or spec.loader is None:
        raise RuntimeError(
            f"Could not import Step3 helpers from {step3_path}"
        )

    mod = module_from_spec(spec)
    spec.loader.exec_module(mod)

    mod.apply_runtime_environment()
    mod.activate_repo_source()

    ds = t["dataset"] + "-shipped"

    # ------------------------------------------------------------------
    # Build a factory for THIS family only.
    # ------------------------------------------------------------------
    import phylognn.models as model_module

    original_builder = model_module.build_model

    if t["family"] == "graph":
        # Graph-only compatibility path.
        mod.patch_spektral_keras3_compatibility()

        import Step2_GNNArchitectures as step2

        def isolated_builder(
            model_type,
            dataset=None,
            *builder_args,
            **builder_kwargs,
        ):
            return step2.build_graph_model(
                t["model"],
                dataset=dataset,
                **builder_kwargs,
            )

        say(
            "[MODEL FACTORY]",
            "GRAPH ONLY",
            t["model"],
        )

    else:
        # Author-only path.
        # CRITICAL: no Step2 import occurs anywhere in this branch.
        variant = mod.AUTHOR_VARIANT[t["model"]]

        def isolated_builder(
            model_type,
            dataset=None,
            *builder_args,
            **builder_kwargs,
        ):
            if dataset is None:
                raise RuntimeError(
                    "Author PhyloCNN requires dataset object"
                )

            return mod.build_author_phylocnn(
                dataset,
                ds,
                variant,
            )

        say(
            "[MODEL FACTORY]",
            "AUTHOR PHYLOCNN ONLY",
            t["model"],
        )

    model_module.build_model = isolated_builder

    # The local trainer may hold a separately imported build_model symbol.
    try:
        import phylognn.training.adapters as adapters

        if hasattr(adapters, "build_model"):
            adapters.build_model = isolated_builder

    except Exception as exc:
        say(
            "[WARN] adapters build_model patch unavailable:",
            type(exc).__name__,
            exc,
        )

    from phylognn.config import (
        cli_overrides,
        build_config,
        validate_experiment,
    )

    from phylognn.training import run_experiment

    is_author = t["model"] in AUTHOR

    config_model = mod.config_model_name(
        t["model"]
    )

    opts = cli_overrides(
        dataset=None,
        dataset_config=str(conf),
        model=config_model,
        feature_mode=(
            None
            if is_author
            else mod.FEATURE_MODE
        ),
        edge_mode=None,
        epochs=1000,
        batch_size=(
            1024
            if is_author
            else 64
        ),
        learning_rate=.001,
        seed=SEED,
        hidden_dim=(
            None
            if is_author
            else 32
        ),
        num_layers=(
            None
            if is_author
            else 3
        ),
        dropout=(
            None
            if is_author
            else .1
        ),
        pooling=(
            None
            if is_author
            else "mean"
        ),
        attention_heads=(
            None
            if is_author
            else 4
        ),
        sage_aggregation=(
            None
            if is_author
            else "mean"
        ),
        residual=False,
        phylocnn_variant=(
            mod.AUTHOR_VARIANT.get(
                t["model"]
            )
        ),
        optimizer="adam",
        patience=100,
        loss=None,
        max_trees=None,
        validation_fraction=None,
        output_dir=str(dest),
        device="auto",
        workers=None,
        log_every=1,
    )

    cfg = build_config(
        None,
        opts,
    )

    errors, warnings = validate_experiment(
        cfg
    )

    for warning in warnings:
        say(
            "[CONFIG WARNING]",
            warning,
        )

    if errors:
        raise RuntimeError(
            "Config rejected full-data dataset config: "
            +
            "; ".join(errors)
        )

    split_meta = read_json(
        split_info(
            t["dataset"]
        )
    )

    raw = {
        "experiment":
            "Experiment2_FullDataset",

        "source_data_config":
            str(conf),

        "source_data_config_sha256":
            sha(conf),

        "seed":
            SEED,

        "frozen_split_file":
            str(
                data_dir(t["dataset"])
                /
                "split_indices.npz"
            ),

        "train_n":
            split_meta["train"],

        "validation_n":
            split_meta["validation"],

        "test_n":
            0,

        "author_released_weights_loaded":
            False
            if is_author
            else None,

        "execution_family":
            t["family"],

        "python_executable":
            sys.executable,
    }

    if t["family"] == "graph" and not command_template and (PROJECT / "Step4_E2_FastFamily.py").is_file():
        from Step4_E2_FastFamily import install_gnn_training_adapter
        install_gnn_training_adapter(t["dataset"], data_dir(t["dataset"]),
                                     ALIASES, list(TARGETS[t["dataset"]]), cfg)
    if t["family"] == "graph" and not command_template:
        _e2_check_graph_backend_ready()
        _e2_capture_best_weights_in_process(t)

    signature = inspect.signature(
        run_experiment
    )

    accepts_raw = (
        "raw_config"
        in signature.parameters
        or any(
            p.kind
            ==
            inspect.Parameter.VAR_KEYWORD
            for p
            in signature.parameters.values()
        )
    )

    if accepts_raw:
        result = run_experiment(
            cfg,
            raw_config=raw,
        )
    else:
        result = run_experiment(
            cfg
        )

    if getattr(
        result,
        "status",
        None,
    ) != "SUCCESS":
        raise RuntimeError(
            "Backend training status "
            f"{getattr(result, 'status', 'unknown')}: "
            f"{getattr(result, 'error', '')}"
        )

    return Path(
        result.run_dir
    )


def find_result(folder, name):
    matches=list(folder.rglob(name))
    return max(matches,key=lambda p:p.stat().st_mtime) if matches else None


def validate_prediction_count(path, dataset):
    """Reject truncated predictions or an accidental internal test partition.

    Accept either one row per validation tree (wide table) or one row per
    (tree,target) pair (long format), with full validation coverage.
    """
    with open(path, "r", newline="") as handle:
        reader=csv.reader(handle)
        header=next(reader, None)
        if not header: raise RuntimeError("Empty prediction header")
        n=sum(1 for row in reader if row)
    v=read_json(split_info(dataset))["validation"]
    allowed={v,v*len(TARGETS[dataset])}
    if n not in allowed:
        raise RuntimeError(f"Prediction file contains {n:,} records, expected one of {sorted(allowed)} for full validation set; refusing to accept partial/test-subset results")
    return n


def worker(tid,args):
    t=task(tid)
    say("="*110)
    say(f"STEP 4 EXPERIMENT 2 | TASK={tid}/43 | DATASET={t['dataset']} | MODEL={t['model']} | {stamp()}")
    say("NO AUTHOR TEST RESOURCE ACCESSED BY WORKER")
    say("="*110)

    worker_preflight(t)

    if existing_valid(t):
        say("[SKIP] Valid SUCCESS result already exists",marker(t))
        return
    out=task_dir(t)
    out.mkdir(parents=True,exist_ok=True)
    failfile(t).unlink(missing_ok=True)
    meta={**t,"started_utc":stamp(),"host":os.uname().nodename,
          "python":sys.executable,"slurm_job_id":os.getenv("SLURM_JOB_ID"),
          "dataset_config_sha256":sha(stage_config(t["dataset"])),
          "split_file_sha256":sha(data_dir(t["dataset"])/"split_indices.npz"),
          "n_train":read_json(split_info(t["dataset"]))["train"],"n_validation":read_json(split_info(t["dataset"]))["validation"],"n_external_test_seen":0}
    try:
        run_dir=train_existing_backend(t,args.backend_command)
        proof=verify_backend_indices(run_dir,t["dataset"])
        met=find_result(run_dir,"metrics.csv")
        preds=find_result(run_dir,"predictions.csv")
        if met is None or preds is None or met.stat().st_size==0 or preds.stat().st_size==0:
            raise RuntimeError("Trainer must save non-empty metrics.csv and predictions.csv. Validate these are VALIDATION metrics; external author test is not available yet.")
        predicted_n=validate_prediction_count(preds,t["dataset"])
        # Copy outputs into stable files for aggregation; don't overwrite run logs/checkpoints.
        stable=out/"VALIDATION_METRICS.csv"
        shutil.copy2(met,stable) if met.resolve()!=stable.resolve() else None
        predictions_out=out/"VALIDATION_PREDICTIONS.csv"
        if preds.resolve()!=predictions_out.resolve():
            shutil.copy2(preds,predictions_out)
        meta.update({"status":"SUCCESS","finished_utc":stamp(),
                     "run_dir":str(run_dir),"package_metrics":str(stable),
                     "predictions":str(predictions_out),
                     "backend_split_proof":proof,"validation_prediction_rows":predicted_n,
                     "evaluation_partition":"VALIDATION_NOT_EXTERNAL_TEST"})
        save_json(marker(t),meta)
        say("[SUCCESS]",t["dataset"],t["model"],"validated exact frozen 70%/30% available-record indices")
    except BaseException as e:
        meta.update({"status":"FAILED","failed_utc":stamp(),"error":str(e),"traceback":traceback.format_exc()})
        save_json(failfile(t),meta)
        say(meta["traceback"])
        raise


def status():
    say("="*110)
    say("STEP 4 — EXPERIMENT 2 STATUS | all available records, 70% train / 30% validation")
    say("="*110)
    counts=Counter()
    for t in all_tasks():
        state="DONE" if existing_valid(t) else ("FAILED" if failfile(t).exists() else "WAIT")
        counts[state]+=1
        say(f"{t['id']:02d} {t['dataset']:<6} {t['model']:<25} {state}")
    say(f"Completed={counts['DONE']}/44  Failed={counts['FAILED']}  Waiting={counts['WAIT']}")
    say("Author external test status: NOT EVALUATED by Step 4 training worker")


def compact_ids(ids):
    return ",".join(map(str,ids))


def submit(args):
    """
    Submit ONLY incomplete Experiment-2 tasks.

    The controller runs the global data-integrity preflight once.
    GPU workers run only worker_preflight(current_task).

    Software families are routed independently:
      author -> --author-python
      graph  -> --gnn-python

    A missing interpreter for one family does NOT prevent the other family
    from being submitted.
    """
    preflight()

    LOGS.mkdir(
        parents=True,
        exist_ok=True,
    )

    if args.max_parallel < 1:
        raise ValueError(
            "--max-parallel must be >= 1"
        )

    smoke_ids = [0, 3]
    smoke_verified = all(existing_valid(task(i)) for i in smoke_ids)
    if args.smoke_submit:
        candidate_ids = [
            0,  # BD PhyloCNN-2gen
            3,  # BD GCN
        ]

        say("[SMOKE] Starting one PhyloCNN and one GCN first. Run --submit again after they succeed.")

    else:
        candidate_ids = [
            t["id"]
            for t
            in all_tasks()
        ]

    ids = [
        i
        for i
        in candidate_ids
        if not existing_valid(
            task(i)
        )
    ]

    if not ids:
        say(
            "[SKIP] All selected Experiment-2 "
            "tasks already have validated outputs"
        )
        return

    requested_family = getattr(
        args,
        "only_family",
        "all",
    )

    family_specs = (
        (
            "author",
            args.author_python,
        ),
        (
            "graph",
            args.gnn_python,
        ),
    )

    submitted_any = False

    for family, interpreter in family_specs:
        if (
            requested_family != "all"
            and requested_family != family
        ):
            continue

        selected = [
            i
            for i
            in ids
            if task(i)["family"]
            ==
            family
        ]

        if not selected:
            say(
                "[SKIP FAMILY]",
                family,
                "no incomplete tasks",
            )
            continue

        py = Path(
            interpreter
        ).expanduser().resolve()

        if not py.is_file():
            say()
            say("=" * 110)
            say(
                "[DEFER FAMILY]",
                family.upper(),
            )
            say("=" * 110)
            say(
                "Missing interpreter",
                py,
            )
            say(
                "The other software family "
                "will still be submitted."
            )
            say(
                "Deferred task IDs:",
                compact_ids(selected),
            )
            say(
                "Later run:"
            )
            say(
                "  python Step4_Experiment2_FullDataset.py "
                "--resume --only-family "
                + family
            )
            say()
            continue

        extra = ""

        if args.backend_command:
            extra = (
                " --backend-command "
                +
                shlex.quote(
                    args.backend_command
                )
            )

        wrapper = (
            f"{shlex.quote(str(py))} "
            "-u "
            f"{shlex.quote(str(Path(__file__).resolve()))} "
            "--worker $SLURM_ARRAY_TASK_ID"
            f"{extra}"
        )

        array_spec = (
            compact_ids(
                selected
            )
            +
            f"%{args.max_parallel}"
        )

        cmd = [
            "sbatch",
            "--parsable",
            "--job-name",
            f"PhyloGNN_E2_{family}",
            "--array",
            array_spec,
            "--partition",
            args.partition,
            *(["--qos", args.qos] if args.qos else []),
            "--nodes",
            "1",
            "--ntasks",
            "1",
            "--cpus-per-task",
            str(args.cpus),
            "--mem",
            args.mem,
            "--time",
            args.time,
            "--gres",
            args.gres,
            "--chdir",
            str(PROJECT),
            "--output",
            str(
                LOGS
                /
                f"E2_{family}.%A_%a.out"
            ),
            "--error",
            str(
                LOGS
                /
                f"E2_{family}.%A_%a.err"
            ),
            "--export",
            (
                "ALL,"
                "PYTHONUNBUFFERED=1,"
                "TF_CPP_MIN_LOG_LEVEL=2,"
                "OMP_NUM_THREADS="
                +
                str(args.cpus)
            ),
            "--wrap",
            (
                "set -euo pipefail; "
                "echo NODE=$(hostname); "
                "echo TASK=$SLURM_ARRAY_TASK_ID; "
                "echo PYTHON="
                + shlex.quote(str(py))
                + "; "
                "nvidia-smi || true; "
                + wrapper
            ),
        ]

        say(
            "[SUBMIT]",
            family,
            "jobs",
            len(selected),
            "array",
            array_spec,
            "python",
            py,
        )

        proc = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            check=True,
        )

        job_id = (
            proc.stdout
            .strip()
            .split(";")[0]
        )

        submitted_any = True

        say(
            "[SUBMITTED]",
            family,
            "Slurm",
            job_id,
        )

        save_json(
            STATE
            /
            f"last_{family}_submission.json",
            {
                "job_id":
                    job_id,

                "task_ids":
                    selected,

                "array":
                    array_spec,

                "submitted_utc":
                    stamp(),

                "python":
                    str(py),
            },
        )

    if submitted_any:
        say()
        say(
            "[ROUTED SUBMISSION COMPLETE]"
        )
        say(
            "Monitor: squeue --me"
        )
        say(
            "Status : python "
            "Step4_Experiment2_FullDataset.py "
            "--status"
        )
    else:
        say(
            "[NO JOBS SUBMITTED] "
            "No requested family had a usable interpreter "
            "and incomplete tasks."
        )


def aggregate():
    import pandas as pd
    OUTPUT.mkdir(parents=True,exist_ok=True)
    results=[];metrics=[]
    for t in all_tasks():
        if not existing_valid(t):continue
        m=read_json(marker(t))
        results.append(m)
        p=Path(m["package_metrics"])
        f=pd.read_csv(p)
        if f.empty:raise RuntimeError(f"Empty metrics {p}")
        f.insert(0,"Model",t["model"])
        f.insert(0,"Dataset",t["dataset"])
        f.insert(0,"TaskID",t["id"])
        f.insert(3,"EvaluationPartition","INTERNAL_VALIDATION")
        metrics.append(f)
    save_json(OUTPUT/"Experiment2_ValidatedRunManifest.json",results)
    if metrics:
        pd.concat(metrics,ignore_index=True,sort=False).to_csv(OUTPUT/"Experiment2_AllValidationMetrics.csv",index=False)
    say(f"[AGGREGATE] {len(results)}/44 completed; validation metrics only")
    say("[IMPORTANT] Independent authors' test remains unevaluated; do not label validation results 'held-out test'.")


# =============================================================================
# EXTERNAL TEST EVALUATION -- separate, inference-only, full shipped datasets
# =============================================================================
# Never import these paths from any training/validation code. The only commands
# that read this test-map are --evaluate-submit and --evaluate-worker.

def evaluation_dir(t):
    return OUTPUT / "external_test" / t["dataset"] / t["model"]


def external_test_done(t):
    return evaluation_dir(t) / "TEST_DONE.json"


def discover_original_author_tests():
    """Locate ORIGINAL PhyloCNN test archives. Does not train, tune, or load models.

    BD/BDEI/BDSS each consist of TWO gzip Newick archives and ONE gzip
    parameter CSV. BiSSE has ONE tree gzip and ONE parameter gzip.
    NEVER silently omit the second archive or read Step 3 internal folds.
    """
    if not ORIGINAL_TEST_ROOT.is_dir():
        raise FileNotFoundError(
            f"Missing original PhyloCNN Test_Sets directory: {ORIGINAL_TEST_ROOT}. "
            "The checked-in repository lives under PhyloGNN_FINAL/PhyloGNN/Test_Sets."
        )
    found = {}
    for d in DATASETS:
        sub = ORIGINAL_TEST_ROOT / d
        tree_files = [sub / name for name in ORIGINAL_TEST_ARCHIVES[d]]
        params = sub / f"{d}.csv.gz"
        for path in [*tree_files, params]:
            if not path.is_file() or path.stat().st_size == 0:
                raise FileNotFoundError(f"{d}: author test file missing/empty: {path}")
            # Verify gzip readability without loading the full test set.
            with gzip.open(path, "rb") as handle:
                handle.read(1)
        found[d] = {
            "test_source": str(sub.resolve()),
            "tree_files": [str(x.resolve()) for x in tree_files],
            "parameters_file": str(params.resolve()),
            "expected_usable_trees": AUTHOR_TEST_USABLE[d],
            "note": "Original PhyloCNN full test set; expected_usable_trees from E1 loader QC; no internal fold",
        }
    return found


def author_test_audit():
    """Read-only archive inventory, outside the Experiment 2 training workflow.

    Tree counts are physical Newick lines; a separate model data loader must
    align test trees with parameter rows and apply its documented QC before
    inference. This audit does NOT constitute inference/evaluation.
    """
    discovered = discover_original_author_tests()
    say("="*115)
    say("ORIGINAL PHYLOCNN TEST RESOURCES — READ ONLY / NOT TRAINING DATA")
    say("GitHub: https://github.com/manolofperez/phyloCNN/tree/main/Test_Sets")
    say("HPC path:", ORIGINAL_TEST_ROOT)
    say("="*115)
    for d in DATASETS:
        spec=discovered[d]
        n_trees=0
        say(f"[{d}] ALL {len(spec['tree_files'])} tree archives plus matching parameter archive:")
        for source in spec["tree_files"]:
            path=Path(source)
            with gzip.open(path, "rt", encoding="utf-8", errors="replace") as f:
                n=sum(bool(line.strip()) for line in f)
            n_trees += n
            say(f"   TREE   {path.name:<22} records={n:>7,}  compressed={path.stat().st_size:,} bytes")
        parameter=Path(spec["parameters_file"])
        with gzip.open(parameter, "rt", encoding="utf-8", errors="replace") as f:
            first=f.readline().strip()
            n_rows=sum(bool(line.strip()) for line in f)
        # The authors use CSV headers for some test sets. Report both
        # interpretations instead of making an unverified off-by-one claim.
        say(f"   PARAM  {parameter.name:<22} remaining-lines={n_rows:>7,}  first-line={first[:120]!r}")
        say(f"   TOTAL  physical nonempty tree lines={n_trees:,}; E1 usable after loader QC={AUTHOR_TEST_USABLE[d]:,}")
        if n_rows not in (n_trees, n_trees - 1):
            say(f"   [CHECK] Parameter row count does not obviously match tree count; inspect CSV schema/alignment!")
    map_path = STATE / "author_test_sources.json"
    save_json(map_path, discovered)
    say("[FOUND] Exact full author test archive map:", map_path)
    say("[SAFE] No training, validation, checkpoint changes or test inference performed.")
    return discovered


def read_external_test_map(path):
    # Automatic discovery is restricted to the EXTERNAL evaluation functions.
    # Training paths never call this helper.
    obj = read_json(Path(path).expanduser().resolve()) if path else discover_original_author_tests()
    if set(obj) != set(DATASETS):
        raise ValueError(f"Test map must contain exactly {DATASETS}")
    for d,x in obj.items():
        if not isinstance(x,dict):
            raise ValueError(f"{d}: missing test set descriptor")
        source=Path(x.get("test_source", "")).expanduser().resolve()
        if not source.is_dir():
            raise FileNotFoundError(f"{d}: expected ORIGINAL author test folder: {source}")
        if str(source).startswith(str((OUTPUT / "data").resolve())):
            raise RuntimeError(f"{d}: simulated train/validation source is FORBIDDEN as test")
        tree_files=x.get("tree_files")
        if not isinstance(tree_files,list) or len(tree_files)!=len(ORIGINAL_TEST_ARCHIVES[d]):
            raise ValueError(f"{d}: specify ALL tree archives, expected {len(ORIGINAL_TEST_ARCHIVES[d])} (not one/internal test fold)")
        actual_names={Path(z).name for z in tree_files}
        if actual_names != set(ORIGINAL_TEST_ARCHIVES[d]):
            raise ValueError(f"{d}: wrong full original tree archive set: {actual_names}")
        params=Path(x.get("parameters_file", "")).resolve()
        if params.name!=f"{d}.csv.gz":
            raise ValueError(f"{d}: expected {d}.csv.gz parameter archive")
        for file_path in [*tree_files,params]:
            fp=Path(file_path).resolve()
            if not fp.is_file() or fp.stat().st_size==0:
                raise FileNotFoundError(f"{d}: missing/empty author test archive: {fp}")
            if not fp.is_relative_to(source):
                raise ValueError(f"{d}: test archive outside dataset's original Test_Sets folder: {fp}")
        expected=x.get("expected_usable_trees")
        if expected is None or int(expected)<=0:
            raise ValueError(f"{d}: missing expected_usable_trees (after QC)")
        if int(expected)!=AUTHOR_TEST_USABLE[d]:
            say(f"[WARN] {d}: expected {expected}, previously observed post-QC {AUTHOR_TEST_USABLE[d]}; verify loader QC")
    return obj


def find_unique_frozen_checkpoint(t):
    """Find one unambiguous frozen trained model artifact or require an explicit manifest entry."""
    m=read_json(marker(t))
    candidates=[]
    for key in ("checkpoint", "best_checkpoint", "model_path", "weights_path"):
        if m.get(key):
            candidates.append(Path(m[key]))
    if not candidates:
        run_dir=Path(m.get("run_dir", task_dir(t)))
        patterns=("best_model.keras", "best_model.h5", "model.keras", "model.h5", "best.weights.h5", "best_model.weights.h5")
        for pattern in patterns:
            candidates.extend(run_dir.rglob(pattern))
    candidates=[c.resolve() for c in candidates if c.is_file() and c.stat().st_size>0]
    unique=list(dict.fromkeys(candidates))
    if len(unique) != 1:
        raise RuntimeError(f"{t['dataset']}/{t['model']}: expected a SINGLE frozen best checkpoint; found {len(unique)}. Supply exact checkpoint in training DONE.json key 'best_checkpoint'. Candidates: {unique[:8]}")
    return unique[0]


def test_valid(t):
    mfile=external_test_done(t)
    if not mfile.is_file() or not existing_valid(t): return False
    try:
        meta=read_json(mfile)
        if meta.get("status")!="SUCCESS" or meta.get("model")!=t["model"] or meta.get("dataset")!=t["dataset"]: return False
        return all(Path(meta[k]).is_file() and Path(meta[k]).stat().st_size>0
                   for k in ("test_metrics", "test_predictions", "frozen_checkpoint"))
    except Exception:return False


def evaluate_worker(tid, args):
    t=task(tid)
    if not all(existing_valid(other) for other in all_tasks()):
        raise RuntimeError("Test evaluation blocked until ALL 44 training/validation jobs have frozen, validated SUCCESS results; otherwise test results could guide architecture selection")
    tests=read_external_test_map(args.test_map)
    if not args.eval_command:
        raise ValueError("Supply --eval-command for your read-only, inference-only backend. No verified checkpoint loader was uploaded; Step4 will not pretend to evaluate without one.")
    item=tests[t["dataset"]]
    # Save a concrete, explicit full-dataset manifest for the inference backend.
    # The manifest contains BOTH tree shards where applicable + params CSV.
    test_manifest_file=evaluation_dir(t) / "author_test_manifest.json"
    save_json(test_manifest_file, item)
    eval_out=evaluation_dir(t)
    eval_out.mkdir(parents=True,exist_ok=True)
    if test_valid(t):
        say("[SKIP] Completed independently evaluated author test",t["dataset"],t["model"])
        return
    ckpt=find_unique_frozen_checkpoint(t)
    train_run=read_json(marker(t))
    # The template is a single command, no shell interpolation; no model fitting
    # is performed in this wrapper. Evaluator code must implement inference only.
    args_vars={
        "test_source":str(Path(item["test_source"]).resolve()),
        "test_manifest":str(test_manifest_file),
        "test_parameters":str(Path(item["parameters_file"]).resolve()),
        "test_tree_files":" ".join(shlex.quote(str(Path(z).resolve())) for z in item["tree_files"]),
        "test_dataset":t["dataset"]+"-shipped",
        "dataset":t["dataset"], "model":t["model"],
        "checkpoint":str(ckpt), "output":str(eval_out),
        "train_run":str(train_run["run_dir"]),
        "train_config":str(stage_config(t["dataset"])),
        "expected_n":str(int(item["expected_usable_trees"])),
    }
    argv=shlex.split(args.eval_command.format(**args_vars))
    if not argv:
        raise ValueError("Empty evaluation command")
    say("[EVALUATION ONLY]",t["dataset"],t["model"],"FULL authors' test",item["expected_usable_trees"])
    say("[FROZEN MODEL]",ckpt,"sha256",sha(ckpt))
    say("[COMMAND]",shlex.join(argv))
    # Save immutable metadata before prediction; never pass author test paths
    # through the training configuration, nor call run_experiment here.
    before=sha(ckpt)
    subprocess.run(argv,cwd=PROJECT,check=True)
    after=sha(ckpt)
    if before!=after:
        raise RuntimeError("Checkpoint changed during test evaluation -- evaluation is invalid; never optimise on test data")
    preds=eval_out/"test_predictions.csv"
    metrics=eval_out/"test_metrics.csv"
    if not preds.is_file() or not metrics.is_file():
        raise RuntimeError("Inference evaluator must write test_predictions.csv and test_metrics.csv. Neither file found.")
    import pandas as pd
    pf=pd.read_csv(preds)
    mf=pd.read_csv(metrics)
    if mf.empty or len(pf)!=int(item["expected_usable_trees"]):
        raise RuntimeError(f"Test evaluator returned {len(pf)} prediction rows, expected {item['expected_usable_trees']} for entire shipped resource")
    proof={**t,"status":"SUCCESS","evaluated_utc":stamp(),
           "test_source":str(Path(item["test_source"]).resolve()),
           "expected_usable_trees":int(item["expected_usable_trees"]),
           "test_prediction_rows":len(pf),"test_metrics":str(metrics),
           "test_predictions":str(preds),"frozen_checkpoint":str(ckpt),
           "frozen_checkpoint_sha256":before,
           "training_done_sha256":sha(marker(t)),
           "protocol":"POST-TRAINING INFERENCE ONLY - FULL ORIGINAL AUTHORS TEST"}
    save_json(external_test_done(t),proof)
    say("[SUCCESS] Independent full authors' test",t["dataset"],t["model"],"N",len(pf))


def evaluate_submit(args):
    preflight()
    if not all(existing_valid(t) for t in all_tasks()):
        raise RuntimeError("Complete, verify and freeze all 44 training/validation runs before touching author tests")
    read_external_test_map(args.test_map)
    if not args.eval_command:
        raise RuntimeError("Specify --eval-command with a verified INFERENCE ONLY evaluator; no checkpoint/training loader available in uploaded files")
    LOGS.mkdir(parents=True,exist_ok=True)
    todo=[t["id"] for t in all_tasks() if not test_valid(t)]
    if not todo:
        say("[SKIP] All 44 full author tests completed");return
    for family,interpreter in (("author",args.author_python),("graph",args.gnn_python)):
        selected=[i for i in todo if task(i)["family"]==family]
        if not selected:continue
        py=Path(interpreter).resolve()
        if not py.is_file():raise FileNotFoundError(py)
        wrapper=f"{shlex.quote(str(py))} -u {shlex.quote(str(Path(__file__).resolve()))} --evaluate-worker $SLURM_ARRAY_TASK_ID --test-map {shlex.quote(str(Path(args.test_map).resolve()))} --eval-command {shlex.quote(args.eval_command)}"
        cmd=["sbatch","--parsable","--job-name",f"PhyloGNN_E2_TEST_{family}",
             "--array",compact_ids(selected)+f"%{args.max_parallel}","--partition",args.partition,
             "--qos",args.qos,"--nodes","1","--ntasks","1", "--cpus-per-task",str(args.cpus),
             "--mem",args.mem,"--time",args.time,"--gres",args.gres,"--chdir",str(PROJECT),
             "--output",str(LOGS/f"E2_TEST_{family}.%A_%a.out"),
             "--error",str(LOGS/f"E2_TEST_{family}.%A_%a.err"),
             "--wrap",f"set -e; nvidia-smi; {wrapper}"]
        say("[SUBMIT TEST ONLY]",family,"tasks",len(selected),"expected full shipped evaluation")
        print(subprocess.run(cmd,capture_output=True,text=True,check=True).stdout.strip())


def aggregate_test():
    import pandas as pd
    frames=[]
    for t in all_tasks():
        if not test_valid(t):continue
        m=read_json(external_test_done(t))
        f=pd.read_csv(m["test_metrics"])
        f.insert(0,"model",t["model"])
        f.insert(0,"dataset",t["dataset"])
        f.insert(0,"test_count",m["test_prediction_rows"])
        f.insert(0,"source","FULL_ORIGINAL_AUTHORS_TEST")
        frames.append(f)
    say("[INDEPENDENT TEST]",len(frames),"of 44 complete model results")
    if frames:
        pd.concat(frames,ignore_index=True).to_csv(OUTPUT/"Experiment2_AllExternalTestMetrics.csv",index=False)


def plan():
    say("="*95)
    say("STEP 4 — PHYLOGNN EXPERIMENT 2 FULL DATA")
    say("="*95)
    say("Datasets BD/BDEI/BDSS/BiSSE: 100,000 generated each; omit invalid and use 70% train + 30% validation of usable")
    say("Models:",", ".join(MODELS))
    say("44 Slurm tasks: 12 author PhyloCNN (separate interpreter) + 32 GNN")
    say("Seed=42, Adam lr=0.001, epochs<=1000, patience=100, author batch=1024, GNN batch=64")
    say("Two protections: frozen split manifest for every dataset; backend split indices MUST exactly match")
    say("The PhyloCNN authors' external test is NEVER included in staged train/validation partitions")
    say("Outputs:",OUTPUT)
    say("IMPORTANT: --prepare now SKIPS invalid trees; --preflight/--submit only after preparation completes")
    say("Commands: --discover --prepare --preflight --backend-probe --smoke-submit --submit --status --resume --aggregate")
    say("Workers validate only their own frozen dataset; --smoke-submit remains optional, while --submit runs all incomplete tasks")
    say("Source default: .step1_generate_data/chunks/*/out_*/; BiSSE aligned params_final.txt")
    say("Backend contract: MUST honour custom train/val data and save matching split_indices.npz; fail rather than train the authors test data")
    say("External evaluation: AFTER freezing all 44 jobs, --evaluate-submit; ALL author-shipped cases, never an internal fold")
    say("CAUTION: underlying phylognn custom loader and inference-only checkpoint evaluator require verification on HPC")



_original_step4_status = status

def status():
    """Show live epochs where the installed fast family dashboard is available."""
    if (PROJECT / "Step4_E2_FastFamily.py").is_file():
        try:
            from Step4_E2_FastFamily import live_status
            return live_status(sys.modules[__name__])
        except ImportError:
            pass
    return _original_step4_status()


# ===========================================================================
# EXPERIMENT 2 V7 -- RESUME, FORCED RESTART, AND DIRECT EMPTY-TEST BACKEND FIX
# Everything required for the controller and repair resides in this Step4.
# No separate installation or resume Python script is required.
# ===========================================================================

_E2_TEST_GUARD = "E2_SKIP_EMPTY_INTERNAL_TEST_V6"
_E2_BEST_GUARD = "E2_SAVE_BEST_GNN_WEIGHTS_V6"
_E2_BACKEND_ROOT = PROJECT / "PhyloGNN" / "src" / "phylognn" / "training"


def _e2_function_nodes(source):
    return ast.parse(source)


def _e2_backend_plan():
    """Find unambiguous edit points; never guess training/evaluation semantics."""
    adapters = _E2_BACKEND_ROOT / "adapters.py"
    if not adapters.is_file():
        raise FileNotFoundError(f"No backend adapters found: {adapters}")
    adapter_source = adapters.read_text(encoding="utf-8")
    if _E2_BEST_GUARD not in adapter_source:
        exp = re.compile(r'(?m)^(?P<ind>[ \t]*)return pd\.DataFrame\(history\), best_epoch, epoch\s*$')
        matches = list(exp.finditer(adapter_source))
        if len(matches) != 1 or len(matches[0].group('ind')) != 8:
            raise RuntimeError("Cannot safely locate unique GraphAdapter.fit return. No patch applied.")
        m=matches[0]
        insert="""# E2_SAVE_BEST_GNN_WEIGHTS_V6
_recovery_root = __import__("os").environ.get("PHYLOGNN_E2_CHECKPOINT_DIR")
if _recovery_root:
    _recover_dir = __import__("pathlib").Path(_recovery_root)
    _recover_dir.mkdir(parents=True, exist_ok=True)
    _model_spec = self.save(_recover_dir)
    _weight_path = _recover_dir / "weights.weights.h5"
    if not _weight_path.is_file() or not _weight_path.stat().st_size:
        raise RuntimeError("Failed to preserve best GNN weights after fit")
    import json as _e2_json
    _ready = _recover_dir / "E2_CHECKPOINT_READY.json"
    _tmp = _recover_dir / ("E2_CHECKPOINT_READY.json.tmp." + str(__import__("os").getpid()))
    _tmp.write_text(_e2_json.dumps({"best_epoch": int(best_epoch),
        "last_epoch": int(epoch), "checkpoint": str(_weight_path),
        "model_spec": _model_spec}, default=str, indent=2) + "\\n")
    _tmp.replace(_ready)
"""
        replacement=''.join(m.group('ind')+line if line.strip() else line
                            for line in insert.splitlines(keepends=True))
        adapter_source=adapter_source[:m.start()]+replacement+adapter_source[m.start():]
        compile(adapter_source,str(adapters),'exec')
    # E2 V7: trainer.py performs a DIRECT post-fit prediction on splits["test"].
    # There is no role-based evaluation loop; seeking one blocked backend repair.
    # For Experiment 2 only, redirect an EMPTY internal test index to the
    # explicitly frozen validation partition, never the authors independent test.
    testpath = _E2_BACKEND_ROOT / "trainer.py"
    if not testpath.is_file():
        raise FileNotFoundError(f"Missing exact training runner {testpath}")
    testsource = testpath.read_text(encoding="utf-8")
    if _E2_TEST_GUARD not in testsource:
        tree = ast.parse(testsource)
        matches = []
        for node in ast.walk(tree):
            if not isinstance(node, ast.Assign) or len(node.targets) != 1:
                continue
            target = node.targets[0]
            if not (isinstance(target, ast.Tuple) and len(target.elts) == 2
                    and all(isinstance(t, ast.Name) for t in target.elts)
                    and [t.id for t in target.elts] == ["test_role", "test_idx"]):
                continue
            val = node.value
            if not (isinstance(val, ast.Subscript)
                    and isinstance(val.value, ast.Name) and val.value.id == "splits"
                    and isinstance(val.slice, ast.Constant) and val.slice.value == "test"):
                continue
            matches.append(node)
        if len(matches) != 1:
            raise RuntimeError(
                f"Expected exactly one direct test_role,test_idx=splits[\"test\"] "
                f"assignment in trainer.py; found {len(matches)}. No edits applied."
            )
        node = matches[0]
        lines = testsource.splitlines(keepends=True)
        line = lines[node.lineno - 1]
        indent = line[:len(line) - len(line.lstrip(" "))]
        if len(indent) < 4:
            raise RuntimeError("Unexpected direct prediction indentation in trainer.py")
        guard = (
            indent + "# " + _E2_TEST_GUARD + "\n"
            + indent + 'if (__import__("os").environ.get("PHYLOGNN_E2_ALLOW_EMPTY_TEST") == "1"\n'
            + indent + '        and len(test_idx) == 0):\n'
            + indent + '    test_role, test_idx = splits["val"]\n'
            + indent + '    if len(test_idx) == 0:\n'
            + indent + '        raise RuntimeError("Experiment 2 validation partition is empty; refusing evaluation")\n'
            + indent + '    log.info("Experiment 2: empty internal test; evaluating frozen validation indices (NOT authors external test)")\n'
        )
        testsource = ''.join(lines[:node.end_lineno]) + guard + ''.join(lines[node.end_lineno:])
        compile(testsource, str(testpath), 'exec')
        # Prove the redirect is after fit and before predict, not in some other API.
        revised = ast.parse(testsource)
        methods = [n for n in ast.walk(revised) if isinstance(n,(ast.FunctionDef,ast.AsyncFunctionDef))
                   and any(isinstance(c,ast.Call) and isinstance(c.func,ast.Attribute)
                           and c.func.attr == 'fit' for c in ast.walk(n))
                   and any(isinstance(c,ast.Call) and isinstance(c.func,ast.Attribute)
                           and c.func.attr == 'predict' for c in ast.walk(n))]
        if len(methods) != 1 or not (methods[0].lineno <= node.lineno <= methods[0].end_lineno):
            raise RuntimeError("Could not prove the patch belongs to one fit/predict method")
        say('[BACKEND EMPTY-TEST GUARD]', testpath.name, 'line', node.lineno,
            'uses frozen validation only when internal test is empty')
    return {adapters: adapter_source, testpath: testsource}


def _e2_fix_backend(dry_run=False):
    """Transactional validation then backups + atomic changes, integrated into Step4."""
    proposed=_e2_backend_plan()
    from datetime import datetime
    edits=[]
    for path,changed in proposed.items():
        old=path.read_text(encoding='utf-8')
        compile(changed,str(path),'exec')
        if old!=changed:
            edits.append((path,old,changed))
    if not edits:
        say('[BACKEND READY] checkpoint-before-evaluation and skip-empty-test guards installed')
        return
    if dry_run:
        for path,old,changed in edits:
            say('[WOULD FIX BACKEND]',path)
        return
    timestamp=datetime.utcnow().strftime('%Y%m%dT%H%M%SZ')
    for path,old,changed in edits:
        backup=path.with_name(path.name+'.before_E2_V6_'+timestamp)
        if backup.exists():
            raise FileExistsError(backup)
        shutil.copy2(path,backup)
        say('[BACKED UP]',backup)
    for path,old,changed in edits:
        tmp=path.with_name(path.name+'.e2tmp.'+str(os.getpid()))
        try:
            tmp.write_text(changed,encoding='utf-8')
            os.replace(tmp,path)
        finally:
            tmp.unlink(missing_ok=True)
        say('[FIXED BACKEND]',path)
    say('[READY] New Slurm workers will evaluate frozen validation when internal test is empty, and preserve graph best weights after fit')


def _e2_check_graph_backend_ready():
    adapter=_E2_BACKEND_ROOT/'adapters.py'
    if not adapter.is_file() or _E2_BEST_GUARD not in adapter.read_text(encoding='utf-8'):
        raise RuntimeError('GNN best-weight checkpoint guard not installed. Run --resume from login first.')
    if not any(_E2_TEST_GUARD in p.read_text(encoding='utf-8')
               for p in _E2_BACKEND_ROOT.glob('*.py') if p.name!='adapters.py'):
        raise RuntimeError('GNN internal test-skip guard not installed. Run --resume from login first.')


def _e2_capture_best_weights_in_process(t):
    """Choose output path BEFORE fit; monkeypatch not needed after adapters guard."""
    os.environ['PHYLOGNN_E2_ALLOW_EMPTY_TEST']='1'
    os.environ['PHYLOGNN_E2_CHECKPOINT_DIR']=str((task_dir(t)/'recovery_checkpoint').resolve())


def _e2_parse_ids(s):
    if not s: return set()
    chosen=set()
    for part in s.split(','):
        part=part.strip()
        if not part:continue
        if '-' in part:
            pair=part.split('-',1)
            if len(pair)!=2:raise ValueError('Invalid --ids range')
            a,b=map(int,pair)
            if b<a:raise ValueError('Invalid descending --ids range')
            chosen.update(range(a,b+1))
        else:
            chosen.add(int(part))
    if not chosen.issubset(set(range(44))):
        raise ValueError('IDs must be 0..43')
    return chosen


def _e2_model_artifact_roots(t):
    """Live task directory and its own timestamped force backups, if present."""
    root=task_dir(t)
    paths=[root]
    if root.parent.is_dir():
        paths.extend(sorted((p for p in root.parent.glob(root.name+'.force_backup_*')
                             if p.is_dir()), key=lambda p:p.stat().st_mtime, reverse=True))
    return paths


def _e2_checkpoint_paths(t):
    """Find model weights in this task or its own backups, never split NPZs.

    Recognition is intentionally broader than the original hard-coded filenames.
    It does not assert that weights alone are enough for faithful inference.
    """
    found = {}
    for root in _e2_model_artifact_roots(t):
        if not root.is_dir():
            continue
        for p in root.rglob('*'):
            if not p.is_file() or not p.stat().st_size:
                continue
            name = p.name.lower()
            if (name.endswith(('.weights.h5', '.keras', '.pt', '.pth', '.ckpt', '.ckpt.index'))
                    or name in ('best_model.h5', 'model.h5', 'weights.h5')):
                found[str(p.resolve())] = p
    return sorted(found.values(), key=lambda p: p.stat().st_mtime, reverse=True)


def _e2_check_validation_metrics(path, t):
    """Reject unrelated or external TEST results when recovering validation."""
    with path.open(newline='') as h:
        rows = list(csv.DictReader(h))
    if not rows:
        raise RuntimeError('Empty validation metrics CSV')
    if 'split' in rows[0]:
        roles = {str(r['split']).strip().lower() for r in rows}
        if roles != {'validation'}:
            raise RuntimeError(f'Non-validation metrics partition(s): {roles}')
    if 'target' in rows[0]:
        got = {r['target'] for r in rows}
        expect = set(TARGETS[t['dataset']])
        if got != expect:
            raise RuntimeError(f'Metric target columns do not match expected targets: {got} != {expect}')


def _e2_recoverable_outputs(t):
    """Recover only COMPLETE validation metrics/predictions with frozen split proof.

    Handles both native backend naming and Step4 stable output filenames.
    Historical FAILED.json alone is not evidence that a run is unrecoverable.
    """
    roots = [p for p in _e2_model_artifact_roots(t) if p.is_dir()]
    if not roots:
        return None
    pred_names = ('predictions.csv', 'validation_predictions.csv',
                  'val_predictions.csv', 'VALIDATION_PREDICTIONS.csv')
    met_names = ('metrics.csv', 'validation_metrics.csv',
                 'val_metrics.csv', 'VALIDATION_METRICS.csv')
    candidates = []
    for root in roots:
        for name in pred_names:
            candidates.extend((p,root) for p in root.rglob(name) if p.is_file() and p.stat().st_size)
    candidates.sort(key=lambda item: item[0].stat().st_mtime, reverse=True)
    for preds,root in candidates:
        for mn in met_names:
            met = preds.parent / mn
            if not met.is_file() or not met.stat().st_size:
                continue
            try:
                _e2_check_validation_metrics(met, t)
                count = validate_prediction_count(preds, t['dataset'])
                proof = None
                # Check only this task's run tree; don't accept another model's split.
                for parent in (preds.parent, *preds.parent.parents):
                    if not parent.is_relative_to(root):
                        break
                    try:
                        proof = verify_backend_indices(parent, t['dataset'])
                        break
                    except (OSError, ValueError, RuntimeError):
                        continue
                if proof is None:
                    raise RuntimeError('Missing matching frozen train/validation split proof')
                return met, preds, proof, count
            except (OSError, ValueError, RuntimeError, KeyError) as exc:
                say('[REJECT UNPROVEN VALIDATION]', t['id'], str(exc)[:150])
    return None


def _e2_log_training_evidence(t):
    """Read the tail of task logs without generating or rewriting files."""
    if not LOGS.is_dir():
        return None
    candidates = []
    for p in LOGS.glob('*.out'):
        m = re.search(r'_(\d+)\.out$', p.name)
        if m and int(m.group(1)) == t['id']:
            candidates.append(p)
    if not candidates:
        return None
    latest = max(candidates, key=lambda p:p.stat().st_mtime)
    with latest.open('rb') as h:
        h.seek(max(0, latest.stat().st_size - 300000))
        chunk = h.read().decode('utf-8', 'replace')
    stops = list(re.finditer(r'early stopping at epoch\s+(\d+)\s*\(best epoch\s+(\d+)\)', chunk, re.I))
    epoch = re.findall(r'Epoch\s+(\d+)/(?:(?:\d+))', chunk, re.I)
    if stops:
        last = stops[-1]
        return {'kind':'EARLY_STOPPED', 'epoch':int(last.group(1)), 'best_epoch':int(last.group(2)), 'log':str(latest)}
    if epoch:
        return {'kind':'TRAINING_LOG', 'epoch':int(epoch[-1]), 'best_epoch':None, 'log':str(latest)}
    return {'kind':'LOG_ONLY', 'epoch':None, 'best_epoch':None, 'log':str(latest)}


def _e2_recovery_classification(t, allow_proof=True):
    """Classify actual recoverability, not only presence of FAILED.json."""
    if existing_valid(t):
        return 'DONE', '-', '-'
    if allow_proof:
        result = _e2_recoverable_outputs(t)
        if result is not None:
            return 'VALIDATION_READY', 'YES', '-'
    ckpts = _e2_checkpoint_paths(t)
    ev = _e2_log_training_evidence(t)
    if ckpts:
        return 'WEIGHTS_ONLY', str(len(ckpts)), ('best=' + str(ev['best_epoch']) if ev and ev['best_epoch'] is not None else '-')
    if ev and ev['kind'] == 'EARLY_STOPPED':
        return 'TRAINED_LOG_ONLY', 'NO', 'best=' + str(ev['best_epoch'])
    if ev and ev['kind'] == 'TRAINING_LOG':
        return 'HISTORY_ONLY', 'NO', 'epoch=' + str(ev['epoch'])
    if failfile(t).is_file():
        return 'FAILED_NO_OUTPUT', 'NO', '-'
    return 'NOT_STARTED', 'NO', '-'



def _e2_recover_done(t, candidate):
    met,preds,proof,count=candidate
    root=task_dir(t);root.mkdir(parents=True,exist_ok=True)
    metout=root/'VALIDATION_METRICS.csv'
    predout=root/'VALIDATION_PREDICTIONS.csv'
    if met.resolve()!=metout.resolve():shutil.copy2(met,metout)
    if preds.resolve()!=predout.resolve():shutil.copy2(preds,predout)
    save_json(marker(t),{**t,'status':'SUCCESS','finished_utc':stamp(),
             'dataset_config_sha256':sha(stage_config(t['dataset'])),
             'split_file_sha256':sha(data_dir(t['dataset'])/'split_indices.npz'),
             'run_dir':str(preds.parent),'package_metrics':str(metout),
             'predictions':str(predout),'backend_split_proof':str(proof),
             'validation_prediction_rows':count,
             'evaluation_partition':'VALIDATION_NOT_EXTERNAL_TEST',
             'recovery':'Verified existing outputs; no retraining'})
    if not existing_valid(t):
        marker(t).unlink(missing_ok=True)
        raise RuntimeError('Recovered outputs failed frozen-data contract')
    failfile(t).unlink(missing_ok=True)
    say('[RESULTS RECOVERED; SKIPPED TRAINING]',t['id'],t['model'])


def _e2_predict_command(t,ckpt,template):
    fields={'checkpoint':str(ckpt.resolve()),'checkpoint_dir':str(ckpt.parent.resolve()),
        'dataset':t['dataset'],'model':t['model'],'task_dir':str(task_dir(t).resolve()),
        'dataset_config':str(stage_config(t['dataset']).resolve()),
        'validation_trees':str((data_dir(t['dataset'])/'validation.trees.nwk.gz').resolve()),
        'validation_targets':str((data_dir(t['dataset'])/'validation.targets.csv.gz').resolve()),
        'expected_n':str(read_json(split_info(t['dataset']))['validation'])}
    argv=shlex.split(template.format(**fields))
    if not argv:raise ValueError('Empty --predict-command')
    before=sha(ckpt)
    say('[PREDICTION ONLY]',t['id'],t['model'],shlex.join(argv))
    subprocess.run(argv,cwd=PROJECT,check=True)
    if sha(ckpt)!=before:raise RuntimeError('Prediction modified checkpoint; refusing recovery')
    data=_e2_recoverable_outputs(t)
    if data is None:raise RuntimeError('No verified validation predictions after inference-only command')
    _e2_recover_done(t,data)


def _e2_archive_task(t):
    root=task_dir(t)
    if root.is_dir():
        suffix=dt.datetime.now(dt.timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
        archive=root.with_name(root.name+'.force_backup_'+suffix)
        root.rename(archive)
        say('[FORCE ARCHIVED]',archive)


def _e2_active_ids():
    """Do not silently guess if Slurm is unavailable."""
    try:
        cp=subprocess.run(['squeue','--me','-r','-h','-o','%i|%j|%T'],
                          capture_output=True,text=True,check=True)
    except (OSError,subprocess.CalledProcessError) as exc:
        raise RuntimeError('Cannot check Slurm queue; will not risk duplicate training') from exc
    result={}
    for line in cp.stdout.splitlines():
        parts=line.split('|',2)
        if len(parts)!=3:continue
        m=re.fullmatch(r'(\d+)_(\d+)',parts[0])
        if not m:continue
        if not parts[1].lower().startswith(('phylognn','e2_')):continue
        if parts[2].upper() in {'RUNNING','PENDING','CONFIGURING','COMPLETING','SUSPENDED'}:
            result.setdefault(int(m.group(2)),[]).append((parts[0],parts[2]))
    return result


_E2_OLD_WORKER=worker
_E2_OLD_TRAIN_BACKEND=train_existing_backend


def worker(tid,args):
    """Check frozen model state before invoking any training run."""
    t=task(tid)
    worker_preflight(t)
    ids=_e2_parse_ids(getattr(args,'ids',''))
    force=bool(getattr(args,'force',False))
    if force:
        if tid not in ids:
            raise RuntimeError('Force retraining requires explicit --ids containing this worker')
        _e2_archive_task(t)
    else:
        if existing_valid(t):
            say('[SKIP COMPLETED]',tid,t['model']);return
        recovered=_e2_recoverable_outputs(t)
        if recovered is not None:
            _e2_recover_done(t,recovered);return
        ckpts=_e2_checkpoint_paths(t)
        if ckpts:
            cmd=getattr(args,'predict_command','')
            if cmd:return _e2_predict_command(t,ckpts[0],cmd)
            say('[CHECKPOINT FOUND; TRAINING DISABLED]',tid,ckpts[0])
            raise RuntimeError('Weights exist, but no verified validation predictions. '
                'Pass --predict-command with a validated inference-only evaluator or --force --ids.')
    return _E2_OLD_WORKER(tid,args)


def submit(args):
    """Submit only uncompleted jobs; never duplicate active Slurm tasks."""
    preflight()
    if args.max_parallel<1:raise ValueError('--max-parallel must be positive')
    selected=_e2_parse_ids(args.ids)
    force=bool(args.force)
    if force and not selected:raise ValueError('--force requires explicit --ids')
    if not selected:selected={t['id'] for t in all_tasks()}
    request=args.only_family
    active=_e2_active_ids()
    todo={'author':[],'graph':[]}
    to_predict=[]
    for i in sorted(selected):
        t=task(i)
        if request!='all' and request!=t['family']:continue
        if i in active:
            say('[ACTIVE; NO DUPLICATE]',i,t['model'],active[i]);continue
        if not force:
            if existing_valid(t):say('[DONE; SKIPPED]',i,t['model']);continue
            recovered=_e2_recoverable_outputs(t)
            if recovered is not None:
                if args.dry_run:say('[RECOVERABLE; DRY RUN]',i,t['model'])
                else:_e2_recover_done(t,recovered)
                continue
            ckpts=_e2_checkpoint_paths(t)
            if ckpts:
                if args.predict_command:
                    if args.dry_run:say('[INFERENCE ONLY; DRY RUN]',i,str(ckpts[0]))
                    else:_e2_predict_command(t,ckpts[0],args.predict_command)
                else:
                    say('[CHECKPOINT KEPT; NEEDS PREDICT COMMAND]',i,str(ckpts[0]))
                continue
        # An old FAILED marker may describe a POST-TRAINING evaluation crash.
        # Do not automatically spend GPU hours recreating its trained weights.
        # The user must opt in via --force --ids for any failed run lacking
        # verified validation results/checkpoints.
        if not force and failfile(t).is_file():
            evidence = _e2_log_training_evidence(t)
            detail = ('best_epoch='+str(evidence['best_epoch'])
                      if evidence and evidence.get('best_epoch') is not None else 'no best epoch confirmed')
            say('[FAILED; PRESERVED; NO AUTOMATIC RETRAIN]', i, t['model'], detail,
                '| use --force --ids '+str(i)+' to retrain')
            continue
        todo[t['family']].append(i)
    # Only touch the backend if we would actually be submitting GNN training tasks.
    if todo['graph']:
        _e2_fix_backend(dry_run=args.dry_run)
    if not any(todo.values()):
        say('[NO NEW TRAINING SUBMITTED] Tasks are completed, active, checkpoint-only, or preserved FAILED pending explicit --force.');return
    LOGS.mkdir(parents=True,exist_ok=True)
    for family,ids in todo.items():
        if not ids:continue
        interp=Path(args.author_python if family=='author' else args.gnn_python).expanduser().resolve()
        if not interp.is_file():raise FileNotFoundError(f'Missing {family} interpreter {interp}')
        spec=compact_ids(ids)+f'%{args.max_parallel}'
        extra=''
        if force:extra+=' --force --ids '+shlex.quote(args.ids)
        if args.predict_command:extra+=' --predict-command '+shlex.quote(args.predict_command)
        if args.backend_command:extra+=' --backend-command '+shlex.quote(args.backend_command)
        wrapper=(f'set -e; {shlex.quote(str(interp))} -u '
                 f'{shlex.quote(str(Path(__file__).resolve()))} '
                 '--worker "$SLURM_ARRAY_TASK_ID"'+extra)
        command=['sbatch','--parsable','--job-name',f'PhyloGNN_E2_{family}_resume',
                 '--array',spec,'--partition',args.partition,
                 *(['--qos',args.qos] if args.qos else []),
                 '--nodes','1','--ntasks','1','--cpus-per-task',str(args.cpus),
                 '--mem',args.mem,'--time',args.time,'--gres',args.gres,
                 '--chdir',str(PROJECT),
                 '--export','ALL,PHYLOGNN_E2_ALLOW_EMPTY_TEST=1,PYTHONUNBUFFERED=1',
                 '--output',str(LOGS/f'E2_{family}_resume.%A_%a.out'),
                 '--error',str(LOGS/f'E2_{family}_resume.%A_%a.err'),
                 '--wrap',wrapper]
        if args.dry_run:
            say('[WOULD SUBMIT]',shlex.join(command));continue
        job=subprocess.check_output(command,text=True).strip()
        say('[SUBMITTED]',family,'Slurm job',job,'array',spec)
        save_json(STATE/f'last_{family}_resume.json',
                  {'jobid':job,'ids':ids,'date':stamp(),'python':str(interp)})
    say('[RECONCILED] all verified DONE models preserved; active jobs untouched')



# =============================================================================
# EXPERIMENT 2: FROZEN AUTHOR MODEL -> ORIGINAL INDEPENDENT TEST INFERENCE
# Uses the SAME PhyloCNN encoder, covariates, and target scaling as E2 training.
# Test observations are never imported into training/validation.
# =============================================================================

def _e2_test_import_setup():
    """Make the local PhyloGNN package visible without modifying it."""
    source = str(PROJECT / 'PhyloGNN' / 'src')
    if source not in sys.path:
        sys.path.insert(0, source)


def _e2_test_folder(t, limit):
    part = 'external_test' if limit == 0 else f'external_test_preview_{limit}'
    return OUTPUT / part / t['dataset'] / t['model']


def _e2_test_fingerprint(t, limit):
    _e2_test_import_setup()
    import Step4_E2_FastFamily as fast
    ds = t['dataset']
    spec = fast.source_metadata(ds)
    ref = discover_original_author_tests()[ds]
    names = [Path(p) for p in (*ref['tree_files'], ref['parameters_file'])]
    run = task_dir(t) / 'fast_author_run'
    weights = run / 'best_model.weights.h5'
    scales = run / 'rescaling_spec.json'
    for p in [weights, scales, *names]:
        if not p.is_file() or p.stat().st_size == 0:
            raise FileNotFoundError(f'Required frozen inference artifact is missing: {p}')
    return {'test_files': [[str(p.resolve()), p.stat().st_size, p.stat().st_mtime_ns] for p in names],
            'checkpoint_sha256': sha(weights), 'scales_sha256': sha(scales),
            'dataset': ds, 'model': t['model'], 'encoder': spec.encoder, 'limit': limit}


def _e2_test_complete(t, limit=0):
    if not existing_valid(t):
        return False
    folder = _e2_test_folder(t, limit)
    meta = folder / 'TEST_DONE.json'
    if not meta.is_file():
        return False
    try:
        d = read_json(meta)
        if d.get('status') != 'SUCCESS' or d.get('fingerprint') != _e2_test_fingerprint(t, limit):
            return False
        n = int(d['n_test'])
        if limit == 0:
            if n != AUTHOR_TEST_USABLE[t['dataset']]: return False
        elif n != limit:
            return False
        for filename in ('test_predictions.csv', 'test_metrics.csv'):
            p = folder / filename
            if not p.is_file() or p.stat().st_size == 0: return False
        with (folder/'test_predictions.csv').open(newline='') as f:
            if sum(1 for _ in csv.DictReader(f)) != n: return False
        return True
    except (OSError, ValueError, TypeError, KeyError, ImportError):
        return False


def _e2_original_test_cache(dsid, limit):
    """Encode independent test rows once per dataset, shared across 3 author CNNs.

    A cache is TEST-ONLY and never reused for fitting or validation.
    """
    _e2_test_import_setup()
    import fcntl
    from itertools import zip_longest, islice
    from numpy.lib.format import open_memmap
    from phylognn.models.phylocnn import encode_trees, assemble_inputs
    import Step4_E2_FastFamily as fast

    source = discover_original_author_tests()[dsid]
    spec = fast.source_metadata(dsid)
    cov = fast.sample_covariate(spec, dsid)
    files = [Path(p) for p in (*source['tree_files'], source['parameters_file'])]
    fingerprint = {'files': [[str(p), p.stat().st_size, p.stat().st_mtime_ns] for p in files],
                   'encoder': spec.encoder, 'covariate': cov, 'limit': limit,
                   'features_version': 1}
    cachedir = OUTPUT / 'external_test' / '_encoded_author_cache' / dsid / ('full' if not limit else f'preview_{limit}')
    cachedir.mkdir(parents=True, exist_ok=True)
    targets = list(TARGETS[dsid]); n_expected = limit or AUTHOR_TEST_USABLE[dsid]
    width = 20 if spec.encoder == 'bisse' else 19
    fp = cachedir/'X_2gen.npy'; rp=cachedir/'rescale.npy'; yp=cachedir/'Y_natural.npy'; manifest=cachedir/'meta.json'

    with (cachedir/'.lock').open('a') as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
        if all(p.is_file() for p in (fp,rp,yp,manifest)):
            try:
                old=read_json(manifest)
                if old.get('fingerprint')==fingerprint and old.get('n')==n_expected:
                    X=np.load(fp,mmap_mode='r'); R=np.load(rp,mmap_mode='r'); Y=np.load(yp,mmap_mode='r')
                    if X.shape==(n_expected,500,width,2) and R.shape==(n_expected,) and Y.shape==(n_expected,len(targets)):
                        say('[INDEPENDENT TEST ENCODING CACHE HIT]',dsid,'N=',n_expected)
                        return fp,rp,yp,n_expected
            except (OSError,ValueError,KeyError):
                pass
        temps=[p.with_name(p.stem+f'.tmp.{os.getpid()}.npy') for p in (fp,rp,yp)]
        X=open_memmap(temps[0],mode='w+',dtype=np.float32,shape=(n_expected,500,width,2))
        R=open_memmap(temps[1],mode='w+',dtype=np.float64,shape=(n_expected,))
        Y=open_memmap(temps[2],mode='w+',dtype=np.float64,shape=(n_expected,len(targets)))
        count=0; rejected=Counter(); observed=0
        def source_trees():
            for p in files[:-1]:
                with gzip.open(p,'rt') as h:
                    for line in h: yield line.strip()
        try:
            with gzip.open(files[-1],'rt',newline='') as h:
                reader=csv.DictReader(h)
                if not reader.fieldnames:raise RuntimeError('Original test parameters are missing CSV header')
                cmap=match_targets(reader.fieldnames,dsid)
                rawmap={key_name(c):c for c in reader.fieldnames}
                found=rawmap.get(key_name(cov))
                if found is None:
                    alternatives=('sampling_probability','sampling_prob','sampling_fraction','sampling_proba','sampling_frac')
                    possible=[rawmap[k] for k in alternatives if k in rawmap]
                    if len(set(possible))!=1:raise RuntimeError(f'Unknown sampling covariate columns: {reader.fieldnames}')
                    found=possible[0]
                def accept(tree,row,idx):
                    try:
                        check_newick(tree)
                        vals=np.array([float(row[cmap[t]]) for t in targets],dtype=np.float64)
                        v=float(row[found])
                        if not np.isfinite(vals).all() or not np.isfinite(v):raise ValueError('Nonfinite input')
                        return vals,v
                    except (ValueError,TypeError,KeyError) as e:
                        rejected[type(e).__name__+':'+str(e)[:80]]+=1
                        return None
                batch_t=[]; batch_v=[]; batch_y=[]
                def flush():
                    nonlocal count
                    if not batch_t:return
                    enc,factors=encode_trees(batch_t,encoder=spec.encoder,max_leaves=500,workers=1)
                    features=assemble_inputs(enc,np.asarray(batch_v,dtype=np.float32).reshape(-1,1),spec.encoder,'2gen')
                    m=len(batch_t)
                    if features.shape!=(m,500,width,2):raise RuntimeError(f'Test encoder shape mismatch: {features.shape}')
                    factors=np.asarray(factors,dtype=np.float64).reshape(-1)
                    if factors.shape!=(m,) or not np.isfinite(factors).all() or np.any(factors<=0):
                        raise RuntimeError('Invalid test rescale factors')
                    if not np.isfinite(features).all():raise RuntimeError('Nonfinite encoded features')
                    X[count:count+m]=features
                    R[count:count+m]=factors
                    Y[count:count+m]=np.asarray(batch_y,dtype=np.float64)
                    count+=m
                    batch_t.clear();batch_v.clear();batch_y.clear()
                    if count%1024 < m or count==n_expected:
                        say(f'[TEST ENCODE] {dsid}: {count:,}/{n_expected:,} ({100*count/n_expected:.1f}%)')
                for tree,row in zip_longest(source_trees(),reader):
                    if tree is None or row is None:
                        raise RuntimeError(f'{dsid}: author test tree/CSV row count mismatch; refused unaligned labels')
                    observed+=1
                    accepted=accept(tree,row,observed)
                    if accepted is None:continue
                    vals,covar=accepted
                    batch_t.append(tree);batch_y.append(vals);batch_v.append(covar)
                    if len(batch_t)>=64:flush()
                    if count+len(batch_t)>=n_expected:
                        flush()
                        if limit:break
                flush()
            if count != n_expected:
                raise RuntimeError(f'{dsid}: encoded {count} valid test rows; expected {n_expected}. Rejected: {dict(rejected)}')
            X.flush();R.flush();Y.flush()
            del X,R,Y
            for tmp,original in zip(temps,(fp,rp,yp)):os.replace(tmp,original)
            save_json(manifest,{'fingerprint':fingerprint,'n':count,'observed':observed,
                                'rejected':dict(rejected),'partition':'INDEPENDENT_AUTHOR_TEST_ONLY',
                                'original_tree_archives':source['tree_files']})
            say('[TEST ENCODING READY]',dsid,'valid=',count,'QC rejected=',sum(rejected.values()))
        finally:
            for p in temps:p.unlink(missing_ok=True)
    return fp,rp,yp,n_expected


def _e2_author_infer_test(t, limit=0, preview=False):
    if t['family'] != 'author':
        raise RuntimeError('This bundled inference loader is verified for PhyloCNN only; GNN requires its graph checkpoint loader')
    if not existing_valid(t):
        raise RuntimeError(f'Training not DONE: {t["dataset"]}/{t["model"]}')
    if _e2_test_complete(t,limit):
        say('[TEST ALREADY COMPLETE; SKIP]',t['id'],t['dataset'],t['model'])
        _e2_print_existing_test_metrics(t,limit)
        return
    _e2_test_import_setup()
    import Step4_E2_FastFamily as fast
    from types import SimpleNamespace
    import Step3_Experiment1_LimitedTest as step3

    dsid=t['dataset']; variant=step3.AUTHOR_VARIANT[t['model']]
    savedir=task_dir(t)/'fast_author_run'
    weights=savedir/'best_model.weights.h5'; meta_path=savedir/'rescaling_spec.json'
    if not weights.is_file() or not meta_path.is_file():
        raise FileNotFoundError(f'Frozen trained model incomplete: {weights} / {meta_path}')
    spec=fast.source_metadata(dsid)
    conf=read_json(meta_path)
    cov=fast.sample_covariate(spec,dsid)
    targets=list(TARGETS[dsid])
    scaling=fast.canonical_scaling(spec,targets,ALIASES)
    if conf.get('encoder')!=spec.encoder or conf.get('covariate')!=cov or conf.get('target_scaling')!=scaling:
        raise RuntimeError('Saved training encoder/covariate/scaling mismatch; no unverified inference')
    checkpoint_before=sha(weights)
    Xpath,Rpath,Ypath,n=_e2_original_test_cache(dsid,limit)
    X=np.load(Xpath,mmap_mode='r'); R=np.load(Rpath,mmap_mode='r');Y=np.load(Ypath,mmap_mode='r')
    base=(19 if spec.encoder=='bisse' else 18)
    select={'2gen':list(range(base+1)),
            '1gen':list(range(10 if spec.encoder=='bisse' else 9))+[base],
            'nocontext':list(range(5 if spec.encoder=='bisse' else 4))+[base]}[variant]
    # CPU-only on the evaluation worker. Never call fit(), train(), or run_experiment().
    os.environ['CUDA_VISIBLE_DEVICES']=''
    os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
    import tensorflow as tf
    tf.config.set_visible_devices([],'GPU')
    model=step3.build_author_phylocnn(SimpleNamespace(task=SimpleNamespace(encoder=spec.encoder)),dsid+'-shipped',variant)
    if tuple(model.input_shape[1:]) != (500,len(select),2):
        raise RuntimeError(f'Model feature shape mismatch: {model.input_shape} != (500,{len(select)},2)')
    model.load_weights(str(weights))
    raw=np.empty((n,len(targets)),dtype=np.float64)
    for lo in range(0,n,64):
        hi=min(n,lo+64)
        features=np.ascontiguousarray(np.asarray(X[lo:hi],dtype=np.float32)[:,:,select,:])
        raw[lo:hi]=np.asarray(model(features,training=False),dtype=np.float64)
    pred=fast.model_scale_targets(raw,np.asarray(R),targets,scaling,inverse=True)
    if not np.isfinite(pred).all() or pred.shape!=(n,len(targets)):
        raise RuntimeError('Invalid model predictions')
    if sha(weights)!=checkpoint_before:
        raise RuntimeError('Checkpoint modified during test inference')
    out=_e2_test_folder(t,limit)
    out.mkdir(parents=True,exist_ok=True)
    fp=_e2_test_fingerprint(t,limit)
    temp_pred=out/f'test_predictions.{os.getpid()}.tmp.csv';temp_met=out/f'test_metrics.{os.getpid()}.tmp.csv'
    try:
        with temp_pred.open('w',newline='') as f:
            w=csv.writer(f)
            w.writerow(['record_index']+[f'true_{x}' for x in targets]+[f'pred_{x}' for x in targets])
            for i in range(n):
                w.writerow([i]+[float(v) for v in Y[i]]+[float(v) for v in pred[i]])
        records=[]
        for j,name in enumerate(targets):
            vals=step3.regression_metrics(np.asarray(Y[:,j]),pred[:,j])
            records.append({'dataset':dsid,'model':t['model'], 'split':'external_test',
                            'target':name, 'N':n,**vals})
        with temp_met.open('w',newline='') as f:
            writer=csv.DictWriter(f,fieldnames=list(dict.fromkeys(k for r in records for k in r)))
            writer.writeheader();writer.writerows(records)
        os.replace(temp_pred,out/'test_predictions.csv')
        os.replace(temp_met,out/'test_metrics.csv')
        final_locked=all(existing_valid(z) for z in all_tasks())
        save_json(out/'TEST_DONE.json',{'status':'SUCCESS','dataset':dsid,'model':t['model'],
                'fingerprint':fp,'n_test':n,'full_test':limit==0,
                'all_44_frozen_at_evaluation':final_locked,'timestamp':stamp(),
                'checkpoint':str(weights),'checkpoint_sha256':checkpoint_before,
                'frozen_checkpoint':str(weights),
                'test_metrics':str(out/'test_metrics.csv'),
                'test_predictions':str(out/'test_predictions.csv'),
                'test_prediction_rows':n,
                'source':'Original authors Test_Sets, independent of E2 training and validation',
                'interpretation':'FINAL LOCKED' if final_locked and limit==0 else 'PRELIMINARY - do not tune on test'})
    finally:
        temp_pred.unlink(missing_ok=True); temp_met.unlink(missing_ok=True)
    say('='*112)
    say('[INDEPENDENT TEST RESULTS]',dsid,t['model'],'N=',n,
        'FINAL LOCKED' if final_locked and limit==0 else 'PRELIMINARY')
    for r in records:
        say('  ',r['target'], ' MAE=',round(float(r['MAE']),5),
            ' RMSE=',round(float(r['RMSE']),5),' R2=',round(float(r['R2']),5),
            ' Bias=',round(float(r['Bias']),5))
    say('[TEST OUTPUT]',out)



def _e2_print_existing_test_metrics(t, limit):
    path = _e2_test_folder(t, limit) / 'test_metrics.csv'
    if not path.is_file():
        return
    with path.open(newline='') as handle:
        rows = list(csv.DictReader(handle))
    for r in rows:
        say('[SAVED TEST]', t['dataset'], t['model'], r.get('target'),
            'N=',r.get('N','?'),'MAE=',r.get('MAE','?'),
            'RMSE=',r.get('RMSE','?'),'R2=',r.get('R2','?'))


def _e2_author_test_active_ids():
    try:
        r=subprocess.run(['squeue','--me','-r','-h','-o','%i|%j|%T'],capture_output=True,text=True,check=True)
    except (OSError,subprocess.CalledProcessError) as e:
        raise RuntimeError('Slurm unavailable: cannot avoid duplicate test jobs') from e
    pending={}
    for line in r.stdout.splitlines():
        m=re.match(r'^(\d+)_(\d+)\|PhyloGNN_E2_TEST_AUTO\|(\S+)',line.strip())
        if m and m.group(3).upper() in {'RUNNING','PENDING','CONFIGURING','COMPLETING','SUSPENDED'}:
            pending[int(m.group(2))]=m.group(1)
    return pending


def _e2_resume_author_tests(args):
    """Resume independent test inference without touching running trainings."""
    ids=_e2_parse_ids(args.ids)
    if not ids:ids={t['id'] for t in all_tasks()}
    targets=[t for t in all_tasks() if t['id'] in ids and t['family']=='author' and existing_valid(t)]
    if not targets:
        say('[INDEPENDENT TEST] No completed PhyloCNN models selected yet');return
    active=_e2_author_test_active_ids()
    todo=[]
    for t in targets:
        if _e2_test_complete(t,args.test_limit):
            say('[TEST DONE; SKIPPED]',t['id'],t['dataset'],t['model'])
            _e2_print_existing_test_metrics(t,args.test_limit)
        elif t['id'] in active:
            say('[TEST ACTIVE; NO DUPLICATE]',t['id'],active[t['id']])
        else:todo.append(t['id'])
    if not todo:
        say('[INDEPENDENT TEST] Nothing to submit');return
    script=Path(__file__).resolve()
    interpreter=Path(args.author_python).expanduser().resolve()
    if not interpreter.is_file():raise FileNotFoundError(f'Test interpreter missing: {interpreter}')
    LOGS.mkdir(parents=True,exist_ok=True)
    wrapper=(f'{shlex.quote(str(interpreter))} -B -u {shlex.quote(str(script))} '
             '--test-worker "$SLURM_ARRAY_TASK_ID" --test-limit '+str(args.test_limit))
    cmd=['sbatch','--parsable','--job-name','PhyloGNN_E2_TEST_AUTO','--array',compact_ids(todo)+f'%{args.test_max_parallel}',
         '--partition',args.test_partition,'--nodes','1','--ntasks','1','--cpus-per-task',str(args.test_cpus),
         '--mem',args.test_mem,'--time',args.test_time,'--chdir',str(PROJECT),
         '--export','ALL,CUDA_VISIBLE_DEVICES=,TF_CPP_MIN_LOG_LEVEL=2,OPENBLAS_NUM_THREADS=2',
         '--output',str(LOGS/'E2_test_auto.%A_%a.out'),
         '--error',str(LOGS/'E2_test_auto.%A_%a.err'),'--wrap',wrapper]
    if args.dry_run:
        say('[TEST DRY RUN]',shlex.join(cmd));return
    job=subprocess.check_output(cmd,text=True).strip()
    say('[INDEPENDENT TEST SUBMITTED] job',job,'models',todo,'full original test' if args.test_limit==0 else f'first {args.test_limit} valid rows')


def _e2_test_resume_entry(args):
    # Test submission is independent from training submission. All training
    # checkpoints and frozen 70/30 partitions are untouched by inference.
    if args.only_family=='graph':
        say('[TEST] Graph test evaluation requires a verified graph checkpoint loader; author testing skipped')
    else:
        _e2_resume_author_tests(args)


# ============================================================================
# V7 RECOVERY AUDIT: expose intermediate artifacts and avoid unnecessary redo.
# These helpers DO NOT turn training logs into independent validation results.
# ============================================================================
_E2_EARLIER_STATUS = status

def status():
    _E2_EARLIER_STATUS()
    say('\n' + '='*124)
    say('GNN RECOVERY AUDIT — check whether FAILED models have usable artifacts')
    say('='*124)
    say('{:<4} {:<6} {:<27} {:<19} {:<10} {}'.format('ID','DATA','MODEL','RECOVERY','WEIGHTS','LOG EVIDENCE'))
    say('-'*124)
    counts = Counter()
    for t in all_tasks():
        if t['family'] != 'graph':
            continue
        classification, weights, log = _e2_recovery_classification(t)
        counts[classification] += 1
        say('{:<4} {:<6} {:<27} {:<19} {:<10} {}'.format(
            t['id'], t['dataset'], t['model'], classification, weights, log))
    say('-'*124)
    say('[RECOVERY BREAKDOWN]', dict(counts))
    say('VALIDATION_READY = existing outputs may be finalized using --recover-only.')
    say('WEIGHTS_ONLY = actual saved weights exist; requires verified inference-only loader.')
    say('TRAINED_LOG_ONLY = epochs/early stopping logged, but saved weights NOT found inside this task.')
    say('FAILED_NO_OUTPUT = no qualifying checkpoint or validation results were found.')
    say('A training history is NOT a reconstructable trained model and cannot yield predictions.')


def _e2_recover_only(args):
    """Repair DONE state only when existing validation artifacts pass strict checks."""
    selected = _e2_parse_ids(args.ids)
    if not selected:
        selected = {t['id'] for t in all_tasks() if t['family']=='graph'}
    counts = Counter()
    for i in sorted(selected):
        t = task(i)
        if args.only_family != 'all' and t['family'] != args.only_family:
            continue
        if existing_valid(t):
            say('[UNCHANGED DONE]', i, t['dataset'], t['model'])
            counts['DONE'] += 1
            continue
        candidate = _e2_recoverable_outputs(t)
        if candidate is not None:
            if args.dry_run:
                say('[DRY-RUN RECOVERABLE]', i, t['dataset'], t['model'])
            else:
                _e2_recover_done(t,candidate)
            counts['RECOVERABLE'] += 1
            continue
        ckpts = _e2_checkpoint_paths(t)
        if ckpts:
            say('[WEIGHTS PRESERVED; INFERENCE REQUIRED]', i, t['dataset'], t['model'], ckpts[0])
            counts['WEIGHTS_ONLY'] += 1
            continue
        ev = _e2_log_training_evidence(t)
        if ev and ev.get('best_epoch') is not None:
            say('[TRAINING HISTORY ONLY; NO WEIGHTS]', i,t['dataset'],t['model'],
                'best_epoch='+str(ev['best_epoch']))
            counts['HISTORY_ONLY'] += 1
        else:
            say('[NO VERIFIED MODEL ARTIFACT]',i,t['dataset'],t['model'])
            counts['NO_MODEL'] += 1
    say('[RECOVERY SUMMARY]',dict(counts))
    say('No training performed. Unrecoverable tasks were not deleted or reclassified DONE.')


def main():
    p=argparse.ArgumentParser(description=__doc__,formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    mode=p.add_mutually_exclusive_group(required=True)
    for flag in ("plan","discover","prepare","count","audit-simulations","preflight","backend-probe","smoke-submit","submit","status","resume","aggregate","evaluate-submit","aggregate-test","author-test-audit","test-only","recover-only"):
        mode.add_argument("--"+flag,action="store_true")
    mode.add_argument("--worker",type=int)
    mode.add_argument("--evaluate-worker",type=int)
    mode.add_argument("--test-worker",type=int,help="Internal inference-only CPU Slurm worker")
    mode.add_argument("--test-one",type=int,help="Run inference-only test for one completed author model on this node")
    p.add_argument("--data-root",default=str(PROJECT),help="PhyloGNN_FINAL directory containing .step1_generate_data/chunks")
    p.add_argument("--source-map",default="")
    p.add_argument("--deep-qc",action="store_true")
    p.add_argument("--min-tips",type=int,default=0,help="Optional lower tip-count QC; zero accepts all plausible Newick trees")
    p.add_argument("--max-tips",type=int,default=0,help="Optional upper tip-count QC; zero disables")
    p.add_argument("--author-python",default=os.environ.get("PHYLOCNN_AUTHOR_PYTHON",sys.executable))
    p.add_argument("--gnn-python",default=os.environ.get("PHYLOGNN_PYTHON",sys.executable))
    p.add_argument("--backend-command",default="",help="Verified custom-data TRAIN/VALIDATE backend command. NEVER supply external test paths.")
    p.add_argument(
        "--only-family",
        choices=("all","author","graph"),
        default="all",
        help="Submit only one software family; default submits all available incomplete families.",
    )
    p.add_argument("--test-map",default="",help="EXTERNAL EVALUATION ONLY: optional manual manifest; omitted means auto-detect PhyloGNN/Test_Sets with every tree archive and params CSV")
    p.add_argument("--eval-command",default="",help="EXTERNAL EVALUATION ONLY: inference-only backend command. Placeholders {test_manifest} {test_source} {test_tree_files} {test_parameters} {test_dataset} {dataset} {model} {checkpoint} {output} {expected_n} {train_run}")
    p.add_argument("--partition",default="ascher")
    p.add_argument("--qos",default="")
    p.add_argument("--gres",default="gpu:1")
    p.add_argument("--cpus",type=int,default=4)
    p.add_argument("--mem",default="120G")
    p.add_argument("--time",default="48:00:00")
    p.add_argument("--max-parallel",type=int,default=2)
    p.add_argument("--ids",default="",help="Select task IDs 0..43: e.g. 3,4,5,6 or 3-10")
    p.add_argument("--force",action="store_true",help="Archive previous task outputs and explicitly retrain selected --ids")
    p.add_argument("--dry-run",action="store_true",help="Preview resume/submit decisions; no backend edits or Slurm submission")
    p.add_argument("--predict-command",default="",help="Verified inference-only command: {checkpoint}, {task_dir}, {dataset}, {model}, {validation_trees}, {validation_targets}, {expected_n}")
    p.add_argument("--test-limit",type=int,default=0,help="Independent test: 0=complete original test, e.g. 100=preliminary sample")
    p.add_argument("--test-partition",default="general",help="CPU partition for automatic test inference; no GPU requested")
    p.add_argument("--test-cpus",type=int,default=2)
    p.add_argument("--test-mem",default="16G")
    p.add_argument("--test-time",default="12:00:00")
    p.add_argument("--test-max-parallel",type=int,default=1)
    args=p.parse_args()
    if args.test_limit<0 or args.test_max_parallel<1: p.error("Invalid --test-limit or --test-max-parallel")
    if args.plan:plan()
    elif args.discover:discover(args)
    elif args.prepare:prepare(args)
    elif args.count:audit_simulation_inputs(args)
    elif args.audit_simulations:audit_simulation_inputs(args)
    elif args.preflight:preflight()
    elif args.backend_probe:backend_probe()
    elif args.smoke_submit or args.submit:submit(args)
    elif args.resume:
        submit(args)
        _e2_test_resume_entry(args)
    elif args.test_only:_e2_test_resume_entry(args)
    elif args.status:status()
    elif args.recover_only:_e2_recover_only(args)
    elif args.aggregate:aggregate()
    elif args.evaluate_submit:evaluate_submit(args)
    elif args.aggregate_test:aggregate_test()
    elif args.author_test_audit:author_test_audit()
    elif args.worker is not None:worker(args.worker,args)
    elif args.evaluate_worker is not None:evaluate_worker(args.evaluate_worker,args)
    elif args.test_worker is not None:_e2_author_infer_test(task(args.test_worker),args.test_limit)
    elif args.test_one is not None:_e2_author_infer_test(task(args.test_one),args.test_limit)

if __name__=="__main__":
    main()
