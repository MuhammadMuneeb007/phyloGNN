#!/usr/bin/env python3
"""Standalone, topology-faithful phylogenetic graph encoding, PhyloCNN encoding and data readers.

This module never imports the phylognn package or a GNN library. Encoding uses
NumPy (+ ETE3 for Newick parsing); PyTorch / PyTorch Geometric are imported only
when PyG batches are requested.

Newick is the SOLE source of graph edges. Adjacency is never inferred from a
PhyloCNN feature matrix.

Data sources (``root`` = the PhyloGNN_FINAL folder):
  train / validation : <repo>/Training_Data/<NAME>/parameters_<NAME>.txt + <NAME>_trees.nwk
                       (Step 1 simulations, assembled by Step4_Experiment2_FullDataset.py --assemble)
                       split 70/30 with the same seeded rule as the PhyloGNN trainer (seed 42),
                       so the split equals Step 4's.
  author_test        : <repo>/Test_Sets/<NAME>/ — the authors' test set, never used for fitting.
  <repo> is root/PhyloGNN or root/phyloCNN.

PhyloCNN inputs are computed from the same Newick strings with the authors'
original encoders (<repo>/Encoding/PhyloCNN_Encoding_*.py, ``process_tree``).

Conventions:
  edge_index[e] = [source_node, destination_node]
  both directions of every phylogenetic branch are included
  edge_attr[e] contains REAL branch lengths and direction information
  tip states are read from BiSSE Newick NHX annotations, not from targets
  all node and edge features are finite float32
  graph_features are per-tree summaries and the observed sampling covariate
"""
from __future__ import annotations

import csv
import gzip
import hashlib
import importlib.util
import io
import json
import math
import os
import re
import warnings
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterable, Iterator, List, Mapping, Optional, Sequence, Tuple

import numpy as np

ENCODING_VERSION = 2

DATASETS = ("BD", "BDEI", "BDSS", "BiSSE")
TARGETS = {
    "BD": ("r_nought", "infectious_period"),
    "BDEI": ("r_nought", "infectious_period", "incubation_period"),
    "BDSS": ("r_nought", "infectious_period", "transmission_factor", "fraction"),
    "BiSSE": ("turnover", "lambda_1", "lambda_2", "q_01"),
}
ALIASES = {
    "r_nought": ("R_nought", "r0", "r_0", "basic_reproduction_number"),
    "infectious_period": ("infectious_period", "infectious_time", "inf_period", "gamma_inverse", "1_gamma"),
    # The authors' BDEI notebook trains on `incubation_period` if present, otherwise
    # on the simulator's `incubation_rate` column. `incubation_ratio` is a DIFFERENT
    # quantity and must never be used.
    "incubation_period": ("incubation_period", "incubation_rate"),
    "transmission_factor": ("transmission_factor", "x_ss", "xss", "x_transmission", "superspreading_factor"),
    "fraction": ("fraction", "f_ss", "fss", "fraction_1", "superspreader_fraction"),
    "turnover": ("turnover", "tau", "turnover_rate"),
    "lambda_1": ("lambda1", "lambda_1"),
    "lambda_2": ("lambda2", "lambda_2"),
    "q_01": ("q01", "q_01", "transition_rate"),
    "sampling": ("sampling_proba", "sampling_frac", "sampling_probability", "sampling_fraction", "sampling_prob"),
    "tree_size": ("tree_size",),
}
NODE_FEATURES = (
    "is_leaf", "is_root", "branch_scaled", "branch_log_scaled",
    "root_distance_scaled", "remaining_height_scaled", "depth_scaled",
    "out_degree_scaled", "subtree_tips_scaled", "log_subtree_tips",
    "parent_branch_scaled", "sibling_imbalance", "state_one", "state_two",
    "state_unknown", "root_distance_squared",
)
EDGE_FEATURES = (
    "branch_scaled", "log_branch_scaled", "direction", "child_is_tip",
    "parent_root_distance_scaled", "child_root_distance_scaled",
)
GRAPH_FEATURES = (
    "log_tips", "log_nodes", "height_log", "mean_branch_scaled",
    "std_branch_scaled", "leaf_fraction", "mean_leaf_depth_scaled",
    "state_one_fraction", "state_two_fraction", "sampling",
)

VALIDATION_FRACTION = 0.30
SPLIT_SEED = 42
MAX_TIPS = 500


# =============================================================================
# TARGETS / COLUMNS
# =============================================================================

def canonical(name: str) -> str:
    return re.sub(r"[^a-z0-9]", "", str(name).casefold())


def choose_column(columns: Sequence[str], target: str) -> str:
    """First alias (in priority order) that exists; refuses ambiguous matches."""
    available: Dict[str, List[str]] = {}
    for col in columns:
        available.setdefault(canonical(col), []).append(col)
    for synonym in ALIASES.get(target, (target,)):
        hits = available.get(canonical(synonym), [])
        if len(hits) == 1:
            return hits[0]
        if len(hits) > 1:
            raise ValueError(f"Ambiguous columns for {target!r}: {hits}")
    raise ValueError(
        f"Cannot resolve {target!r} (aliases {ALIASES.get(target, (target,))}); "
        f"available columns={list(columns)}. Refusing to guess a target definition."
    )


def observed_labels(dataset: str, row: Mapping[str, str]) -> np.ndarray:
    columns = list(row)
    out = []
    for target in TARGETS[dataset]:
        value = float(row[choose_column(columns, target)])
        if not math.isfinite(value):
            raise ValueError(f"Non-finite {target}: {value}")
        out.append(value)
    return np.asarray(out, dtype=np.float32)


def sampling_from_row(row: Mapping[str, str]) -> float:
    value = float(row[choose_column(tuple(row), "sampling")])
    if not math.isfinite(value) or not 0 <= value <= 1:
        raise ValueError(f"Invalid sampling covariate {value}")
    return value


def tree_size_from_row(row: Mapping[str, str]) -> Optional[int]:
    try:
        return int(float(row[choose_column(tuple(row), "tree_size")]))
    except ValueError:
        return None


def open_text(path: Path):
    return gzip.open(path, "rt", encoding="utf-8", newline="") if str(path).endswith(".gz") \
        else open(path, "r", encoding="utf-8", newline="")


def file_hash(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as handle:
        for part in iter(lambda: handle.read(1 << 20), b""):
            h.update(part)
    return h.hexdigest()


# =============================================================================
# NEWICK -> GRAPH
# =============================================================================

@dataclass(frozen=True)
class Graph:
    x: np.ndarray
    edge_index: np.ndarray
    edge_attr: np.ndarray
    graph_features: np.ndarray
    n_tips: int
    tree_hash: str

    def validate(self):
        if self.x.ndim != 2 or self.x.shape[1] != len(NODE_FEATURES):
            raise ValueError(f"Invalid node matrix {self.x.shape}")
        e = self.edge_index
        if e.ndim != 2 or e.shape[1] != 2 or e.dtype.kind not in "iu":
            raise ValueError(f"Invalid edge index {e.shape}")
        if len(e) != 2 * (len(self.x) - 1):
            raise ValueError("Edges must be bidirectional for a connected tree")
        if len(e) and (np.min(e) < 0 or np.max(e) >= len(self.x)):
            raise ValueError("Edge out of bounds")
        if self.edge_attr.shape != (len(e), len(EDGE_FEATURES)):
            raise ValueError("Edge attributes do not align with edges")
        if self.graph_features.shape != (len(GRAPH_FEATURES),):
            raise ValueError("Invalid graph-level features")
        if (not np.isfinite(self.x).all() or not np.isfinite(self.edge_attr).all()
                or not np.isfinite(self.graph_features).all()):
            raise ValueError("Non-finite graph features")
        return self


class _TreeNode:
    __slots__ = ("branch", "name", "state", "children")

    def __init__(self, branch=0.0, name="", state=0):
        self.branch = float(branch)
        self.name = str(name)
        self.state = int(state)
        self.children: List[_TreeNode] = []


def _state_text(name: str, nhx: str = "") -> int:
    for text in (nhx, name):
        match = re.search(r"(?:^|[\[\]:;&_-])t_s\s*[-=]\s*([12])(?:$|[^0-9])", str(text))
        if match:
            return int(match.group(1))
    return 0


def parse_tree(text: str) -> _TreeNode:
    """Parse with ETE3 (as the authors' encoders do), Bio.Phylo as fallback."""
    if not text or not text.strip().endswith(";"):
        raise ValueError("Missing/incomplete Newick tree")
    try:
        from ete3 import Tree
    except ImportError:
        from Bio import Phylo
        orig = Phylo.read(io.StringIO(text), "newick")

        def convert(node):
            raw = str(getattr(node, "comment", "") or "")
            z = _TreeNode(node.branch_length or 0., node.name or "", _state_text(str(node.name or ""), raw))
            z.children = [convert(k) for k in node.clades]
            return z
        return convert(orig.root)
    orig = Tree(text, format=1)

    def convert(node):
        attr = str(getattr(node, "t_s", ""))
        state = int(attr) if attr in ("1", "2") else _state_text(str(node.name))
        z = _TreeNode(node.dist if node.up is not None else 0., node.name, state)
        z.children = [convert(k) for k in node.children]
        return z
    return convert(orig)


def tree_to_graph(newick: str, sampling: float, dataset: str = "BD",
                  max_tips: int = MAX_TIPS, min_tips: int = 2,
                  expected_tips: Optional[int] = None) -> Graph:
    """Encode a REAL phylogeny into a bidirectional graph with 16 node features.

    One vertex per phylogenetic node, including internal nodes. Branches are NOT
    constructed from PhyloCNN matrices. BiSSE tip traits come from the Newick
    NHX annotations. No supervision targets enter the graph features.
    """
    if dataset not in DATASETS:
        raise ValueError(f"Unknown dataset {dataset}")
    if not math.isfinite(float(sampling)) or not 0 <= sampling <= 1:
        raise ValueError("sampling must be a finite probability in [0,1]")
    root = parse_tree(newick)
    nodes: List[_TreeNode] = []
    parents: List[int] = []
    depth: List[int] = []
    dist: List[float] = []
    todo = [(root, -1, 0, 0.)]
    while todo:
        node, par, d, accumulated = todo.pop()
        if not math.isfinite(node.branch) or node.branch < 0:
            raise ValueError("Invalid negative/non-finite branch length")
        idx = len(nodes)
        nodes.append(node)
        parents.append(par)
        depth.append(d)
        dist.append(accumulated)
        for child in reversed(node.children):
            todo.append((child, idx, d + 1, accumulated + child.branch))
    if len(nodes) < 3:
        raise ValueError("Tree has too few nodes")
    tips = [i for i, n in enumerate(nodes) if not n.children]
    if not min_tips <= len(tips) <= max_tips:
        raise ValueError(f"Expected {min_tips}..{max_tips} tips, found {len(tips)}")
    if expected_tips is not None and len(tips) != expected_tips:
        raise ValueError(f"Tip count {len(tips)} != tree_size {expected_tips} (tree/parameter misalignment)")
    if len(parents) - 1 != sum(p >= 0 for p in parents):
        raise ValueError("Malformed rooted tree")
    leaf_names = [nodes[i].name for i in tips]
    if all(leaf_names) and len(set(leaf_names)) != len(leaf_names):
        raise ValueError("Duplicate tip identifiers")
    if dataset == "BiSSE" and not any(nodes[i].state in (1, 2) for i in tips):
        raise ValueError("BiSSE tip-state annotations missing from Newick")
    height = max(dist[i] for i in tips)
    if height <= 0:
        raise ValueError("Zero phylogenetic tree height")
    max_depth = max(depth) or 1
    n_tips = len(tips)
    descendants = np.zeros(len(nodes), dtype=np.int32)
    for i in range(len(nodes) - 1, -1, -1):
        descendants[i] += (not nodes[i].children)
        if parents[i] >= 0:
            descendants[parents[i]] += descendants[i]
    children_index = [[] for _ in nodes]
    for child, parent in enumerate(parents):
        if parent >= 0:
            children_index[parent].append(child)
    x = np.zeros((len(nodes), len(NODE_FEATURES)), dtype=np.float32)
    scaled_lengths = np.asarray([n.branch / height for n in nodes[1:]], dtype=np.float32)
    n_state1 = sum(nodes[i].state == 1 for i in tips)
    n_state2 = sum(nodes[i].state == 2 for i in tips)
    for i, node in enumerate(nodes):
        par = parents[i]
        parent_length = nodes[par].branch if par >= 0 else 0.
        children_desc = [descendants[j] for j in children_index[i]]
        imbalance = (max(children_desc) - min(children_desc)) / n_tips if len(children_desc) >= 2 else 0.
        d = dist[i] / height
        x[i] = (
            float(not node.children), float(par < 0), node.branch / height,
            math.log1p(node.branch) / math.log1p(height), d,
            max(0., 1. - d), depth[i] / max_depth,
            len(node.children) / 4., descendants[i] / n_tips,
            math.log1p(descendants[i]) / math.log1p(n_tips),
            parent_length / height, imbalance,
            float(node.state == 1 and not node.children),
            float(node.state == 2 and not node.children),
            float(node.state == 0 and not node.children), d * d,
        )
    edges, attrs = [], []
    for child_idx in range(1, len(nodes)):
        parent_idx = parents[child_idx]
        node = nodes[child_idx]
        length = node.branch
        base = (length / height, math.log1p(length) / math.log1p(height))
        edges.append((parent_idx, child_idx))
        attrs.append((*base, 1., float(not node.children), dist[parent_idx] / height, dist[child_idx] / height))
        edges.append((child_idx, parent_idx))
        attrs.append((*base, -1., float(not node.children), dist[parent_idx] / height, dist[child_idx] / height))
    gf = np.asarray((
        math.log1p(n_tips), math.log1p(len(nodes)), math.log1p(height),
        float(np.mean(scaled_lengths)), float(np.std(scaled_lengths)),
        n_tips / len(nodes), float(np.mean([depth[i] / max_depth for i in tips])),
        n_state1 / n_tips, n_state2 / n_tips, float(sampling),
    ), dtype=np.float32)
    return Graph(x, np.asarray(edges, np.int32), np.asarray(attrs, np.float32), gf,
                 n_tips, hashlib.sha256(newick.strip().encode("utf-8")).hexdigest()).validate()


def collate_graphs(graphs: Sequence[Graph]) -> Dict[str, np.ndarray]:
    """Disjoint-union batching of Graph objects (NumPy)."""
    if not graphs:
        raise ValueError("Cannot batch an empty list")
    x_list, e_list, a_list, ids, gf = [], [], [], [], []
    offset = 0
    for gidx, graph in enumerate(graphs):
        x_list.append(graph.x)
        e_list.append(graph.edge_index + offset)
        a_list.append(graph.edge_attr)
        ids.append(np.full(len(graph.x), gidx, dtype=np.int32))
        gf.append(graph.graph_features)
        offset += len(graph.x)
    return {
        "x": np.concatenate(x_list).astype(np.float32, copy=False),
        "edge_index": np.concatenate(e_list).astype(np.int64, copy=False),
        "edge_attr": np.concatenate(a_list).astype(np.float32, copy=False),
        "graph_index": np.concatenate(ids),
        "graph_features": np.stack(gf).astype(np.float32),
    }


# =============================================================================
# NEWICK -> PHYLOCNN INPUT (authors' original encoders)
# =============================================================================

_ENCODERS: Dict[str, object] = {}


def find_repo(root: Path) -> Path:
    """The PhyloCNN/PhyloGNN repository folder (Test_Sets/, Encoding/, Training_Data/)."""
    env = os.environ.get("PHYLOCNN_REPO")
    candidates = ([Path(env)] if env else []) + [Path(root) / "PhyloGNN", Path(root) / "phyloCNN", Path(root)]
    for c in candidates:
        if (c / "Test_Sets").is_dir():
            return c.resolve()
    raise FileNotFoundError(f"No repository with Test_Sets/ among {[str(c) for c in candidates]} "
                            "(set PHYLOCNN_REPO)")


def _encoder_module(dataset: str, encoding_dir: Path):
    name = "BiSSE" if dataset == "BiSSE" else "PhyloDyn"
    if name not in _ENCODERS:
        path = Path(encoding_dir) / f"PhyloCNN_Encoding_{name}.py"
        if not path.is_file():
            raise FileNotFoundError(f"Authors' PhyloCNN encoder not found: {path}")
        spec = importlib.util.spec_from_file_location(f"_phylocnn_encoding_{name.lower()}", path)
        module = importlib.util.module_from_spec(spec)
        with warnings.catch_warnings():  # the encoder silences all warnings at import time
            spec.loader.exec_module(module)
        _ENCODERS[name] = module
    return _ENCODERS[name]


def phylocnn_raw_encoding(newick: str, dataset: str, encoding_dir: Path) -> np.ndarray:
    """1000 x F matrix of the authors' encoder (999 node rows + rescaling-factor row)."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        frame = _encoder_module(dataset, encoding_dir).process_tree(newick)
    raw = frame.values
    # The phylodynamics encoder writes float32 to its CSV, as the original pipeline does.
    raw = raw.astype(np.float32) if dataset != "BiSSE" else raw.astype(np.float64).astype(np.float32)
    if raw.shape != (1000, 19 if dataset == "BiSSE" else 18):
        raise ValueError(f"Unexpected PhyloCNN encoding shape {raw.shape}")
    if not np.isfinite(raw).all():
        raise ValueError("Non-finite PhyloCNN encoding")
    return raw


def build_phylocnn_input(raw: np.ndarray, sampling: float) -> np.ndarray:
    """Notebook assembly: drop rescale row, append sampling, sort leaves/internal nodes, pad.

    Shape: (500,19,2) for epidemic models; (500,20,2) for BiSSE.
    """
    if raw.ndim != 2 or raw.shape[0] != 1000 or raw.shape[1] not in (18, 19):
        raise ValueError(f"Invalid raw PhyloCNN encoding {raw.shape}")
    tree = np.concatenate((raw[:-1], np.full((999, 1), sampling, np.float32)), axis=1)
    leaves = tree[tree[:, 3] == 1]
    internal = tree[tree[:, 3] > 1]
    if not len(leaves) or not len(internal):
        raise ValueError("PhyloCNN encoding missing leaves or internal nodes")
    leaves = leaves[np.argsort(leaves[:, 1], kind="stable")]
    internal = internal[np.argsort(internal[:, 1], kind="stable")]
    internal = np.vstack((internal, internal[-1:]))
    if len(leaves) > MAX_TIPS or len(internal) > MAX_TIPS:
        raise ValueError("Encoded phylogeny exceeds 500 tips/internal rows")
    result = np.zeros((MAX_TIPS, tree.shape[-1], 2), dtype=np.float32)
    result[:len(leaves), :, 0] = leaves
    result[:len(internal), :, 1] = internal
    return result


def cnn_feature_columns(dataset: str, variant: str) -> List[int]:
    n = 19 if dataset == "BiSSE" else 18
    if variant == "phylocnn_2gen":
        return list(range(n + 1))
    if variant == "phylocnn_1gen":
        return list(range(10 if dataset == "BiSSE" else 9)) + [n]
    if variant == "phylocnn_nocontext":
        return list(range(5 if dataset == "BiSSE" else 4)) + [n]
    raise ValueError(f"Unknown PhyloCNN variant: {variant}")


# =============================================================================
# DATA SOURCES
# =============================================================================

@dataclass
class TreeRecord:
    source: str
    record_id: int
    dataset: str
    partition: str
    newick: str
    parameters: Mapping[str, str]

    @property
    def y(self):
        return observed_labels(self.dataset, self.parameters)

    @property
    def sampling(self):
        return sampling_from_row(self.parameters)


def split_train_validation(n: int, validation_fraction: float = VALIDATION_FRACTION,
                           seed: int = SPLIT_SEED) -> Tuple[np.ndarray, np.ndarray]:
    """Identical to the PhyloGNN trainer's seeded random split (training/splits.py),
    so train/validation equal Step 4's for the same Training_Data file."""
    perm = np.random.default_rng(seed).permutation(n)
    n_held = max(1, int(round(n * validation_fraction))) if n > 1 else 0
    return np.sort(perm[n_held:]), np.sort(perm[:n_held])


def _read_table(path: Path) -> List[Dict[str, str]]:
    with open_text(path) as handle:
        delim = "\t" if "\t" in handle.readline() else ","
    with open_text(path) as handle:
        reader = csv.DictReader(handle, delimiter=delim)
        if not reader.fieldnames:
            raise ValueError(f"Missing header in {path}")
        return [{k.strip(): v for k, v in row.items()} for row in reader]


def training_files(repo: Path, dataset: str) -> Tuple[Path, Path]:
    folder = repo / "Training_Data" / dataset
    params, trees = folder / f"parameters_{dataset}.txt", folder / f"{dataset}_trees.nwk"
    if not params.is_file() or not trees.is_file():
        raise FileNotFoundError(
            f"Training data for {dataset} not found in {folder}. Build it from the Step 1 simulations with: "
            "python Step4_Experiment2_FullDataset.py --assemble")
    return params, trees


def author_test_files(repo: Path, dataset: str) -> Tuple[List[Path], Path]:
    folder = repo / "Test_Sets" / dataset
    numbered = sorted(folder.glob(f"{dataset}_*.nwk.gz"), key=lambda p: int(p.name.split("_")[-1].split(".")[0]))
    trees = numbered or [folder / f"{dataset}.nwk.gz"]
    params = folder / f"{dataset}.csv.gz"
    for p in [*trees, params]:
        if not p.is_file():
            raise FileNotFoundError(f"Author test file missing: {p}")
    return trees, params


def _newick_lines(paths: Sequence[Path]) -> Iterator[str]:
    for p in paths:
        with open_text(p) as handle:
            for line in handle:
                if line.strip():
                    yield line.strip()


def iter_records(root: Path, dataset: str, partition: str,
                 validation_fraction: float = VALIDATION_FRACTION, seed: int = SPLIT_SEED) -> Iterator[TreeRecord]:
    """Yield aligned (Newick, parameter row) records of one partition.

    partition: 'train', 'validation' (Step 1 simulations) or 'author_test' (authors' Test_Sets).
    record_id is the row number in its source (training file or author test set).
    """
    repo = find_repo(root)
    if partition == "author_test":
        tree_files, params_file = author_test_files(repo, dataset)
        rows = _read_table(params_file)
        n = 0
        for i, (tree, row) in enumerate(zip(_newick_lines(tree_files), rows)):
            n += 1
            yield TreeRecord(str(params_file.parent), i, dataset, partition, tree, row)
        if n != len(rows):
            raise ValueError(f"{dataset} author test: {n} trees but {len(rows)} parameter rows")
        return
    if partition not in ("train", "validation"):
        raise ValueError(f"Unknown partition {partition}")
    params_file, trees_file = training_files(repo, dataset)
    rows = _read_table(params_file)
    train_idx, val_idx = split_train_validation(len(rows), validation_fraction, seed)
    wanted = np.zeros(len(rows), dtype=bool)
    wanted[train_idx if partition == "train" else val_idx] = True
    n = 0
    for i, tree in enumerate(_newick_lines([trees_file])):
        n += 1
        if i < len(rows) and wanted[i]:
            yield TreeRecord(str(trees_file), i, dataset, partition, tree, rows[i])
    if n != len(rows):
        raise ValueError(f"{dataset} training data: {n} trees but {len(rows)} parameter rows")


# Backwards-compatible name used by earlier scripts.
iter_step3_records = iter_records


# =============================================================================
# ENCODING A PARTITION (packed arrays, parallel, cached)
# =============================================================================

@dataclass
class GraphSet:
    """All graphs of one partition, packed into flat arrays (fast batching)."""
    ids: np.ndarray             # (B,)  int64  source row numbers
    y: np.ndarray               # (B,T) float32 targets, natural units
    x: np.ndarray               # (N,F) float32 node features
    edge_index: np.ndarray      # (E,2) int32  LOCAL node indices per graph
    edge_attr: np.ndarray       # (E,D) float32
    node_ptr: np.ndarray        # (B+1,) int64
    edge_ptr: np.ndarray        # (B+1,) int64
    graph_features: np.ndarray  # (B,G) float32
    tree_hashes: np.ndarray     # (B,) str
    cnn: Optional[np.ndarray] = None   # (B,500,F_cnn,2) float32, all PhyloCNN columns
    invalid: List[Dict[str, str]] = field(default_factory=list)

    def __len__(self):
        return len(self.ids)

    @staticmethod
    def pack(items, invalid) -> "GraphSet":
        graphs = [g for _, _, g, _ in items]
        node_counts = np.array([len(g.x) for g in graphs], np.int64)
        edge_counts = np.array([len(g.edge_index) for g in graphs], np.int64)
        cnn = [c for *_, c in items]
        return GraphSet(
            ids=np.array([i for i, *_ in items], np.int64),
            y=np.stack([y for _, y, *_ in items]).astype(np.float32),
            x=np.concatenate([g.x for g in graphs]),
            edge_index=np.concatenate([g.edge_index for g in graphs]).astype(np.int32),
            edge_attr=np.concatenate([g.edge_attr for g in graphs]),
            node_ptr=np.concatenate([[0], np.cumsum(node_counts)]),
            edge_ptr=np.concatenate([[0], np.cumsum(edge_counts)]),
            graph_features=np.stack([g.graph_features for g in graphs]),
            tree_hashes=np.array([g.tree_hash for g in graphs]),
            cnn=np.stack(cnn).astype(np.float32) if cnn and cnn[0] is not None else None,
            invalid=list(invalid),
        )

    def with_graph_features(self, graph_features: np.ndarray) -> "GraphSet":
        return GraphSet(self.ids, self.y, self.x, self.edge_index, self.edge_attr, self.node_ptr,
                        self.edge_ptr, graph_features.astype(np.float32), self.tree_hashes, self.cnn,
                        self.invalid)

    def graph(self, i: int) -> Graph:
        n0, n1, e0, e1 = self.node_ptr[i], self.node_ptr[i + 1], self.edge_ptr[i], self.edge_ptr[i + 1]
        return Graph(self.x[n0:n1], self.edge_index[e0:e1], self.edge_attr[e0:e1],
                     self.graph_features[i], int(self.x[n0:n1, 0].sum()), str(self.tree_hashes[i]))

    def pyg_batch(self, indices, device=None):
        """One torch_geometric Data holding the selected graphs as a disjoint union.

        Attributes: x, edge_index (2,E), edge_attr, batch, graph_features (B,G), num_graphs.
        """
        import torch
        from torch_geometric.data import Data

        idx = np.asarray(indices, dtype=np.int64)
        n_nodes = self.node_ptr[idx + 1] - self.node_ptr[idx]
        n_edges = self.edge_ptr[idx + 1] - self.edge_ptr[idx]
        node_sel = np.concatenate([np.arange(self.node_ptr[i], self.node_ptr[i + 1]) for i in idx])
        edge_sel = np.concatenate([np.arange(self.edge_ptr[i], self.edge_ptr[i + 1]) for i in idx])
        offsets = np.repeat(np.cumsum(n_nodes) - n_nodes, n_edges)
        edge_index = self.edge_index[edge_sel].astype(np.int64) + offsets[:, None]
        data = Data(
            x=torch.from_numpy(self.x[node_sel]),
            edge_index=torch.from_numpy(np.ascontiguousarray(edge_index.T)),
            edge_attr=torch.from_numpy(self.edge_attr[edge_sel]),
            batch=torch.from_numpy(np.repeat(np.arange(len(idx), dtype=np.int64), n_nodes)),
            graph_features=torch.from_numpy(self.graph_features[idx]),
        )
        data.num_graphs = int(len(idx))
        return data.to(device) if device is not None else data


def in_degree_histogram(gs: GraphSet, max_graphs: int = 5000) -> np.ndarray:
    """Node in-degree histogram (needed by PNA), from the TRAINING graphs only."""
    hist = np.zeros(1, np.int64)
    for i in range(min(len(gs), max_graphs)):
        e0, e1, n = gs.edge_ptr[i], gs.edge_ptr[i + 1], gs.node_ptr[i + 1] - gs.node_ptr[i]
        deg = np.bincount(gs.edge_index[e0:e1, 1], minlength=n)
        h = np.bincount(deg)
        if len(h) > len(hist):
            hist = np.pad(hist, (0, len(h) - len(hist)))
        hist[:len(h)] += h
    return hist


def _encode_one(args):
    record, need_cnn, encoding_dir, check_size = args
    try:
        label = record.y
        sampling = record.sampling
        expected = tree_size_from_row(record.parameters) if check_size else None
        graph = tree_to_graph(record.newick, sampling, record.dataset, expected_tips=expected)
        cnn = build_phylocnn_input(phylocnn_raw_encoding(record.newick, record.dataset, encoding_dir), sampling) \
            if need_cnn else None
        return True, (record.record_id, label, graph, cnn)
    except Exception as exc:  # recorded per tree; the caller decides whether to stop
        return False, {"id": str(record.record_id), "source": record.source, "error": f"{type(exc).__name__}: {exc}"}


def prepare_records(records: Iterable[TreeRecord], count: int, need_cnn: bool, skip_invalid: bool,
                    encoding_dir: Optional[Path] = None, workers: int = 1, check_size: bool = True):
    """Encode records (graph + optional PhyloCNN input). Returns (list of tuples, failures).

    Each tuple is (record_id, y, Graph, cnn or None). Encoding stops after `count`
    valid trees when count > 0.
    """
    if need_cnn and encoding_dir is None:
        raise ValueError("encoding_dir (the authors' Encoding/ folder) is required for PhyloCNN inputs")
    collected, failures = [], []
    stream = iter(records)
    chunk = max(64, 16 * workers)
    pool = ProcessPoolExecutor(max_workers=workers) if workers > 1 else None
    try:
        while not count or len(collected) < count:
            batch = []
            for item in stream:
                batch.append((item, need_cnn, encoding_dir, check_size))
                if len(batch) >= chunk:
                    break
            if not batch:
                break
            results = pool.map(_encode_one, batch, chunksize=4) if pool else map(_encode_one, batch)
            for ok, value in results:
                if ok:
                    if not count or len(collected) < count:
                        collected.append(value)
                else:
                    failures.append(value)
                    if not skip_invalid:
                        raise RuntimeError(f"Invalid tree {value['id']} ({value['source']}): {value['error']}")
    finally:
        if pool:
            pool.shutdown(cancel_futures=True)
    if not collected:
        raise RuntimeError("No valid samples; first errors: " + repr(failures[:5]))
    return collected, failures


def _source_fingerprint(repo: Path, dataset: str, partition: str) -> List:
    if partition == "author_test":
        trees, params = author_test_files(repo, dataset)
        files = [*trees, params]
    else:
        files = list(training_files(repo, dataset))
    return [[str(p), p.stat().st_size, p.stat().st_mtime_ns] for p in files]


def encode_partition(root: Path, dataset: str, partition: str, limit: int = 0, need_cnn: bool = False,
                     skip_invalid: bool = True, workers: int = 1, cache_dir: Optional[Path] = None,
                     validation_fraction: float = VALIDATION_FRACTION, seed: int = SPLIT_SEED) -> GraphSet:
    """Encode one partition into a GraphSet, reusing an on-disk cache when the inputs are unchanged."""
    repo = find_repo(root)
    key = {"version": ENCODING_VERSION, "dataset": dataset, "partition": partition, "limit": limit,
           "need_cnn": need_cnn, "skip_invalid": skip_invalid, "validation_fraction": validation_fraction,
           "seed": seed, "sources": _source_fingerprint(repo, dataset, partition)}
    digest = hashlib.sha256(json.dumps(key, sort_keys=True).encode()).hexdigest()[:16]
    cache = Path(cache_dir) / f"{dataset}_{partition}_{digest}.npz" if cache_dir else None
    if cache is not None and cache.is_file():
        with np.load(cache, allow_pickle=False) as z:
            gs = GraphSet(**{k: z[k] for k in ("ids", "y", "x", "edge_index", "edge_attr", "node_ptr",
                                                 "edge_ptr", "graph_features", "tree_hashes")},
                          cnn=z["cnn"] if "cnn" in z.files else None,
                          invalid=json.loads(str(z["invalid_json"])))
        return gs
    items, invalid = prepare_records(iter_records(root, dataset, partition, validation_fraction, seed),
                                     limit, need_cnn, skip_invalid, repo / "Encoding", workers)
    gs = GraphSet.pack(items, invalid)
    if cache is not None:
        cache.parent.mkdir(parents=True, exist_ok=True)
        tmp = cache.with_name(cache.stem + f".{os.getpid()}.tmp.npz")
        arrays = {k: getattr(gs, k) for k in ("ids", "y", "x", "edge_index", "edge_attr", "node_ptr",
                                               "edge_ptr", "graph_features", "tree_hashes")}
        if gs.cnn is not None:
            arrays["cnn"] = gs.cnn
        np.savez(tmp, invalid_json=np.array(json.dumps(gs.invalid)), **arrays)
        os.replace(tmp, cache)
    return gs


# =============================================================================
# LABELS AND METRICS
# =============================================================================

@dataclass
class LabelScaler:
    mean: np.ndarray
    std: np.ndarray
    targets: Tuple[str, ...]

    @classmethod
    def fit(cls, labels, targets):
        values = np.asarray(labels, np.float64)
        if values.ndim != 2 or len(values) == 0:
            raise ValueError("Need nonempty two-dimensional training labels")
        means = values.mean(axis=0)
        stds = values.std(axis=0)
        stds = np.where(stds < 1e-7, 1., stds)
        return cls(means.astype(np.float32), stds.astype(np.float32), tuple(targets))

    def transform(self, y):
        return (np.asarray(y, np.float32) - self.mean) / self.std

    def inverse(self, y):
        return np.asarray(y, np.float32) * self.std + self.mean

    def json(self):
        return {"mean": self.mean.tolist(), "std": self.std.tolist(),
                "targets": list(self.targets), "fitted_on": "TRAIN_ONLY"}


def summarize_metrics(actual: np.ndarray, predicted: np.ndarray,
                      scaler: LabelScaler) -> Tuple[Dict, List[Dict]]:
    """Metrics in ORIGINAL parameter units; normalized error for macro ranking."""
    actual = np.asarray(actual, np.float64)
    predicted = np.asarray(predicted, np.float64)
    if actual.shape != predicted.shape or actual.ndim != 2:
        raise ValueError("Prediction/target shape mismatch")
    if not np.isfinite(actual).all() or not np.isfinite(predicted).all():
        raise ValueError("Non-finite predictions/targets")
    err = predicted - actual
    per_target = []
    for j, name in enumerate(scaler.targets):
        y = actual[:, j]
        e = err[:, j]
        mse = np.mean(e ** 2)
        denom = np.sum((y - y.mean()) ** 2)
        r2 = (1 - np.sum(e ** 2) / denom) if denom > 0 else float("nan")
        nonzero = np.abs(y) > 1e-8
        mape = 100 * np.mean(np.abs(e[nonzero] / y[nonzero])) if nonzero.any() else float("nan")
        per_target.append({
            "target": name, "n": len(y), "mae": float(np.mean(np.abs(e))),
            "rmse": float(np.sqrt(mse)), "r2": float(r2),
            "mape_nonzero_percent": float(mape),
            "mape_n_nonzero": int(nonzero.sum()),
            "normalized_mae": float(np.mean(np.abs(e)) / max(float(scaler.std[j]), 1e-7)),
        })
    aggregate = {
        "n": len(actual), "n_targets": len(scaler.targets),
        "macro_mae_original_units": float(np.abs(err).mean()),
        "macro_normalized_mae": float(np.mean(np.abs(err) / scaler.std[None, :])),
        "macro_rmse_original_units": float(np.sqrt(np.mean(err ** 2))),
    }
    return aggregate, per_target


if __name__ == "__main__":
    sample = "((a:0.1,b:0.2):0.3,(c:0.2,d:0.1):0.3);"
    g = tree_to_graph(sample, .7, min_tips=2)
    batch = collate_graphs([g, g])
    print("ENCODING OK", g.x.shape, g.edge_index.shape, g.edge_attr.shape,
          batch["x"].shape, batch["graph_features"].shape)
