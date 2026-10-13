#!/usr/bin/env python3
"""Standalone, topology-faithful phylogenetic graph encoding, PhyloCNN inputs and Step 3 readers.

This module never imports the phylognn package or a GNN library. Encoding uses
NumPy (+ ETE3 for Newick parsing); PyTorch / PyTorch Geometric are imported only
when PyG batches are requested. Python 3.8 compatible.

Newick is the SOLE source of graph edges. Adjacency is never inferred from a
PhyloCNN feature matrix.

Data sources (``root`` = the project folder that holds Data/ and phyloCNN/):
  train / validation : Step 3 output, Data/<NAME>/full_<N>/ (e.g. 100000) or
                       Data/<NAME>/subset_<N>/ (e.g. 10000), all chunks verified.
                       Targets come from raw/parameters_final.tsv.gz (for BiSSE the
                       simulator's state-aligned values), trees from raw/trees.tsv.gz,
                       PhyloCNN inputs from Step 3's encoded/encoded.tsv.gz.
                       validation="step3" uses Step 3's split file
                       (Data/<NAME>/full_trainval_assignments.tsv or subset_...);
                       a fraction such as "0.2" re-splits that pool with a seed.
  author_test        : Data/<NAME>/author_test/ (encoded by Step 3), or, if absent,
                       phyloCNN/Test_Sets/<NAME>/ encoded here with the authors' encoders.
                       Never used for fitting.

Encoded partitions are cached (one directory of .npy files per partition) and
read memory-mapped. A lock directory makes sure only one process encodes a given
partition; other processes wait for it.

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
import shutil
import socket
import sys
import time
import warnings
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Dict, Iterable, Iterator, List, Mapping, Optional, Sequence, Tuple

import numpy as np

ENCODING_VERSION = 3

# Interface version checked by Step4-CheckPhyloCNN.py: 4 = Step 3 exclusions handled and the
# extended metrics (R2, correlations, bias, ...). Bump when Step 4 starts relying on new features.
STEP4_API = 7   # 5 = per-tree rescaling factors (GraphSet.resc); 6 = node_features; 7 = multiscale edges, need_resc

# Newick strings of 500-tip trees can exceed csv's default 128 kB field limit.
csv.field_size_limit(2 ** 31 - 1)

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
# PhyloCNN-derived node features (ablation: the authors' per-node encoder features, attached to the
# real tree topology): the definitions of phyloCNN/Encoding/PhyloCNN_Encoding_*.py, computed for every
# node of the graph. Distances in units of the mean branch length (the authors' rescaling), -1 where a
# relative does not exist (as in the encoder); leaf counts divided by the number of tips; t_s = mean tip
# state below the node (BiSSE; 0 without states).
PHYLOCNN_NODE_FEATURES = (
    "is_leaf", "is_root", "dist_to_present", "dist_to_root", "dist_to_anc", "leaves",
    "dist_to_grand_anc", "dist_to_minor_child", "dist_to_major_child", "minor_leaves", "major_leaves",
    "dist_to_great_grand_anc", "dist_to_minor_minor_grandchild", "dist_to_minor_major_grandchild",
    "dist_to_major_minor_grandchild", "dist_to_major_major_grandchild", "minor_minor_leaves",
    "minor_major_leaves", "major_minor_leaves", "major_major_leaves", "t_s",
)
NODE_FEATURE_SETS = ("standard", "phylocnn", "both")
# Graph connectivity: "tree" = the phylogeny's edges; "multiscale" = also an edge from every node to
# its ancestors 2, 4, 8, 16, 32 and 64 steps up (long-range paths in a few message-passing layers).
# Jump edges carry the same 6 edge features (path length instead of branch length) and
# direction = +-(1 + log2(steps)), so edge-aware models can tell them from tree edges (+-1).
EDGE_SETS = ("tree", "multiscale")
ANCESTOR_JUMPS = (2, 4, 8, 16, 32, 64)


def node_feature_names(node_features: str = "standard") -> Tuple[str, ...]:
    """Names of the node-feature columns: 'standard' (16), 'phylocnn' (21) or 'both' (16 + 19)."""
    if node_features == "standard":
        return tuple(NODE_FEATURES)
    if node_features == "phylocnn":
        return PHYLOCNN_NODE_FEATURES
    if node_features == "both":
        return tuple(NODE_FEATURES) + PHYLOCNN_NODE_FEATURES[2:]
    raise ValueError(f"node_features must be one of {NODE_FEATURE_SETS}")


GRAPH_FEATURES = (
    "log_tips", "log_nodes", "height_log", "mean_branch_scaled",
    "std_branch_scaled", "leaf_fraction", "mean_leaf_depth_scaled",
    "state_one_fraction", "state_two_fraction", "sampling",
)

RAW_CNN_WIDTH = {"BD": 18, "BDEI": 18, "BDSS": 18, "BiSSE": 19}
# Targets measured in units of time. The authors' encoder rescales every tree to a mean branch
# length of 1 (factor r = mean branch length, stored in the last row of the raw encoding), so their
# notebooks train PhyloCNN on rescaled targets (periods / r, rates * r) and convert the predictions
# back with the tree's own r. Scale-free targets (R0, x, f, turnover) are unchanged.
TIME_SCALED = {"infectious_period": "period", "incubation_period": "period",
               "lambda_1": "rate", "lambda_2": "rate", "q_01": "rate"}
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


def _int_id(value) -> int:
    number = float(value)
    if not math.isfinite(number) or not number.is_integer():
        raise ValueError(f"Invalid tree index {value!r}")
    return int(number)


def open_text(path: Path):
    return gzip.open(path, "rt", encoding="utf-8", newline="") if str(path).endswith(".gz") \
        else open(path, "r", encoding="utf-8", newline="")


def _read_table(path: Path) -> List[Dict[str, str]]:
    with open_text(path) as handle:
        delim = "\t" if "\t" in handle.readline() else ","
    with open_text(path) as handle:
        reader = csv.DictReader(handle, delimiter=delim)
        if not reader.fieldnames:
            raise ValueError(f"Missing header in {path}")
        return [{k.strip(): v for k, v in row.items()} for row in reader]


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
        if self.x.ndim != 2 or self.x.shape[1] not in {len(node_feature_names(n)) for n in NODE_FEATURE_SETS}:
            raise ValueError(f"Invalid node matrix {self.x.shape}")
        e = self.edge_index
        if e.ndim != 2 or e.shape[1] != 2 or e.dtype.kind not in "iu":
            raise ValueError(f"Invalid edge index {e.shape}")
        if len(e) < 2 * (len(self.x) - 1) or len(e) % 2:
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
                  expected_tips: Optional[int] = None, node_features: str = "standard",
                  edges: str = "tree") -> Graph:
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
    if node_features != "standard":
        root_length = re.search(r"\)[^():,;\[]*:\s*([-+0-9.eE]+)\s*(?:\[[^\]]*\])?\s*;\s*$", newick.strip())
        extra = _phylocnn_node_features(nodes, parents, children_index, descendants, dist, height, n_tips,
                                        dataset, float(root_length.group(1)) if root_length else 0.0)
        x = extra if node_features == "phylocnn" else np.concatenate([x, extra[:, 2:]], axis=1)
    edges_list, attrs = [], []
    for child_idx in range(1, len(nodes)):
        parent_idx = parents[child_idx]
        node = nodes[child_idx]
        length = node.branch
        base = (length / height, math.log1p(length) / math.log1p(height))
        edges_list.append((parent_idx, child_idx))
        attrs.append((*base, 1., float(not node.children), dist[parent_idx] / height, dist[child_idx] / height))
        edges_list.append((child_idx, parent_idx))
        attrs.append((*base, -1., float(not node.children), dist[parent_idx] / height, dist[child_idx] / height))
    if edges == "multiscale":
        for child_idx in range(1, len(nodes)):
            ancestor, steps = child_idx, 0
            for jump in ANCESTOR_JUMPS:
                while steps < jump and ancestor > 0:
                    ancestor, steps = parents[ancestor], steps + 1
                if steps < jump:
                    break
                length = dist[child_idx] - dist[ancestor]
                base = (length / height, math.log1p(length) / math.log1p(height))
                direction = 1. + math.log2(jump)
                is_tip = float(not nodes[child_idx].children)
                tail = (dist[ancestor] / height, dist[child_idx] / height)
                edges_list.append((ancestor, child_idx))
                attrs.append((*base, direction, is_tip, *tail))
                edges_list.append((child_idx, ancestor))
                attrs.append((*base, -direction, is_tip, *tail))
    elif edges != "tree":
        raise ValueError(f"edges must be one of {EDGE_SETS}")
    gf = np.asarray((
        math.log1p(n_tips), math.log1p(len(nodes)), math.log1p(height),
        float(np.mean(scaled_lengths)), float(np.std(scaled_lengths)),
        n_tips / len(nodes), float(np.mean([depth[i] / max_depth for i in tips])),
        n_state1 / n_tips, n_state2 / n_tips, float(sampling),
    ), dtype=np.float32)
    return Graph(x, np.asarray(edges_list, np.int32), np.asarray(attrs, np.float32), gf,
                 n_tips, hashlib.sha256(newick.strip().encode("utf-8")).hexdigest()).validate()


def _phylocnn_node_features(nodes, parents, children_index, descendants, dist, height, n_tips,
                            dataset="BD", root_length=0.0) -> np.ndarray:
    """The PhyloCNN encoder's per-node features (see PHYLOCNN_NODE_FEATURES) for every graph node,
    with the encoders' conventions: mean branch length over all nodes including the root's own
    branch; the epidemic encoder breaks ties between equal-sized children towards the first child as
    major and gives the root a branch of 0, the BiSSE encoder sorts children by size (ties: first is
    minor) and gives the root -1."""
    n = len(nodes)
    mean_branch = (sum(node.branch for node in nodes[1:]) + max(root_length, 0.0)) / n
    bisse = dataset == "BiSSE"
    scale = mean_branch if mean_branch > 0 else 1.0
    states = np.zeros(n)                                                 # sum of tip states below each node
    for i in range(n - 1, -1, -1):                                       # children come after their parents
        if not nodes[i].children:
            states[i] = nodes[i].state
        if parents[i] >= 0:
            states[parents[i]] += states[i]
    out = np.full((n, len(PHYLOCNN_NODE_FEATURES)), -1.0, dtype=np.float64)

    def split(i):
        """(minor, major) child of node i by number of tips, ties as in the encoder."""
        kids = children_index[i]
        if not kids:
            return None, None
        if len(kids) == 1:
            return kids[0], kids[0]
        if len(kids) == 2 and not bisse:
            a, b = kids
            return (b, a) if descendants[a] >= descendants[b] else (a, b)
        ordered = sorted(kids, key=lambda k: descendants[k])          # stable, as list.sort in the encoder
        return ordered[0], ordered[-1]

    for i, node in enumerate(nodes):
        leaf = not node.children
        p = parents[i]
        gp = parents[p] if p >= 0 else -1
        ggp = parents[gp] if gp >= 0 else -1
        row = out[i]
        row[0], row[1] = float(leaf), float(p < 0)
        row[2] = (height - dist[i]) / scale
        row[3] = dist[i] / scale
        row[4] = node.branch / scale if p >= 0 else (-1.0 if bisse else 0.0)
        row[5] = descendants[i] / n_tips
        row[6] = (dist[i] - dist[gp]) / scale if gp >= 0 else -1.0
        minor, major = split(i)
        if minor is not None:
            row[7], row[8] = nodes[minor].branch / scale, nodes[major].branch / scale
            row[9], row[10] = descendants[minor] / n_tips, descendants[major] / n_tips
        row[11] = (dist[i] - dist[ggp]) / scale if ggp >= 0 else -1.0
        if minor is not None:
            for k, child in enumerate((minor, major)):
                g_minor, g_major = split(child)
                if g_minor is not None:
                    row[12 + 2 * k] = (dist[g_minor] - dist[i]) / scale
                    row[13 + 2 * k] = (dist[g_major] - dist[i]) / scale
                    row[16 + 2 * k] = descendants[g_minor] / n_tips
                    row[17 + 2 * k] = descendants[g_major] / n_tips
        row[20] = states[i] / max(descendants[i], 1)
    return out.astype(np.float32)


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
# PHYLOCNN INPUTS (authors' original encoders and notebook assembly)
# =============================================================================

_ENCODERS: Dict[str, object] = {}


def find_repo(root: Path) -> Path:
    """The PhyloCNN repository folder (Test_Sets/, Encoding/)."""
    env = os.environ.get("PHYLOCNN_REPO")
    candidates = ([Path(env)] if env else []) + [Path(root) / "phyloCNN", Path(root) / "PhyloGNN", Path(root)]
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
    raw = frame.values.astype(np.float32)
    if raw.shape != (1000, RAW_CNN_WIDTH[dataset]):
        raise ValueError(f"Unexpected PhyloCNN encoding shape {raw.shape}")
    if not np.isfinite(raw).all():
        raise ValueError("Non-finite PhyloCNN encoding")
    return raw


def read_encoded_block(path: Path, n_trees: int, dataset: str) -> np.ndarray:
    """Step 3's encoded.tsv.gz: n_trees x 1000 rows of the authors' encoder, one header."""
    import pandas as pd
    frame = pd.read_csv(path, sep="\t", index_col=0)
    raw = frame.to_numpy(dtype=np.float32)
    if raw.shape != (1000 * n_trees, RAW_CNN_WIDTH[dataset]):
        raise ValueError(f"{path}: encoded shape {raw.shape}, expected {(1000 * n_trees, RAW_CNN_WIDTH[dataset])}")
    if not np.isfinite(raw).all():
        raise ValueError(f"{path}: non-finite PhyloCNN encodings")
    return raw.reshape(n_trees, 1000, RAW_CNN_WIDTH[dataset])


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


def phylocnn_rescale_factor(raw: np.ndarray) -> float:
    """The authors' rescaling factor r (mean branch length) from a raw 1000 x F encoding."""
    value = float(raw[-1, -1])
    if not math.isfinite(value) or value <= 0:
        raise ValueError(f"Invalid PhyloCNN rescaling factor {value}")
    return value


def to_rescaled(y, resc, targets: Sequence[str]) -> np.ndarray:
    """Natural-unit targets -> the authors' PhyloCNN training units (periods / r, rates * r)."""
    out = np.array(y, dtype=np.float64, copy=True)
    r = np.asarray(resc, dtype=np.float64).reshape(-1)
    for j, target in enumerate(targets):
        kind = TIME_SCALED.get(target)
        if kind == "period":
            out[:, j] /= r
        elif kind == "rate":
            out[:, j] *= r
    return out


def from_rescaled(y, resc, targets: Sequence[str]) -> np.ndarray:
    """Inverse of to_rescaled: PhyloCNN outputs -> natural units of each tree."""
    out = np.array(y, dtype=np.float64, copy=True)
    r = np.asarray(resc, dtype=np.float64).reshape(-1)
    for j, target in enumerate(targets):
        kind = TIME_SCALED.get(target)
        if kind == "period":
            out[:, j] *= r
        elif kind == "rate":
            out[:, j] /= r
    return out


def cnn_feature_columns(dataset: str, variant: str) -> List[int]:
    n = RAW_CNN_WIDTH[dataset]
    if variant == "phylocnn_2gen":
        return list(range(n + 1))
    if variant == "phylocnn_1gen":
        return list(range(10 if dataset == "BiSSE" else 9)) + [n]
    if variant == "phylocnn_nocontext":
        return list(range(5 if dataset == "BiSSE" else 4)) + [n]
    raise ValueError(f"Unknown PhyloCNN variant: {variant}")


# =============================================================================
# STEP 3 DATA
# =============================================================================

@dataclass
class TreeRecord:
    source: str
    record_id: int
    dataset: str
    partition: str
    newick: str
    parameters: Mapping[str, str]
    encoded: Optional[np.ndarray] = None   # Step 3's raw 1000 x F PhyloCNN encoding, if read

    @property
    def y(self):
        return observed_labels(self.dataset, self.parameters)

    @property
    def sampling(self):
        return sampling_from_row(self.parameters)


def split_train_validation(n: int, validation_fraction: float, seed: int = SPLIT_SEED
                           ) -> Tuple[np.ndarray, np.ndarray]:
    """Seeded random split (same rule as the PhyloGNN trainer's training/splits.py)."""
    perm = np.random.default_rng(seed).permutation(n)
    n_held = max(1, int(round(n * validation_fraction))) if n > 1 else 0
    return np.sort(perm[n_held:]), np.sort(perm[:n_held])


def step3_set(root: Path, dataset: str, trees: int) -> Tuple[Path, Path]:
    """Step 3 tree set with `trees` trees per model -> (set folder, its train/validation split file)."""
    base = Path(root) / "Data" / dataset
    for kind in ("full", "subset"):
        folder = base / f"{kind}_{trees}"
        if (folder / "chunks").is_dir():
            assignments = base / f"{kind}_trainval_assignments.tsv"
            if not assignments.is_file():
                raise FileNotFoundError(f"Step 3 split file missing: {assignments}")
            return folder, assignments
    available = sorted(p.name for p in base.glob("*_*") if (p / "chunks").is_dir()) if base.is_dir() else []
    raise FileNotFoundError(f"No Step 3 tree set with {trees} trees for {dataset} in {base}. "
                            f"Available: {available or 'none'} (run Step3-GenerateData.py first)")


def partition_map(root: Path, dataset: str, trees: int, validation: str = "step3",
                  seed: int = SPLIT_SEED) -> Dict[int, str]:
    """tree index -> 'train' / 'validation' for one Step 3 tree set."""
    _, assignment_file = step3_set(root, dataset, trees)
    rows = _read_table(assignment_file)
    ids = [_int_id(r["index"]) for r in rows]
    if len(set(ids)) != len(ids):
        raise ValueError(f"Duplicate tree IDs in {assignment_file}")
    if str(validation).lower() == "step3":
        out = {i: r["partition"].strip().lower() for i, r in zip(ids, rows)}
        bad = set(out.values()) - {"train", "validation"}
        if bad:
            raise ValueError(f"Unexpected partitions {bad} in {assignment_file}")
        return out
    fraction = float(validation)
    if not 0 < fraction < 1:
        raise ValueError("validation must be 'step3' or a fraction between 0 and 1")
    ordered = sorted(ids)
    train_pos, val_pos = split_train_validation(len(ordered), fraction, seed)
    out = {ordered[k]: "train" for k in train_pos}
    out.update({ordered[k]: "validation" for k in val_pos})
    return out


def _step3_chunks(root: Path, dataset: str, folder: Path) -> List[Path]:
    chunks = sorted(p for p in (folder / "chunks").iterdir() if p.is_dir())
    unfinished = [c.name for c in chunks if not (c / "DONE.json").is_file()]
    expected = None
    tasks_file = Path(root) / "Data" / "_step3" / "tasks.json"
    if tasks_file.is_file():
        tasks = json.loads(tasks_file.read_text())
        expected = sum(1 for t in tasks if t.get("model") == dataset and t.get("split") == "trainval")
    if unfinished or not chunks or (expected is not None and len(chunks) != expected):
        raise RuntimeError(
            f"{folder}: {len(chunks) - len(unfinished)} of {expected or len(chunks)} chunks verified. "
            "Finish Step 3 first (python Step3-GenerateData.py --status).")
    return chunks


def _chunk_records(chunk: Path, dataset: str, with_cnn: bool, param_file: str):
    trees = _read_table(chunk / "raw" / "trees.tsv.gz")
    params = _read_table(chunk / "raw" / param_file)
    if len(trees) != len(params):
        raise ValueError(f"{chunk}: {len(trees)} trees but {len(params)} parameter rows")
    raw = read_encoded_block(chunk / "encoded" / "encoded.tsv.gz", len(trees), dataset) if with_cnn else None
    for k, (t, p) in enumerate(zip(trees, params)):
        tid = _int_id(t["index"])
        pid = _int_id(p["tree_row"] if "tree_row" in p else p["index"])
        if tid != pid:
            raise ValueError(f"{chunk}: tree id {tid} != parameter id {pid} at row {k}")
        yield tid, t["tree"], p, (raw[k] if raw is not None else None)


def _author_test_tree_files(repo: Path, dataset: str) -> Tuple[List[Path], Path]:
    folder = repo / "Test_Sets" / dataset
    numbered = sorted(folder.glob(f"{dataset}_*.nwk.gz"), key=lambda p: int(p.name.split("_")[-1].split(".")[0]))
    trees = numbered or [folder / f"{dataset}.nwk.gz"]
    params = folder / f"{dataset}.csv.gz"
    for p in [*trees, params]:
        if not p.is_file():
            raise FileNotFoundError(f"Author test file missing: {p}")
    return trees, params


def _newick_records(paths: Sequence[Path]) -> Iterator[str]:
    """Semicolon-terminated Newick trees in archive order."""
    for path in paths:
        buffer = ""
        with open_text(path) as handle:
            for line in handle:
                buffer += line.strip()
                while ";" in buffer:
                    tree, buffer = buffer.split(";", 1)
                    if tree.strip():
                        yield tree.strip() + ";"
        if buffer.strip():
            raise ValueError(f"Unterminated Newick record in {path}")


def iter_records(root: Path, dataset: str, partition: str, trees: int = 100000, validation: str = "step3",
                 seed: int = SPLIT_SEED, with_cnn: bool = False) -> Iterator[TreeRecord]:
    """Yield aligned (Newick, parameter row[, PhyloCNN raw encoding]) records of one partition.

    partition: 'train' / 'validation' (Step 3 tree set with `trees` trees) or 'author_test'.
    record_id: Step 3 tree index (train/validation) or row number (author test).
    """
    root = Path(root)
    if partition == "author_test":
        folder = root / "Data" / dataset / "author_test"
        if (folder / "DONE.json").is_file():
            for tid, tree, row, raw in _chunk_records(folder, dataset, with_cnn, "parameters_original.tsv.gz"):
                yield TreeRecord(str(folder), tid, dataset, partition, tree, row, raw)
            return
        tree_files, params_file = _author_test_tree_files(find_repo(root), dataset)
        rows = _read_table(params_file)
        n = 0
        for i, tree in enumerate(_newick_records(tree_files)):
            if i >= len(rows):
                raise ValueError(f"{dataset} author test: more trees than parameter rows")
            n += 1
            yield TreeRecord(str(params_file.parent), i, dataset, partition, tree, rows[i])
        if n != len(rows):
            raise ValueError(f"{dataset} author test: {n} trees but {len(rows)} parameter rows")
        return

    if partition not in ("train", "validation"):
        raise ValueError(f"Unknown partition {partition}")
    folder, _ = step3_set(root, dataset, trees)
    assignment = partition_map(root, dataset, trees, validation, seed)
    seen, excluded = 0, set()
    for chunk in _step3_chunks(root, dataset, folder):
        # Step 3 drops (and records) parameter rows it could not simulate.
        done = json.loads((chunk / "DONE.json").read_text())
        excluded.update(int(e["index"]) for e in done.get("excluded", []))
        for tid, tree, row, raw in _chunk_records(chunk, dataset, with_cnn, "parameters_final.tsv.gz"):
            if tid not in assignment:
                raise ValueError(f"{chunk}: tree {tid} is not in the Step 3 split file")
            seen += 1
            if assignment[tid] == partition:
                yield TreeRecord(str(chunk), tid, dataset, partition, tree, row, raw)
    expected = len(assignment) - len(excluded & set(assignment))
    if seen != expected:
        raise RuntimeError(f"{folder}: read {seen} trees, expected {expected} "
                           f"({len(assignment)} in the split file, {len(excluded & set(assignment))} "
                           "excluded by Step 3)")


# Backwards-compatible name used by earlier scripts.
iter_step3_records = iter_records


# =============================================================================
# ENCODING A PARTITION (packed arrays, parallel, cached, locked)
# =============================================================================

GRAPH_ARRAYS = ("ids", "y", "x", "edge_index", "edge_attr", "node_ptr", "edge_ptr", "graph_features",
                "tree_hashes")


@dataclass
class GraphSet:
    """All graphs of one partition, packed into flat arrays (fast batching)."""
    ids: np.ndarray             # (B,)  int64  Step 3 tree indices / author-test rows
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
    resc: Optional[np.ndarray] = None  # (B,) float64 PhyloCNN rescaling factor of each tree

    def __len__(self):
        return len(self.ids)

    @staticmethod
    def pack(items, invalid) -> "GraphSet":
        graphs = [g for _, _, g in items]
        node_counts = np.array([len(g.x) for g in graphs], np.int64)
        edge_counts = np.array([len(g.edge_index) for g in graphs], np.int64)
        return GraphSet(
            ids=np.array([i for i, _, _ in items], np.int64),
            y=np.stack([y for _, y, _ in items]).astype(np.float32),
            x=np.concatenate([g.x for g in graphs]),
            edge_index=np.concatenate([g.edge_index for g in graphs]).astype(np.int32),
            edge_attr=np.concatenate([g.edge_attr for g in graphs]),
            node_ptr=np.concatenate([[0], np.cumsum(node_counts)]),
            edge_ptr=np.concatenate([[0], np.cumsum(edge_counts)]),
            graph_features=np.stack([g.graph_features for g in graphs]),
            tree_hashes=np.array([g.tree_hash for g in graphs]),
            invalid=list(invalid),
        )

    def with_graph_features(self, graph_features: np.ndarray) -> "GraphSet":
        return GraphSet(self.ids, self.y, self.x, self.edge_index, self.edge_attr, self.node_ptr,
                        self.edge_ptr, graph_features.astype(np.float32), self.tree_hashes, self.cnn,
                        self.invalid, self.resc)

    def graph(self, i: int) -> Graph:
        n0, n1, e0, e1 = self.node_ptr[i], self.node_ptr[i + 1], self.edge_ptr[i], self.edge_ptr[i + 1]
        return Graph(np.asarray(self.x[n0:n1]), np.asarray(self.edge_index[e0:e1]),
                     np.asarray(self.edge_attr[e0:e1]), np.asarray(self.graph_features[i]),
                     int(self.x[n0:n1, 0].sum()), str(self.tree_hashes[i]))

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
        edge_index = np.asarray(self.edge_index[edge_sel], dtype=np.int64) + offsets[:, None]
        data = Data(
            x=torch.from_numpy(np.ascontiguousarray(self.x[node_sel], dtype=np.float32)),
            edge_index=torch.from_numpy(np.ascontiguousarray(edge_index.T)),
            edge_attr=torch.from_numpy(np.ascontiguousarray(self.edge_attr[edge_sel], dtype=np.float32)),
            batch=torch.from_numpy(np.repeat(np.arange(len(idx), dtype=np.int64), n_nodes)),
            graph_features=torch.from_numpy(np.ascontiguousarray(self.graph_features[idx], dtype=np.float32)),
        )
        data.num_graphs = int(len(idx))
        return data.to(device) if device is not None else data


def in_degree_histogram(gs: GraphSet, max_graphs: int = 5000) -> np.ndarray:
    """Node in-degree histogram (needed by PNA), from the TRAINING graphs only."""
    hist = np.zeros(1, np.int64)
    for i in range(min(len(gs), max_graphs)):
        e0, e1, n = gs.edge_ptr[i], gs.edge_ptr[i + 1], gs.node_ptr[i + 1] - gs.node_ptr[i]
        deg = np.bincount(np.asarray(gs.edge_index[e0:e1, 1]), minlength=int(n))
        h = np.bincount(deg)
        if len(h) > len(hist):
            hist = np.pad(hist, (0, len(h) - len(hist)))
        hist[:len(h)] += h
    return hist


def _graph_one(args):
    record, check_size, node_features, edges = args
    try:
        label = record.y
        expected = tree_size_from_row(record.parameters) if check_size else None
        graph = tree_to_graph(record.newick, record.sampling, record.dataset, expected_tips=expected,
                              node_features=node_features, edges=edges)
        return True, (record.record_id, label, graph)
    except Exception as exc:  # recorded per tree; the caller decides whether to stop
        return False, {"id": str(record.record_id), "source": record.source, "error": f"{type(exc).__name__}: {exc}"}


def _cnn_one(args):
    record, encoding_dir = args
    raw = record.encoded if record.encoded is not None else \
        phylocnn_raw_encoding(record.newick, record.dataset, encoding_dir)
    return record.record_id, build_phylocnn_input(raw, record.sampling), phylocnn_rescale_factor(raw)


def _resc_one(args):
    record, encoding_dir = args
    raw = record.encoded if record.encoded is not None else \
        phylocnn_raw_encoding(record.newick, record.dataset, encoding_dir)
    return record.record_id, phylocnn_rescale_factor(raw)


def _chunked(iterable, size):
    batch = []
    for item in iterable:
        batch.append(item)
        if len(batch) >= size:
            yield batch
            batch = []
    if batch:
        yield batch


def prepare_records(records: Iterable[TreeRecord], count: int, skip_invalid: bool, workers: int = 1,
                    check_size: bool = True, node_features: str = "standard", edges: str = "tree"):
    """Encode graphs. Returns ([(record_id, y, Graph)], failures); stops after `count` valid trees (0 = all)."""
    collected, failures = [], []
    pool = ProcessPoolExecutor(max_workers=workers) if workers > 1 else None
    try:
        for batch in _chunked(((r, check_size, node_features, edges) for r in records), max(64, 16 * workers)):
            results = list(pool.map(_graph_one, batch, chunksize=4)) if pool else [_graph_one(b) for b in batch]
            for ok, value in results:
                if ok:
                    if not count or len(collected) < count:
                        collected.append(value)
                else:
                    failures.append(value)
                    if not skip_invalid:
                        raise RuntimeError(f"Invalid tree {value['id']} ({value['source']}): {value['error']}")
            if count and len(collected) >= count:
                break
    finally:
        if pool:
            pool.shutdown()
    if not collected:
        raise RuntimeError("No valid samples; first errors: " + repr(failures[:5]))
    return collected, failures


def _fingerprint(root: Path, dataset: str, partition: str, trees: int) -> List:
    """Cheap description of the inputs (paths, sizes, modification times)."""
    root = Path(root)
    if partition == "author_test":
        folder = root / "Data" / dataset / "author_test"
        if (folder / "DONE.json").is_file():
            files = [folder / "DONE.json"]
        else:
            tree_files, params = _author_test_tree_files(find_repo(root), dataset)
            files = [*tree_files, params]
    else:
        folder, assignments = step3_set(root, dataset, trees)
        files = [assignments, *sorted((folder / "chunks").glob("*/DONE.json"))]
    return [[str(p), p.stat().st_size, p.stat().st_mtime_ns] for p in files]


def _locked_build(final: Path, build: Callable[[Path], None], wait_minutes: float):
    """Build `final` (a directory) exactly once across processes; others wait for it."""
    lock = final.with_name(final.name + ".lock")
    final.parent.mkdir(parents=True, exist_ok=True)
    deadline = time.time() + wait_minutes * 60
    while True:
        if (final / "meta.json").is_file():
            return
        try:
            lock.mkdir()
        except FileExistsError:
            try:
                age = time.time() - lock.stat().st_mtime
            except FileNotFoundError:
                continue
            if age > wait_minutes * 60:            # owner died: take over
                shutil.rmtree(lock, ignore_errors=True)
                continue
            if time.time() > deadline:
                raise RuntimeError(f"Timed out waiting for {final}; delete {lock} if no job is encoding")
            print(f"[WAIT] another task is encoding {final.name}", file=sys.stderr, flush=True)
            time.sleep(30)
            continue
        try:
            (lock / "owner.txt").write_text(f"{socket.gethostname()} pid={os.getpid()} "
                                            f"job={os.environ.get('SLURM_JOB_ID')}\n")
            if not (final / "meta.json").is_file():
                tmp = final.with_name(final.name + f".tmp{os.getpid()}")
                shutil.rmtree(tmp, ignore_errors=True)
                shutil.rmtree(final, ignore_errors=True)
                tmp.mkdir(parents=True)
                try:
                    build(tmp)
                    os.replace(tmp, final)
                finally:
                    shutil.rmtree(tmp, ignore_errors=True)
            return
        finally:
            shutil.rmtree(lock, ignore_errors=True)


def encode_partition(root: Path, dataset: str, partition: str, trees: int = 100000, validation: str = "step3",
                     limit: int = 0, need_cnn: bool = False, skip_invalid: bool = True, workers: int = 1,
                     cache_dir: Optional[Path] = None, seed: int = SPLIT_SEED,
                     wait_minutes: float = 240, log: Callable[..., None] = print,
                     node_features: str = "standard", edges: str = "tree", need_resc: bool = False) -> GraphSet:
    """Encode one partition into a GraphSet (graphs, and PhyloCNN inputs when need_cnn).

    With cache_dir, results are stored once and memory-mapped afterwards; concurrent
    jobs wait for the one that is encoding.
    """
    root = Path(root)
    key = {"version": ENCODING_VERSION, "dataset": dataset, "partition": partition, "limit": limit,
           "skip_invalid": skip_invalid, "seed": seed,
           "trees": None if partition == "author_test" else trees,
           "validation": None if partition == "author_test" else str(validation),
           "sources": _fingerprint(root, dataset, partition, trees)}
    # PhyloCNN inputs do not depend on the graph node features: their cache keeps the standard key.
    cnn_digest = hashlib.sha256(json.dumps(key, sort_keys=True).encode()).hexdigest()[:12]
    if node_features != "standard":
        node_feature_names(node_features)
        key["node_features"] = node_features
    if edges != "tree":
        if edges not in EDGE_SETS:
            raise ValueError(f"edges must be one of {EDGE_SETS}")
        key["edges"] = edges
    digest = hashlib.sha256(json.dumps(key, sort_keys=True).encode()).hexdigest()[:12]
    label = f"{dataset}_{partition}" + ("" if partition == "author_test" else f"_{trees}_{validation}")

    def records(with_cnn: bool):
        return iter_records(root, dataset, partition, trees, validation, seed, with_cnn)

    def build_graphs(folder: Path):
        start = time.time()
        log(f"  encoding graphs: {label} (limit={limit or 'all'}, workers={workers}, node features={node_features})")
        items, invalid = prepare_records(records(False), limit, skip_invalid, workers, node_features=node_features,
                                         edges=edges)
        gs = GraphSet.pack(items, invalid)
        for name in GRAPH_ARRAYS:
            np.save(folder / f"{name}.npy", getattr(gs, name))
        (folder / "invalid.json").write_text(json.dumps(gs.invalid))
        (folder / "meta.json").write_text(json.dumps({**key, "n": len(gs), "seconds": time.time() - start}))

    def load_graphs(folder: Path) -> GraphSet:
        arrays = {name: np.load(folder / f"{name}.npy", mmap_mode="r") for name in GRAPH_ARRAYS}
        arrays["tree_hashes"] = np.load(folder / "tree_hashes.npy")
        return GraphSet(**arrays, invalid=json.loads((folder / "invalid.json").read_text()))

    if cache_dir is None:
        items, invalid = prepare_records(records(False), limit, skip_invalid, workers, node_features=node_features,
                                         edges=edges)
        gs = GraphSet.pack(items, invalid)
    else:
        graph_dir = Path(cache_dir) / f"{label}_{digest}"
        _locked_build(graph_dir, build_graphs, wait_minutes)
        gs = load_graphs(graph_dir)

    if not need_cnn and not need_resc:
        return gs

    resc = np.zeros(len(gs), dtype=np.float64)

    def build_cnn(target):
        start = time.time()
        log(f"  assembling PhyloCNN inputs: {label} ({len(gs)} trees)")
        position = {int(i): k for k, i in enumerate(np.asarray(gs.ids))}
        width = RAW_CNN_WIDTH[dataset] + 1
        out = np.lib.format.open_memmap(target / "cnn.npy", mode="w+", dtype=np.float32,
                                        shape=(len(gs), MAX_TIPS, width, 2)) \
            if isinstance(target, Path) else target
        filled = np.zeros(len(gs), dtype=bool)
        encoding_dir = find_repo(root) / "Encoding"
        wanted = (r for r in records(True) if r.record_id in position)
        pool = ProcessPoolExecutor(max_workers=workers) if workers > 1 else None
        try:
            for batch in _chunked(((r, encoding_dir) for r in wanted), max(64, 16 * workers)):
                results = pool.map(_cnn_one, batch, chunksize=4) if pool else map(_cnn_one, batch)
                for record_id, matrix, factor in results:
                    k = position[record_id]
                    out[k] = matrix
                    resc[k] = factor
                    filled[k] = True
                if filled.all():
                    break
        finally:
            if pool:
                pool.shutdown()
        if not filled.all():
            raise RuntimeError(f"{label}: PhyloCNN inputs missing for {int((~filled).sum())} trees")
        if isinstance(target, Path):
            out.flush()
            del out
            np.save(target / "resc.npy", resc)
            (target / "meta.json").write_text(json.dumps({**key, "graph_digest": cnn_digest,
                                                          "seconds": time.time() - start}))

    def read_resc():
        """The authors' rescaling factor of every tree (from Step 3's encodings, or the encoder)."""
        position = {int(i): k for k, i in enumerate(np.asarray(gs.ids))}
        values = np.full(len(gs), np.nan)
        encoding_dir = find_repo(root) / "Encoding"
        wanted = (r for r in records(True) if r.record_id in position)
        pool = ProcessPoolExecutor(max_workers=workers) if workers > 1 else None
        try:
            for batch in _chunked(((r, encoding_dir) for r in wanted), max(64, 16 * workers)):
                results = pool.map(_resc_one, batch, chunksize=4) if pool else map(_resc_one, batch)
                for record_id, factor in results:
                    values[position[record_id]] = factor
        finally:
            if pool:
                pool.shutdown()
        if not np.isfinite(values).all():
            raise RuntimeError(f"{label}: rescaling factors missing for {int((~np.isfinite(values)).sum())} trees")
        return values

    def build_resc(folder: Path):
        """Rescaling factors on their own (old PhyloCNN caches, or GNNs with PhyloCNN-style targets)."""
        start = time.time()
        log(f"  reading PhyloCNN rescaling factors: {label} ({len(gs)} trees)")
        values = read_resc()
        np.save(folder / "resc.npy", values)
        (folder / "meta.json").write_text(json.dumps({**key, "graph_digest": cnn_digest,
                                                      "seconds": time.time() - start}))

    if not need_cnn:    # rescaling factors only: from the PhyloCNN cache when it exists
        if cache_dir is None:
            gs.resc = read_resc()
            return gs
        cnn_dir = Path(cache_dir) / f"{label}_{cnn_digest}_cnn"
        if (cnn_dir / "meta.json").is_file() and (cnn_dir / "resc.npy").is_file():
            gs.resc = np.load(cnn_dir / "resc.npy").astype(np.float64)
        else:
            resc_dir = Path(cache_dir) / f"{label}_{cnn_digest}_resc"
            _locked_build(resc_dir, build_resc, wait_minutes)
            gs.resc = np.load(resc_dir / "resc.npy").astype(np.float64)
        return gs

    if cache_dir is None:
        cnn = np.zeros((len(gs), MAX_TIPS, RAW_CNN_WIDTH[dataset] + 1, 2), np.float32)
        build_cnn(cnn)
    else:
        cnn_dir = Path(cache_dir) / f"{label}_{cnn_digest}_cnn"
        _locked_build(cnn_dir, build_cnn, wait_minutes)
        cnn = np.load(cnn_dir / "cnn.npy", mmap_mode="r")
        if (cnn_dir / "resc.npy").is_file():
            resc = np.load(cnn_dir / "resc.npy")
        else:   # cache from an older Graph_Encoding.py: read the factors once
            resc_dir = Path(cache_dir) / f"{label}_{cnn_digest}_resc"
            _locked_build(resc_dir, build_resc, wait_minutes)
            resc = np.load(resc_dir / "resc.npy")
    gs.cnn = cnn
    gs.resc = np.asarray(resc, dtype=np.float64)
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


def _correlation(a: np.ndarray, b: np.ndarray) -> float:
    if len(a) < 2 or np.std(a) == 0 or np.std(b) == 0:
        return float("nan")
    return float(np.corrcoef(a, b)[0, 1])


def _ranks(values: np.ndarray) -> np.ndarray:
    """Average ranks (ties share their mean rank), as used by Spearman's rho."""
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(len(values), np.float64)
    ranks[order] = np.arange(1, len(values) + 1)
    _, inverse, counts = np.unique(values, return_inverse=True, return_counts=True)
    sums = np.bincount(inverse, weights=ranks)
    return (sums / counts)[inverse]


def summarize_metrics(actual: np.ndarray, predicted: np.ndarray,
                      scaler: LabelScaler) -> Tuple[Dict, List[Dict]]:
    """Regression metrics in ORIGINAL parameter units (+ normalised versions for ranking).

    Per target: MAE, RMSE, median absolute error, bias (mean signed error), relative
    bias and MAPE (PhyloDeep/PhyloCNN-style relative errors, on non-zero targets),
    median absolute percentage error, R2, Pearson r, Spearman rho, and MAE/RMSE/bias
    divided by the training-set standard deviation of that target.
    Aggregate: the same, averaged over targets (macro).
    """
    actual = np.asarray(actual, np.float64)
    predicted = np.asarray(predicted, np.float64)
    if actual.shape != predicted.shape or actual.ndim != 2:
        raise ValueError("Prediction/target shape mismatch")
    if not np.isfinite(actual).all() or not np.isfinite(predicted).all():
        raise ValueError("Non-finite predictions/targets")
    err = predicted - actual
    per_target = []
    for j, name in enumerate(scaler.targets):
        y, p, e = actual[:, j], predicted[:, j], err[:, j]
        std = max(float(scaler.std[j]), 1e-7)
        rmse = float(np.sqrt(np.mean(e ** 2)))
        denom = np.sum((y - y.mean()) ** 2)
        r2 = (1 - np.sum(e ** 2) / denom) if denom > 0 else float("nan")
        nonzero = np.abs(y) > 1e-8
        relative = e[nonzero] / y[nonzero] if nonzero.any() else np.array([np.nan])
        per_target.append({
            "target": name, "n": len(y),
            "mae": float(np.mean(np.abs(e))),
            "rmse": rmse,
            "median_ae": float(np.median(np.abs(e))),
            "bias": float(np.mean(e)),
            "relative_bias_percent": float(100 * np.mean(relative)),
            "mape_nonzero_percent": float(100 * np.mean(np.abs(relative))),
            "median_ape_percent": float(100 * np.median(np.abs(relative))),
            "mape_n_nonzero": int(nonzero.sum()),
            "r2": float(r2),
            "pearson_r": _correlation(y, p),
            "spearman_rho": _correlation(_ranks(y), _ranks(p)),
            "normalized_mae": float(np.mean(np.abs(e)) / std),
            "normalized_rmse": rmse / std,
            "normalized_bias": float(np.mean(e)) / std,
            "true_mean": float(np.mean(y)), "pred_mean": float(np.mean(p)),
        })

    def macro(key):
        values = [t[key] for t in per_target if np.isfinite(t[key])]
        return float(np.mean(values)) if values else float("nan")

    aggregate = {
        "n": len(actual), "n_targets": len(scaler.targets),
        "macro_mae_original_units": float(np.abs(err).mean()),
        "macro_normalized_mae": float(np.mean(np.abs(err) / scaler.std[None, :])),
        "macro_rmse_original_units": float(np.sqrt(np.mean(err ** 2))),
        "macro_normalized_rmse": macro("normalized_rmse"),
        "macro_r2": macro("r2"),
        "macro_pearson_r": macro("pearson_r"),
        "macro_spearman_rho": macro("spearman_rho"),
        "macro_mape_percent": macro("mape_nonzero_percent"),
        "macro_median_ape_percent": macro("median_ape_percent"),
        "macro_abs_normalized_bias": float(np.mean([abs(t["normalized_bias"]) for t in per_target])),
    }
    return aggregate, per_target


if __name__ == "__main__":
    sample = "((a:0.1,b:0.2):0.3,(c:0.2,d:0.1):0.3);"
    g = tree_to_graph(sample, .7, min_tips=2)
    batch = collate_graphs([g, g])
    print("ENCODING OK", g.x.shape, g.edge_index.shape, g.edge_attr.shape,
          batch["x"].shape, batch["graph_features"].shape)
