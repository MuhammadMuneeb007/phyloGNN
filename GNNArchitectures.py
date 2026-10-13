#!/usr/bin/env python3
"""
GNNArchitectures.py  (Step 2) — PyTorch Geometric

Self-contained definitions of the PhyloGNN graph architectures. Imports only
PyTorch and PyTorch Geometric (no Spektral, no TensorFlow, no phylognn).

Message-passing baselines
  gcn          GCNConv          symmetric-normalised adjacency with self-loops
  graphsage    SAGEConv         self + aggregated neighbours, L2-normalised
  gat          GATConv          multi-head attention over neighbours
  gin          GINConv          MLP((1 + eps) h_i + sum of neighbours), learnable eps
  ecc          NNConv           edge-conditioned filters from the edge features

Edge-aware models (PyTorch Geometric)
  gine         GINEConv         GIN with edge features added to every message
  pna          PNAConv          mean/min/max/std aggregators + degree scalers
  transformer  TransformerConv  graph transformer: attention over edges, edge features in keys/values
  gps          GPSConv          GraphGPS: local GINE message passing + global self-attention per tree

Fusion models
  trignn                = GraphSAGE + GAT + GIN, one head per branch, target-specific fusion weights
  gnn_fusion            = GCN + GraphSAGE + GAT + GIN, fused embeddings, one shared head
  graphsage_gat_fusion  = GraphSAGE + GAT, one head per branch, target-specific fusion weights

Every model:  node features -> [conv -> norm -> ELU (+ residual) -> dropout] x L
              -> global pooling -> concat graph covariates
              -> Dense 64 -> 32 -> 16 -> 8 (ELU; the PhyloCNN head) -> output

Input: one torch_geometric Data / Batch holding a disjoint union of trees with
  x               (N, F)  float   node features
  edge_index      (2, E)  long    [source; target], both directions of every branch
  edge_attr       (E, D)  float   edge features (used by ecc, gine, pna, transformer, gps)
  batch           (N,)    long    graph of each node
  graph_features  (B, G)  float   tree-level covariates (e.g. sampling probability)
Output: (B, output_dim).

Run `python GNNArchitectures.py --check` to build every model and run a
forward/backward pass on a toy batch.
"""

from __future__ import annotations

import inspect
from typing import Optional, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import (
    GATConv,
    GCNConv,
    GINConv,
    GINEConv,
    GPSConv,
    NNConv,
    PNAConv,
    SAGEConv,
    TransformerConv,
    global_add_pool,
    global_max_pool,
    global_mean_pool,
)


# Interface version checked by Step4-CheckPhyloCNN.py (catches mismatched copies of the files).
API_VERSION = 4   # 3 = pooling mean_max / mean_max_root; 4 = mean_max_sum, attention pooling, jk='cat'

# GPSConv gained `attn_type` in newer PyTorch Geometric releases; older ones always use
# multi-head attention, which is what this module asks for anyway.
_GPS_HAS_ATTN_TYPE = "attn_type" in inspect.signature(GPSConv.__init__).parameters


# =============================================================================
# ARCHITECTURE NAMES
# =============================================================================

BASELINE_GNNS = ("gcn", "graphsage", "gat", "gin", "ecc")
EDGE_AWARE_GNNS = ("gine", "pna", "transformer", "gps")
FUSION_GNNS = ("trignn", "gnn_fusion", "graphsage_gat_fusion")
OUR_ARCHITECTURES = BASELINE_GNNS + EDGE_AWARE_GNNS + FUSION_GNNS

# Layers that read edge_attr.
USES_EDGE_FEATURES = {"ecc", "gine", "pna", "transformer", "gps"}

ALIASES = {
    "sage": "graphsage",
    "graph_sage": "graphsage",
    "nnconv": "ecc",
    "graph_transformer": "transformer",
    "graphgps": "gps",
    "tri_gnn": "trignn",
    "tri-gnn": "trignn",
    "fusion": "gnn_fusion",
    "gnnfusion": "gnn_fusion",
    "sage_gat_fusion": "graphsage_gat_fusion",
    "graphsage-gat-fusion": "graphsage_gat_fusion",
}

POOLINGS = {"mean": global_mean_pool, "sum": global_add_pool, "max": global_max_pool}
# Combined readouts (concatenated): mean + max, and mean + max + the root node's embedding.
# The root is the FIRST node of every graph (Graph_Encoding.tree_to_graph adds it first).
POOLING_PARTS = {"mean": ("mean",), "sum": ("sum",), "max": ("max",), "mean_max": ("mean", "max"),
                 "mean_max_root": ("mean", "max", "root"), "mean_max_sum": ("mean", "max", "sum"),
                 "attention": ("attention",)}
# jk ("jumping knowledge"): "cat" pools the concatenation of every layer's node embeddings
# (local and wider neighbourhoods together) instead of the last layer only.
JK_MODES = ("none", "cat")


def _attention_pool(dim):
    """Attention readout: softmax(gate(h)) weighted sum of the node embeddings of each graph."""
    try:
        from torch_geometric.nn.aggr import AttentionalAggregation
        return AttentionalAggregation(gate_nn=nn.Linear(dim, 1)), "aggr"
    except ImportError:  # older PyTorch Geometric
        from torch_geometric.nn import GlobalAttention
        return GlobalAttention(gate_nn=nn.Linear(dim, 1)), "global"
SAGE_AGGREGATIONS = ("mean", "max", "sum")
HEAD_UNITS = (64, 32, 16, 8)

# Binary rooted trees with bidirectional edges: leaves have in-degree 1, the root 2,
# other internal nodes 3. Used for PNA only when no training histogram is given.
BINARY_TREE_DEGREE_HISTOGRAM = (0, 250, 1, 248)


def canonical_architecture(name: str) -> str:
    key = str(name).strip().lower().replace(" ", "_")
    key = ALIASES.get(key, key)
    if key not in OUR_ARCHITECTURES:
        raise ValueError(f"Unknown graph architecture {name!r}. Available: {', '.join(OUR_ARCHITECTURES)}")
    return key


# =============================================================================
# LAYERS
# =============================================================================

def _mlp(in_dim: int, hidden: Sequence[int], out_dim: int) -> nn.Sequential:
    layers, d = [], in_dim
    for h in hidden:
        layers += [nn.Linear(d, h), nn.ELU()]
        d = h
    layers.append(nn.Linear(d, out_dim))
    return nn.Sequential(*layers)


def make_conv(kind: str, in_dim: int, out_dim: int, *, edge_dim: Optional[int] = None, heads: int = 4,
              sage_aggregation: str = "mean", gin_mlp_hidden: Optional[Sequence[int]] = None,
              degree_histogram=None, attention_dropout: float = 0.0) -> nn.Module:
    """One message-passing layer mapping in_dim -> out_dim features (no activation)."""
    if kind in USES_EDGE_FEATURES and not edge_dim:
        raise ValueError(f"{kind} uses edge features: pass edge_dim")
    hidden = list(gin_mlp_hidden) if gin_mlp_hidden else [out_dim]
    if kind == "gcn":
        return GCNConv(in_dim, out_dim)
    if kind == "graphsage":
        if sage_aggregation not in SAGE_AGGREGATIONS:
            raise ValueError(f"sage aggregation must be one of {SAGE_AGGREGATIONS}")
        return SAGEConv(in_dim, out_dim, aggr=sage_aggregation, normalize=True)
    if kind in ("gat", "transformer"):
        if out_dim % heads:
            raise ValueError(f"hidden_dim={out_dim} must be divisible by heads={heads}")
        if kind == "gat":
            return GATConv(in_dim, out_dim // heads, heads=heads, concat=True, dropout=attention_dropout)
        return TransformerConv(in_dim, out_dim // heads, heads=heads, concat=True, edge_dim=edge_dim,
                               dropout=attention_dropout)
    if kind == "gin":
        return GINConv(_mlp(in_dim, hidden, out_dim), train_eps=True)
    if kind == "gine":
        return GINEConv(_mlp(in_dim, hidden, out_dim), train_eps=True, edge_dim=edge_dim)
    if kind == "ecc":
        return NNConv(in_dim, out_dim, nn.Linear(edge_dim, in_dim * out_dim), aggr="add", root_weight=True)
    if kind == "pna":
        deg = torch.as_tensor(degree_histogram if degree_histogram is not None else BINARY_TREE_DEGREE_HISTOGRAM,
                              dtype=torch.long)
        return PNAConv(in_dim, out_dim, aggregators=["mean", "min", "max", "std"],
                       scalers=["identity", "amplification", "attenuation"], deg=deg,
                       edge_dim=edge_dim, towers=1, divide_input=False)
    raise ValueError(f"No single convolution for {kind!r}")


def _norm(kind: str, dim: int) -> nn.Module:
    if kind == "batch":
        return nn.BatchNorm1d(dim)
    if kind == "layer":
        return nn.LayerNorm(dim)
    if kind == "none":
        return nn.Identity()
    raise ValueError("norm must be 'batch', 'layer' or 'none'")


class MessagePassingTower(nn.Module):
    """[conv -> norm -> ELU (+ residual) -> dropout] x L, then graph pooling -> (B, hidden_dim)."""

    def __init__(self, kind: str, input_dim: int, hidden_dim: int = 32, num_layers: int = 3,
                 dropout: float = 0.1, pooling: str = "mean", norm: str = "batch", residual: bool = False,
                 heads: int = 4, sage_aggregation: str = "mean", gin_mlp_hidden=None, edge_dim=None,
                 degree_histogram=None, jk: str = "none"):
        super().__init__()
        if num_layers < 1:
            raise ValueError("num_layers must be >= 1")
        if jk not in JK_MODES:
            raise ValueError(f"jk must be one of {JK_MODES}")
        self.jk = jk
        if pooling not in POOLING_PARTS:
            raise ValueError(f"pooling must be one of {sorted(POOLING_PARTS)}")
        self.kind = kind
        self.uses_edges = kind in USES_EDGE_FEATURES
        self.residual = bool(residual)
        self.dropout = float(dropout)
        self.pool_parts = POOLING_PARTS[pooling]
        node_dim = hidden_dim * (num_layers if jk == "cat" else 1)
        self.out_dim = node_dim * len(self.pool_parts)
        if "attention" in self.pool_parts:
            self.attention, self._attention_api = _attention_pool(node_dim)

        if kind == "gps":
            # GPSConv keeps the width fixed and has its own norms, residuals and MLP.
            self.encoder = nn.Linear(input_dim, hidden_dim)
            gps_options = {"heads": heads, "dropout": dropout}
            if _GPS_HAS_ATTN_TYPE:
                gps_options["attn_type"] = "multihead"
            self.convs = nn.ModuleList(
                GPSConv(hidden_dim,
                        GINEConv(_mlp(hidden_dim, [hidden_dim], hidden_dim), train_eps=True, edge_dim=edge_dim),
                        **gps_options)
                for _ in range(num_layers)
            )
            self.norms = None
            return

        self.encoder = None
        dims = [input_dim] + [hidden_dim] * num_layers
        self.convs = nn.ModuleList(
            make_conv(kind, dims[i], dims[i + 1], edge_dim=edge_dim, heads=heads,
                      sage_aggregation=sage_aggregation, gin_mlp_hidden=gin_mlp_hidden,
                      degree_histogram=degree_histogram)
            for i in range(num_layers)
        )
        self.norms = nn.ModuleList(_norm(norm, hidden_dim) for _ in range(num_layers))

    def forward(self, x, edge_index, edge_attr, batch, num_graphs):
        if self.kind == "gps":
            h = self.encoder(x)
            layers = []
            for conv in self.convs:
                h = conv(h, edge_index, batch, edge_attr=edge_attr)
                layers.append(h)
            return self.pool(torch.cat(layers, dim=-1) if self.jk == "cat" else h, batch, num_graphs)

        h = x
        layers = []
        last = len(self.convs) - 1
        for i, (conv, norm) in enumerate(zip(self.convs, self.norms)):
            z = conv(h, edge_index, edge_attr) if self.uses_edges else conv(h, edge_index)
            z = F.elu(norm(z))
            if self.residual and i > 0:
                z = z + h
            if i < last:
                z = F.dropout(z, p=self.dropout, training=self.training)
            h = z
            layers.append(h)
        return self.pool(torch.cat(layers, dim=-1) if self.jk == "cat" else h, batch, num_graphs)

    def pool(self, h, batch, num_graphs):
        parts = []
        for part in self.pool_parts:
            if part == "root":
                counts = torch.bincount(batch, minlength=num_graphs)
                first = torch.cumsum(counts, 0) - counts          # index of each graph's first node
                parts.append(h[first])
            elif part == "attention":
                if self._attention_api == "aggr":
                    parts.append(self.attention(h, batch, dim_size=num_graphs))
                else:
                    parts.append(self.attention(h, batch, size=num_graphs))
            else:
                parts.append(POOLINGS[part](h, batch, num_graphs))
        return parts[0] if len(parts) == 1 else torch.cat(parts, dim=-1)


class DenseHead(nn.Module):
    """PhyloCNN-style 64 -> 32 -> 16 -> 8 (ELU) -> output head."""

    def __init__(self, in_dim: int, output_dim: int, head_units=HEAD_UNITS):
        super().__init__()
        layers, d = [], in_dim
        for u in head_units:
            layers += [nn.Linear(d, int(u)), nn.ELU()]
            d = int(u)
        layers.append(nn.Linear(d, int(output_dim)))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


def _output_activation(name: Optional[str]):
    name = (name or "linear").lower()
    table = {"linear": lambda t: t, "identity": lambda t: t, "elu": F.elu, "softplus": F.softplus,
             "softmax": lambda t: torch.softmax(t, dim=-1)}
    if name not in table:
        raise ValueError(f"output_activation must be one of {sorted(table)}")
    return table[name]


# =============================================================================
# MODELS
# =============================================================================

class _GraphModel(nn.Module):
    def __init__(self, output_dim: int, num_graph_features: int, output_activation: Optional[str]):
        super().__init__()
        self.output_dim = int(output_dim)
        self.num_graph_features = int(num_graph_features)
        self.output_activation = output_activation or "linear"
        self._activate = _output_activation(output_activation)

    @staticmethod
    def _unpack(data):
        num_graphs = int(data.graph_features.size(0))
        return data.x, data.edge_index, getattr(data, "edge_attr", None), data.batch, num_graphs

    def with_graph_features(self, z, data):
        if self.num_graph_features:
            z = torch.cat([z, data.graph_features.to(z.dtype)], dim=-1)
        return z


class PhyloGNN(_GraphModel):
    """One message-passing tower + graph covariates + dense head."""

    def __init__(self, kind, input_dim, output_dim, num_graph_features=0, head_units=HEAD_UNITS,
                 output_activation=None, **tower_kwargs):
        super().__init__(output_dim, num_graph_features, output_activation)
        self.kind = kind
        self.tower = MessagePassingTower(kind, input_dim, **tower_kwargs)
        self.head = DenseHead(self.tower.out_dim + self.num_graph_features, output_dim, head_units)

    def forward(self, data):
        x, edge_index, edge_attr, batch, b = self._unpack(data)
        z = self.tower(x, edge_index, edge_attr, batch, b)
        return self._activate(self.head(self.with_graph_features(z, data)))


class _PerBranchFusion(_GraphModel):
    """Each branch has its own tower and head; predictions are mixed with
    target-specific softmax weights (one weight vector per output target)."""

    BRANCHES: Sequence[str] = ()

    def __init__(self, input_dim, output_dim, num_graph_features=0, head_units=HEAD_UNITS,
                 output_activation=None, **tower_kwargs):
        super().__init__(output_dim, num_graph_features, output_activation)
        self.towers = nn.ModuleList(MessagePassingTower(b, input_dim, **tower_kwargs) for b in self.BRANCHES)
        self.heads = nn.ModuleList(DenseHead(t.out_dim + self.num_graph_features, output_dim, head_units)
                                   for t in self.towers)
        self.fusion_logits = nn.Parameter(torch.zeros(int(output_dim), len(self.BRANCHES)))

    def forward(self, data):
        x, edge_index, edge_attr, batch, b = self._unpack(data)
        preds = [head(self.with_graph_features(tower(x, edge_index, edge_attr, batch, b), data))
                 for tower, head in zip(self.towers, self.heads)]
        stacked = torch.stack(preds, dim=-1)                      # [batch, target, branch]
        weights = torch.softmax(self.fusion_logits, dim=-1)       # [target, branch]
        return self._activate((stacked * weights.unsqueeze(0)).sum(dim=-1))

    def fusion_weights(self):
        return torch.softmax(self.fusion_logits, dim=-1).detach().cpu().numpy()


class TriGNN(_PerBranchFusion):
    """GraphSAGE + GAT + GIN with target-specific trainable fusion weights."""
    BRANCHES = ("graphsage", "gat", "gin")


class GraphSAGEGATFusion(_PerBranchFusion):
    """GraphSAGE + GAT with target-specific trainable fusion weights."""
    BRANCHES = ("graphsage", "gat")


class GNNFusion(_GraphModel):
    """GCN + GraphSAGE + GAT + GIN: softmax-weighted sum of the branch embeddings, one shared head."""

    BRANCHES = ("gcn", "graphsage", "gat", "gin")

    def __init__(self, input_dim, output_dim, num_graph_features=0, head_units=HEAD_UNITS,
                 output_activation=None, **tower_kwargs):
        super().__init__(output_dim, num_graph_features, output_activation)
        self.towers = nn.ModuleList(MessagePassingTower(b, input_dim, **tower_kwargs) for b in self.BRANCHES)
        self.fusion_logits = nn.Parameter(torch.zeros(len(self.BRANCHES)))
        self.head = DenseHead(self.towers[0].out_dim + self.num_graph_features, output_dim, head_units)

    def forward(self, data):
        x, edge_index, edge_attr, batch, b = self._unpack(data)
        embeds = torch.stack([t(x, edge_index, edge_attr, batch, b) for t in self.towers], dim=1)
        weights = torch.softmax(self.fusion_logits, dim=-1)
        z = (embeds * weights.view(1, -1, 1)).sum(dim=1)
        return self._activate(self.head(self.with_graph_features(z, data)))

    def fusion_weights(self):
        return torch.softmax(self.fusion_logits, dim=-1).detach().cpu().numpy()


# =============================================================================
# BUILDING
# =============================================================================

def build_graph_model(
    architecture: str,
    *,
    input_dim: int,
    output_dim: int,
    num_graph_features: int = 0,
    edge_dim: Optional[int] = None,
    hidden_dim: int = 32,
    num_layers: int = 3,
    dropout: float = 0.1,
    pooling: str = "mean",
    norm: str = "batch",
    residual: bool = False,
    heads: int = 4,
    sage_aggregation: str = "mean",
    gin_mlp_hidden: Optional[Sequence[int]] = None,
    degree_histogram=None,
    head_units=HEAD_UNITS,
    output_activation: Optional[str] = "linear",
    jk: str = "none",
    **_ignored,
) -> nn.Module:
    """Build one graph architecture as a torch.nn.Module.

    edge_dim: number of edge features (required by ecc, gine, pna, transformer, gps
              and by gnn_fusion-style models only if a branch uses them).
    degree_histogram: in-degree histogram of the TRAINING graphs (PNA); defaults to
              that of binary trees.
    output_activation: 'linear' for standardised regression targets (default),
              'elu' / 'softplus' / 'softmax' otherwise.
    """
    architecture = canonical_architecture(architecture)
    tower_kwargs = dict(hidden_dim=hidden_dim, num_layers=num_layers, dropout=dropout, pooling=pooling,
                        norm=norm, residual=residual, heads=heads, sage_aggregation=sage_aggregation,
                        gin_mlp_hidden=gin_mlp_hidden, edge_dim=edge_dim, degree_histogram=degree_histogram,
                        jk=jk)
    common = dict(input_dim=int(input_dim), output_dim=int(output_dim),
                  num_graph_features=int(num_graph_features), head_units=head_units,
                  output_activation=output_activation)
    if architecture in BASELINE_GNNS or architecture in EDGE_AWARE_GNNS:
        model = PhyloGNN(architecture, **common, **tower_kwargs)
    elif architecture == "trignn":
        model = TriGNN(**common, **tower_kwargs)
    elif architecture == "gnn_fusion":
        model = GNNFusion(**common, **tower_kwargs)
    else:
        model = GraphSAGEGATFusion(**common, **tower_kwargs)
    model.phylognn_meta = {
        "model_type": architecture,
        "framework": "pytorch_geometric",
        "architecture_source": "GNNArchitectures.py",
        **{k: v for k, v in tower_kwargs.items() if k != "degree_histogram"},
        **{k: v for k, v in common.items() if k != "head_units"},
    }
    return model


def count_params(model: nn.Module) -> dict:
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    return {"trainable": int(trainable), "non_trainable": int(total - trainable), "total": int(total)}


# =============================================================================
# SELF-CHECK
# =============================================================================

def _toy_batch(input_dim=7, edge_dim=2, num_graph_features=1):
    """Two trees (5 and 3 nodes) as one disjoint batch."""
    from torch_geometric.data import Data

    parent = [-1, 0, 0, 1, 1, -1, 5, 5]
    src, dst = [], []
    for child, par in enumerate(parent):
        if par >= 0:
            src += [par, child]
            dst += [child, par]
    g = torch.Generator().manual_seed(0)
    data = Data(
        x=torch.randn(len(parent), input_dim, generator=g),
        edge_index=torch.tensor([src, dst], dtype=torch.long),
        edge_attr=torch.rand(len(src), edge_dim, generator=g),
        batch=torch.tensor([0, 0, 0, 0, 0, 1, 1, 1]),
        graph_features=torch.rand(2, num_graph_features, generator=g),
    )
    data.num_graphs = 2
    return data


def self_check():
    import torch_geometric

    print(f"torch {torch.__version__} | torch_geometric {torch_geometric.__version__}")
    batch = _toy_batch()
    configs = (dict(), dict(pooling="mean_max_root", num_layers=6, residual=True),
               dict(pooling="mean_max_sum", jk="cat"), dict(pooling="attention"))
    for config in configs:
        print(f"  settings: {config or 'defaults (mean pooling, 3 layers)'}")
        for name in OUR_ARCHITECTURES:
            torch.manual_seed(42)
            model = build_graph_model(name, input_dim=7, output_dim=2, num_graph_features=1, edge_dim=2, **config)
            model.train()
            out = model(batch)
            out.square().mean().backward()
            no_grad = [n for n, p in model.named_parameters() if p.requires_grad and p.grad is None]
            ok = out.shape == (2, 2) and torch.isfinite(out).all() and not no_grad
            print(f"  {'PASS' if ok else 'FAIL'}  {name:22s} params={count_params(model)['trainable']:7,d}  "
                  f"output={tuple(out.shape)}" + (f"  no gradient: {no_grad[:3]}" if no_grad else ""))
            if not ok:
                raise SystemExit(1)
    print(f"All {len(OUR_ARCHITECTURES)} architectures build, run and train.")


if __name__ == "__main__":
    import sys

    print("Step 2 — PhyloGNN graph architectures (PyTorch Geometric)")
    for i, name in enumerate(OUR_ARCHITECTURES, 1):
        print(f"{i:2d}. {name}")
    if "--check" in sys.argv:
        self_check()
