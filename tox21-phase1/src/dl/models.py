"""Phase 2 models. All take a PyG ``Batch`` and return raw logits of shape (B, n_tasks).

* ``GCN``    - graph convolution baseline (ignores bond features).
* ``MPNN``   - Gilmer et al. style: edge-network messages, GRU update, Set2Set readout.
               ``extra_dim > 0`` turns it into the hybrid (graph embedding + engineered features).
* ``FPMLP``  - MLP on the per-molecule ``extra`` vector only (DL control for the fingerprints).
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import (GCNConv, NNConv, Set2Set, global_add_pool,
                                global_max_pool, global_mean_pool)
from torch_geometric.utils import dropout_edge

N_TASKS = 12


def _pool(kind: str, x: torch.Tensor, batch: torch.Tensor) -> torch.Tensor:
    if kind == "sum":
        return global_add_pool(x, batch)
    if kind == "mean":
        return global_mean_pool(x, batch)
    if kind == "max":
        return global_max_pool(x, batch)
    raise ValueError(kind)


class GCN(nn.Module):
    def __init__(self, atom_dim: int, hidden: int = 128, layers: int = 3, dropout: float = 0.2,
                 readout: str = "mean", n_tasks: int = N_TASKS, **_):
        super().__init__()
        self.readout = readout
        self.dropout = dropout
        dims = [atom_dim] + [hidden] * layers
        self.convs = nn.ModuleList(GCNConv(dims[i], dims[i + 1]) for i in range(layers))
        self.norms = nn.ModuleList(nn.BatchNorm1d(hidden) for _ in range(layers))
        self.head = nn.Sequential(nn.Linear(hidden, hidden), nn.ReLU(), nn.Dropout(dropout),
                                  nn.Linear(hidden, n_tasks))

    def embed(self, data) -> torch.Tensor:
        x = data.x
        for conv, norm in zip(self.convs, self.norms):
            x = F.dropout(F.relu(norm(conv(x, data.edge_index))), self.dropout, self.training)
        return _pool(self.readout, x, data.batch)

    def forward(self, data) -> torch.Tensor:
        return self.head(self.embed(data))


class MPNN(nn.Module):
    """Message passing with an edge network (Gilmer et al. 2017), T steps, shared weights."""

    def __init__(self, atom_dim: int, bond_dim: int, hidden: int = 64, steps: int = 3,
                 readout: str = "set2set", dropout: float = 0.2, edge_drop: float = 0.0,
                 extra_dim: int = 0, extra_hidden: int = 128, use_edges: bool = True,
                 n_tasks: int = N_TASKS, **_):
        super().__init__()
        self.steps, self.readout, self.edge_drop, self.use_edges = steps, readout, edge_drop, use_edges
        self.dropout = dropout
        self.lin0 = nn.Linear(atom_dim, hidden)
        edge_net = nn.Sequential(nn.Linear(bond_dim, 128), nn.ReLU(),
                                 nn.Linear(128, hidden * hidden))
        self.conv = NNConv(hidden, hidden, edge_net, aggr="mean")   # M_t(h_v, h_w, e_vw) = A(e_vw) h_w
        self.gru = nn.GRU(hidden, hidden)                           # U_t
        if readout == "set2set":
            self.set2set = Set2Set(hidden, processing_steps=3)      # R
            emb_dim = 2 * hidden
        else:
            emb_dim = hidden
        self.emb_dim = emb_dim
        self.extra = None
        head_in = emb_dim
        if extra_dim > 0:
            self.extra = nn.Sequential(nn.Linear(extra_dim, extra_hidden), nn.ReLU(),
                                       nn.Dropout(dropout))
            head_in += extra_hidden
        self.head = nn.Sequential(nn.Linear(head_in, 2 * hidden), nn.ReLU(), nn.Dropout(dropout),
                                  nn.Linear(2 * hidden, n_tasks))

    def embed(self, data) -> torch.Tensor:
        """Graph-level embedding (B, emb_dim). Encoder interface for hybrids."""
        edge_index, edge_attr = data.edge_index, data.edge_attr
        if not self.use_edges:                       # ablation: remove bond information
            edge_attr = torch.zeros_like(edge_attr)
        if self.training and self.edge_drop > 0:     # DropEdge regularization
            edge_index, mask = dropout_edge(edge_index, p=self.edge_drop, training=True)
            edge_attr = edge_attr[mask]
        out = F.relu(self.lin0(data.x))
        h = out.unsqueeze(0)
        for _ in range(self.steps):
            m = F.relu(self.conv(out, edge_index, edge_attr))
            out, h = self.gru(m.unsqueeze(0), h)
            out = out.squeeze(0)
        if self.readout == "set2set":
            return self.set2set(out, data.batch)
        return _pool(self.readout, out, data.batch)

    def forward(self, data) -> torch.Tensor:
        z = self.embed(data)
        z = F.dropout(z, self.dropout, self.training)
        if self.extra is not None:
            z = torch.cat([z, self.extra(data.extra)], dim=1)
        return self.head(z)


class FPMLP(nn.Module):
    """MLP on data.extra (Morgan bits and/or descriptors). Same loaders as the GNNs."""

    def __init__(self, extra_dim: int, hidden: int = 512, dropout: float = 0.3,
                 n_tasks: int = N_TASKS, **_):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(extra_dim, hidden), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(hidden, hidden // 2), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(hidden // 2, n_tasks))

    def forward(self, data) -> torch.Tensor:
        return self.net(data.extra)
