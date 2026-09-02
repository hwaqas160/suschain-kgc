"""Heterogeneous GNN encoders: GraphSAGE, GAT(v2), and an RGCN-style
relation-aware encoder, all built as stacked `HeteroConv` layers so they
share one training loop / API (`forward(x_dict, edge_index_dict) -> x_dict`).

Design note on the "RGCN" variant: PyG's `RGCNConv` assumes a single
node-type graph with typed edges (the classic KG-completion setting). Our
graphs are heterogeneous with *multiple node types* (food/manufacturer/
ingredient/cert), so a direct RGCNConv doesn't apply without first
collapsing to a homogeneous representation. Instead we follow the standard
PyG pattern for relation-aware heterogeneous GNNs: `GraphConv` (a
relation-specific linear transform of the mean-aggregated neighborhood,
plus a root/self weight) applied independently per edge type inside
`HeteroConv(aggr='sum')`, which sums per-relation messages at each
destination node -- structurally the same mechanism as RGCN's
`h_v' = W_0 h_v + sum_r sum_{u in N_r(v)} (1/c_v,r) W_r h_u`, adapted to a
multi-node-type graph. This is stated explicitly here (and belongs in the
paper's methods section) rather than silently claimed as an exact
reproduction of Schlichtkrull et al. (2018).
"""
from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn
from torch_geometric.nn import GATv2Conv, GraphConv, HeteroConv, SAGEConv

Metadata = tuple[list[str], list[tuple[str, str, str]]]


class HeteroGNNEncoder(nn.Module):
    def __init__(
        self,
        metadata: Metadata,
        hidden_channels: int,
        out_channels: int,
        num_layers: int,
        conv_kind: str,
        cross_relation_aggr: str,
        heads: int = 2,
        dropout: float = 0.2,
    ):
        super().__init__()
        node_types, edge_types = metadata
        self.dropout = dropout
        self.convs = nn.ModuleList()

        for layer_idx in range(num_layers):
            out_dim = out_channels if layer_idx == num_layers - 1 else hidden_channels
            conv_dict = {
                et: self._make_conv(conv_kind, out_dim, heads) for et in edge_types
            }
            self.convs.append(HeteroConv(conv_dict, aggr=cross_relation_aggr))

    @staticmethod
    def _make_conv(conv_kind: str, out_dim: int, heads: int):
        if conv_kind == "sage":
            return SAGEConv((-1, -1), out_dim, aggr="mean")
        if conv_kind == "gat":
            # concat=False -> output stays `out_dim` regardless of head count,
            # so layers can be stacked without manual dim bookkeeping.
            return GATv2Conv(
                (-1, -1), out_dim, heads=heads, concat=False,
                add_self_loops=False, dropout=0.1,
            )
        if conv_kind == "rgcn":
            return GraphConv((-1, -1), out_dim, aggr="mean")
        raise ValueError(f"unknown conv_kind: {conv_kind}")

    def forward(self, x_dict: dict[str, torch.Tensor], edge_index_dict) -> dict[str, torch.Tensor]:
        for i, conv in enumerate(self.convs):
            x_dict = conv(x_dict, edge_index_dict)
            if i < len(self.convs) - 1:
                x_dict = {k: F.relu(x) for k, x in x_dict.items()}
                x_dict = {k: F.dropout(x, p=self.dropout, training=self.training) for k, x in x_dict.items()}
        return x_dict


def build_encoder(kind: str, metadata: Metadata, hidden_channels: int = 128, out_channels: int = 64, num_layers: int = 2) -> HeteroGNNEncoder:
    """kind in {'sage', 'gat', 'rgcn'} -- matches the base paper's three
    compared architectures (GraphSAGE baseline, GAT, RGCN), with GAT
    upgraded to GATv2 (strictly more expressive attention mechanism) as a
    documented deviation."""
    cross_relation_aggr = "sum" if kind == "rgcn" else "mean"
    return HeteroGNNEncoder(
        metadata=metadata,
        hidden_channels=hidden_channels,
        out_channels=out_channels,
        num_layers=num_layers,
        conv_kind=kind,
        cross_relation_aggr=cross_relation_aggr,
    )
