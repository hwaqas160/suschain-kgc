"""Node classification model: the same per-node-type embedding + HeteroGNNEncoder
stack used for link prediction (models/link_predictor.py), with a linear
classification head on a single target node type instead of a decoder.
Sharing the encoder implementation means the same GraphSAGE/GAT/RGCN
architectures compared for link prediction are directly comparable here.
"""
from __future__ import annotations

from torch import nn

from suschain.models.encoders import Metadata, build_encoder


class HeteroNodeClassifier(nn.Module):
    def __init__(
        self,
        metadata: Metadata,
        num_nodes_dict: dict[str, int],
        target_node_type: str,
        num_classes: int,
        embedding_dim: int = 128,
        hidden_channels: int = 128,
        out_channels: int = 64,
        num_layers: int = 2,
        conv_kind: str = "sage",
    ):
        super().__init__()
        self.target_node_type = target_node_type
        self.embeddings = nn.ModuleDict(
            {ntype: nn.Embedding(n, embedding_dim) for ntype, n in num_nodes_dict.items()}
        )
        self.encoder = build_encoder(
            conv_kind, metadata, hidden_channels=hidden_channels,
            out_channels=out_channels, num_layers=num_layers,
        )
        self.classifier = nn.Linear(out_channels, num_classes)

    def forward(self, node_id_dict, edge_index_dict):
        x_dict = {ntype: self.embeddings[ntype](ids) for ntype, ids in node_id_dict.items()}
        z_dict = self.encoder(x_dict, edge_index_dict)
        return self.classifier(z_dict[self.target_node_type])
