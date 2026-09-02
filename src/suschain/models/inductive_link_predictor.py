"""Inductive-capable link predictor (plan C4): identical encoder/decoder
architecture to models/link_predictor.py::HeteroLinkPredictor, but for
node types with `text_features` provided, uses a trainable Linear
projection of a FIXED (frozen, registered as a buffer -- never updated by
gradient descent) sentence-transformer embedding instead of a trainable
nn.Embedding lookup table.

This is the entire mechanism that makes inductive generalization possible
at all: an nn.Embedding row for an entity never seen during training is
whatever it was at random initialization (no gradient ever touched it) --
structurally meaningless. A text feature exists and is meaningful for any
entity with a name, seen or not; only the (shared, trained-on-seen-nodes)
projection Linear layer needs to generalize, not a per-entity parameter.
"""
from __future__ import annotations

import torch
from torch import nn

from suschain.models.decoders import build_decoder
from suschain.models.encoders import Metadata, build_encoder


class HeteroInductiveLinkPredictor(nn.Module):
    def __init__(
        self,
        metadata: Metadata,
        num_nodes_dict: dict[str, int],
        text_features: dict[str, torch.Tensor] | None = None,
        embedding_dim: int = 128,
        hidden_channels: int = 128,
        out_channels: int = 64,
        num_layers: int = 2,
        conv_kind: str = "sage",
        decoder_kind: str = "dot",
    ):
        super().__init__()
        text_features = text_features or {}

        self.embeddings = nn.ModuleDict()
        self.text_projections = nn.ModuleDict()
        for ntype, n in num_nodes_dict.items():
            if ntype in text_features:
                feat = text_features[ntype]
                assert feat.shape[0] == n, f"{ntype}: text_features has {feat.shape[0]} rows, expected {n}"
                self.register_buffer(f"text_feat_{ntype}", feat)
                self.text_projections[ntype] = nn.Linear(feat.shape[1], embedding_dim)
            else:
                self.embeddings[ntype] = nn.Embedding(n, embedding_dim)

        self.encoder = build_encoder(
            conv_kind, metadata, hidden_channels=hidden_channels,
            out_channels=out_channels, num_layers=num_layers,
        )
        self.decoder = build_decoder(decoder_kind, out_channels)

    def _input_features(self, ntype: str, ids: torch.Tensor) -> torch.Tensor:
        if ntype in self.text_projections:
            feat = getattr(self, f"text_feat_{ntype}")[ids]
            return self.text_projections[ntype](feat)
        return self.embeddings[ntype](ids)

    def encode(self, node_id_dict: dict[str, torch.Tensor], edge_index_dict) -> dict[str, torch.Tensor]:
        x_dict = {ntype: self._input_features(ntype, ids) for ntype, ids in node_id_dict.items()}
        return self.encoder(x_dict, edge_index_dict)

    def decode(self, z_dict, edge_label_index: torch.Tensor, src_type: str, dst_type: str) -> torch.Tensor:
        z_src = z_dict[src_type][edge_label_index[0]]
        z_dst = z_dict[dst_type][edge_label_index[1]]
        return self.decoder(z_src, z_dst)
