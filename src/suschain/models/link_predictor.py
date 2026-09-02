"""Link-prediction model: per-node-type trainable embedding table (the
"one-hot/categorical representation" stand-in, see graph_builder.py) feeding
a HeteroGNNEncoder, with a swappable decoder head (dot-product by default;
Phase 1 adds MLP/DistMult/ComplEx decoder alternatives per the plan's
suspicion that the decoder, not the encoder, is the base paper's ranking
bottleneck).
"""
from __future__ import annotations

from torch import nn

from suschain.models.decoders import build_decoder
from suschain.models.encoders import Metadata, build_encoder


class HeteroLinkPredictor(nn.Module):
    def __init__(
        self,
        metadata: Metadata,
        num_nodes_dict: dict[str, int],
        embedding_dim: int = 128,
        hidden_channels: int = 128,
        out_channels: int = 64,
        num_layers: int = 2,
        conv_kind: str = "sage",
        decoder_kind: str = "dot",
    ):
        super().__init__()
        self.embeddings = nn.ModuleDict(
            {ntype: nn.Embedding(n, embedding_dim) for ntype, n in num_nodes_dict.items()}
        )
        self.encoder = build_encoder(
            conv_kind, metadata, hidden_channels=hidden_channels,
            out_channels=out_channels, num_layers=num_layers,
        )
        self.decoder = build_decoder(decoder_kind, out_channels)

    def encode(self, node_id_dict: dict[str, torch.Tensor], edge_index_dict) -> dict[str, torch.Tensor]:
        x_dict = {ntype: self.embeddings[ntype](ids) for ntype, ids in node_id_dict.items()}
        return self.encoder(x_dict, edge_index_dict)

    def decode(self, z_dict: dict[str, torch.Tensor], edge_label_index: torch.Tensor, src_type: str, dst_type: str) -> torch.Tensor:
        z_src = z_dict[src_type][edge_label_index[0]]
        z_dst = z_dict[dst_type][edge_label_index[1]]
        return self.decoder(z_src, z_dst)
