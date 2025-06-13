import torch
import torch.nn as nn


class TransformerEncoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_heads, num_layers, dropout=0.1):
        super(TransformerEncoder, self).__init__()

        self.embedding = nn.Linear(input_dim, hidden_dim)  # 映射到 Transformer 维度
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim * 4,
            dropout=dropout,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        self.output_layer = nn.Linear(hidden_dim, input_dim)

    def forward(self, x, mask):
        x = self.embedding(x)
        mask = (mask == 0)
        out = self.transformer(x, src_key_padding_mask=mask)
        out = self.output_layer(out)
        return out


class GraphDecoder2(nn.Module):
    def __init__(self, hidden_dim, node_dim):

        super(GraphDecoder2, self).__init__()

        self.mlp = nn.Sequential(
            nn.Linear(hidden_dim, 2 * hidden_dim),
            nn.ReLU(),
            nn.Linear(2 * hidden_dim, node_dim)
        )

    def forward(self, h, mask_nodes):

        h_masked = h[mask_nodes]
        x_reconstructed = self.mlp(h_masked)
        return x_reconstructed
