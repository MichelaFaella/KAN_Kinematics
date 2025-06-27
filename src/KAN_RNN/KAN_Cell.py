import torch
import torch.nn as nn
from src.KAN.KAN_Block import KAN_Block


class KAN_Cell(nn.Module):
    def __init__(self, input_dim, hidden_dim, layer_configs):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim

        self.blocks = nn.ModuleList()
        in_dim = input_dim + hidden_dim  # concatenazione input + hidden
        for cfg in layer_configs:
            block = KAN_Block(
                in_features=in_dim,
                out_features=cfg["out_features"],
                n_knots=cfg["n_knots"],
                x_min=cfg["x_min"],
                x_max=cfg["x_max"],
                use_bn=cfg["use_bn"],
                dropout=cfg["dropout"]
            )
            self.blocks.append(block)
            in_dim = cfg["out_features"]

        self.out_layer = nn.Linear(in_dim, hidden_dim)

    def forward(self, x_t, h_prev):
        x_cat = torch.cat([x_t, h_prev], dim=1)
        out = x_cat
        for block in self.blocks:
            out = block(out)
        h_next = self.out_layer(out)
        return h_next
