import torch
import torch.nn as nn
from src.KAN.KAN_Block import KAN_Block


class KAN_Cell(nn.Module):

    def __init__(self, input_dim: int, hidden_dim: int, layer_configs: list
                 ):
        super().__init__()

        # Mapping x_t -> hidden
        cfg_in = layer_configs[0]
        self.input_block = KAN_Block(
            in_features=input_dim,
            out_features=hidden_dim,
            n_knots=cfg_in["n_knots"],
            x_min=cfg_in["x_min"],
            x_max=cfg_in["x_max"],
            use_bn=cfg_in["use_bn"],
            activation=cfg_in["activation"],
            dropout=cfg_in["dropout"]
        )

        # Mapping h_{t-1} -> hidden
        cfg_hid = layer_configs[1]
        self.hidden_block = KAN_Block(
            in_features=hidden_dim,
            out_features=hidden_dim,
            n_knots=cfg_hid["n_knots"],
            x_min=cfg_hid["x_min"],
            x_max=cfg_hid["x_max"],
            use_bn=cfg_hid["use_bn"],
            activation=cfg_hid["activation"],
            dropout=cfg_hid["dropout"]
        )

        self.activation = nn.Tanh()

    def forward(self, x_t: torch.Tensor, h_prev: torch.Tensor) -> torch.Tensor:
        # Trasforma input e stato precedente
        i2h = self.input_block(x_t)  # [B, hidden_dim]
        h2h = self.hidden_block(h_prev)  # [B, hidden_dim]
        # Somma + non-linearità
        h_new = self.activation(i2h + h2h)
        return h_new
