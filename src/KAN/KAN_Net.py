import torch
import torch.nn as nn
from src.KAN.KAN_Block import KAN_Block


class KAN_Net(nn.Module):
    def __init__(
        self,
        input_dim: int,
        layer_configs: list,
        output_dim: int = 3,
    ):
        super().__init__()
        self.blocks = nn.ModuleList()
        in_dim = input_dim
        for cfg in layer_configs:
            block = KAN_Block(
                in_features=in_dim,
                out_features=cfg["out_features"],
                n_knots=cfg["n_knots"],
                x_min=cfg["x_min"],
                x_max=cfg["x_max"],
                use_bn=cfg["use_bn"],
                dropout=cfg["dropout"],
            )
            self.blocks.append(block)
            in_dim = cfg["out_features"]
        self.out = nn.Linear(in_dim, output_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for block in self.blocks:
            x = block(x)
        return self.out(x)
