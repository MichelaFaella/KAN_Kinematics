import torch.nn as nn
from src.KAN.KAN_Block import KAN_Block


class KAN_Net(nn.Module):
    """
    KAN_Net: a configurable stack of KAN blocks + final linear output layer.

    Args:
        input_dim (int):       dimensionality of input (e.g., 9 actuators)
        layer_configs (list):  list of dicts, one for each KAN_Block.
        
                               Each dict must contain:
                                 - "out_features": int
                                 - "n_knots": int
                                 - "x_min": float
                                 - "x_max": float
                                 - "use_bn": bool
                                 - "activation": nn.Module
                                 - "dropout": float
        output_dim (int):      dimensionality of output (e.g., 3D tip position)
    """

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
            # Each cfg is a dict describing a KAN_Block
            out_dim = cfg["out_features"]

            # Create and append a new block
            block = KAN_Block(
                in_features=in_dim,
                out_features=out_dim,
                n_knots=cfg["n_knots"],
                x_min=cfg["x_min"],
                x_max=cfg["x_max"],
                use_bn=cfg["use_bn"],
                activation=cfg["activation"],
                dropout=cfg["dropout"]
            )
            self.blocks.append(block)

            in_dim = out_dim  # Update for next block

        # Final linear layer: maps last hidden to output_dim
        self.out = nn.Linear(in_dim, output_dim)

    def forward(self, x):
        for block in self.blocks:
            x = block(x)
        return self.out(x)