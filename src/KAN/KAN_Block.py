from torch import nn
from src.KAN.Kan_Layer import KAN_Layer


class KAN_Block(nn.Module):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        n_knots: int = 16,
        x_min: float = -1.0,
        x_max: float = 1.0,
        use_bn: bool = True,
        activation: nn.Module = None,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.kan = KAN_Layer(in_features, out_features, n_knots, x_min, x_max)
        self.bn = nn.BatchNorm1d(out_features) if use_bn else nn.Identity()
        self.act = activation if activation is not None else nn.Identity()
        self.drop = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

    def forward(self, x):
        x = self.kan(x)
        x = self.bn(x)
        x = self.act(x)
        x = self.drop(x)
        return x
