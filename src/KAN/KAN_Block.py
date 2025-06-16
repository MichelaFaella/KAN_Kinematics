import torch.nn as nn
from src.KAN.Kan_Layer import KAN_Layer


class KAN_Block(nn.Module):
    """
    KAN_Block:
    A modular block combining:
      - KAN_Layer (learnable univariate functions on each connection)
      - Batch Normalization (optional)
      - Activation Function (e.g., ReLU)
      - Dropout (optional)

    Args:
        in_features:     input dimension
        out_features:    output dimension
        n_knots:         number of spline knots per connection
        x_min, x_max:    spline domain range
        use_bn:          enable BatchNorm1d after KAN_Layer
        activation:      non-linear activation function (e.g., ReLU)
        dropout:         dropout probability (set 0.0 to disable)
    """

    def __init__(
            self,
            in_features: int,
            out_features: int,
            n_knots: int = 16,
            x_min: float = -1.0,
            x_max: float = 1.0,
            use_bn: bool = True,
            activation: nn.Module = nn.ReLU(),
            dropout: float = 0.1,
    ):
        super().__init__()

        # KAN Layer: applies piecewise-defined univariate functions per connection
        self.kan = KAN_Layer(in_features, out_features, n_knots, x_min, x_max)

        # BatchNorm: stabilizes training and speeds up convergence
        self.bn = nn.BatchNorm1d(out_features) if use_bn else nn.Identity()

        # Activation: introduces non-linearity
        self.act = activation

        # Dropout: prevents overfitting by randomly deactivating neurons during training
        self.drop = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

    def forward(self, x):
        """
        Forward pass through the block:
          x --KAN--> y --BN--> y --ACT--> y --DROPOUT--> output
        """
        y = self.kan(x)  # Apply KAN layer
        y = self.bn(y)  # Apply BatchNorm (or Identity if disabled)
        y = self.act(y)  # Apply activation function
        return self.drop(y)  # Apply dropout (or Identity)
