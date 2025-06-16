import torch.nn as nn


class MLP_Net(nn.Module):
    """
    MLP_Net: A configurable feedforward neural network (fully connected).

    Args:
        input_dim (int):    input size (e.g., 9 actuator signals)
        hidden_dims (list): list of hidden layer sizes (e.g., [64, 32])
        output_dim (int):   output size (e.g., 3 for tip position XYZ)
        use_bn (bool):      whether to apply BatchNorm after each linear layer
        activation (nn.Module): activation function (e.g. ReLU, GELU)
        dropout (float):    dropout probability (0.0 to disable)
    """

    def __init__(
            self,
            input_dim: int,
            hidden_dims: list,
            output_dim: int = 3,
            use_bn: bool = True,
            activation: nn.Module = nn.ReLU(),
            dropout: float = 0.1
    ):
        super().__init__()

        layers = []
        in_dim = input_dim

        for h in hidden_dims:
            layers.append(nn.Linear(in_dim, h))
            if use_bn:
                layers.append(nn.BatchNorm1d(h))
            layers.append(activation)
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            in_dim = h

        # Final output layer: no activation
        layers.append(nn.Linear(in_dim, output_dim))

        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)
