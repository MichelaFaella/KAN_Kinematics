import torch
import torch.nn as nn
from src.KAN_RNN.KAN_Cell import KAN_Cell

class KAN_Rnn(nn.Module):
    """
    Rete ricorrente con una sola cella KAN.
    Input: sequenze di attuazione [B, T, input_dim]
    Output: predizione della posizione finale [B, output_dim]
    """

    def __init__(self, input_dim: int, hidden_dim: int, layer_configs: list, output_dim: int = 3):
        super().__init__()
        self.hidden_dim = hidden_dim

        # Singola cella ricorrente
        self.cell = KAN_Cell(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            layer_configs=layer_configs
        )

        # Mappatura dallo stato finale all'output
        self.out = nn.Linear(hidden_dim, output_dim)

    def forward(self, x_seq: torch.Tensor) -> torch.Tensor:
        # x_seq: [B, T, input_dim]
        B, T, _ = x_seq.size()
        h = torch.zeros(B, self.hidden_dim, device=x_seq.device)

        for t in range(T):
            h = self.cell(x_seq[:, t, :], h)

        return self.out(h)
