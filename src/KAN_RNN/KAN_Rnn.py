import torch
import torch.nn as nn

from src.KAN_RNN.KAN_Cell import KAN_Cell


class KAN_Rnn(nn.Module):
    """
    Rete ricorrente che usa KANCell per modellazione dinamica.
    Input: sequenze di attuazione [B, T, input_dim]
    Output: predizione della posizione finale [B, output_dim]
    """

    def __init__(self,
                 input_dim: int,
                 hidden_dim: int,
                 layer_configs: list,
                 output_dim: int = 3):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.cell = KAN_Cell(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            layer_configs=layer_configs,
        )
        # Mapping dallo stato finale all'output 3D
        self.out = nn.Linear(hidden_dim, output_dim)

    def forward(self, x_seq: torch.Tensor) -> torch.Tensor:
        # x_seq: [B, T, input_dim]
        B, T, _ = x_seq.size()
        h = torch.zeros(B, self.hidden_dim, device=x_seq.device)
        # Loop temporale
        for t in range(T):
            x_t = x_seq[:, t, :]
            h = self.cell(x_t, h)
        # Mappa stato finale a output
        return self.out(h)
