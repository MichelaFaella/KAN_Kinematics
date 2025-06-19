import torch
import torch.nn as nn

from src.KAN_RNN.KAN_Cell import KAN_Cell


class KAN_Rnn(nn.Module):
    """
    Rete ricorrente che usa KANCell per modellazione dinamica.
    Input: sequenze di attuazione [B, T, input_dim]
    Output: predizione della posizione finale [B, output_dim]
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        layer_configs: list,
        output_dim: int = 3
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        # Prima cella: mapping input_dim -> hidden_dim
        self.cell1 = KAN_Cell(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            layer_configs=layer_configs,
        )
        # Seconda cella: mapping hidden_dim -> hidden_dim
        self.cell2 = KAN_Cell(
            input_dim=hidden_dim,
            hidden_dim=hidden_dim,
            layer_configs=layer_configs,
        )
        # Terza cella: mapping hidden_dim -> hidden_dim
        self.cell3 = KAN_Cell(
            input_dim=hidden_dim,
            hidden_dim=hidden_dim,
            layer_configs=layer_configs,
        )
        # Mappatura dallo stato finale all'output 3D
        self.out = nn.Linear(hidden_dim, output_dim)

    def forward(self, x_seq: torch.Tensor) -> torch.Tensor:
        # x_seq: [B, T, input_dim]
        B, T, _ = x_seq.size()
        # inizializza stati a zero
        h1 = torch.zeros(B, self.hidden_dim, device=x_seq.device)
        h2 = torch.zeros(B, self.hidden_dim, device=x_seq.device)
        h3 = torch.zeros(B, self.hidden_dim, device=x_seq.device)

        # Loop temporale
        for t in range(T):
            h1 = self.cell1(x_seq[:, t, :], h1)
            h2 = self.cell2(h1, h2)
            h3 = self.cell3(h2, h3)

        # Proiezione finale sullo spazio output
        return self.out(h3)
