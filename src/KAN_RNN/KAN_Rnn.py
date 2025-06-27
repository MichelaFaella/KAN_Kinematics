import torch
import torch.nn as nn
from src.KAN_RNN.KAN_Cell import KAN_Cell

class KAN_Rnn(nn.Module):
    def __init__(self, input_dim, hidden_dim, layer_configs, output_dim):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.kan_cell = KAN_Cell(input_dim, hidden_dim, layer_configs)
        self.output_layer = nn.Linear(hidden_dim, output_dim)

    def forward(self, x_seq):
        """
        x_seq shape: [B, T, input_dim]
        """
        B, T, _ = x_seq.shape
        h = torch.zeros(B, self.hidden_dim, device=x_seq.device)

        for t in range(T):
            x_t = x_seq[:, t, :]
            h = self.kan_cell(x_t, h)

        out = self.output_layer(h)
        return out
