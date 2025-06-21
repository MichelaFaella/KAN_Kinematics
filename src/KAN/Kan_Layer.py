import torch
import torch.nn as nn


class KAN_Layer(nn.Module):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        n_knots: int = 16,
        x_min: float = -1.0,
        x_max: float = 1.0,
    ):
        super().__init__()
        self.in_f = in_features
        self.out_f = out_features
        self.nk = n_knots
        # Fixed knot positions
        self.register_buffer(
            "knots",
            torch.linspace(x_min, x_max, n_knots)
        )
        # Learnable spline coefficients per connection
        self.coeffs = nn.Parameter(
            torch.randn(out_features, in_features, n_knots) * 0.05
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B = x.size(0)
        # Expand input to [B, out_f, in_f]
        x_exp = x.view(B, 1, self.in_f).expand(-1, self.out_f, -1)
        # Map x onto knot index range [0, nk-1]
        pos = (
            (x_exp - self.knots[0]) /
            (self.knots[-1] - self.knots[0]) *
            (self.nk - 1)
        )
        # Indices for interpolation
        id_x0 = pos.floor().long().clamp(0, self.nk - 2)
        id_x1 = id_x0 + 1
        # Gather coefficients
        C = self.coeffs.unsqueeze(0).expand(B, -1, -1, -1)
        i0 = id_x0.unsqueeze(-1)
        i1 = id_x1.unsqueeze(-1)
        c0 = torch.gather(C, 3, i0).squeeze(-1)
        c1 = torch.gather(C, 3, i1).squeeze(-1)
        # Interpolation weight
        t = (pos - id_x0.float()).unsqueeze(-1)
        phi = c0 * (1 - t.squeeze(-1)) + c1 * t.squeeze(-1)
        # Sum over inputs
        return phi.sum(dim=2)
