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
        super().__init__()  # Important: call parent constructor
        self.in_f = in_features
        self.out_f = out_features
        self.nk = n_knots

        # Register knot locations as a constant buffer: shape [n_knots]
        self.register_buffer(
            "knots",
            torch.linspace(x_min, x_max, n_knots)
        )

        # Learnable spline coefficients for each arc (output ← input): shape [out_f, in_f, n_knots]
        self.coeffs = nn.Parameter(
            torch.randn(out_features, in_features, n_knots) * 0.05
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [B, in_features] - batch of input vectors

        Returns:
            y: [B, out_features] - output of the KAN layer
        --------------------------------------------
        Legend:
            B         - batch size
            in_f      - number of input dimensions
            out_f     - number of output dimensions
            nk        - number of spline knots
            knots     - tensor of [nk] spline node positions (fixed)
            coeffs    - tensor of shape [out_f, in_f, nk] (learnable)
        --------------------------------------------
        """
        B = x.size(0)  # batch size

        # Prepare input for broadcasting: shape [B, 1, in_f]
        x_exp = x.view(B, 1, self.in_f).expand(-1, self.out_f, -1)  # [B, out_f, in_f]

        # Normalize x values into [0, nk - 1] range
        pos = (x_exp - self.knots[0]) / (self.knots[-1] - self.knots[0]) * (self.nk - 1)

        # Clamp position to valid range and get left/right indices
        id_x0 = pos.floor().long().clamp(0, self.nk - 2)  # [B, out_f, in_f]
        id_x1 = id_x0 + 1

        # Expand coefficients: shape [B, out_f, in_f, nk]
        C = self.coeffs.unsqueeze(0).expand(B, -1, -1, -1)

        # Index selection needs shape [B, out_f, in_f, 1]
        index_x0 = id_x0.unsqueeze(-1)
        index_x1 = id_x1.unsqueeze(-1)

        # Gather coefficients at both ends of the interpolation interval
        c_0 = torch.gather(C, 3, index_x0).squeeze(-1)  # [B, out_f, in_f]
        c_1 = torch.gather(C, 3, index_x1).squeeze(-1)  # [B, out_f, in_f]

        # Linear interpolation weight t in [0, 1]
        t = (pos - id_x0.float()).unsqueeze(-1)  # [B, out_f, in_f, 1]

        # Linear interpolation
        phi = c_0 * (1 - t.squeeze(-1)) + c_1 * t.squeeze(-1)  # [B, out_f, in_f]

        # Sum over input features to produce output: [B, out_f]
        return phi.sum(dim=2)

