import math
import torch
from torch import nn
from einops import rearrange, einsum

class Linear(nn.Module):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.mean = 0.0
        self.std = math.sqrt(2.0/(self.in_features + self.out_features))
        self.W = nn.init.trunc_normal_(
            torch.empty(self.out_features, self.in_features),
            mean = self.mean,
            std = self.std,
            a = -3.0,
            b = 3.0,    
        )
        self.W = nn.Parameter(self.W)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return einsum(self.W, x, "... d_out d_in, ... d_in -> ... d_out")