import math
import torch
from torch import nn
from einops import rearrange, reduce, einsum

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

class Embedding(nn.Module):
    def __init__(
            self, 
            num_embeddings: int,
            embedding_dim: int,
            device: torch.device | None = None,
            dtype: torch.dtype | None = None
        ) -> torch.Tensor:
        super().__init__()
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.device = device
        self.dtype = dtype
        mean_ = 0.0
        std_ = 1.0
        self.embeddings = nn.init.trunc_normal_(
            torch.empty(self.num_embeddings, self.embedding_dim),
            mean = mean_,
            std = std_,
            a = -3.0 * std_,
            b = 3.0 * std_,    
        )
        self.embeddings = nn.Parameter(self.embeddings).to(self.device)
    
    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        return self.embeddings[token_ids]

class RMSNorm(nn.Module):
    def __init__(
            self,
            d_model:int,
            eps: float = 1e-5,
            device: torch.device | None = None,
            dtype: torch.dtype | None = None
        ):
        super().__init__()
        self.d_model = d_model
        self.eps = eps
        self.device = device
        self.dtype = dtype
        self.gains = nn.Parameter(torch.ones(self.d_model))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # save input dtype before upcast
        in_dtype = x.dtype
        x = x.to(torch.float32)
        rms = torch.sqrt(
            reduce(
                x.pow(2), "batch_size sequence d_model -> batch_size sequence", "mean"
            ) + self.eps)
        x = einsum(
            x, self.gains,
            "batch_size sequence d_model, d_model -> batch_size sequence d_model"
        )
        result = einsum(
            x, 1 / rms,
            "batch_size sequence d_model, batch_size sequence -> batch_size sequence d_model"
        )
        return result.to(in_dtype)

class SwiGLU(nn.Module):
    def __init__(self, d_model: int, d_ff: int):
        super().__init__()
        self.d_model = d_model
        self.d_ff = d_ff
        self.w1_weight = nn.Parameter(torch.randn(self.d_ff, self.d_model))
        self.w2_weight = nn.Parameter(torch.randn(self.d_model, self.d_ff))
        self.w3_weight = nn.Parameter(torch.randn(self.d_ff, self.d_model))

    @staticmethod
    def silu(x: torch.Tensor):
        return x * torch.sigmoid(x)

    def forward(self, x: torch.Tensor):
        w3x = einsum(
            self.w3_weight, x, 
            "d_ff d_model, ... d_model -> ... d_ff"
        )
        w1x = einsum(
            self.w1_weight, x, 
            "d_ff d_model, ... d_model -> ... d_ff"
        )
        sw1x = SwiGLU.silu(w1x)
        sw1xw3x = einsum(
            sw1x, w3x,
            "... d_ff, ... d_ff -> ... d_ff"
        )
        result = einsum(
            self.w2_weight, sw1xw3x,
            "d_model d_ff, ... d_ff -> ... d_model"
        )
        return result