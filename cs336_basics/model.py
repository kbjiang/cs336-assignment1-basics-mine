import math
import torch
from torch import nn
from einops import rearrange, reduce, einsum
from jaxtyping import Float
from nn_utils import softmax

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

class SiLU(nn.Module):
    def __init__(self, d_model: int, d_ff: int):
        super().__init__()
        self.d_model = d_model
        self.d_ff = d_ff
        self.w1_weight = Linear(self.d_model, self.d_ff).W
        self.w2_weight = Linear(self.d_ff, self.d_model).W

    @staticmethod
    def silu(x: torch.Tensor):
        return x * torch.sigmoid(x)

    def forward(self, x: torch.Tensor):
        w1x = einsum(
            self.w1_weight, x, 
            "d_ff d_model, ... d_model -> ... d_ff"
        )
        sw1x = SiLU.silu(w1x)
        result = einsum(
            self.w2_weight, sw1x,
            "d_model d_ff, ... d_ff -> ... d_model"
        )
        return result

class SwiGLU(nn.Module):
    def __init__(self, d_model: int, d_ff: int):
        super().__init__()
        self.d_model = d_model
        self.d_ff = d_ff
        self.w1_weight = Linear(self.d_model, self.d_ff).W
        self.w2_weight = Linear(self.d_ff, self.d_model).W
        self.w3_weight = Linear(self.d_model, self.d_ff).W
        # self.w1_weight = nn.Parameter(torch.randn(self.d_ff, self.d_model))
        # self.w2_weight = nn.Parameter(torch.randn(self.d_model, self.d_ff))
        # self.w3_weight = nn.Parameter(torch.randn(self.d_ff, self.d_model))

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

class RoPE(nn.Module):
    def __init__(
        self,
        theta: float,
        d_k: int,
        max_seq_len: int,
        device: torch.device | None = None,
    ):
        super().__init__()
        self.theta = theta
        self.d_k = d_k
        self.max_seq_len = max_seq_len
        self.device = device
        rotation_matrices = []
        for i in range(max_seq_len):
            for k in range(d_k//2):
                rotation_matrices.append(
                    RoPE.get_rotation_matrix_(i, k, self.theta, self.d_k)
                )
        rotation_matrices = rearrange(
            torch.tensor(rotation_matrices), 
            "(seq_len d_model_half) row col -> seq_len d_model_half row col", 
            seq_len = self.max_seq_len,
        )
        self.register_buffer("rotation_matrices", rotation_matrices, persistent=False)

    @staticmethod
    def get_rotation_matrix_(i, k, theta, d_k):
        angle = i / theta**(2*k/d_k)
        return [[math.cos(angle), -math.sin(angle)], [math.sin(angle), math.cos(angle)]]

    def forward(self, x: torch.Tensor, token_positions: torch.Tensor) -> torch.Tensor:
        rotation_matrices = self.rotation_matrices[token_positions]
        x = rearrange(
            x, "... seq_len (d_model_half m) -> ... seq_len d_model_half m", m = 2
        )
        result = einsum(
            rotation_matrices, x,
            "... seq_len d_model_half row col, ... seq_len d_model_half col -> ... seq_len d_model_half row"
        )
        result = rearrange(
            result, "... seq_len d_model_half row -> ... seq_len (d_model_half row)"
        )
        return result

def scaled_dot_product_attention(
    Q: Float[torch.Tensor, " ... queries d_k"],
    K: Float[torch.Tensor, " ... keys d_k"],
    V: Float[torch.Tensor, " ... values d_v"],
    mask: Float[torch.Tensor, " ... queries keys"] | None = None,
) -> Float[torch.Tensor, " ... queries d_v"]:
    # torch.save(Q, "q.pt")
    # torch.save(K, "k.pt")
    # torch.save(V, "v.pt")
    # torch.save(mask, "mask.pt")
    attention = einsum(
        Q, K,
        "... queries d_k, ... keys d_k -> ... queries keys") / math.sqrt(K.shape[-1])
    if mask is not None:
        attention = attention.masked_fill(~mask, -float("inf"))
    attention = softmax(attention, -1)

    result = einsum(
        attention, V,
        "batch_size ... queries keys, batch_size ... keys d_v-> batch_size ... queries d_v"
    )
    return result

class multihead_self_attention(nn.Module):
    def __init__(
        self, d_model: int, num_heads: int, max_seq_len: int
    ) -> torch.Tensor:
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.max_seq_len = max_seq_len
        self.d_k = self.d_v =  self.d_model // self.num_heads
        self.q_proj_weight = Linear(self.d_model, self.d_model).W
        self.k_proj_weight = Linear(self.d_model, self.d_model).W
        self.v_proj_weight = Linear(self.d_model, self.d_model).W
        self.o_proj_weight = Linear(self.d_model, self.d_model).W
        # self.q_proj_weight = nn.Parameter(torch.randn(self.d_model, self.d_model))
        # self.k_proj_weight = nn.Parameter(torch.randn(self.d_model, self.d_model))
        # self.v_proj_weight = nn.Parameter(torch.randn(self.d_model, self.d_model))
        # self.o_proj_weight = nn.Parameter(torch.randn(self.d_model, self.d_model))
        # Register mask as a buffer so it moves with the model to different devices
        self.register_buffer('mask', ~torch.triu(torch.ones(
            self.max_seq_len, self.max_seq_len), diagonal = 1).bool())
        self.rope = None

    def forward(self, x: torch.Tensor, token_positions=None):
        # Rearrange weights on-the-fly to ensure they're on the correct device
        q_mha = rearrange(
            self.q_proj_weight,
            "(num_h d_q) d -> num_h d_q d",
            num_h = self.num_heads
        )
        k_mha = rearrange(
            self.k_proj_weight,
            "(num_h d_k) d -> num_h d_k d",
            num_h = self.num_heads
        )
        v_mha = rearrange(
            self.v_proj_weight,
            "(num_h d_v) d -> num_h d_v d",
            num_h = self.num_heads
        )
        
        qx = einsum(
            q_mha, x,
            "num_h d_q d, ... seq_len d -> ... num_h seq_len d_q",
        )
        kx = einsum(
            k_mha, x,
            "num_h d_k d, ... seq_len d -> ... num_h seq_len d_k",
        )
        vx = einsum(
            v_mha, x,
            "num_h d_v d, ... seq_len d -> ... num_h seq_len d_v",
        )
        if token_positions is None:
            token_positions = torch.arange(qx.shape[-2], device=qx.device).unsqueeze(dim=0)
        scaled_vx = scaled_dot_product_attention(
            qx, kx, vx, self.mask[:token_positions.shape[-1], :token_positions.shape[-1]])
        scaled_vx = rearrange(
            scaled_vx,
            "batch_size h seq_len d -> batch_size seq_len (h d)"
        )
        res = einsum(
            self.o_proj_weight, scaled_vx,
            "d_out d_in, batch_size seq_len d_in -> batch_size seq_len d_out"
        )
        return res

class multihead_self_attention_with_rope(nn.Module):
    def __init__(
        self, d_model: int, num_heads: int, max_seq_len: int, theta: float
    ) -> torch.Tensor:
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.max_seq_len = max_seq_len
        self.theta = theta
        self.d_k = self.d_v =  self.d_model // self.num_heads
        self.q_proj_weight = Linear(self.d_model, self.d_model).W
        self.k_proj_weight = Linear(self.d_model, self.d_model).W
        self.v_proj_weight = Linear(self.d_model, self.d_model).W
        self.o_proj_weight = Linear(self.d_model, self.d_model).W
        # self.q_proj_weight = nn.Parameter(torch.randn(self.d_model, self.d_model))
        # self.k_proj_weight = nn.Parameter(torch.randn(self.d_model, self.d_model))
        # self.v_proj_weight = nn.Parameter(torch.randn(self.d_model, self.d_model))
        # self.o_proj_weight = nn.Parameter(torch.randn(self.d_model, self.d_model))
        # Register mask as a buffer so it moves with the model to different devices
        self.register_buffer('mask', ~torch.triu(torch.ones(
            self.max_seq_len, self.max_seq_len), diagonal = 1).bool())
        self.rope = RoPE(self.theta, self.d_k, self.max_seq_len)

    def forward(self, x: torch.Tensor, token_positions: torch.Tensor = None):
        # Rearrange weights on-the-fly to ensure they're on the correct device
        q_mha = rearrange(
            self.q_proj_weight,
            "(num_h d_q) d -> num_h d_q d",
            num_h = self.num_heads
        )
        k_mha = rearrange(
            self.k_proj_weight,
            "(num_h d_k) d -> num_h d_k d",
            num_h = self.num_heads
        )
        v_mha = rearrange(
            self.v_proj_weight,
            "(num_h d_v) d -> num_h d_v d",
            num_h = self.num_heads
        )
        
        qx = einsum(
            q_mha, x,
            "num_h d_q d, ... seq_len d -> ... num_h seq_len d_q",
        )
        kx = einsum(
            k_mha, x,
            "num_h d_k d, ... seq_len d -> ... num_h seq_len d_k",
        )
        vx = einsum(
            v_mha, x,
            "num_h d_v d, ... seq_len d -> ... num_h seq_len d_v",
        )
        if token_positions is None:
            token_positions = torch.arange(qx.shape[-2], device=qx.device).unsqueeze(dim=0)
        qx = self.rope(qx, token_positions)
        kx = self.rope(kx, token_positions)
        scaled_vx = scaled_dot_product_attention(
            qx, kx, vx, self.mask[:token_positions.shape[-1], :token_positions.shape[-1]])
        scaled_vx = rearrange(
            scaled_vx,
            "batch_size h seq_len d -> batch_size seq_len (h d)"
        )
        res = einsum(
            self.o_proj_weight, scaled_vx,
            "d_out d_in, batch_size seq_len d_in -> batch_size seq_len d_out"
        )
        return res

class transformer_block(nn.Module):
    def __init__(
        self, d_model: int, num_heads: int, d_ff: int, max_seq_len: int, theta: float):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_ff = d_ff
        self.rmsnorm1 = RMSNorm(self.d_model)
        self.rmsnorm2 = RMSNorm(self.d_model)
        self.attn = multihead_self_attention_with_rope(
            d_model, num_heads, max_seq_len, theta
        # self.attn = multihead_self_attention(
        #     d_model, num_heads, max_seq_len,
        )
        self.ffn = SwiGLU(self.d_model, self.d_ff)
        # self.ffn = SiLU(self.d_model, self.d_ff)

    def forward(self, x:torch.Tensor):
        y = x + self.attn(self.rmsnorm1(x), token_positions=None)
        z = y + self.ffn(self.rmsnorm2(y))
        return z

class transformer_lm(nn.Module):
    def __init__(
        self,
        d_model: int,
        d_ff: int,
        num_heads: int,
        rope_theta: float,
        num_layers: int,
        vocab_size: int,
        context_length: int,
    ):
        super().__init__()
        self.d_model = d_model
        self.d_ff = d_ff
        self.num_heads = num_heads
        self.num_layers = num_layers
        self.vocab_size = vocab_size
        self.context_length = context_length
        self.rope_theta = rope_theta
        # self.token_embeddings = nn.Parameter(torch.randn(self.vocab_size, self.d_model))
        self.token_embeddings = Linear(self.d_model, self.vocab_size).W
        self.layers = nn.ModuleList([
            transformer_block(
                self.d_model, self.num_heads, self.d_ff, self.context_length, self.rope_theta
            ) for i in range(num_layers)])
        self.rmsnorm_final = RMSNorm(self.d_model)
        self.lm_head = Linear(
            out_features = self.vocab_size,
            in_features = self.d_model
        )

    def forward(self, in_indices: torch.Tensor):
        x = self.token_embeddings[in_indices]
        for i in range(self.num_layers):
            x = self.layers[i](x)
        x = self.rmsnorm_final(x)
        x = self.lm_head(x)
        return x
