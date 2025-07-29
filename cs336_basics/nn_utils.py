import torch
from typing import Iterable

def softmax(x:torch.Tensor, dim:int):
    x -= torch.amax(x, dim=dim, keepdim=True)
    x_exp = torch.exp(x)
    return x_exp/torch.sum(x_exp, axis=dim, keepdim=True)

def cross_entropy(inputs: torch.Tensor, targets:torch.Tensor):
    o = inputs - torch.amax(inputs, dim=-1, keepdim=True)
    o_i = o[torch.arange(inputs.shape[0]), targets]
    z = torch.sum(torch.exp(o), axis=-1)
    loss = - (o_i - torch.log(z))
    return loss.mean()

# def cross_entropy(inputs: torch.Tensor, targets:torch.Tensor):
#     inputs = rearrange(
#         inputs, "batch_size seq_len vocab_size -> (batch_size seq_len) vocab_size"
#     )
#     targets = rearrange(
#         targets, "batch_size seq_len -> (batch_size seq_len)"
#     )
#     o = inputs - torch.amax(inputs, dim=-1, keepdim=True)
#     z = torch.sum(torch.exp(o), axis=-1)
#     o_i = o[torch.arange(o.size(0)), targets]
#     loss = - (o_i - torch.log(z))
#     return loss.mean()

def gradient_clipping(parameters: Iterable[torch.nn.Parameter], max_l2_norm: float, eps: float=1e-6):
    total_norm_2 = sum([torch.sum(p.grad**2) for p in parameters if p.grad is not None])
    total_norm = total_norm_2 ** 0.5

    clip_coef = max_l2_norm / (total_norm + eps)
    if total_norm > max_l2_norm:
        for p in parameters:
            if p.grad is not None:
                p.grad.mul_(clip_coef)