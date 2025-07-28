import torch

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