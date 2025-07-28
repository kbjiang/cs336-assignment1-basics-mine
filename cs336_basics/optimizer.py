from collections.abc import Callable, Iterable
from typing import Optional
import torch
import math

class AdamW(torch.optim.Optimizer):
    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8, weight_decay=0.001):
        if lr < 0:
            raise ValueError(f"Invalid learning rate: {alpha}")
        defaults = {"lr": lr, "betas": betas, "eps": eps, "weight_decay": weight_decay}
        super().__init__(params, defaults)

    def step(self, closure: Optional[Callable] = None):
        loss = None if closure is None else closure()
        for group in self.param_groups:
            lr = group['lr']
            beta1, beta2 = group['betas']
            eps = group['eps']
            weight_decay = group["weight_decay"]
            for p in group["params"]:
                if p.grad is None:
                    continue

                state = self.state[p]
                grad = p.grad.data
                t = state.get("t", 0)
                # print(t)
                if t == 0:
                    # state["t"] = 0
                    state["m"] = torch.zeros_like(grad)
                    state["v"] = torch.zeros_like(grad)
                
                # +1 to coz we count from 1.
                t += 1
                m, v = state["m"], state["v"]
                m = beta1 * m + (1 - beta1) * grad
                v = beta2 * v + (1 - beta2) * grad **2
                lr_t = lr * (1-beta2**t)**0.5 / (1-beta1**t)
                p.data -= lr_t * m / (v**0.5 + eps)
                p.data -= lr * weight_decay * p.data
                state["t"] = t
                state["m"] = m
                state["v"] = v
        return loss