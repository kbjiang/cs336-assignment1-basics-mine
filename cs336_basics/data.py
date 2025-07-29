import torch
import numpy as np
import random

def get_batch(
    dataset: np.ndarray,
    batch_size: int,
    context_length: int,
    device:str = "cpu"
) -> tuple[torch.tensor, torch.tensor]:
    starting_indices = torch.randint(0, len(dataset) - context_length, (batch_size,))
    x = torch.stack([torch.from_numpy(dataset[sid:sid+context_length]) for sid in starting_indices])
    y = torch.stack([torch.from_numpy(dataset[sid+1:sid+context_length+1]) for sid in starting_indices])
    return x.to(device), y.to(device)

class Dataset:
    def __init__(self, src: str):
        self.dataset = np.load(src, mmap_mode = "r")

    def get_batch(self, batch_size, context_length, device="cpu"):
        return get_batch(self.dataset, batch_size, context_length, device)
