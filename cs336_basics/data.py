import torch
import numpy as np
import random

def get_batch(
    dataset: np.ndarray,
    batch_size: int,
    context_length: int,
    device:str = "cpu"
) -> tuple[torch.tensor, torch.tensor]:
    """get batch with replacement"""
    starting_indices = torch.randint(0, len(dataset) - context_length, (batch_size,))
    x = torch.stack([torch.from_numpy(dataset[sid:sid+context_length].copy()) for sid in starting_indices])
    y = torch.stack([torch.from_numpy(dataset[sid+1:sid+context_length+1].copy()) for sid in starting_indices])
    return x.to(device), y.to(device)

class Dataset:
    def __init__(self, src: str, wo_repl: bool = False):
        self.dataset = np.load(src, mmap_mode = "r")
        self.wo_repl = wo_repl

    def get_batch(self, batch_size, context_length, device="cpu"):
        if self.wo_repl:
            if not hasattr(self, 'starting_indices') or len(self.starting_indices) < batch_size:
                # Create non-overlapping sequences stepping by context_length
                random_start_id = random.choice(range(context_length))
                self.starting_indices = list(range(random_start_id, len(self.dataset) - context_length, context_length))
                random.shuffle(self.starting_indices)
            
            # Collect batch_size sequences
            selected_indices = []
            for _ in range(batch_size):
                selected_indices.append(self.starting_indices.pop())
            
            x = torch.stack([torch.from_numpy(self.dataset[sid:sid+context_length].copy()) for sid in selected_indices])
            y = torch.stack([torch.from_numpy(self.dataset[sid+1:sid+context_length+1].copy()) for sid in selected_indices])
            return x.to(device), y.to(device)
        else:
            return get_batch(self.dataset, batch_size, context_length, device)
