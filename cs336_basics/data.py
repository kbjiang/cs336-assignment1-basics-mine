import torch
import numpy as np
def get_batch(
    dataset: np.ndarray,
    batch_size: int,
    context_length: int,
    device:str = "cpu"
) -> tuple[torch.tensor, torch.tensor]:
    valid_range = len(dataset) - context_length
    starting_indices = torch.randint(0, valid_range, (batch_size,))
    x = torch.tensor([dataset[sid:sid+context_length] for sid in starting_indices])
    y = torch.tensor([dataset[sid+1:sid+context_length+1] for sid in starting_indices])
    return x.to(device), y.to(device)