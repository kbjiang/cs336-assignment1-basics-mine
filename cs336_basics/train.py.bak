import numpy as np
import torch
from einops import rearrange, einsum
from tqdm import tqdm
import argparse

from model import *
from nn_utils import *
from data import *
from optimizer import * 
from tokenizer import *
from serialization import *

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train a transformer language model")
    
    # Model hyperparameters
    parser.add_argument("--d_model", type=int, default=512, help="Model dimension")
    parser.add_argument("--d_ff", type=int, default=2048, help="Feed-forward dimension")
    parser.add_argument("--num_heads", type=int, default=8, help="Number of attention heads")
    parser.add_argument("--num_layers", type=int, default=6, help="Number of transformer layers")
    parser.add_argument("--context_length", type=int, default=256, help="Maximum sequence length")
    parser.add_argument("--rope_theta", type=float, default=1e4, help="RoPE theta parameter")
    
    # Training hyperparameters
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    parser.add_argument("--weight_decay", type=float, default=0.01, help="Weight decay")
    parser.add_argument("--num_steps", type=int, default=1000, help="Number of training steps")
    
    # Data and model paths
    parser.add_argument("--data_path", type=str, required=True, help="Path to training data (.npy file)")
    parser.add_argument("--vocab_path", type=str, required=True, help="Path to vocabulary file")
    parser.add_argument("--merges_path", type=str, required=True, help="Path to merges file")
    parser.add_argument("--checkpoint_path", type=str, default="checkpoint.pt", help="Path to save checkpoint")
    
    # Device
    parser.add_argument("--device", type=str, default="cpu", help="Device to use (cpu, cuda, cuda:0, etc.)")
    
    args = parser.parse_args()
    
    # training code starts here
    vocab, _ = get_vocab_and_merges_from_files(args.vocab_path, args.merges_path)
    model = transformer_lm(
        d_model = args.d_model,
        d_ff = args.d_ff,
        num_heads = args.num_heads,
        rope_theta = args.rope_theta,
        num_layers = args.num_layers,
        vocab_size = len(vocab),
        context_length = args.context_length
    )
    model.to(args.device)
    optimizer = AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )

    dataset = Dataset(args.data_path)
    for i in tqdm(range(args.num_steps), total=args.num_steps):
        batch = dataset.get_batch(args.batch_size, args.context_length, args.device)
        optimizer.zero_grad()
        x, y = batch
        y_hat = model(x)
        loss = cross_entropy_with_batch(y_hat, y)
        loss.backward()
        optimizer.step()
        if (i+1) % 10 == 0:
            print(f"Loss at step {i}: {loss.data}")

    print("Training finished")

    save_checkpoint(model, optimizer, i, args.checkpoint_path)
    



