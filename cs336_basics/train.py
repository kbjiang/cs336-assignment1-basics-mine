import numpy as np
import torch
from einops import rearrange, einsum
from tqdm import tqdm
import argparse
import wandb
import os

from model import *
from nn_utils import *
from data import *
from optimizer import * 
from tokenizer import *
from serialization import *

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train a transformer language model with wandb logging")
    
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
    parser.add_argument("--eval_data_path", type=str, required=True, help="Path to evaluation data (.npy file)")
    parser.add_argument("--vocab_path", type=str, required=True, help="Path to vocabulary file")
    parser.add_argument("--merges_path", type=str, required=True, help="Path to merges file")
    parser.add_argument("--checkpoint_path", type=str, default="/home/azureuser/02-fun/cs336-assignment1-basics/data/checkpoint.pt", help="Path to save checkpoint")
    
    # Device
    parser.add_argument("--device", type=str, default="cpu", help="Device to use (cpu, cuda, cuda:0, etc.)")
    
    # Wandb arguments
    parser.add_argument("--wandb_project", type=str, default="transformer-training", help="Wandb project name")
    parser.add_argument("--wandb_run_name", type=str, default=None, help="Wandb run name")
    parser.add_argument("--wandb_entity", type=str, default=None, help="Wandb entity/team name")
    parser.add_argument("--log_interval", type=int, default=10, help="Log metrics every N steps")
    parser.add_argument("--no_wandb", action="store_true", help="Disable wandb logging")
    
    args = parser.parse_args()
    
    # Initialize wandb
    if not args.no_wandb:
        wandb.init(
            project=args.wandb_project,
            name=args.wandb_run_name,
            entity=args.wandb_entity,
            config=vars(args),
            tags=["transformer", "language-model"]
        )
    
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

    # Log model architecture info
    if not args.no_wandb:
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        wandb.log({
            "model/total_params": total_params,
            "model/trainable_params": trainable_params,
            "model/vocab_size": len(vocab)
        })
        # Watch the model for gradients and parameters
        wandb.watch(model, log="all", log_freq=args.log_interval)

    dataset = Dataset(args.data_path)
    dataset_eval = Dataset(args.eval_data_path)
    
    print(f"Starting training for {args.num_steps} steps...")
    for i in tqdm(range(args.num_steps), total=args.num_steps):
        batch = dataset.get_batch(args.batch_size, args.context_length, args.device)
        optimizer.zero_grad()
        x, y = batch
        y_hat = model(x)
        loss = cross_entropy_with_batch(y_hat, y)
        loss.backward()
        
        # Compute gradient norm for monitoring (`inf` so it doesn't clip)
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), float('inf'))
        # gradient_clipping(model.parameters(), max_l2_norm = 1e-2)
        
        optimizer.step()
        
        # Log metrics
        if (i + 1) % args.log_interval == 0:
            x_eval, y_eval = dataset_eval.get_batch(args.batch_size * 4, args.context_length, args.device)
            with torch.no_grad():
                y_hat_eval = model(x_eval)
                loss_eval = cross_entropy_with_batch(y_hat_eval, y_eval).item()
            loss_train = loss.item()
            print(f"Step {i+1}: Loss_eval = {loss_eval:.4f}, Loss_train = {loss_train:.4f}, Grad Norm = {grad_norm:.4f}")
            
            if not args.no_wandb:
                wandb.log({
                    "train_loss": loss_train,
                    "eval_loss": loss_eval,
                    "grad_norm": grad_norm,
                    "step": i + 1,
                    "learning_rate": optimizer.param_groups[0]['lr']
                }, step=i + 1)

    print("Training finished")

    # Save checkpoint
    save_checkpoint(model, optimizer, i, args.checkpoint_path)
    
    # Log final checkpoint to wandb
    if not args.no_wandb:
        artifact = wandb.Artifact(
            name=f"model-checkpoint-{wandb.run.id}",
            type="model",
            description=f"Final model checkpoint after {args.num_steps} steps"
        )
        artifact.add_file(args.checkpoint_path)
        wandb.log_artifact(artifact)
        
        wandb.finish()
        print(f"Training completed. Checkpoint saved to {args.checkpoint_path}")
