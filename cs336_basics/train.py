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
    parser.add_argument("--d_ff", type=int, default=1344, help="Feed-forward dimension")
    parser.add_argument("--num_heads", type=int, default=16, help="Number of attention heads")
    parser.add_argument("--num_layers", type=int, default=4, help="Number of transformer layers")
    parser.add_argument("--context_length", type=int, default=256, help="Maximum sequence length")
    parser.add_argument("--rope_theta", type=float, default=1e4, help="RoPE theta parameter")
    
    # Training hyperparameters
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    parser.add_argument("--weight_decay", type=float, default=0.01, help="Weight decay")
    parser.add_argument("--num_steps", type=int, default=1000, help="Number of training steps")
    
    # Learning rate scheduling
    parser.add_argument("--lr_scheduling", action="store_true", help="Enable learning rate scheduling")
    parser.add_argument("--lr_max", type=float, default=1e-3, help="Maximum learning rate for scheduling")
    parser.add_argument("--lr_min", type=float, default=1e-5, help="Minimum learning rate for scheduling")
    parser.add_argument("--warmup_iters", type=int, default=100, help="Number of warmup iterations")
    parser.add_argument("--cosine_cycle_iters", type=int, default=1000, help="Number of iterations for cosine cycle")
    
    # Data and model paths
    parser.add_argument("--data_path", type=str, required=True, help="Path to training data (.npy file)")
    parser.add_argument("--eval_data_path", type=str, required=True, help="Path to evaluation data (.npy file)")
    parser.add_argument("--vocab_path", type=str, required=True, help="Path to vocabulary file")
    parser.add_argument("--merges_path", type=str, required=True, help="Path to merges file")
    parser.add_argument("--checkpoint_path", type=str, default="/home/azureuser/02-fun/cs336-assignment1-basics/data/checkpoint.pt", help="Path to save checkpoint")
    parser.add_argument("--log_path", type=str, default="log.jsonl", help="Path to save experiment logs")
    parser.add_argument("--without_replacement", action="store_true", help="Disable replacement in batch generation")
    
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
        lr_scheduling=args.lr_scheduling,
        lr_max=args.lr_max,
        lr_min=args.lr_min,
        warmup_iters=args.warmup_iters,
        cosine_cycle_iters=args.cosine_cycle_iters,
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

    dataset = Dataset(args.data_path, args.without_replacement)
    dataset_eval = Dataset(args.eval_data_path, args.without_replacement)
    
    print(f"Starting training for {args.num_steps} steps...")
    
    # Start timing
    import time
    training_start_time = time.time()
    
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
                    "step": i + 1,
                    "train_loss": loss_train,
                    "eval_loss": loss_eval,
                    "grad_norm": grad_norm,
                    "learning_rate": optimizer.param_groups[0]['lr']
                }, step=i+1)

    # Calculate training time
    training_end_time = time.time()
    total_training_time = training_end_time - training_start_time
    
    print(f"Training finished in {total_training_time:.2f} seconds ({total_training_time/60:.2f} minutes)")

    # Save checkpoint
    save_checkpoint(model, optimizer, i, args.checkpoint_path)
    
    # Log final checkpoint to wandb
    if not args.no_wandb:
        # Log final training time
        wandb.log({
            "training_time_seconds": total_training_time,
            "training_time_minutes": total_training_time / 60,
            "steps_per_second": args.num_steps / total_training_time
        })
        
        artifact = wandb.Artifact(
            name=f"model-checkpoint-{wandb.run.id}",
            type="model",
            description=f"Final model checkpoint after {args.num_steps} steps"
        )
        artifact.add_file(args.checkpoint_path)
        wandb.log_artifact(artifact)
        
        wandb.finish()

    # log experiment params/result to a local file.
    # params should include all `model parameters` and `training parameters`, date/time of completion, the final losses, grad_norm and learning_rate
    # and the id of this entry should be the same as args.wandb_run_name

    import json
    from datetime import datetime
    
    # Get final metrics
    x_eval, y_eval = dataset_eval.get_batch(args.batch_size * 4, args.context_length, args.device)
    with torch.no_grad():
        y_hat_eval = model(x_eval)
        final_loss_eval = cross_entropy_with_batch(y_hat_eval, y_eval).item()
    final_loss_train = loss.item()
    final_grad_norm = grad_norm.item()
    final_lr = optimizer.param_groups[0]['lr']
    
    # Prepare experiment log entry
    experiment_log = {
        "id": args.wandb_run_name if args.wandb_run_name else f"run_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
        "completion_time": datetime.now().isoformat(),
        "wandb_url": wandb.run.url if not args.no_wandb and wandb.run else None,
        "model_parameters": {
            "d_model": args.d_model,
            "d_ff": args.d_ff,
            "num_heads": args.num_heads,
            "num_layers": args.num_layers,
            "context_length": args.context_length,
            "rope_theta": args.rope_theta,
            "vocab_size": len(vocab)
        },
        "training_parameters": {
            "batch_size": args.batch_size,
            "lr": args.lr,
            "lr_scheduling": args.lr_scheduling,
            "lr_max": args.lr_max if args.lr_scheduling else None,
            "lr_min": args.lr_min if args.lr_scheduling else None,
            "warmup_iters": args.warmup_iters if args.lr_scheduling else None,
            "cosine_cycle_iters": args.cosine_cycle_iters if args.lr_scheduling else None,
            "weight_decay": args.weight_decay,
            "num_steps": args.num_steps,
            "log_interval": args.log_interval
        },
        "final_metrics": {
            "final_train_loss": final_loss_train,
            "final_eval_loss": final_loss_eval,
            "final_grad_norm": final_grad_norm,
            "final_learning_rate": final_lr,
            "training_time_seconds": total_training_time,
            "training_time_minutes": total_training_time / 60,
            "steps_per_second": args.num_steps / total_training_time
        },
        "data_paths": {
            "data_path": args.data_path,
            "eval_data_path": args.eval_data_path,
            "vocab_path": args.vocab_path,
            "merges_path": args.merges_path,
            "checkpoint_path": args.checkpoint_path
        }
    }
    
    # Save to local experiments log file
    with open(args.log_path, "a") as f:
        f.write(json.dumps(experiment_log) + "\n")
    
    print(f"Training completed. Checkpoint saved to {args.checkpoint_path}")
    print(f"Experiment logged to {args.log_path}")
