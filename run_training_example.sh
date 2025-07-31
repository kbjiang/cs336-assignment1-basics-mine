#!/bin/bash

# Example script to run training with wandb logging
# Make sure to set your wandb API key first: wandb login

# Training with wandb (default settings)
python cs336_basics/train_with_wandb.py \
    --data_path data/TinyStoriesV2-train.npy \
    --eval_data_path data/TinyStoriesV2-valid.npy \
    --vocab_path data/train_bpe_vocab_ts.json \
    --merges_path data/train_bpe_merges_ts.txt \
    --num_steps 10000 \
    --batch_size 128 \
    --lr 1e-4 \
    --device cuda \
    --wandb_project "cs336-transformer-training" \
    --wandb_run_name "test-run-$(date +%Y%m%d-%H%M%S)" \
    --log_interval 100

# Training without wandb
# python cs336_basics/train_with_wandb.py \
#     --data_path data/TinyStoriesV2-train.npy \
#     --vocab_path data/train_bpe_vocab_ts.json \
#     --merges_path data/train_bpe_merges_ts.txt \
#     --num_steps 100 \
#     --batch_size 16 \
#     --lr 1e-4 \
#     --no_wandb

echo "Training completed!"
