#!/bin/bash

# Example script to run training with wandb logging
# Make sure to set your wandb API key first: wandb login

# Configuration flags - set to empty string to disable, or uncomment/comment as needed
USE_WANDB=""              # Set to "--no_wandb" to disable wandb
BATCH_SIZE=128
LR_SCHEDULING="--lr_scheduling"  # Set to "" to disable lr scheduling
LR=1e-3
LR_MAX=1e-3
LR_MIN=1e-4
WANDB_PROJECT="cs336-assign1-ts" 
WANDB_RUN_NAME="test-run-bs-$BATCH_SIZE"
# WANDB_RUN_NAME="--wandb_run_name test-run-$(date +%Y%m%d-%H%M%S)" 

# Training with wandb (default settings)
python cs336_basics/train.py \
    --data_path data/TinyStoriesV2-train.npy \
    --eval_data_path data/TinyStoriesV2-valid.npy \
    --vocab_path data/train_bpe_vocab_ts.json \
    --merges_path data/train_bpe_merges_ts.txt \
    --log_path data/log-ts.jsonl \
    --checkpoint_path /home/azureuser/02-fun/cs336-assignment1-basics/data/checkpoint_ts.pt \
    --num_steps 10000 \
    --batch_size $BATCH_SIZE \
    $LR_SCHEDULING \
    --lr $LR \
    --lr_max $LR_MAX \
    --lr_min $LR_MIN \
    --warmup_iters 1000 \
    --cosine_cycle_iters 10000 \
    --device cuda \
    --log_interval 100 \
    $USE_WANDB \
    --wandb_project $WANDB_PROJECT \
    --wandb_run_name $WANDB_RUN_NAME \

echo "Training completed!"
