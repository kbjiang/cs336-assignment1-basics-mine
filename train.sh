#!/bin/bash

# Example script to run training with wandb logging
# Make sure to set your wandb API key first: wandb login

# Configuration flags - set to empty string to disable, or uncomment/comment as needed
D_MODEL=512
D_FF=1344
NUM_LAYERS=4
NUM_HEADS=16
BATCH_SIZE=128
CONTEXT_LENGTH=256
GRADIENT_ACCUMULATION_STEPS=1
WITHOUT_REPLACEMENT=""            # Set to "--without_replacement" to disable replacement in batch generation
# LR_SCHEDULING="" 
LR_SCHEDULING="--lr_scheduling"  # Set to "" to disable lr scheduling
LR=1e-3
LR_MAX=1e-3
LR_MIN=1e-4
# USE_WANDB="--no_wandb"              # Set to "--no_wandb" to disable wandb
USE_WANDB=""              # Set to "--no_wandb" to disable wandb
WANDB_PROJECT="cs336-assign1-ts" 

SUFFIX="d512-dff1344-l4-h16-cl256-bs128"
WANDB_RUN_NAME="run-$SUFFIX"
# WANDB_RUN_NAME="--wandb_run_name test-run-$(date +%Y%m%d-%H%M%S)" 

# Training with wandb (default settings)
python cs336_basics/train.py \
    --d_model $D_MODEL \
    --d_ff $D_FF \
    --num_layers $NUM_LAYERS \
    --num_heads $NUM_HEADS \
    --data_path data/ts_train.npy \
    --eval_data_path data/ts_valid.npy \
    --vocab_path data/bpe_vocab_ts.json \
    --merges_path data/bpe_merges_ts.txt \
    --log_path data/log-ts.jsonl \
    $WITHOUT_REPLACEMENT \
    --checkpoint_path data/checkpoint-ts-$SUFFIX.pt \
    --num_steps 10000 \
    --batch_size $BATCH_SIZE \
    --context_length $CONTEXT_LENGTH \
    --gradient_accumulation_steps $GRADIENT_ACCUMULATION_STEPS \
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
