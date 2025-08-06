#!/bin/bash

# Example script to run training with wandb logging
# Make sure to set your wandb API key first: wandb login

# Configuration flags - set to empty string to disable, or uncomment/comment as needed
D_MODEL=768
D_FF=2048
NUM_LAYERS=8
NUM_HEADS=8
BATCH_SIZE=64
CONTEXT_LENGTH=512
GRADIENT_ACCUMULATION_STEPS=2
WITHOUT_REPLACEMENT=""            # Set to "--without_replacement" to disable replacement in batch generation
LR_SCHEDULING="" 
# LR_SCHEDULING="--lr_scheduling"  # Set to "" to disable lr scheduling
LR=1e-3
LR_MAX=1e-3
LR_MIN=1e-4
# USE_WANDB="--no_wandb"              # Set to "--no_wandb" to disable wandb
USE_WANDB=""              # Set to "--no_wandb" to disable wandb
WANDB_PROJECT="cs336-assign1-owt" 

SUFFIX="d768-dff2048-l8-h8-cl512-bs128"
WANDB_RUN_NAME="test-run-$SUFFIX"
# WANDB_RUN_NAME="--wandb_run_name test-run-$(date +%Y%m%d-%H%M%S)" 

# Training with wandb (default settings)
python cs336_basics/train.py \
    --d_model $D_MODEL \
    --d_ff $D_FF \
    --num_layers $NUM_LAYERS \
    --num_heads $NUM_HEADS \
    --data_path data/owt_train.npy \
    --eval_data_path data/owt_valid.npy \
    --vocab_path data/train_bpe_vocab_owt.json \
    --merges_path data/train_bpe_merges_owt.txt \
    --log_path data/log-owt.jsonl \
    $WITHOUT_REPLACEMENT \
    --checkpoint_path data/checkpoint-owt-$SUFFIX.pt \
    --num_steps 40000 \
    --batch_size $BATCH_SIZE \
    --context_length $CONTEXT_LENGTH \
    --gradient_accumulation_steps $GRADIENT_ACCUMULATION_STEPS \
    $LR_SCHEDULING \
    --lr $LR \
    --lr_max $LR_MAX \
    --lr_min $LR_MIN \
    --warmup_iters 2000 \
    --cosine_cycle_iters 40000 \
    --device cuda \
    --log_interval 200 \
    $USE_WANDB \
    --wandb_project $WANDB_PROJECT \
    --wandb_run_name $WANDB_RUN_NAME \

echo "Training completed!"
