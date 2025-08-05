#!/bin/bash

# Example script to run training with wandb logging
# Make sure to set your wandb API key first: wandb login

# Configuration flags - set to empty string to disable, or uncomment/comment as needed
USE_WANDB=""              # Set to "--no_wandb" to disable wandb
WITHOUT_REPLACEMENT=""            # Set to "--without_replacement" to disable replacement in batch generation
BATCH_SIZE=128
GRADIENT_ACCUMULATION_STEPS=4
# LR_SCHEDULING="" 
LR_SCHEDULING="--lr_scheduling"  # Set to "" to disable lr scheduling
LR=1e-3
LR_MAX=1e-3
LR_MIN=5e-4
WANDB_PROJECT="cs336-assign1-owt" 
WANDB_RUN_NAME="test-run-grad-accum-4"
# WANDB_RUN_NAME="--wandb_run_name test-run-$(date +%Y%m%d-%H%M%S)" 

# Training with wandb (default settings)
python cs336_basics/train.py \
    --data_path data/owt_train.npy \
    --eval_data_path data/owt_valid.npy \
    --vocab_path data/train_bpe_vocab_owt.json \
    --merges_path data/train_bpe_merges_owt.txt \
    $WITHOUT_REPLACEMENT \
    --log_path data/log-ts.jsonl \
    --checkpoint_path data/checkpoint_owt_grad_accum_2.pt \
    --num_steps 10000 \
    --batch_size $BATCH_SIZE \
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
