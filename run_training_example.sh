#!/bin/bash

# Example script to run training with wandb logging
# Make sure to set your wandb API key first: wandb login

# Training with wandb (default settings)
python cs336_basics/train.py \
    --data_path data/owt_train.npy \
    --eval_data_path data/owt_valid.npy \
    --vocab_path data/train_bpe_vocab_owt.json \
    --merges_path data/train_bpe_merges_owt.txt \
    --log_path data/log.jsonl \
    --num_steps 10000 \
    --batch_size 128 \
    --lr 1e-2 \
    --device cuda \
    --wandb_project "cs336-assign1-owt" \
    --wandb_run_name "test-run-$(date +%Y%m%d-%H%M%S)" \
    --log_interval 100 \
    --checkpoint_path /home/azureuser/02-fun/cs336-assignment1-basics/data/checkpoint_owt.pt
    --lr_scheduling \
    --lr_max 1e-2 \
    --lr_min 1e-3 \
    --warmup_iters 1000 \
    --cosine_cycle_iters 10000 \

# test
# python cs336_basics/train.py \
#     --data_path data/owt_train.npy \
#     --eval_data_path data/owt_valid.npy \
#     --vocab_path data/train_bpe_vocab_owt.json \
#     --merges_path data/train_bpe_merges_owt.txt \
#     --log_path data/log.jsonl \
#     --num_steps 1000 \
#     --batch_size 128 \
#     --lr_scheduling \
#     --lr_max 1e-3 \
#     --lr_min 1e-4 \
#     --warmup_iters 100 \
#     --cosine_cycle_iters 800 \
#     --device cuda \
#     --wandb_project "cs336-assign1-owt" \
#     --wandb_run_name "test-run-$(date +%Y%m%d-%H%M%S)" \
#     --log_interval 20 \
#     --checkpoint_path /home/azureuser/02-fun/cs336-assignment1-basics/data/test.pt \
    # --no_wandb

echo "Training completed!"
