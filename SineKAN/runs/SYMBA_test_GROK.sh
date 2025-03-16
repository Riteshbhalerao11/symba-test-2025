#!/bin/bash

nvidia-smi

torchrun --standalone --nproc_per_node 2 main.py \
    --project_name "SYMBA_test" \
    --run_name "64x2_Skanformer_GROK_$SLURM_JOB_ID" \
    --model_name "skanformer_64x2" \
    --root_dir "$SCRATCH/SYMBA_test/Skanformer_GROK" \
    --data_dir "$SCRATCH/SYMBA_test/SYMBA_test" \
    --device "cuda" \
    --epochs 300 \
    --training_batch_size 64 \
    --valid_batch_size 64 \
    --num_workers 32 \
    --embedding_size 512 \
    --ff_dims 4096 \
    --d_ff 4096 \
    --nhead 8 \
    --num_layers 3 \
    --warmup_ratio 0 \
    --weight_decay 1e-3 \
    --clip_grad_norm 1 \
    --dropout 0.1 \
    --src_max_len 280 \
    --tgt_max_len 323 \
    --curr_epoch 0 \
    --optimizer_lr 5e-5\
    --train_shuffle True \
    --pin_memory True \
    --world_size 2 \
    --save_freq 9 \
    --test_freq 3 \
    --seed 42 \
    --log_freq 20 \
    --save_last True \
    --save_limit 3 \
    --is_constant_lr \
