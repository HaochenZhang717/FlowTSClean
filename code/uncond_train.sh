#!/bin/bash

# =============================
# GPU
# =============================
export CUDA_VISIBLE_DEVICES=0

# =============================
# Paths
# =============================
TRAIN_DATA="/playpen/haochenz/LitsDatasets/128_len_ts/synthetic_u/train_ts.npy"
VALID_DATA="/playpen/haochenz/LitsDatasets/128_len_ts/synthetic_u/valid_ts.npy"
CKPT_DIR="/playpen/haochenz/FlowTS/unconditional/synth_u_0310"


# =============================
# WandB
# =============================
WANDB_PROJECT="FlowTS"
WANDB_RUN="synth_u"

# =============================
# Run training
# =============================
python train_dspflow.py \
    --seq_len 128 \
    --feature_size 1 \
    \
    --n_layer_enc 4 \
    --n_layer_dec 4 \
    --d_model 128 \
    --n_heads 8 \
    \
    --train_data_path "\"$TRAIN_DATA\"" \
    --valid_data_path "\"$VALID_DATA\"" \
    \
    --lr 1e-4 \
    --batch_size 64 \
    --max_epochs 500 \
    --grad_clip_norm 1.0 \
    --grad_accum_steps 1 \
    --early_stop "True" \
    --patience 20 \
    \
    --wandb_project $WANDB_PROJECT \
    --wandb_run $WANDB_RUN \
    \
    --gpu_id 0