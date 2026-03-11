#!/bin/bash

# =============================
# GPU
# =============================
export CUDA_VISIBLE_DEVICES=6
export hucfg_t_sampling="logitnorm"
export hucfg_num_steps="100"

# =============================
# Paths
# =============================
TEST_DATA="/playpen/haochenz/LitsDatasets/128_len_ts/stock/test_ts.npy"
CKPT_PATH="/playpen/haochenz/FlowTS/unconditional/stock_0311/ema_ckpt_best.pth"
OUTPUT_PATH="/playpen/haochenz/FlowTS/unconditional/stock_0311/sample_results.pth"


# =============================
# Run sampling
# =============================
python uncond_sample.py \
    --seq_len 128 \
    --feature_size 6 \
    \
    --n_layer_enc 4 \
    --n_layer_dec 4 \
    --d_model 128 \
    --n_heads 8 \
    \
    --test_data_path ${TEST_DATA} \
    \
    --batch_size 64 \
    \
    --gpu_id 0 \
    \
    --ckpt_path ${CKPT_PATH} \
    --output_path ${OUTPUT_PATH}