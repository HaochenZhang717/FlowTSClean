#!/bin/bash

# =============================
# GPU
# =============================
export CUDA_VISIBLE_DEVICES=1
export hucfg_t_sampling="logitnorm"
export hucfg_num_steps="100"




# =============================
# Paths
# =============================
TS_PATH_TRAIN="/playpen/haochenz/LitsDatasets/128_len_ts/synthetic_u/train_ts.npy"
TEXT_EMBED_PATH_TRAIN="/playpen/haochenz/LitsDatasets/128_len_caps_one_per_channel_0309/synth_u/train_embeds.pt"

TS_PATH_VALID="/playpen/haochenz/LitsDatasets/128_len_ts/synthetic_u/valid_ts.npy"
TEXT_EMBED_PATH_VALID="/playpen/haochenz/LitsDatasets/128_len_caps_one_per_channel_0309/synth_u/valid_embeds.pt"


CKPT_DIR="/playpen/haochenz/FlowTS/unconditional/synth_u_0310"


# =============================
# WandB
# =============================
WANDB_PROJECT="FlowTS"
WANDB_RUN="synth_u"

# =============================
# Run training
# =============================
python uncond_train.py \
    --seq_len 128 \
    --feature_size 1 \
    \
    --n_layer_enc 4 \
    --n_layer_dec 4 \
    --d_model 128 \
    --n_heads 8 \
    \
    --train_ts_path ${TS_PATH_TRAIN} \
    --train_embed_path ${TEXT_EMBED_PATH_TRAIN} \
    --valid_ts_path ${TS_PATH_VALID} \
    --valid_embed_path ${TEXT_EMBED_PATH_VALID} \
    \
    --lr 1e-4 \
    --batch_size 64 \
    --max_epochs 1000 \
    --grad_clip_norm 1.0 \
    --grad_accum_steps 1 \
    \
    --wandb_project $WANDB_PROJECT \
    --wandb_run $WANDB_RUN \
    \
    --gpu_id 0 \
    \
    --ckpt_dir ${CKPT_DIR}