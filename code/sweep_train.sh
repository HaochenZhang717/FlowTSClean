#!/bin/bash

export hucfg_t_sampling="logitnorm"
export hucfg_num_steps="100"

TRAIN_DATA="/playpen/haochenz/LitsDatasets/128_len_ts/synthetic_u/train_ts.npy"
VALID_DATA="/playpen/haochenz/LitsDatasets/128_len_ts/synthetic_u/valid_ts.npy"

BASE_CKPT="/playpen/haochenz/FlowTS/unconditional"

WANDB_PROJECT="FlowTS"

# hyperparameters
LRS=(1e-4 3e-4 5e-4 1e-3)
BATCHES=(32 64 128 256)

GPU_LIST=(1 2 3 4 6 7)

NUM_GPU=${#GPU_LIST[@]}

job_id=0

for lr in "${LRS[@]}"
do
for bs in "${BATCHES[@]}"
do

gpu=${GPU_LIST[$((job_id % NUM_GPU))]}

run_name="synth_lr${lr}_bs${bs}"

ckpt_dir="${BASE_CKPT}/${run_name}"

echo "Launching $run_name on GPU $gpu"

CUDA_VISIBLE_DEVICES=$gpu \
python uncond_train.py \
    --seq_len 128 \
    --feature_size 1 \
    \
    --n_layer_enc 4 \
    --n_layer_dec 4 \
    --d_model 128 \
    --n_heads 8 \
    \
    --train_data_path ${TRAIN_DATA} \
    --valid_data_path ${VALID_DATA} \
    \
    --lr ${lr} \
    --batch_size ${bs} \
    --max_epochs 1000 \
    --grad_clip_norm 1.0 \
    --grad_accum_steps 1 \
    --early_stop "True" \
    --patience 20 \
    \
    --wandb_project ${WANDB_PROJECT} \
    --wandb_run ${run_name} \
    \
    --gpu_id 0 \
    --ckpt_dir ${ckpt_dir} \
    &

job_id=$((job_id+1))

# 等待一轮GPU跑完
if (( job_id % NUM_GPU == 0 ))
then
    wait
fi

done
done

wait

echo "All jobs finished"