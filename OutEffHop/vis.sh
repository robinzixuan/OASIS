#!/bin/bash
export PYTHONNOUSERSITE=1
export CUDA_VISIBLE_DEVICES=0
module load cuda/12.6.2-gcc-12.4.0
export HF_HOME="/scratch/hlv8980/.cache/"
export HF_TOKEN="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"
export WANDB_PROJECT="residual"
export WANDB_ENABLED="true"
export PYTHONNOUSERSITE=1

# Vanilla softmax (baseline)
python visulization.py \
    --model_name_or_path /scratch/hlv8980/residual/output/oasis_llama3/ \
    --attn_softmax softmax1 --attn_res_softmax_fn softmax1 \
    --oasis --save_dir attn_vis_oasis