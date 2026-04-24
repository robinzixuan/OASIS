#!/bin/bash
module load cuda/12.6.2-gcc-12.4.0
export HF_HOME="/scratch/hlv8980/.cache/" 
export WANDB_PROJECT="residual"
export WANDB_ENABLED="true"
export HF_TOKEN="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"
source ~/.bashrc && conda activate outlier && which python && python -V



export LC_ALL=C.UTF-8
export LANG=C.UTF-8
export CUDA_HOME=/software/cuda/cuda-12.1.0 
# export MASTER_PORT=$(expr 10000 + $(echo -n $SLURM_JOBID | tail -c 4))
# export WORLD_SIZE=$(($SLURM_NNODES * $SLURM_NTASKS_PER_NODE))
# echo "WORLD_SIZE="$WORLD_SIZE

export HF_DATASETS_OFFLINE=1
export PYTHONNOUSERSITE=1
export CUDA_VISIBLE_DEVICES=0
# ~/.conda/envs/outlier/bin/python -m accelerate.commands.launch --config_file accelerate_configs/1gpu_fp16.yaml run_clm_oasis.py \
# --pad_to_max_length \
# --wd_LN_gamma \
# --with_tracking \
# --report_to wandb \
# --run_name test_oasis_llama3_1b \
# --extra_tb_stats \
# --seed 1000 \
# --dataset_setup bookcorpus_and_wiki \
# --preprocessing_num_workers 10 \
# --data_cache_dir /scratch/hlv8980/residual/.hf_data \
# --model_cache_dir /scratch/hlv8980/residual/.hf_cache \
# --model_type llama \
# --tokenizer_name meta-llama/Llama-3.2-1B \
# --max_seq_length 2048 \
# --block_size 512 \
# --learning_rate 0.0004 \
# --lr_scheduler_type linear \
# --max_train_steps 500 \
# --num_warmup_steps 2000 \
# --per_device_train_batch_size 6 \
# --per_device_eval_batch_size 6 \
# --gradient_accumulation_steps 32 \
# --max_grad_norm 1.0 \
# --weight_decay 0.1 \
# --checkpointing_steps 100 \
# --tb_scalar_log_interval 20000 \
# --tb_hist_log_interval 40000 \
# --model_name_or_path meta-llama/Llama-3.2-1B \
# --attn_softmax softmax1 \
# --attn_res_softmax_fn softmax1 \
# --max_checkpointing_number 2 \
# --output_dir /scratch/hlv8980/residual/output/oasis_llama3 

# ~/.conda/envs/outlier/bin/python -m accelerate.commands.launch --config_file accelerate_configs/1gpu_fp16.yaml run_clm_oasis.py \
# --pad_to_max_length \
# --wd_LN_gamma \
# --with_tracking \
# --report_to wandb \
# --run_name test_oasis_qwen3_0.6b \
# --extra_tb_stats \
# --seed 1000 \
# --dataset_setup bookcorpus_and_wiki \
# --preprocessing_num_workers 10 \
# --data_cache_dir /scratch/hlv8980/residual/qwen/.hf_data \
# --model_cache_dir /scratch/hlv8980/residual/qwen/.hf_cache \
# --model_type qwen3 \
# --tokenizer_name Qwen/Qwen3-0.6B \
# --max_seq_length 2048 \
# --block_size 512 \
# --learning_rate 0.000001 \
# --lr_scheduler_type linear \
# --max_train_steps 100000 \
# --num_warmup_steps 2000 \
# --per_device_train_batch_size 3 \
# --per_device_eval_batch_size 3 \
# --gradient_accumulation_steps 16 \
# --max_grad_norm 1.0 \
# --weight_decay 0.1 \
# --checkpointing_steps 5000 \
# --tb_scalar_log_interval 10000 \
# --tb_hist_log_interval 20000 \
# --model_name_or_path Qwen/Qwen3-0.6B \
# --attn_softmax softmax1 \
# --attn_res_softmax_fn softmax1 \
# --output_dir /scratch/hlv8980/residual/output/oasis_qwen3_0.6b 


# ~/.conda/envs/outlier/bin/python -m accelerate.commands.launch --config_file accelerate_configs/1gpu_fp16.yaml validate_clm.py \
# --quantize \
# --quant_setup fp32_head \
# --ranges_acts running_minmax \
# --qmethod_acts asymmetric_uniform \
# --percentile 99.999 \
# --est_num_batches 4 \
# --seed 6789 \
# --dataset_setup bookcorpus_and_wiki \
# --preprocessing_num_workers 16 \
# --model_type llama \
# --block_size 512 \
# --per_device_eval_batch_size 4 \
# --attn_softmax vanilla \
# --attn_res_softmax_fn vanilla \
# --data_cache_dir /scratch/hlv8980/residual/qwen/.hf_data \
# --model_cache_dir /scratch/hlv8980/residual/qwen/.hf_cache \
# --model_name_or_path /scratch/hlv8980/residual/output/vanilla_qwen3_0.6b/ \
# --output_dir  output_metrics/vanilla_qwen


# ~/.conda/envs/outlier/bin/python -m accelerate.commands.launch --config_file accelerate_configs/1gpu_fp16.yaml validate_clm_oasis.py \
# --quantize \
# --quant_setup fp32_head \
# --ranges_acts running_minmax \
# --qmethod_acts asymmetric_uniform \
# --percentile 99.999 \
# --est_num_batches 4 \
# --seed 6789 \
# --dataset_setup bookcorpus_and_wiki \
# --preprocessing_num_workers 8 \
# --model_type llama \
# --block_size 512 \
# --per_device_eval_batch_size 32 \
# --attn_softmax sparsemax \
# --attn_res_softmax_fn sparsemax \
# --data_cache_dir /scratch/hlv8980/residual/.hf_data \
# --model_cache_dir /scratch/hlv8980/residual/.hf_cache \
# --model_name_or_path /scratch/hlv8980/residual/output/oasis_llama3_sparsemax_xc \
# --output_dir output_metrics/oasis_llama3_sparsemax_quantized



~/.conda/envs/outlier/bin/python -m accelerate.commands.launch --config_file accelerate_configs/1gpu_fp16.yaml validate_clm_oasis.py \
--seed 6789 \
--dataset_setup bookcorpus_and_wiki \
--preprocessing_num_workers 8 \
--model_type llama \
--block_size 512 \
--per_device_eval_batch_size 8 \
--attn_softmax sparsemax \
--attn_res_softmax_fn sparsemax \
--data_cache_dir /scratch/hlv8980/residual/.hf_data \
--model_cache_dir /scratch/hlv8980/residual/.hf_cache \
--model_name_or_path /scratch/hlv8980/residual/output/oasis_llama3_sparsemax_xc \
--output_dir output_metrics/oasis_llama3_sparsemax

