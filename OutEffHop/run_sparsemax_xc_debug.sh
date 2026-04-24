#!/bin/bash
module load cuda/12.6.2-gcc-12.4.0

export HF_HOME="/scratch/hlv8980/.cache/"
export WANDB_PROJECT="residual"
export WANDB_ENABLED="false"

export LC_ALL=C.UTF-8
export LANG=C.UTF-8
export CUDA_VISIBLE_DEVICES=""
export PYTHONNOUSERSITE=1
export PYTHONPATH=""

source /software/miniconda3/4.12.0/etc/profile.d/conda.sh
conda activate outlier

which python
python -V
python -c "import torch; print(torch.__file__)"
python -c "import datasets; print(datasets.__file__)"

python run_clm_oasis.py \
--pad_to_max_length \
--wd_LN_gamma \
--seed 1000 \
--dataset_setup bookcorpus_and_wiki \
--preprocessing_num_workers 2 \
--data_cache_dir /scratch/hlv8980/residual/.hf_data \
--model_cache_dir /scratch/hlv8980/residual/.hf_cache \
--model_type llama \
--tokenizer_name meta-llama/Llama-3.2-1B \
--max_seq_length 256 \
--block_size 128 \
--learning_rate 0.0004 \
--lr_scheduler_type linear \
--max_train_steps 5 \
--num_warmup_steps 1 \
--per_device_train_batch_size 1 \
--per_device_eval_batch_size 1 \
--gradient_accumulation_steps 1 \
--max_grad_norm 1.0 \
--weight_decay 0.1 \
--checkpointing_steps 100 \
--model_name_or_path meta-llama/Llama-3.2-1B \
--attn_softmax sparsemax \
--attn_res_softmax_fn sparsemax \
--max_checkpointing_number 1 \
--output_dir /scratch/hlv8980/residual/output/oasis_llama3_sparsemax_cpu_debug
