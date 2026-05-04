#!/bin/bash
set -euo pipefail

if command -v module >/dev/null 2>&1; then
  module load "${CUDA_MODULE:-cuda/12.6.2-gcc-12.4.0}" || true
fi

export HF_HOME="${HF_HOME:-${PWD}/.cache/huggingface}"
export WANDB_PROJECT="${WANDB_PROJECT:-residual}"
export WANDB_ENABLED="${WANDB_ENABLED:-false}"
export LC_ALL=C.UTF-8
export LANG=C.UTF-8
export CUDA_HOME="${CUDA_HOME:-}"
export PYTHONNOUSERSITE=1
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

PYTHON_BIN="${PYTHON_BIN:-python}"
DATA_CACHE_DIR="${DATA_CACHE_DIR:-${PWD}/.cache/hf_data}"
MODEL_CACHE_DIR="${MODEL_CACHE_DIR:-${PWD}/.cache/hf_cache}"
OUTPUT_DIR="${OUTPUT_DIR:-output/outeffhop_llama3}"
RESUME_FROM_CHECKPOINT="${RESUME_FROM_CHECKPOINT:-}"

TRACK_ARGS=()
if [[ "${WANDB_ENABLED}" == "true" ]]; then
  TRACK_ARGS=(--with_tracking --report_to wandb --run_name "${RUN_NAME:-test_outeffhop_llama3_1b}" --extra_tb_stats)
fi

RESUME_ARGS=()
if [[ -n "${RESUME_FROM_CHECKPOINT}" ]]; then
  RESUME_ARGS=(--resume_from_checkpoint "${RESUME_FROM_CHECKPOINT}")
fi

"${PYTHON_BIN}" -m accelerate.commands.launch --config_file accelerate_configs/1gpu_fp16.yaml run_clm_ddp.py \
--pad_to_max_length \
--wd_LN_gamma \
"${TRACK_ARGS[@]}" \
--seed 1000 \
--dataset_setup bookcorpus_and_wiki \
--preprocessing_num_workers 10 \
--data_cache_dir "${DATA_CACHE_DIR}" \
--model_cache_dir "${MODEL_CACHE_DIR}" \
--model_type llama \
--tokenizer_name meta-llama/Llama-3.2-1B \
--max_seq_length 2048 \
--block_size 512 \
--learning_rate 0.0004 \
--lr_scheduler_type linear \
--max_train_steps 2000 \
--num_warmup_steps 2000 \
--per_device_train_batch_size 6 \
--per_device_eval_batch_size 6 \
--gradient_accumulation_steps 32 \
--max_grad_norm 1.0 \
--weight_decay 0.1 \
--checkpointing_steps 500 \
--tb_scalar_log_interval 500 \
--tb_hist_log_interval 500 \
--model_name_or_path meta-llama/Llama-3.2-1B \
--attn_softmax softmax1 \
--attn_res_softmax_fn vanilla \
--max_checkpointing_number 2 \
--output_dir "${OUTPUT_DIR}" \
"${RESUME_ARGS[@]}"
