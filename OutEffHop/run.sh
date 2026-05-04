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
# export MASTER_PORT=$(expr 10000 + $(echo -n $SLURM_JOBID | tail -c 4))
# export WORLD_SIZE=$(($SLURM_NNODES * $SLURM_NTASKS_PER_NODE))
# echo "WORLD_SIZE="$WORLD_SIZE

export HF_DATASETS_OFFLINE="${HF_DATASETS_OFFLINE:-0}"
export PYTHONNOUSERSITE=1
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

PYTHON_BIN="${PYTHON_BIN:-python}"
DATA_CACHE_DIR="${DATA_CACHE_DIR:-${PWD}/.cache/hf_data}"
MODEL_CACHE_DIR="${MODEL_CACHE_DIR:-${PWD}/.cache/hf_cache}"
MODEL_NAME_OR_PATH="${MODEL_NAME_OR_PATH:-Qwen/Qwen3-0.6B}"
OUTPUT_DIR="${OUTPUT_DIR:-output_metrics/vanilla_qwen}"

"${PYTHON_BIN}" -m accelerate.commands.launch --config_file accelerate_configs/1gpu_fp16.yaml validate_clm.py \
--quantize \
--quant_setup fp32_head \
--ranges_acts running_minmax \
--qmethod_acts asymmetric_uniform \
--percentile 99.999 \
--est_num_batches 4 \
--seed 6789 \
--dataset_setup bookcorpus_and_wiki \
--preprocessing_num_workers 16 \
--model_type llama \
--block_size 512 \
--per_device_eval_batch_size 4 \
--attn_softmax vanilla \
--attn_res_softmax_fn vanilla \
--data_cache_dir "${DATA_CACHE_DIR}" \
--model_cache_dir "${MODEL_CACHE_DIR}" \
--model_name_or_path "${MODEL_NAME_OR_PATH}" \
--output_dir "${OUTPUT_DIR}"