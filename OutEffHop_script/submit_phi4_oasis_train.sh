#!/bin/bash
#===============================================================================
# Phi-3 / Phi-4 文本模型 — run_clm_oasis（phi4_oasis_attention + patch forward）
#
# 你可能还要改的地方：
#   1) #SBATCH -t / --mem / --gres：按模型大小与训练步数调整
#   2) PHI4_MODEL：默认 Hub ID；本地目录则 export PHI4_MODEL=/path/to/model（含 config.json）
#   3) SCRATCH_ROOT、OUTPUT_DIR_NAME、训练超参（可先改小做 smoke test）
#   4) 模块与 conda 环境名若不同请改 module load / conda activate
#
# Smoke test 建议：MAX_TRAIN_STEPS=50，BLOCK_SIZE=256，per_device_train_batch_size=1
#
# 提交：
#   sbatch /path/to/OASIS/OutEffHop_script/submit_phi4_oasis_train.sh
#===============================================================================

#SBATCH -A p32013
#SBATCH -p gengpu
#SBATCH --gres=gpu:1
#SBATCH -t 12:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --job-name=phi4-oasis
#SBATCH --output=slurm_phi4_oasis_%j.out
#SBATCH --error=slurm_phi4_oasis_%j.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=onionzsy@umich.edu

set -euo pipefail

#--------------- 用户配置 ----------------------------------------------------
PHI4_MODEL="${PHI4_MODEL:-microsoft/Phi-3-mini-4k-instruct}"
SCRATCH_ROOT="${SCRATCH_ROOT:-/scratch/${USER}/residual}"
OUTPUT_DIR_NAME="${OUTPUT_DIR_NAME:-oasis_phi4_${SLURM_JOB_ID:-local}}"
DATASET_SETUP="${DATASET_SETUP:-bookcorpus_and_wiki}"

# 试跑改小；正式跑再加大
MAX_TRAIN_STEPS="${MAX_TRAIN_STEPS:-2000}"
BLOCK_SIZE="${BLOCK_SIZE:-512}"
MAX_SEQ_LENGTH="${MAX_SEQ_LENGTH:-2048}"
PER_DEV_TRAIN_BS="${PER_DEV_TRAIN_BS:-2}"
PER_DEV_EVAL_BS="${PER_DEV_EVAL_BS:-2}"
GRAD_ACCUM="${GRAD_ACCUM:-16}"
LEARNING_RATE="${LEARNING_RATE:-2e-4}"
WARMUP_STEPS="${WARMUP_STEPS:-200}"
SEED="${SEED:-1000}"
RUN_NAME="${RUN_NAME:-phi4_oasis_smoke}"

WANDB_PROJECT="${WANDB_PROJECT:-residual}"
# 不需要 wandb 时设为 false
WANDB_ENABLED="${WANDB_ENABLED:-false}"
#------------------------------------------------------------------------------

module purge 2>/dev/null || true

# Lmod 报 luac spiderT.lua 错：rm -rf ~/.cache/lmod 后重登

module load python-miniconda3/4.12.0

# torch 2.7+cu128：wheel 自带 12.8；module 选集群上较新的 12.x（见 submit_phi4_validate.sh 注释）
if ! module load cuda/12.6.2-gcc-12.4.0 2>/dev/null; then
  if ! module load cuda/12.4.1-gcc-12.3.0 2>/dev/null; then
    module load cuda/12.1.0-gcc-11.2.0 || true
  fi
fi

eval "$(conda shell.bash hook 2>/dev/null)" || true
conda activate outlier

# Slurm 拷贝脚本后 BASH_SOURCE 不在仓库内；用 SLURM_SUBMIT_DIR。
if [[ -n "${SLURM_SUBMIT_DIR:-}" ]]; then
  if [[ -d "${SLURM_SUBMIT_DIR}/OutEffHop" ]]; then
    REPO_ROOT="${SLURM_SUBMIT_DIR}"
  elif [[ -d "$(cd "${SLURM_SUBMIT_DIR}/.." && pwd)/OutEffHop" ]]; then
    REPO_ROOT="$(cd "${SLURM_SUBMIT_DIR}/.." && pwd)"
  else
    echo "OutEffHop not found near SLURM_SUBMIT_DIR=${SLURM_SUBMIT_DIR}; sbatch from repo root or OutEffHop_script."
    exit 1
  fi
else
  SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]:-$0}")" && pwd)"
  REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
fi
cd "${REPO_ROOT}/OutEffHop" || { echo "Cannot cd to ${REPO_ROOT}/OutEffHop"; exit 1; }

export LC_ALL=C.UTF-8
export LANG=C.UTF-8
export PYTHONPATH="${PYTHONPATH:-}:${PWD}"
export WANDB_PROJECT
export WANDB_ENABLED

DATA_CACHE="${SCRATCH_ROOT}/phi4_oasis/.hf_data"
MODEL_CACHE="${SCRATCH_ROOT}/phi4_oasis/.hf_cache"
OUTPUT_DIR="${SCRATCH_ROOT}/output/${OUTPUT_DIR_NAME}"
mkdir -p "${DATA_CACHE}" "${MODEL_CACHE}" "${OUTPUT_DIR}"

TRACK_ARGS=()
if [[ "${WANDB_ENABLED}" == "true" ]]; then
  TRACK_ARGS=(--with_tracking --report_to wandb --run_name "${RUN_NAME}" --extra_tb_stats)
else
  TRACK_ARGS=(--run_name "${RUN_NAME}")
fi

accelerate launch --config_file accelerate_configs/1gpu_fp16.yaml run_clm_oasis.py \
  "${TRACK_ARGS[@]}" \
  --pad_to_max_length \
  --wd_LN_gamma \
  --seed "${SEED}" \
  --dataset_setup "${DATASET_SETUP}" \
  --preprocessing_num_workers 8 \
  --data_cache_dir "${DATA_CACHE}" \
  --model_cache_dir "${MODEL_CACHE}" \
  --tokenizer_name "${PHI4_MODEL}" \
  --max_seq_length "${MAX_SEQ_LENGTH}" \
  --block_size "${BLOCK_SIZE}" \
  --learning_rate "${LEARNING_RATE}" \
  --lr_scheduler_type linear \
  --max_train_steps "${MAX_TRAIN_STEPS}" \
  --num_warmup_steps "${WARMUP_STEPS}" \
  --per_device_train_batch_size "${PER_DEV_TRAIN_BS}" \
  --per_device_eval_batch_size "${PER_DEV_EVAL_BS}" \
  --gradient_accumulation_steps "${GRAD_ACCUM}" \
  --max_grad_norm 1.0 \
  --weight_decay 0.1 \
  --checkpointing_steps 500 \
  --tb_scalar_log_interval 500 \
  --tb_hist_log_interval 500 \
  --model_name_or_path "${PHI4_MODEL}" \
  --attn_softmax vanilla \
  --attn_res_softmax_fn vanilla \
  --max_checkpointing_number 2 \
  --output_dir "${OUTPUT_DIR}"

echo "Done. Checkpoints under: ${OUTPUT_DIR}"
