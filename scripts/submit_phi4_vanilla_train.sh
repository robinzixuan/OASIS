#!/bin/bash
#===============================================================================
# Phi-4 文本模型 — run_clm_ddp（phi4_attention，非 OASIS）
#
# 与 submit_phi4_oasis_train.sh 使用同一套「可对齐」默认超参（见该文件顶部注释）。
# 提交：
#   sbatch /path/to/OASIS/scripts/submit_phi4_vanilla_train.sh
#===============================================================================

#SBATCH -A p32013
#SBATCH -p gengpu
#SBATCH --gres=gpu:1
#SBATCH -t 48:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --job-name=phi4-vanilla
#SBATCH --output=slurm_phi4_vanilla_%j.out
#SBATCH --error=slurm_phi4_vanilla_%j.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=onionzsy@umich.edu

set -euo pipefail

#--------------- 用户配置（与 submit_phi4_oasis_train.sh 保持同名变量、同默认值）----
PHI4_MODEL="${PHI4_MODEL:-microsoft/Phi-4-mini-instruct}"
SCRATCH_ROOT="${SCRATCH_ROOT:-/scratch/${USER}/residual}"
OUTPUT_DIR_NAME="${OUTPUT_DIR_NAME:-vanilla_phi4_${SLURM_JOB_ID:-local}}"
DATASET_SETUP="${DATASET_SETUP:-wikitext_2}"

MAX_TRAIN_STEPS="${MAX_TRAIN_STEPS:-2000}"
MAX_SEQ_LENGTH="${MAX_SEQ_LENGTH:-512}"
BLOCK_SIZE="${BLOCK_SIZE:-512}"
PER_DEV_TRAIN_BS="${PER_DEV_TRAIN_BS:-2}"
PER_DEV_EVAL_BS="${PER_DEV_EVAL_BS:-2}"
GRAD_ACCUM="${GRAD_ACCUM:-16}"
LEARNING_RATE="${LEARNING_RATE:-2e-4}"
WARMUP_STEPS="${WARMUP_STEPS:-200}"
SEED="${SEED:-1000}"
RUN_NAME="${RUN_NAME:-phi4_vanilla_512}"

WANDB_PROJECT="${WANDB_PROJECT:-residual}"
WANDB_ENABLED="${WANDB_ENABLED:-false}"
#------------------------------------------------------------------------------

module purge 2>/dev/null || true
module load python-miniconda3/4.12.0

if ! module load cuda/12.6.2-gcc-12.4.0 2>/dev/null; then
  if ! module load cuda/12.4.1-gcc-12.3.0 2>/dev/null; then
    module load cuda/12.1.0-gcc-11.2.0 || true
  fi
fi

export PYTHONUNBUFFERED=1
export PYTHONNOUSERSITE=1
export CUDA_VISIBLE_DEVICES=0
export HF_HOME="${HF_HOME:-${SCRATCH_ROOT}/.hf_home}"
mkdir -p "${HF_HOME}"

set +e
if command -v conda >/dev/null 2>&1; then
  eval "$(conda shell.bash hook 2>/dev/null)" || true
else
  conda init bash 2>/dev/null || true
  [[ -f "${HOME}/.bashrc" ]] && source "${HOME}/.bashrc"
fi
conda activate outlier
_CONDA_RC=$?
set -e
if [[ ${_CONDA_RC} -ne 0 ]]; then
  echo "conda activate outlier failed (exit ${_CONDA_RC})"
  exit 1
fi

if [[ -n "${SLURM_SUBMIT_DIR:-}" ]]; then
  if [[ -d "${SLURM_SUBMIT_DIR}/source_code" ]]; then
    REPO_ROOT="${SLURM_SUBMIT_DIR}"
  elif [[ -d "$(cd "${SLURM_SUBMIT_DIR}/.." && pwd)/source_code" ]]; then
    REPO_ROOT="$(cd "${SLURM_SUBMIT_DIR}/.." && pwd)"
  else
    echo "source_code not found near SLURM_SUBMIT_DIR=${SLURM_SUBMIT_DIR}; sbatch from repo root or scripts."
    exit 1
  fi
else
  SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]:-$0}")" && pwd)"
  REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
fi
cd "${REPO_ROOT}/source_code" || { echo "Cannot cd to ${REPO_ROOT}/source_code"; exit 1; }

export LC_ALL=C.UTF-8
export LANG=C.UTF-8
_PP="$(realpath "${PWD}" 2>/dev/null || pwd -P)"
export PYTHONPATH="${PYTHONPATH:+${PYTHONPATH}:}${_PP}"
export WANDB_PROJECT
export WANDB_ENABLED

DATA_CACHE="${SCRATCH_ROOT}/phi4_vanilla/.hf_data"
MODEL_CACHE="${SCRATCH_ROOT}/phi4_vanilla/.hf_cache"
OUTPUT_DIR="${SCRATCH_ROOT}/output/${OUTPUT_DIR_NAME}"
mkdir -p "${DATA_CACHE}" "${MODEL_CACHE}" "${OUTPUT_DIR}"

TRACK_ARGS=()
if [[ "${WANDB_ENABLED}" == "true" ]]; then
  TRACK_ARGS=(--with_tracking --report_to wandb --run_name "${RUN_NAME}" --extra_tb_stats)
else
  TRACK_ARGS=(--run_name "${RUN_NAME}")
fi

accelerate launch --config_file accelerate_configs/1gpu_fp16.yaml run_clm_ddp.py \
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
