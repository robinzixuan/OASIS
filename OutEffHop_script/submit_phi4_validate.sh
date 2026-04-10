#!/bin/bash
#===============================================================================
# Phi-4 validate_clm.py
#
# - Vanilla checkpoints (run_clm_ddp.py): default; same softmax as submit_phi4_vanilla_train.sh
# - OASIS checkpoints (run_clm_oasis.py): export USE_PHI4_OASIS=1 and set CKPT_PATH to checkpoint dir
#
# Defaults: PHI4_HUB=microsoft/Phi-4-mini-instruct
# Submit: sbatch /path/to/OASIS/OutEffHop_script/submit_phi4_validate.sh
#===============================================================================

#SBATCH -A p32013
#SBATCH -p gengpu
#SBATCH --gres=gpu:1
#SBATCH -t 04:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --job-name=phi4-validate
#SBATCH --output=slurm_phi4_validate_%j.out
#SBATCH --error=slurm_phi4_validate_%j.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=onionzsy@umich.edu

set -euo pipefail

#--------------- User config -------------------------------------------------
# Hub id for config + tokenizer (must match training); weights come from CKPT_PATH if USE_PHI4_OASIS=1
PHI4_HUB="${PHI4_HUB:-microsoft/Phi-4-mini-instruct}"
# Single-path eval (Hub or a full HF folder): still PHI4_MODEL for backward compatibility
PHI4_MODEL="${PHI4_MODEL:-$PHI4_HUB}"
USE_PHI4_OASIS="${USE_PHI4_OASIS:-0}"
# Accelerate checkpoint dir (only when USE_PHI4_OASIS=1), e.g. .../checkpoints/checkpoint_200
CKPT_PATH="${CKPT_PATH:-}"
SCRATCH_ROOT="${SCRATCH_ROOT:-/scratch/${USER}/residual}"
CACHE_ROOT="${CACHE_ROOT:-/scratch/${USER}/.cache/residual}"
DATASET_SETUP="${DATASET_SETUP:-wikitext_2}"
# Match submit_phi4_*_train.sh default unless you override (debug runs may use 256)
BLOCK_SIZE="${BLOCK_SIZE:-512}"
EVAL_BS="${EVAL_BS:-1}"
ATTN_SOFTMAX="${ATTN_SOFTMAX:-vanilla}"
ATTN_RES_SOFTMAX="${ATTN_RES_SOFTMAX:-vanilla}"
SEED="${SEED:-5678}"
#------------------------------------------------------------------------------

module purge 2>/dev/null || true

module load python-miniconda3/4.12.0

# Pick a CUDA module matching cluster (torch wheel ships its own runtime).
if ! module load cuda/12.6.2-gcc-12.4.0 2>/dev/null; then
  if ! module load cuda/12.4.1-gcc-12.3.0 2>/dev/null; then
    module load cuda/12.1.0-gcc-11.2.0 || true
  fi
fi

# ----- ??mentor ??submit_outlier_valid_opt.sh / run.sh ???????????? -----
export PYTHONUNBUFFERED=1
export PYTHONNOUSERSITE=1
export CUDA_VISIBLE_DEVICES=0
export XDG_CACHE_HOME="${XDG_CACHE_HOME:-${CACHE_ROOT}}"
export HF_HOME="${HF_HOME:-${CACHE_ROOT}/huggingface}"
export TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-${HF_HOME}/transformers}"
export HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-${HF_HOME}/datasets}"
export TORCH_HOME="${TORCH_HOME:-${XDG_CACHE_HOME}/torch}"
export MPLCONFIGDIR="${MPLCONFIGDIR:-${XDG_CACHE_HOME}/matplotlib}"
export WANDB_DIR="${WANDB_DIR:-${CACHE_ROOT}/wandb}"
mkdir -p "${XDG_CACHE_HOME}" "${HF_HOME}" "${TRANSFORMERS_CACHE}" "${HF_DATASETS_CACHE}" "${TORCH_HOME}" "${MPLCONFIGDIR}" "${WANDB_DIR}"

# mentor?conda init + source ~/.bashrc?sbatch ????????hook ??
set +e
if command -v conda >/dev/null 2>&1; then
  eval "$(conda shell.bash hook 2>/dev/null)" || true
else
  conda init bash 2>/dev/null || true
  # shellcheck disable=SC1090
  [[ -f "${HOME}/.bashrc" ]] && source "${HOME}/.bashrc"
fi
conda activate outlier
_CONDA_RC=$?
set -e
if [[ ${_CONDA_RC} -ne 0 ]]; then
  echo "conda activate outlier failed (exit ${_CONDA_RC})"
  exit 1
fi

# Slurm ??????/var/spool/slurmd/...???? BASH_SOURCE ??????????cwd??if [[ -n "${SLURM_SUBMIT_DIR:-}" ]]; then
  if [[ -d "${SLURM_SUBMIT_DIR}/OutEffHop" ]]; then
    REPO_ROOT="${SLURM_SUBMIT_DIR}"
  elif [[ -d "$(cd "${SLURM_SUBMIT_DIR}/.." && pwd)/OutEffHop" ]]; then
    REPO_ROOT="$(cd "${SLURM_SUBMIT_DIR}/.." && pwd)"
  else
    echo "OutEffHop not found under SLURM_SUBMIT_DIR=${SLURM_SUBMIT_DIR}; run sbatch from repo root (parent of OutEffHop) or from OutEffHop_script."
    exit 1
  fi
else
  SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]:-$0}")" && pwd)"
  REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
fi
cd "${REPO_ROOT}/OutEffHop" || { echo "Cannot cd to ${REPO_ROOT}/OutEffHop"; exit 1; }

export LC_ALL=C.UTF-8
export LANG=C.UTF-8
_PP="$(realpath "${PWD}" 2>/dev/null || pwd -P)"
export PYTHONPATH="${PYTHONPATH:+${PYTHONPATH}:}${_PP}"

DATA_CACHE="${CACHE_ROOT}/phi4_validate/hf_data"
MODEL_CACHE="${CACHE_ROOT}/phi4_validate/hf_cache"
OUT_DIR="${SCRATCH_ROOT}/output_metrics/phi4_validate_${SLURM_JOB_ID:-local}"
mkdir -p "${DATA_CACHE}" "${MODEL_CACHE}" "$(dirname "${OUT_DIR}")"

EXTRA_ARGS=()
if [[ "${USE_PHI4_OASIS}" == "1" ]]; then
  if [[ -z "${CKPT_PATH}" ]]; then
    echo "USE_PHI4_OASIS=1 requires CKPT_PATH=/path/to/checkpoint_xxx"
    exit 1
  fi
  EXTRA_ARGS+=(--phi4_oasis --config_name "${PHI4_HUB}" --tokenizer_name "${PHI4_HUB}" --model_name_or_path "${CKPT_PATH}")
else
  EXTRA_ARGS+=(--model_name_or_path "${PHI4_MODEL}")
fi

accelerate launch --config_file accelerate_configs/1gpu_no_mp.yaml validate_clm.py \
  "${EXTRA_ARGS[@]}" \
  --seed "${SEED}" \
  --dataset_setup "${DATASET_SETUP}" \
  --preprocessing_num_workers 8 \
  --block_size "${BLOCK_SIZE}" \
  --per_device_eval_batch_size "${EVAL_BS}" \
  --attn_softmax "${ATTN_SOFTMAX}" \
  --attn_res_softmax_fn "${ATTN_RES_SOFTMAX}" \
  --data_cache_dir "${DATA_CACHE}" \
  --model_cache_dir "${MODEL_CACHE}" \
  --output_dir "${OUT_DIR}"

echo "Done. Metrics under: ${OUT_DIR}"
