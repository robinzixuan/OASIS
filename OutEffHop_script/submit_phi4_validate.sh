#!/bin/bash
#===============================================================================
# Phi-4 ? validate_clm.py (phi4_attention + replace_attention_modules)
#
# Defaults: PHI4_MODEL=microsoft/Phi-4-mini-instruct; override with export PHI4_MODEL=/path/to/local_ckpt
# Submit: sbatch /path/to/OASIS/OutEffHop_script/submit_phi4_validate.sh
#===============================================================================

#SBATCH -A xxxx
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
PHI4_MODEL="${PHI4_MODEL:-microsoft/Phi-4-mini-instruct}"
SCRATCH_ROOT="${SCRATCH_ROOT:-/scratch/${USER}/residual}"
DATASET_SETUP="${DATASET_SETUP:-wikitext_2}"
BLOCK_SIZE="${BLOCK_SIZE:-512}"
EVAL_BS="${EVAL_BS:-4}"
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
export HF_HOME="${HF_HOME:-${SCRATCH_ROOT}/.hf_home}"
mkdir -p "${HF_HOME}"

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

DATA_CACHE="${SCRATCH_ROOT}/.hf_data"
MODEL_CACHE="${SCRATCH_ROOT}/.hf_cache"
OUT_DIR="${SCRATCH_ROOT}/output_metrics/phi4_validate_${SLURM_JOB_ID:-local}"
mkdir -p "${DATA_CACHE}" "${MODEL_CACHE}" "$(dirname "${OUT_DIR}")"

# ????????????? mentor ??validate ??????????
accelerate launch --config_file accelerate_configs/1gpu_no_mp.yaml validate_clm.py \
  --seed "${SEED}" \
  --dataset_setup "${DATASET_SETUP}" \
  --preprocessing_num_workers 8 \
  --block_size "${BLOCK_SIZE}" \
  --per_device_eval_batch_size "${EVAL_BS}" \
  --attn_softmax "${ATTN_SOFTMAX}" \
  --attn_res_softmax_fn "${ATTN_RES_SOFTMAX}" \
  --data_cache_dir "${DATA_CACHE}" \
  --model_cache_dir "${MODEL_CACHE}" \
  --model_name_or_path "${PHI4_MODEL}" \
  --output_dir "${OUT_DIR}"

echo "Done. Metrics under: ${OUT_DIR}"
