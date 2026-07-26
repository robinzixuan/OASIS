#!/bin/bash
#===============================================================================
# Paired Phi-4 validation for Assumption D.4.
#
# ATTN_RESIDUAL_CHECKPOINT must be the repository's run_clm_oasis.py model
# trained with standard token/depth Softmax (both flags set to "vanilla").
# OASIS/Softmax1 checkpoints are not valid inputs for this D.4 experiment.
#
# Submit from the repository root or OutEffHop_script:
#   ATTN_RESIDUAL_CHECKPOINT=/path/to/ar \
#   VANILLA_CHECKPOINT=/path/to/vanilla \
#   D4_OUTPUT_ROOT=/path/to/output \
#   sbatch OutEffHop_script/submit_d4_validation.sh
#===============================================================================

#SBATCH -A p32013
#SBATCH -p gengpu
#SBATCH --gres=gpu:1
#SBATCH -t 24:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --job-name=phi4-d4
#SBATCH --output=slurm_phi4_d4_%j.out
#SBATCH --error=slurm_phi4_d4_%j.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=onionzsy@umich.edu

set -euo pipefail

#--------------- User config -------------------------------------------------
ATTN_RESIDUAL_CHECKPOINT="${ATTN_RESIDUAL_CHECKPOINT:-/path/to/attn_residual_checkpoint}"
VANILLA_CHECKPOINT="${VANILLA_CHECKPOINT:-/path/to/vanilla_checkpoint}"
D4_OUTPUT_ROOT="${D4_OUTPUT_ROOT:-/path/to/d4_output}"

BASE_MODEL="${BASE_MODEL:-microsoft/Phi-4-mini-instruct}"
SCRATCH_ROOT="${SCRATCH_ROOT:-/scratch/${USER}/residual}"
PYTHON_BIN="${PYTHON_BIN:-python}"
MODEL_CACHE_DIR="${MODEL_CACHE_DIR:-${SCRATCH_ROOT}/phi4_d4/.hf_cache}"
DATA_CACHE_DIR="${DATA_CACHE_DIR:-${SCRATCH_ROOT}/phi4_d4/.hf_data}"

NUM_SAMPLES="${NUM_SAMPLES:-32}"
SEQUENCE_LENGTH="${SEQUENCE_LENGTH:-256}"
BATCH_SIZE="${BATCH_SIZE:-1}"
LAYERS="${LAYERS:-all}"
HEADS="${HEADS:-all}"
DTYPE="${DTYPE:-float16}"
BOOTSTRAP_ITERATIONS="${BOOTSTRAP_ITERATIONS:-10000}"
OVERWRITE="${OVERWRITE:-false}"
#------------------------------------------------------------------------------

for required_path in \
  "${ATTN_RESIDUAL_CHECKPOINT}" \
  "${VANILLA_CHECKPOINT}" \
  "${D4_OUTPUT_ROOT}"; do
  if [[ "${required_path}" == /path/to/* ]]; then
    echo "Set ATTN_RESIDUAL_CHECKPOINT, VANILLA_CHECKPOINT, and D4_OUTPUT_ROOT before submission."
    exit 2
  fi
done

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
cd "${REPO_ROOT}/OutEffHop" || {
  echo "Cannot cd to ${REPO_ROOT}/OutEffHop"
  exit 1
}

export LC_ALL=C.UTF-8
export LANG=C.UTF-8
_PP="$(realpath "${PWD}" 2>/dev/null || pwd -P)"
export PYTHONPATH="${PYTHONPATH:+${PYTHONPATH}:}${_PP}"

mkdir -p "${D4_OUTPUT_ROOT}" "${MODEL_CACHE_DIR}" "${DATA_CACHE_DIR}"

OVERWRITE_ARGS=()
if [[ "${OVERWRITE}" == "true" ]]; then
  OVERWRITE_ARGS=(--overwrite)
fi

COMMON_ARGS=(
  --base-model "${BASE_MODEL}"
  --num-samples "${NUM_SAMPLES}"
  --sequence-length "${SEQUENCE_LENGTH}"
  --batch-size "${BATCH_SIZE}"
  --layers "${LAYERS}"
  --heads "${HEADS}"
  --dtype "${DTYPE}"
  --prepend-bos
  --analysis-variants identity_outcome attention_only_conditional
  --model-cache-dir "${MODEL_CACHE_DIR}"
  --data-cache-dir "${DATA_CACHE_DIR}"
)

"${PYTHON_BIN}" analysis/d4_collect.py \
  --model-kind attn_residual \
  --checkpoint "${ATTN_RESIDUAL_CHECKPOINT}" \
  --output "${D4_OUTPUT_ROOT}/attn_residual.parquet" \
  "${COMMON_ARGS[@]}" \
  "${OVERWRITE_ARGS[@]}"

"${PYTHON_BIN}" analysis/d4_collect.py \
  --model-kind vanilla \
  --checkpoint "${VANILLA_CHECKPOINT}" \
  --output "${D4_OUTPUT_ROOT}/vanilla.parquet" \
  "${COMMON_ARGS[@]}" \
  "${OVERWRITE_ARGS[@]}"

"${PYTHON_BIN}" analysis/d4_report.py \
  --attn-residual "${D4_OUTPUT_ROOT}/attn_residual.parquet" \
  --vanilla "${D4_OUTPUT_ROOT}/vanilla.parquet" \
  --output-dir "${D4_OUTPUT_ROOT}/report" \
  --bootstrap-iterations "${BOOTSTRAP_ITERATIONS}" \
  "${OVERWRITE_ARGS[@]}"

echo "D.4 report: ${D4_OUTPUT_ROOT}/report/d4_report.md"
