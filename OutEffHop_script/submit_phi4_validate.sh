#!/bin/bash
#===============================================================================
# Phi-3 / Phi-4 文本模型 �?validate_clm（走 phi4_attention + replace_attention_modules�?#
# �?mentor 旧脚本（submit_outlier_valid_opt.sh / run.sh）已对齐的部分：
#   conda init+source 回退、PYTHONUNBUFFERED、PYTHONNOUSERSITE、CUDA_VISIBLE_DEVICES�?#   HF_HOME、PYTHONPATH+realpath、SLURM_SUBMIT_DIR 定位仓库（sbatch 拷贝脚本�?spool）�?# mentor 脚本里另�?moose/gcc/cuda-11.4 等，是否加载取决于你节点�?PyTorch；本脚本用多版本 cuda 回退�?# 代码库隐患（导师脚本无法预见）：vutils/softmax_1.py 曾在 import �?torch.empty(..., cuda)，已删除�?#
# 你可能还要改的地方：
#   1) 下面 module load：已�?Quest �?module avail 里「全名」对齐；�?PyTorch �?cu118 请改�?cuda/11.8 那行
#   2) PHI4_MODEL、SCRATCH_ROOT：见下方「用户配置�?#
# 若每�?module avail 都出�?luac: ... spiderT... unexpected symbol�?#   退出登录后执行  rm -rf ~/.cache/lmod  再重�?ssh（损坏的是本�?Lmod 缓存，不是集群）
#
# 模型�?Hub 还是本地？脚本里 PHI4_MODEL 默认�?Hugging Face Hub ID�?#   - 能联网且接受默认小模型：不用改�?#   - 本地已有权重目录：export PHI4_MODEL=/projects/p32013/.../your_model_dir
#     （该目录里应�?config.json、tokenizer 与权重；�?from_pretrained 要求一致）
#
# 提交（在任意目录均可）：
#   sbatch /path/to/OASIS/OutEffHop_script/submit_phi4_validate.sh
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

#--------------- 用户配置（至少改 PHI4_MODEL；建议改 SCRATCH_ROOT�?--------------
PHI4_MODEL="${PHI4_MODEL:-microsoft/Phi-4-mini-instruct}"
SCRATCH_ROOT="${SCRATCH_ROOT:-/scratch/${USER}/residual}"
# 默认 wikitext_2 与「先冒烟」一致；导师式全�?benchmark：export DATASET_SETUP=bookcorpus_and_wiki
DATASET_SETUP="${DATASET_SETUP:-wikitext_2}"
BLOCK_SIZE="${BLOCK_SIZE:-512}"
EVAL_BS="${EVAL_BS:-4}"
ATTN_SOFTMAX="${ATTN_SOFTMAX:-vanilla}"
ATTN_RES_SOFTMAX="${ATTN_RES_SOFTMAX:-vanilla}"
SEED="${SEED:-5678}"
#------------------------------------------------------------------------------

module purge 2>/dev/null || true

# Conda 初始化（�?mentor 脚本一致）。若你登录后已有 conda，可注释掉下一行避免重复�?module load python-miniconda3/4.12.0

# CUDA toolkit：你的环境为 torch 2.7+cu128（自�?CUDA 12.8 运行时），以�?module 主要提供
# CUDA_HOME / nvcc；与 12.8 不必逐位相同。按 Quest �?module avail 选「最新的 12.x」即可�?# 顺序：先�?rhel8 �?12.6，再 12.4，再 rhel7 spack �?12.1（哪个成功用哪个）�?if ! module load cuda/12.6.2-gcc-12.4.0 2>/dev/null; then
  if ! module load cuda/12.4.1-gcc-12.3.0 2>/dev/null; then
    module load cuda/12.1.0-gcc-11.2.0 || true
  fi
fi

# ----- �?mentor �?submit_outlier_valid_opt.sh / run.sh 对齐，减少批处理环境差异 -----
export PYTHONUNBUFFERED=1
export PYTHONNOUSERSITE=1
export CUDA_VISIBLE_DEVICES=0
export HF_HOME="${HF_HOME:-${SCRATCH_ROOT}/.hf_home}"
mkdir -p "${HF_HOME}"

# mentor：conda init + source ~/.bashrc；sbatch 非交互下比单�?hook 更稳
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

# Slurm 把脚本拷�?/var/spool/slurmd/...，不能用 BASH_SOURCE 找仓库；用提交时�?cwd�?if [[ -n "${SLURM_SUBMIT_DIR:-}" ]]; then
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

# 评测：单卡、无混合精度（与 mentor �?validate 脚本一致，数值更稳）
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
